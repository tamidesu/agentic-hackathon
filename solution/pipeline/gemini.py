"""Провайдер Gemini и адаптер схем.

Подключается к той же абстракции `Provider`, что и Anthropic: пайплайн о
смене поставщика не знает, а имя модели входит в ключ кэша, поэтому
результаты обоих можно держать рядом и сравнивать.

ПОЧЕМУ НУЖЕН АДАПТЕР СХЕМ

Схемы написаны в обычном JSON Schema под forced tool use Anthropic. Gemini
принимает JSON Schema через `response_json_schema`, но поддерживает не всё.
Проверено на установленном SDK:

  * `$defs` и `$ref` поддерживаются — рекурсивное дерево выражений ковенанта
    передаётся как есть. Оговорка из документации: циклические ссылки
    «may only be used within non-required properties», поэтому рекурсивное
    поле обязано быть необязательным;
  * `type: ["string", "null"]` НЕ поддерживается — переписывается в
    `anyOf: [{type: "string"}, {type: "null"}]`;
  * `enum` допустим только для строк и чисел — `null` из перечисления
    выносится в `anyOf`;
  * `additionalProperties`, `minimum`, `maximum`, `items`, `required`
    проходят без изменений.

Адаптер не «чинит на всякий случай», а выполняет ровно эти замены: чем
меньше расхождение между тем, что видит модель, и тем, чем мы проверяем
ответ, тем меньше поводов для тихой ошибки.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Sequence

from .llm import DEFAULT_GEMINI_MODEL, LLMError, LLMRequest, Provider

log = logging.getLogger(__name__)

#: Модели с бесплатным тарифом на момент написания. Список нужен не для
#: логики, а чтобы предупредить: на бесплатном тарифе данные используются
#: для улучшения продуктов поставщика.
FREE_TIER_MODELS = (
    "gemini-3-flash-preview",
    "gemini-3.5-flash",
    "gemini-3.6-flash",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
)

#: DEFAULT_GEMINI_MODEL определён в llm.py и импортирован выше: там он нужен
#: клиенту при загрузке, а держать две копии значит однажды их разъехать.

#: Потолок бюджета вывода при эскалации после обрыва ответа.
MAX_OUTPUT_TOKENS = 32_000

#: Столько параллельных веток выдерживает бесплатный тариф на ОДИН ключ.
#: Шесть веток на один ключ дали сплошные 429 в первом боевом прогоне.
FREE_TIER_WORKERS_PER_KEY = 2

#: Минимальный интервал между запросами К ОДНОМУ КЛЮЧУ, секунды.
#:
#: ПОЧЕМУ САМООГРАНИЧЕНИЕ, А НЕ ПРОСТО РЕТРАИ. Ретрай после 429 — это
#: обнаружение лимита постфактум: запрос уже отправлен, отвергнут, и
#: сверху накинута экспоненциальная задержка. На двенадцати вызовах это
#: терпимо, на шести сотнях (шаг 10) очередь ретраев растёт быстрее,
#: чем разгребается, и прогон занимает не минуты, а десятки минут.
#: Дешевле не превышать лимит, чем узнавать о нём от сервера.
#:
#: Значение подобрано под бесплатный тариф Gemini Flash (порядка 10
#: запросов в минуту на ключ) с запасом. Переопределяется переменной
#: GEMINI_MIN_INTERVAL_S — на платном тарифе ограничение не нужно.
FREE_TIER_MIN_INTERVAL_S = 6.5


# --------------------------------------------------------------------------- #
# Адаптер схем
# --------------------------------------------------------------------------- #


def _split_nullable(types_: list) -> tuple[list, bool]:
    rest = [t for t in types_ if t != "null"]
    return rest, len(rest) != len(types_)


def adapt_schema(node: Any) -> Any:
    """JSON Schema → подмножество, понятное Gemini.

    Обходит дерево целиком; `$defs` и `$ref` сохраняются, чтобы рекурсия
    продолжала работать.
    """
    if isinstance(node, list):
        return [adapt_schema(x) for x in node]
    if not isinstance(node, dict):
        return node

    out: dict[str, Any] = {}
    nullable = False

    for key, value in node.items():
        if key == "type" and isinstance(value, list):
            rest, has_null = _split_nullable(value)
            nullable = nullable or has_null
            if len(rest) == 1:
                out["type"] = rest[0]
            elif rest:
                out["anyOf"] = [{"type": t} for t in rest]
            continue
        if key == "enum" and isinstance(value, list) and any(v is None for v in value):
            cleaned = [v for v in value if v is not None]
            nullable = True
            if cleaned:
                out["enum"] = cleaned
            continue
        out[key] = adapt_schema(value)

    if nullable:
        # Явного `nullable` в поддерживаемом наборе нет — выражаем через anyOf.
        if "anyOf" in out:
            out["anyOf"] = list(out["anyOf"]) + [{"type": "null"}]
        else:
            inner = {k: v for k, v in out.items()
                     if k in {"type", "enum", "items", "properties", "required",
                              "additionalProperties", "minimum", "maximum",
                              "minItems", "maxItems", "format", "$ref"}}
            keep = {k: v for k, v in out.items() if k not in inner}
            out = {**keep, "anyOf": [inner, {"type": "null"}]} if inner else keep
    return out


def _ref_name(ref: str) -> str:
    return ref.rsplit("/", 1)[-1]


def recursive_refs_are_optional(schema: dict) -> list[str]:
    """Проверяет оговорку документации про циклические ссылки.

    «Cyclic references … may only be used within non-required properties.»

    Ищется НАСТОЯЩИЙ цикл по обязательным полям: определение, которое
    через цепочку обязательных свойств ссылается само на себя. Такую
    рекурсию развернуть нельзя — она бесконечна.

    Ссылка на рекурсивное определение из обязательного поля нарушением
    НЕ является: например, `metric` обязателен и указывает на `node`,
    но сам цикл замыкается через необязательное `node.args`, и глубина
    конечна. Первая версия проверки помечала и такой случай — она
    ловила любую ссылку в обязательном поле, а не цикл.
    """
    defs: dict[str, Any] = {}
    for key in ("$defs", "definitions"):
        defs.update(schema.get(key) or {})
    if not defs:
        return []

    problems: list[str] = []

    def required_refs(node: Any, seen_depth: int = 0) -> set[str]:
        """Определения, достижимые ТОЛЬКО через обязательные свойства."""
        if seen_depth > 12 or not isinstance(node, dict):
            return set()
        if "$ref" in node:
            return {_ref_name(node["$ref"])}
        out: set[str] = set()
        required = set(node.get("required") or ())
        for name, sub in (node.get("properties") or {}).items():
            if name in required:
                out |= required_refs(sub, seen_depth + 1)
        # Элементы массива обязательны только если обязателен сам массив,
        # что уже проверено выше при спуске в свойство.
        if "items" in node and node.get("_required_array"):
            out |= required_refs(node["items"], seen_depth + 1)
        return out

    for name, definition in defs.items():
        reachable, frontier = set(), required_refs(definition)
        while frontier:
            current = frontier.pop()
            if current in reachable:
                continue
            reachable.add(current)
            frontier |= required_refs(defs.get(current, {}))
        if name in reachable:
            problems.append(
                f"$defs.{name}: цикл по обязательным свойствам — Gemini "
                f"разворачивает рекурсию только через необязательные, "
                f"сделайте рекурсивное поле необязательным"
            )
    return problems


# --------------------------------------------------------------------------- #
# Провайдер
# --------------------------------------------------------------------------- #


class DailyQuotaExhausted(LLMError):
    """Исчерпана СУТОЧНАЯ квота — ждать бессмысленно.

    Отличать от лимита частоты обязательно. Лимит в минуту рассасывается
    за секунды, и повтор с задержкой — правильное поведение. Суточная
    квота не рассосётся до полуночи по тихоокеанскому времени, и пять
    кругов экспоненциальной задержки — это полторы минуты, потраченные
    впустую, да ещё и в конце всё равно провал. В трёхчасовом окне такая
    ошибка обязана всплывать немедленно и с понятным указанием, что делать.
    """


class TruncatedResponse(LLMError):
    """Ответ оборван по лимиту вывода.

    Отдельный класс, а не JSONDecodeError, потому что причина другая и
    лечится иначе: не «модель ошиблась», а «не хватило бюджета вывода».
    Повтор с бо́льшим лимитом осмыслен, повтор с тем же — нет.
    """


class GeminiProvider(Provider):
    """Провайдер Gemini.

    ДВА ПОДВОДНЫХ КАМНЯ, НА КОТОРЫХ РАЗВАЛИЛСЯ ПЕРВЫЙ БОЕВОЙ ПРОГОН

    1. РАЗМЫШЛЕНИЯ ЕДЯТ БЮДЖЕТ ВЫВОДА. Gemini 3 по умолчанию думает, и
       мысли считаются в `max_output_tokens` вместе с ответом. При лимите
       в 8000 токенов модель тратила почти всё на размышления, а JSON
       обрывался на середине строки — снаружи это выглядело как
       «Unterminated string», то есть как ошибка формата, хотя причина
       совсем другая. Поэтому уровень размышлений задаётся явно.

    2. НЕСКОЛЬКО КЛЮЧЕЙ ПРОТИВ ЛИМИТА ЧАСТОТЫ. На бесплатном тарифе
       лимит запросов в минуту низкий, и шесть параллельных веток
       упираются в него мгновенно. Ключи перебираются по кругу: это
       умножает доступную частоту на их число, ничего не меняя в остальном
       пайплайне.
    """

    name = "gemini"

    def __init__(
        self,
        api_key: str | None = None,
        timeout_s: float = 180.0,
        thinking_level: str | None = "low",
    ):
        import itertools
        import os
        import threading

        from google import genai
        from google.genai import errors, types

        self._genai, self._errors, self._types = genai, errors, types
        self.thinking_level = thinking_level

        # GEMINI_API_KEYS — несколько ключей через запятую; GEMINI_API_KEY —
        # один. Ключи читаются ТОЛЬКО из окружения и никуда не пишутся.
        raw = (api_key
               or os.environ.get("GEMINI_API_KEYS")
               or os.environ.get("GEMINI_API_KEY")
               or os.environ.get("GOOGLE_API_KEY")
               or "")
        keys = [k.strip() for k in raw.split(",") if k.strip()]
        if not keys:
            raise LLMError(
                "GEMINI_API_KEY не задан. Получите ключ в Google AI Studio "
                "или запускайте с LLM_MODE=mock."
            )

        self._clients = [
            genai.Client(api_key=k, http_options=types.HttpOptions(timeout=int(timeout_s * 1000)))
            for k in keys
        ]
        self.n_keys = len(self._clients)
        self._cycle = itertools.cycle(range(self.n_keys))
        self._lock = threading.Lock()

        #: Ключи, чей ПРОЕКТ исчерпал суточную квоту. Выбывают из круга
        #: до конца прогона: квота не рассосётся, а каждый следующий
        #: запрос к такому ключу — это гарантированный отказ, потраченное
        #: время и ещё одна ветка, упавшая на ровном месте.
        #:
        #: Раньше ключи перебирались вслепую, и один исчерпанный проект
        #: отравлял треть всех вызовов.
        self._retired: set[int] = set()

        env_interval = os.environ.get("GEMINI_MIN_INTERVAL_S")
        self.min_interval_s = (
            float(env_interval) if env_interval is not None else FREE_TIER_MIN_INTERVAL_S
        )
        #: Время последнего запроса по каждому ключу — основа самоограничения.
        self._last_call = [0.0] * self.n_keys

    @property
    def _client(self):
        """Совместимость со старым кодом и тестами: первый клиент."""
        return self._clients[0]

    def available_models(self) -> list[str]:
        """Модели, реально доступные ЭТОМУ ключу, для генерации текста.

        ЗАЧЕМ СПРАШИВАТЬ, А НЕ ЗНАТЬ. Имена моделей и их доступность
        меняются без нашего участия: `gemini-2.5-flash` вернул 404
        «no longer available to new users» — при том, что модель
        существует и документирована. Захардкоженное имя — это отложенный
        отказ, который сработает в самый неудобный момент. В трёхчасовом
        боевом окне выяснять это опытным путём непозволительно.
        """
        out: list[str] = []
        for model in self._clients[0].models.list():
            actions = getattr(model, "supported_actions", None) or []
            if actions and "generateContent" not in actions:
                continue
            name = (getattr(model, "name", "") or "").removeprefix("models/")
            if name:
                out.append(name)
        return sorted(out)

    def retire_key(self, index: int, reason: str) -> bool:
        """Выводит ключ из круга. Возвращает True, если остались живые.

        Последний ключ НЕ выводится: тогда работать станет нечем, а
        осмысленное сообщение об исчерпании квоты полезнее, чем ошибка
        «нет доступных ключей».
        """
        with self._lock:
            if len(self._retired) >= self.n_keys - 1:
                return False
            self._retired.add(index)
            alive = self.n_keys - len(self._retired)
        log.warning("Ключ %d выведен из круга (%s); осталось %d", index, reason, alive)
        return True

    @property
    def live_keys(self) -> int:
        return self.n_keys - len(self._retired)

    def _pick_key(self) -> int:
        """Номер следующего живого ключа, с выдержкой паузы под лимит частоты.

        Пауза выдерживается ВНЕ блокировки: держать её во время сна значило
        бы выстроить все ветки в одну очередь и свести параллелизм к нулю.
        Под блокировкой только выбор ключа и отметка времени — то есть
        решение «когда этой ветке можно», а само ожидание идёт параллельно.
        """
        import time

        with self._lock:  # itertools.cycle не потокобезопасен
            idx = 0
            if self.n_keys > 1:
                # Пропускаем выбывшие. Круг конечен, поэтому обход
                # ограничен числом ключей: бесконечного цикла не выйдет
                # даже если выбыли все.
                for _ in range(self.n_keys):
                    idx = next(self._cycle)
                    if idx not in self._retired:
                        break
            if self.min_interval_s > 0:
                earliest = self._last_call[idx] + self.min_interval_s
                wait = max(0.0, earliest - time.monotonic())
                # Отметка ставится авансом: следующая ветка, выбравшая тот же
                # ключ, отсчитает свою паузу от нашей, а не от общего прошлого.
                self._last_call[idx] = max(time.monotonic(), earliest)
            else:
                wait = 0.0

        if wait > 0:
            log.debug("Ключ %d: жду %.1fс до следующего запроса", idx, wait)
            time.sleep(wait)
        return idx

    def _next_client_with_index(self):
        """Клиент и НОМЕР его ключа.

        Номер берётся из самого перебора, а не поиском по списку: искать
        клиент в списке значит выводить из круга «похожий» ключ вместо
        того, который упёрся в квоту. Заодно это делало подмену клиента
        в тестах невозможной.
        """
        idx = self._pick_key()
        return self._clients[idx], idx

    def _next_client(self):
        """Только клиент — для мест, где номер ключа не нужен."""
        return self._clients[self._pick_key()]

    def call(self, req: LLMRequest, extra_user_turns: Sequence[dict] = ()) -> tuple[dict, int, int]:
        types = self._types

        parts = [
            types.Part.from_bytes(data=__import__("base64").b64decode(b64), mime_type=mt)
            for mt, b64 in req.images
        ]
        parts.append(types.Part.from_text(text=req.prompt))
        contents = [types.Content(role="user", parts=parts)]
        for turn in extra_user_turns:
            contents.append(types.Content(
                role="model" if turn.get("role") == "assistant" else "user",
                parts=[types.Part.from_text(text=str(turn.get("content", "")))],
            ))

        # Обрыв по лимиту лечится увеличением лимита, а не повтором того же
        # запроса. Поэтому бюджет поднимается прямо здесь, до того как
        # ошибка уйдёт наверх: это дешевле круга ретраев с задержками.
        budget = req.max_tokens
        last: BaseException | None = None
        for attempt in range(3):
            config = types.GenerateContentConfig(
                temperature=req.temperature,
                max_output_tokens=budget,
                response_mime_type="application/json",
                response_json_schema=adapt_schema(req.schema),
                system_instruction=req.system or None,
                thinking_config=self._thinking_config(req.model),
            )
            client, key_index = self._next_client_with_index()
            try:
                response = client.models.generate_content(
                    model=req.model, contents=contents, config=config
                )
            except Exception as exc:  # noqa: BLE001
                if "no longer available" in str(exc) or "NOT_FOUND" in str(exc):
                    raise LLMError(
                        f"модель {req.model} недоступна этому ключу: {str(exc)[:200]}. "
                        f"Список доступных: scripts/list_models.py"
                    ) from exc
                if self._is_daily_quota(exc):
                    # Квота считается на ПРОЕКТ. Если ключей несколько и
                    # они из разных проектов, соседний может быть свеж —
                    # выводим исчерпанный и пробуем дальше.
                    if self.retire_key(key_index, f"суточная квота {req.model}"):
                        continue
                    raise DailyQuotaExhausted(
                        f"исчерпана СУТОЧНАЯ квота бесплатного тарифа для модели "
                        f"{req.model}. Ждать до полуночи по тихоокеанскому времени "
                        f"бессмысленно в боевом окне. Варианты: взять стабильную "
                        f"модель вместо preview (у preview лимиты жёстче), включить "
                        f"оплату в AI Studio, либо использовать ключ из другого "
                        f"проекта — квота считается НА ПРОЕКТ, а не на ключ. "
                        f"Исходная ошибка: {str(exc)[:200]}"
                    ) from exc
                raise
            try:
                return self._parse(response, budget)
            except TruncatedResponse as exc:
                last = exc
                budget = min(budget * 3, MAX_OUTPUT_TOKENS)
                log.warning("Ответ оборван, поднимаю лимит вывода до %d: %s", budget, exc)
                if budget >= MAX_OUTPUT_TOKENS and attempt:
                    break
        raise last if last else LLMError("Gemini не вернул ответа")

    def _parse(self, response, budget: int) -> tuple[dict, int, int]:

        usage = getattr(response, "usage_metadata", None)
        thoughts = getattr(usage, "thoughts_token_count", 0) or 0
        finish = self._finish_reason(response)

        text = (response.text or "").strip()
        if finish == "MAX_TOKENS":
            # Диагноз, а не догадка: лимит вывода исчерпан. Сообщаем, сколько
            # ушло на размышления — без этого числа причина неочевидна.
            raise TruncatedResponse(
                f"ответ оборван по лимиту вывода: max_output_tokens={budget}, "
                f"на размышления ушло {thoughts} токенов, получено {len(text)} знаков. "
                f"Поднимите max_tokens или снизьте уровень размышлений"
            )
        if not text:
            raise LLMError(f"Gemini вернул пустой ответ (finish_reason={finish})")
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            # response_mime_type=application/json обещает валидный JSON,
            # но обещание не гарантия — обрабатываем как ошибку вызова.
            # Повторяем: обрыв обычно не воспроизводится дважды подряд.
            raise TruncatedResponse(
                f"Gemini вернул невалидный JSON (finish_reason={finish}, "
                f"размышления {thoughts} токенов): {exc}; начало: {text[:160]!r}"
            )

        return (
            payload,
            getattr(usage, "prompt_token_count", 0) or 0,
            getattr(usage, "candidates_token_count", 0) or 0,
        )

    def _thinking_config(self, model: str = ""):
        """Размышления считаются в бюджет вывода вместе с ответом.

        Задача извлечения — перевод формулировки в структуру, а не
        рассуждение: длинная цепочка мыслей здесь не улучшает ответ, зато
        съедает бюджет и обрывает JSON на середине строки.

        СПОСОБ ЗАДАНИЯ РАЗНЫЙ У ПОКОЛЕНИЙ. Модели 2.5 принимают числовой
        `thinking_budget`, и ноль отключает размышления полностью. Модели 3
        отключить нельзя — у них `thinking_level`, минимум «low». Отправить
        не то поле не то модели значит получить 400 на каждом вызове,
        поэтому выбор делается по имени модели, а не наугад.
        """
        if not self.thinking_level:
            return None
        types = self._types
        name = (model or "").lower()
        if name.startswith("gemini-2"):
            return types.ThinkingConfig(thinking_budget=0)
        return types.ThinkingConfig(thinking_level=self.thinking_level)

    @staticmethod
    def _finish_reason(response) -> str:
        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            return "NO_CANDIDATES"
        reason = getattr(candidates[0], "finish_reason", None)
        return getattr(reason, "name", None) or str(reason)

    @staticmethod
    def _is_daily_quota(exc: BaseException) -> bool:
        """Суточная квота или лимит частоты? Оба приходят как 429."""
        text = str(exc)
        return "PerDay" in text or "GenerateRequestsPerDay" in text

    def retryable(self, exc: BaseException) -> bool:
        errors = self._errors
        if isinstance(exc, DailyQuotaExhausted):
            return False
        if isinstance(exc, TruncatedResponse):
            # Обрыв ответа воспроизводится не всегда: повтор дешевле,
            # чем потерянный заёмщик.
            return True
        if isinstance(exc, errors.ServerError):
            return True
        if isinstance(exc, errors.ClientError):
            # 429 RESOURCE_EXHAUSTED бывает двух видов. Лимит частоты —
            # повтор осмыслен. Суточная квота — нет.
            if getattr(exc, "code", None) == 429:
                return not self._is_daily_quota(exc)
            return False
        status = getattr(exc, "code", None) or getattr(exc, "status_code", None)
        return status in {408, 429, 500, 502, 503, 504}


#: Модели, непригодные для структурного извлечения текста.
_SPECIALISED = ("tts", "image", "embedding", "live", "robotics", "veo",
                "lyria", "banana", "omni", "audio", "vision-")


def rank_model(name: str) -> tuple:
    """Насколько модель пригодна для извлечения. Больше — лучше.

    Порядок предпочтений выведен из ограничений задачи, а не из
    «новее значит лучше»:

      1. НЕ preview. Не из-за качества, а из-за квоты: у preview-моделей
         бесплатный тариф даёт 20 запросов в сутки на проект;
      2. flash, а не pro. Задача — перевод формулировки в структуру,
         а не рассуждение; pro дороже и медленнее без выигрыша по сути;
      3. более новое поколение. Поколение важнее «полноты» версии:
         свежая lite-модель разбирает юридический текст лучше, чем
         полная модель на два поколения старше;
      4. НЕ lite при равном поколении — качество разбора выше;
      5. без узкой специализации (tts, image, embedding, live, robotics).
    """
    lowered = name.lower()

    version = 0.0
    match = re.search(r"gemini-(\d+(?:\.\d+)?)", lowered)
    if match:
        version = float(match.group(1))

    return (
        0 if any(x in lowered for x in _SPECIALISED) else 1,
        0 if "preview" in lowered or "-exp" in lowered else 1,
        1 if "flash" in lowered else 0,
        version,
        0 if "lite" in lowered else 1,
    )


def resolve_model(preferred: str | None, available: Sequence[str]) -> tuple[str, list[str]]:
    """Выбирает модель: желаемую, если она есть, иначе лучшую доступную.

    Возвращает (имя, пояснения). Пояснения не для красоты: подмена модели
    меняет и качество, и стоимость, поэтому она обязана быть видимой,
    а не происходить втихую.
    """
    notes: list[str] = []
    catalogue = list(available)
    if not catalogue:
        return (preferred or DEFAULT_GEMINI_MODEL), ["список моделей пуст — беру заявленную"]

    if preferred and preferred in catalogue:
        return preferred, notes

    if preferred:
        notes.append(f"модель {preferred} недоступна этому ключу")

    best = max(catalogue, key=rank_model)
    notes.append(f"выбрана {best} из {len(catalogue)} доступных")
    if "lite" in best.lower():
        notes.append("это lite-версия: лимиты щедрее, качество разбора ниже")
    if "preview" in best.lower():
        notes.append(
            "это preview: бесплатный тариф даёт лишь 20 запросов в сутки на проект"
        )
    return best, notes


def verify_model(
    provider: "GeminiProvider",
    preferred: str | None = None,
    catalogue: Sequence[str] | None = None,
    max_tries: int = 4,
) -> tuple[str, list[str]]:
    """Подбирает модель и ПРОВЕРЯЕТ её настоящим вызовом.

    ПОЧЕМУ НЕДОСТАТОЧНО СПИСКА МОДЕЛЕЙ. Каталог ключа перечислял
    `gemini-2.5-flash`, а генерация по ней отвечала 404 «no longer
    available to new users». То есть листинг и генерация расходятся:
    список говорит о существовании модели, а не о праве её вызывать.
    Единственная надёжная проверка — вызвать.

    ПОЧЕМУ ЗДЕСЬ, А НЕ ВНУТРИ `call`. Подставлять другую модель молча
    посреди прогона нельзя: имя модели входит в ключ кэша, и тогда под
    ключом одной модели лежал бы ответ другой — воспроизводимость
    сломалась бы незаметно. Поэтому модель фиксируется ДО работы, один
    раз, ценой одного крошечного вызова.
    """
    notes: list[str] = []
    try:
        models = list(catalogue if catalogue is not None else provider.available_models())
    except Exception as exc:  # noqa: BLE001
        return (preferred or DEFAULT_GEMINI_MODEL), [
            f"каталог недоступен ({type(exc).__name__}) — беру заявленную модель"
        ]

    candidates: list[str] = []
    if preferred:
        candidates.append(preferred)
    candidates += sorted((m for m in models if m != preferred), key=rank_model, reverse=True)

    probe = {"type": "object", "required": ["ok"],
             "properties": {"ok": {"type": "boolean"}}, "additionalProperties": False}

    for name in candidates[:max_tries]:
        try:
            provider.call(LLMRequest(
                prompt='Верни ровно {"ok": true}.', schema=probe,
                model=name, max_tokens=2000,
            ))
        except Exception as exc:  # noqa: BLE001
            reason = str(exc)
            if "no longer available" in reason or "NOT_FOUND" in reason or "404" in reason:
                notes.append(f"{name}: числится в каталоге, но вызвать нельзя")
                continue
            if provider._is_daily_quota(exc):
                notes.append(f"{name}: суточная квота исчерпана")
                continue
            # Иная ошибка — не повод перебирать модели дальше: она,
            # скорее всего, повторится на любой.
            notes.append(f"{name}: проверка не прошла ({type(exc).__name__}), беру как есть")
            return name, notes
        if name != preferred:
            notes.append(f"вместо {preferred} выбрана {name}")
        if "lite" in name:
            notes.append("это lite-версия: лимиты щедрее, качество разбора ниже")
        if "preview" in name:
            notes.append("это preview: 20 запросов в сутки на проект")
        return name, notes

    notes.append("ни одна модель не ответила — беру заявленную и надеюсь на ретраи")
    return (preferred or DEFAULT_GEMINI_MODEL), notes


def warn_about_free_tier(model: str) -> str | None:
    """На бесплатном тарифе данные идут в обучение — это стоит знать явно."""
    if any(model.startswith(m) for m in FREE_TIER_MODELS):
        return (
            f"модель {model} доступна на бесплатном тарифе, где данные "
            f"используются для улучшения продуктов поставщика; на платном — нет"
        )
    return None
