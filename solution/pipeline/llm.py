"""Инфраструктура LLM-вызовов (шаг 0 плана).

Единственная точка контакта пайплайна с моделью. Ни один шаг не вызывает
API напрямую.

Гарантии:
  * structured output — ответ всегда валиден по JSON Schema (forced tool use);
  * содержательная валидация — схема проверяет форму, validator проверяет смысл;
  * контентно-адресуемый кэш — повторный прогон не делает сетевых вызовов
    и даёт побитово тот же результат; прогон резюмируем после падения;
  * ретраи с экспоненциальным бэкоффом на транспортных ошибках;
  * repair-петля на содержательных ошибках (модели показывают, что не так);
  * учёт расхода токенов и полный JSONL-лог всех вызовов.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

log = logging.getLogger(__name__)

# Меняется вручную, когда правки в промптах/схемах должны инвалидировать кэш.
CACHE_EPOCH = "1"

# Модель по умолчанию. Переопределяется переменной окружения LLM_MODEL, чтобы
# эксперимент со сменой модели на шаге 10 не требовал правки кода.
# Стоимость прогона (~264k входных токенов) не является фактором выбора —
# ставим сильнейшую доступную модель.
#: Модель Gemini по умолчанию. Живёт ЗДЕСЬ, а не в gemini.py, потому что
#: gemini.py импортирует этот модуль: обратная ссылка дала бы цикл при
#: загрузке. Дублировать значение в двух местах нельзя — копии разъезжаются,
#: и умолчание клиента начинает отличаться от умолчания скриптов.
#:
#: Не preview: у preview-моделей бесплатный тариф даёт 20 запросов в СУТКИ
#: на проект. Значение — лишь отправная точка, настоящий выбор делает
#: gemini.verify_model, проверяя модель настоящим вызовом.
DEFAULT_GEMINI_MODEL = "gemini-3.6-flash"


def _default_model() -> str:
    """Модель по умолчанию зависит от провайдера: имена не пересекаются,
    и подстановка чужого имени даёт 404 в самый неподходящий момент."""
    explicit = os.environ.get("LLM_MODEL")
    if explicit:
        return explicit
    if os.environ.get("LLM_MODE", "").lower() in {"gemini", "google"}:
        return DEFAULT_GEMINI_MODEL
    return "claude-opus-5"


#: ВНИМАНИЕ: значение вычисляется ОДИН РАЗ при импорте модуля, то есть до
#: того, как скрипт успевает выставить LLM_MODE. Поэтому оно НЕ должно
#: попадать в запросы как умолчание — именно так извлечение сканов ушло
#: к `claude-opus-5` на Gemini-ключе и получило 404.
#:
#: Запросы оставляют `model` ПУСТЫМ, а подставляет его клиент — уже
#: проверенной моделью. Константа остаётся только для обратной
#: совместимости и диагностики.
DEFAULT_MODEL = _default_model()
DEFAULT_MAX_TOKENS = 8000
DEFAULT_TIMEOUT_S = 180.0

# Модели, пригодные для шага 10, где объём вызовов оправдывает эксперимент.
KNOWN_MODELS = (
    "claude-opus-5", "claude-sonnet-5", "claude-haiku-4-5-20251001", "claude-fable-5",
    "gemini-3-flash-preview", "gemini-3.5-flash", "gemini-3.6-flash",
    "gemini-3.1-pro-preview", "gemini-2.5-flash", "gemini-2.5-flash-lite",
)

# Валидатор возвращает список претензий. Пустой список = всё в порядке.
Validator = Callable[[dict], list[str]]


class LLMError(RuntimeError):
    pass


class ValidationFailed(LLMError):
    def __init__(self, problems: list[str], last_payload: dict | None):
        super().__init__("; ".join(problems))
        self.problems = problems
        self.last_payload = last_payload


@dataclass(frozen=True)
class LLMRequest:
    """Всё, что влияет на ответ, — и всё, что входит в ключ кэша."""

    prompt: str
    schema: dict
    system: str | None = None
    #: Пустая строка означает «подставит клиент». Жёсткое умолчание здесь
    #: замораживало бы имя модели на момент импорта модуля.
    model: str = ""
    max_tokens: int = DEFAULT_MAX_TOKENS
    temperature: float = 0.0
    # Метка версии промпта. Правишь промпт — меняй метку, иначе кэш отдаст старое.
    prompt_version: str = "v1"
    # Изображения страниц для сканов: (media_type, base64). Идут перед текстом
    # промпта — модели легче рассуждать, когда картинка уже в контексте.
    images: tuple[tuple[str, str], ...] = ()

    def cache_key(self) -> str:
        payload = json.dumps(
            {
                "epoch": CACHE_EPOCH,
                "prompt": self.prompt,
                "schema": self.schema,
                "system": self.system,
                "model": self.model,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "prompt_version": self.prompt_version,
                # Хэш, а не сами байты: ключ должен оставаться коротким.
                "images": [
                    (mt, hashlib.sha256(b64.encode("ascii")).hexdigest())
                    for mt, b64 in self.images
                ],
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class LLMResult:
    data: dict
    cached: bool = False
    attempts: int = 1
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    model: str = DEFAULT_MODEL


@dataclass
class Usage:
    calls: int = 0
    cache_hits: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    repairs: int = 0
    retries: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, r: LLMResult, repairs: int = 0, retries: int = 0) -> None:
        with self._lock:
            self.calls += 1
            if r.cached:
                self.cache_hits += 1
            self.input_tokens += r.input_tokens
            self.output_tokens += r.output_tokens
            self.repairs += repairs
            self.retries += retries

    def as_dict(self) -> dict:
        return {
            "calls": self.calls,
            "cache_hits": self.cache_hits,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "repairs": self.repairs,
            "retries": self.retries,
        }


# --------------------------------------------------------------------------- #
# Провайдеры
# --------------------------------------------------------------------------- #


class Provider:
    """Транспорт до модели. Возвращает (payload, in_tokens, out_tokens)."""

    name = "base"

    def call(self, req: LLMRequest, extra_user_turns: Sequence[dict] = ()) -> tuple[dict, int, int]:
        raise NotImplementedError

    def retryable(self, exc: BaseException) -> bool:
        """Транспортная ли ошибка. Решение о повторе принимает провайдер,
        потому что только он знает исключения своего SDK."""
        return False


class AnthropicProvider(Provider):
    name = "anthropic"

    # Имя инструмента, через который модель обязана вернуть структуру.
    TOOL_NAME = "emit"

    def __init__(self, api_key: str | None = None, timeout: float = DEFAULT_TIMEOUT_S):
        import anthropic  # импорт внутри, чтобы mock-режим работал без SDK

        self._anthropic = anthropic
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise LLMError(
                "ANTHROPIC_API_KEY не задан. Установите переменную окружения "
                "или запускайте с LLM_MODE=mock."
            )
        self._client = anthropic.Anthropic(api_key=key, timeout=timeout)

    def call(self, req: LLMRequest, extra_user_turns: Sequence[dict] = ()) -> tuple[dict, int, int]:
        if req.images:
            content: list[dict] = [
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": mt, "data": b64},
                }
                for mt, b64 in req.images
            ]
            content.append({"type": "text", "text": req.prompt})
            messages: list[dict] = [{"role": "user", "content": content}]
        else:
            messages = [{"role": "user", "content": req.prompt}]
        messages.extend(extra_user_turns)
        kwargs: dict[str, Any] = {
            "model": req.model,
            "max_tokens": req.max_tokens,
            "temperature": req.temperature,
            "messages": messages,
            "tools": [
                {
                    "name": self.TOOL_NAME,
                    "description": "Вернуть извлечённые данные строго по схеме.",
                    "input_schema": req.schema,
                }
            ],
            "tool_choice": {"type": "tool", "name": self.TOOL_NAME},
        }
        if req.system:
            kwargs["system"] = req.system

        msg = self._client.messages.create(**kwargs)
        payload = None
        for block in msg.content:
            if getattr(block, "type", None) == "tool_use" and block.name == self.TOOL_NAME:
                payload = block.input
                break
        if payload is None:
            raise LLMError("Модель не вернула tool_use-блок при forced tool_choice")
        usage = getattr(msg, "usage", None)
        return (
            payload,
            getattr(usage, "input_tokens", 0) or 0,
            getattr(usage, "output_tokens", 0) or 0,
        )

    def retryable(self, exc: BaseException) -> bool:
        a = self._anthropic
        transient = (
            a.RateLimitError,
            a.APIConnectionError,
            a.APITimeoutError,
            a.InternalServerError,
        )
        if isinstance(exc, transient):
            return True
        status = getattr(exc, "status_code", None)
        return status in {408, 409, 429, 500, 502, 503, 504}


class MockProvider(Provider):
    """Офлайн-провайдер для тестов и разработки без ключа.

    Ответы регистрируются заранее по ключу запроса либо через предикат.
    Незарегистрированный запрос — явная ошибка, а не тихая заглушка.
    """

    name = "mock"

    def __init__(self) -> None:
        self._by_key: dict[str, dict] = {}
        self._rules: list[tuple[Callable[[LLMRequest], bool], Callable[[LLMRequest], dict]]] = []
        self.calls: list[LLMRequest] = []

    def register(self, req: LLMRequest, payload: dict) -> None:
        self._by_key[req.cache_key()] = payload

    def register_rule(
        self, match: Callable[[LLMRequest], bool], produce: Callable[[LLMRequest], dict]
    ) -> None:
        self._rules.append((match, produce))

    def call(self, req: LLMRequest, extra_user_turns: Sequence[dict] = ()) -> tuple[dict, int, int]:
        self.calls.append(req)
        key = req.cache_key()
        if key in self._by_key:
            return self._by_key[key], 0, 0
        for match, produce in self._rules:
            if match(req):
                return produce(req), 0, 0
        raise LLMError(
            f"MockProvider: нет зарегистрированного ответа для запроса "
            f"(prompt_version={req.prompt_version}, первые 80 симв.: {req.prompt[:80]!r})"
        )

    def retryable(self, exc: BaseException) -> bool:
        return False


# --------------------------------------------------------------------------- #
# Клиент
# --------------------------------------------------------------------------- #


class LLMClient:
    def __init__(
        self,
        cache_dir: str | Path,
        provider: Provider | None = None,
        log_path: str | Path | None = None,
        max_retries: int = 5,
        max_repairs: int = 2,
        base_backoff_s: float = 1.5,
        read_only_cache: bool = False,
        force: bool = False,
        model: str | None = None,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.provider = provider or self._default_provider()
        self.log_path = Path(log_path) if log_path else None
        self.max_retries = max_retries
        self.max_repairs = max_repairs
        self.base_backoff_s = base_backoff_s
        self.read_only_cache = read_only_cache
        #: Игнорировать НАКОПЛЕННЫЙ кэш, но писать новый. Нужно после правки
        #: промпта, которую забыли отразить в prompt_version, и для замера
        #: реального времени прогона: с кэшем шаг 5 идёт за секунду и ничего
        #: не говорит о том, сколько он займёт в боевом окне.
        self.force = force
        #: Модель для запросов, не указавших свою. Задаётся ПОСЛЕ проверки
        #: доступности (`gemini.verify_model`), поэтому все шаги — включая
        #: распознавание сканов — работают одной и той же проверенной моделью.
        self.model = model or _default_model()
        self.usage = Usage()
        self._log_lock = threading.Lock()

    @staticmethod
    def _default_provider() -> Provider:
        """Провайдер выбирается переменной окружения.

        LLM_MODE=mock     — офлайн, без сети (тесты и разработка);
        LLM_MODE=gemini   — Gemini, есть бесплатный тариф;
        по умолчанию      — Anthropic.

        Смена провайдера не требует правок в шагах: имя модели входит
        в ключ кэша, поэтому результаты разных моделей лежат рядом
        и сравниваются без перезаписи.
        """
        mode = os.environ.get("LLM_MODE", "").lower()
        if mode == "mock":
            return MockProvider()
        if mode in {"gemini", "google"}:
            from .gemini import GeminiProvider

            return GeminiProvider()
        return AnthropicProvider()

    # ---------------- кэш ---------------- #

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def _cache_read(self, key: str) -> dict | None:
        p = self._cache_path(key)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            log.warning("Битая запись кэша, игнорирую: %s", p.name)
            return None

    def _cache_write(self, key: str, req: LLMRequest, payload: dict, meta: dict) -> None:
        if self.read_only_cache:
            return
        record = {
            "key": key,
            "request": {
                "model": req.model,
                "prompt_version": req.prompt_version,
                "temperature": req.temperature,
                "system": req.system,
                "prompt": req.prompt,
                "schema": req.schema,
            },
            "response": payload,
            "meta": meta,
        }
        tmp = self._cache_path(key).with_suffix(".tmp")
        tmp.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self._cache_path(key))  # атомарно: не оставляем half-written

    # ---------------- лог ---------------- #

    def _log_call(self, entry: dict) -> None:
        if not self.log_path:
            return
        with self._log_lock:
            with self.log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # ---------------- основной вызов ---------------- #

    def extract(self, req: LLMRequest, validator: Validator | None = None) -> LLMResult:
        """Один структурированный вызов с кэшем, ретраями и repair-петлёй."""
        if not req.model:
            # Подстановка ДО вычисления ключа кэша: имя модели входит в ключ,
            # и подставить его позже значило бы класть ответы под ключом
            # с пустой моделью.
            req = replace(req, model=self.model)
        key = req.cache_key()
        cached = None if self.force else self._cache_read(key)
        if cached is not None:
            payload = cached["response"]
            problems = validator(payload) if validator else []
            if not problems:
                r = LLMResult(data=payload, cached=True, model=req.model)
                self.usage.add(r)
                return r
            # Кэш содержит ответ, не проходящий текущую валидацию — значит
            # валидатор ужесточили после записи. Перезапрашиваем.
            log.info("Кэш не проходит валидацию (%s), перезапрашиваю", "; ".join(problems))

        t0 = time.time()
        repairs = 0
        retries = 0
        extra_turns: list[dict] = []
        last_payload: dict | None = None
        problems: list[str] = []

        while True:
            try:
                payload, in_tok, out_tok = self.provider.call(req, extra_turns)
            except Exception as exc:  # noqa: BLE001 — решение о ретрае за провайдером
                if self.provider.retryable(exc) and retries < self.max_retries:
                    delay = self.base_backoff_s * (2**retries) * (0.5 + random.random())
                    log.warning(
                        "Транспортная ошибка (%s), повтор %d/%d через %.1fс",
                        type(exc).__name__, retries + 1, self.max_retries, delay,
                    )
                    time.sleep(delay)
                    retries += 1
                    continue
                raise

            last_payload = payload
            problems = validator(payload) if validator else []
            if not problems:
                break

            if repairs >= self.max_repairs:
                self._log_call(
                    {
                        "key": key, "status": "validation_failed",
                        "problems": problems, "payload": payload,
                    }
                )
                raise ValidationFailed(problems, payload)

            repairs += 1
            log.info("Содержательная ошибка, repair %d/%d: %s",
                     repairs, self.max_repairs, "; ".join(problems))
            extra_turns = [
                {"role": "assistant", "content": json.dumps(payload, ensure_ascii=False)},
                {
                    "role": "user",
                    "content": (
                        "В предыдущем ответе есть содержательные ошибки:\n"
                        + "\n".join(f"- {p}" for p in problems)
                        + "\n\nИсправь их и верни результат заново через тот же инструмент. "
                        "Опирайся только на текст исходного документа, ничего не домысливай."
                    ),
                },
            ]

        latency = time.time() - t0
        meta = {
            "input_tokens": in_tok, "output_tokens": out_tok,
            "latency_s": round(latency, 3), "repairs": repairs, "retries": retries,
            "provider": self.provider.name,
        }
        self._cache_write(key, req, payload, meta)
        self._log_call({"key": key, "status": "ok", **meta,
                        "prompt_version": req.prompt_version, "model": req.model})

        r = LLMResult(
            data=payload, cached=False, attempts=1 + repairs + retries,
            input_tokens=in_tok, output_tokens=out_tok,
            latency_s=latency, model=req.model,
        )
        self.usage.add(r, repairs=repairs, retries=retries)
        return r

    # ---------------- self-consistency ---------------- #

    def extract_consistent(
        self,
        req: LLMRequest,
        n: int = 3,
        temperature: float = 0.3,
        validator: Validator | None = None,
        key_fn: Callable[[dict], Any] | None = None,
    ) -> tuple[LLMResult, float]:
        """n прогонов с ненулевой температурой + голосование большинством.

        Возвращает (результат-победитель, доля согласия). Доля согласия — это
        и есть сигнал уверенности для флагов шага 15.
        """
        if n <= 1:
            return self.extract(req, validator), 1.0

        results: list[LLMResult] = []
        for i in range(n):
            variant = LLMRequest(
                prompt=req.prompt, schema=req.schema, system=req.system,
                model=req.model, max_tokens=req.max_tokens,
                temperature=temperature,
                # Разные метки, иначе все n попадут в одну запись кэша.
                prompt_version=f"{req.prompt_version}#sc{i}",
            )
            results.append(self.extract(variant, validator))

        def canon(d: dict) -> str:
            v = key_fn(d) if key_fn else d
            return json.dumps(v, sort_keys=True, ensure_ascii=False)

        counts = Counter(canon(r.data) for r in results)
        winner_key, winner_n = counts.most_common(1)[0]
        winner = next(r for r in results if canon(r.data) == winner_key)
        return winner, winner_n / len(results)

    # ---------------- батчинг и параллельный запуск ---------------- #

    @staticmethod
    def chunked(items: Sequence[Any], size: int) -> list[list[Any]]:
        """Нарезка на батчи. Нужна шагу 10: ~56 транзакций заёмщика идут
        не по одной (56 вызовов) и не разом (риск потери строк в длинном
        ответе), а группами."""
        if size <= 0:
            raise ValueError("size должен быть положительным")
        return [list(items[i : i + size]) for i in range(0, len(items), size)]

    @staticmethod
    def map_parallel(
        fn: Callable[[Any], Any], items: Iterable[Any], workers: int = 8
    ) -> list[Any]:
        """Порядок результатов соответствует порядку items.

        Исключение в ветке не роняет остальные: оно возвращается объектом
        и разбирается вызывающей стороной. Это нужно, чтобы падение одного
        заёмщика не обнуляло прогон по остальным одиннадцати.
        """
        items = list(items)
        if not items:
            return []
        out: list[Any] = [None] * len(items)

        def run(idx_item):
            idx, item = idx_item
            try:
                return idx, fn(item)
            except Exception as exc:  # noqa: BLE001
                log.exception("Ветка %s упала", idx)
                return idx, exc

        with ThreadPoolExecutor(max_workers=min(workers, len(items))) as pool:
            for idx, res in pool.map(run, enumerate(items)):
                out[idx] = res
        return out
