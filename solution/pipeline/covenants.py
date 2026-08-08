"""Шаг 5: извлечение ковенантов из действующих договоров.

Вход:  <run>/artifacts/01_texts/, <run>/artifacts/02_doc_index.json
       (тип LOAN_ACTIVE и scenario_id проставлены шагом 4),
       submission_template.json — список ожидаемых пунктов
Выход: <run>/artifacts/04_covenants.json

ЧТО ЗДЕСЬ ДЕЛАЕТ МОДЕЛЬ И ЧЕГО НЕ ДЕЛАЕТ

Модель переводит юридическую формулировку в ИСПОЛНИМОЕ ДЕРЕВО ВЫРАЖЕНИЙ —
и только. Она не считает, не смотрит в реестр, не решает, нарушен ковенант
или нет. Всё это делает код на шаге 12 по дереву, которое здесь получено.
Разделение принципиальное: арифметика в модели непроверяема и невоспроизводима,
арифметика в коде тестируется и даёт одинаковый ответ дважды.

ТРИ ПРОВЕРКИ, КОТОРЫЕ ЗДЕСЬ ЕСТЬ, И ПОЧЕМУ ИМЕННО ОНИ

1. СПИСОК ПУНКТОВ СВЕРЯЕТСЯ С ШАБЛОНОМ. Шаблон submission_template.json
   называет ровно те пункты, за которые начисляют очки. Если модель извлекла
   6.1 и 6.3, а шаблон ждёт ещё 6.2 — это не «неполнота», это гарантированный
   ноль по трети ячеек заёмщика. Проверка бесплатна и абсолютна: расхождение
   всегда ошибка извлечения, потому что шаблон — данность.

2. ЦИТАТА ПРОВЕРЯЕТСЯ НА ВХОЖДЕНИЕ. Дерево выражений невозможно
   верифицировать, не читая договор. Цитата — то, что связывает дерево
   с источником: если её в тексте нет, модель сочинила и пункт.

3. КАТЕГОРИИ СВЕРЯЮТСЯ СО СЛОВАРЁМ. `AGG('capital_expenditure')` вместо
   `AGG('capex')` не упадёт — он вернёт ПУСТОЙ агрегат, то есть ноль,
   и ковенант окажется «соблюдён» с actual=0. Это самый дорогой из тихих
   отказов проекта, поэтому словарь один на шаги 5, 6 и 10, а проверка
   встроена в валидатор схемы.

ЯЗЫК. Организаторы предупредили, что документы могут быть на английском.
Промпт двуязычен, якорь раздела ищется на обоих языках, а при ненайденном
якоре договор уходит в модель целиком — дороже, но не молча.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .classify import DocType
from . import artifacts as A
from .config import DatasetPaths, RunPaths
from .covenant_types import ANY_CATEGORY, CATEGORIES, OBSERVED_FORMS, PARTY_KINDS, SCOPES
from .llm import LLMClient, LLMRequest, ValidationFailed
from .schemas import (
    COVENANT_SPEC_SCHEMA,
    MetricGraphError,
    make_quote_validator,
    metric_from_payload,
    validate_covenant_spec,
)

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Обрезка по разделу
# --------------------------------------------------------------------------- #

#: Якоря раздела о ковенантах. Русские формы первыми: набор «в основном
#: на русском». Порядок влияет только на скорость, не на результат.
SECTION_ANCHORS = (
    "Финансовые ковенанты",
    "ФИНАНСОВЫЕ КОВЕНАНТЫ",
    "FINANCIAL COVENANTS",
    "Financial Covenants",
)

#: Заголовки, на которых раздел заканчивается. Нужны, чтобы не тащить
#: в промпт остаток договора: пункты о неустойках и применимом праве
#: содержат числа и пороги, которые модель может принять за ковенант.
_SECTION_END = re.compile(
    r"Стать[яи]\s*7\b|Article\s*7\b|ПРИЛОЖЕНИЕ\b|APPENDIX\b|SCHEDULE\s+1\b",
    re.IGNORECASE,
)

#: Потолок обрезки. Раздел ковенантов в публичном наборе укладывается
#: в 6 000 знаков; 12 000 — запас вдвое на случай более многословной
#: редакции, но всё ещё вчетверо меньше договора целиком.
SECTION_MAX_CHARS = 12_000

#: Минимальная длина, при которой фрагмент вообще считается разделом.
#: Нужна из-за ОГЛАВЛЕНИЯ: строка «Статья 6 Финансовые ковенанты» стоит
#: в содержании договора раньше самого раздела, и следом за ней немедленно
#: идёт «Статья 7 …». Первое вхождение якоря давало фрагмент в 21 знак —
#: молча, без единого предупреждения. Именно так выглядит тихий отказ:
#: модель получила бы заголовок вместо договора и вернула пустоту.
MIN_SECTION_CHARS = 400


@dataclass
class Section:
    """Фрагмент договора, отданный модели. Цитаты проверяются против него же."""

    text: str
    anchor: str | None
    truncated: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def trimmed(self) -> bool:
        return self.anchor is not None


def find_section(text: str) -> Section:
    """Вырезает раздел о финансовых ковенантах.

    ЗАЧЕМ ОБРЕЗАТЬ. Договор — около 40 000 знаков, раздел — около 4 000.
    Разница в десять раз это и деньги, и качество: чем меньше постороннего
    текста, тем меньше поводов принять порог из пункта о неустойке за порог
    ковенанта.

    ПОЧЕМУ ПЕРЕБИРАЮТСЯ ВСЕ ВХОЖДЕНИЯ, А НЕ БЕРЁТСЯ ПЕРВОЕ. Заголовок
    раздела встречается в договоре дважды: сначала в ОГЛАВЛЕНИИ, потом
    в самом тексте. В оглавлении следующая строка — «Статья 7 …», то есть
    признак конца раздела срабатывает сразу, и первое вхождение даёт
    фрагмент в два десятка знаков. Поэтому рассматриваются все вхождения
    всех якорей, а выбирается самое содержательное.

    ЧЕГО ЗДЕСЬ СОЗНАТЕЛЬНО НЕ ДЕЛАЕТСЯ. Если якорь не найден или все
    кандидаты вырождены, текст НЕ режется по догадке. Обрезка вслепую
    способна отсечь половину пунктов, и обнаружится это только по
    расхождению с шаблоном. Целый договор дороже, но полон; о переходе
    на этот путь сообщается.
    """
    candidates: list[tuple[int, str, str, bool]] = []
    for anchor in SECTION_ANCHORS:
        start = 0
        while (i := text.find(anchor, start)) >= 0:
            start = i + 1
            tail = text[i:]
            end = _SECTION_END.search(tail, pos=len(anchor))
            body = tail[: end.start()] if end else tail
            candidates.append((len(body), anchor, body, end is not None))

    usable = [c for c in candidates if c[0] >= MIN_SECTION_CHARS]
    if usable:
        # Самый длинный кандидат и есть раздел: оглавление всегда короче
        # текста, который оно описывает.
        length, anchor, body, closed = max(usable, key=lambda c: c[0])
        section = Section(text=body[:SECTION_MAX_CHARS], anchor=anchor)
        if len(candidates) > 1:
            section.notes.append(
                f"якорь встретился {len(candidates)} раз (обычно оглавление и сам "
                f"раздел) — взят фрагмент длиной {length} знаков"
            )
        if length > SECTION_MAX_CHARS:
            section.truncated = True
            section.notes.append(
                f"раздел длиннее {SECTION_MAX_CHARS} знаков и обрезан — "
                f"проверьте, что все пункты попали"
            )
        if not closed:
            section.notes.append("конец раздела не опознан — взят текст до конца документа")
        return section

    note = (
        "заголовок раздела о ковенантах не найден ни на одном языке — "
        "договор отправлен целиком; формулировка заголовка, вероятно, иная"
        if not candidates else
        f"якорь найден ({len(candidates)} раз), но все фрагменты короче "
        f"{MIN_SECTION_CHARS} знаков — похоже, это только оглавление; "
        f"договор отправлен целиком"
    )
    return Section(text=text, anchor=None, notes=[note])


# --------------------------------------------------------------------------- #
# Промпт
# --------------------------------------------------------------------------- #


def _forms_block() -> str:
    """Наблюдённые формы как few-shot материал.

    Берётся из OBSERVED_FORMS, а не пишется в промпте отдельно: список форм
    и проверочный список движка обязаны быть одним и тем же объектом, иначе
    они разойдутся при первой же правке.
    """
    from .schemas import flatten_metric

    lines = []
    for name, form in OBSERVED_FORMS.items():
        nodes, root = flatten_metric(form["дерево"])
        lines.append(f"* {name} — {form['описание']}")
        lines.append(f"  пример: {form['пример']}")
        lines.append(f"  metric_nodes: {json.dumps(nodes, ensure_ascii=False)}")
        lines.append(f"  metric_root: {root!r}")
        if "условие" in form:
            cond_nodes, cond_root = flatten_metric(form["условие"]["metric"])
            lines.append(
                f"  condition_nodes: {json.dumps(cond_nodes, ensure_ascii=False)}, "
                f"condition_root: {cond_root!r}, "
                f"порог {form['условие']['threshold']} {form['условие']['direction']}"
            )
    return "\n".join(lines)


PROMPT_VERSION = "covenants-v2-flat"

_PROMPT = """Ты разбираешь раздел о финансовых ковенантах из кредитного договора. \
Документ может быть на русском или на английском языке — отвечай одинаково в обоих случаях.

ЗАДАЧА: для КАЖДОГО пункта раздела верни порог, направление и ДЕРЕВО ВЫРАЖЕНИЯ, \
по которому показатель можно вычислить из бухгалтерского реестра.

Ты НЕ считаешь значения и НЕ решаешь, нарушен ковенант. Ты только переводишь \
формулировку в дерево. Считать будет код.

ФОРМА ОТВЕТА: дерево передаётся ПЛОСКИМ СПИСКОМ узлов metric_nodes. У каждого \
узла есть короткий id. Связи задаются полем args — это массив СТРОК, то есть \
id дочерних узлов из этого же списка. Вложенных объектов быть не должно. \
metric_root — id корневого узла.

Пример для «Выручка минус Операционные расходы, делить на Процентные расходы»:
  metric_nodes: [
    {{"id": "root", "op": "DIV", "args": ["ebitda", "int"]}},
    {{"id": "ebitda", "op": "SUB", "args": ["rev", "opx"]}},
    {{"id": "int", "op": "AGG", "category": "interest"}},
    {{"id": "rev", "op": "AGG", "category": "revenue"}},
    {{"id": "opx", "op": "AGG", "category": "opex"}}
  ]
  metric_root: "root"

УЗЛЫ ДЕРЕВА:
  AGG      — сумма операций по статье за период; это ЛИСТ, args не нужен.
             Поля: category, при необходимости scope, party, period
  ADD SUB MUL DIV MAX MIN — операции. args ОБЯЗАТЕЛЕН и содержит id
             дочерних узлов. DIV: [числитель, знаменатель].
             SUB: [уменьшаемое, вычитаемое]
  ABS      — модуль значения, args из одного id
  CONST    — константа, поле value
  DISCLOSED — величина, раскрытая в документах и НЕ отражённая операцией
             в реестре. Поле key — короткое имя на латинице

СЛОВАРЬ СТАТЕЙ (category) — использовать ТОЛЬКО эти значения:
{categories}
  {any_category} — любая статья; так задаются ковенанты, где важен получатель
  платежа, а не статья расхода
Если формулировка не ложится в словарь — возьми ближайшую статью и опиши
расхождение в metric_definition. Придуманная статья даст ПУСТУЮ сумму
и неверный ответ.

scope: {scopes} — group только если в пункте прямо сказано про Группу
       или консолидированную отчётность
party: {parties} — только если пункт ограничивает круг контрагентов

НА ЧТО ОБРАТИТЬ ВНИМАНИЕ:
  * «наибольшая из статей» — это MAX, а НЕ сумма. Признак: оговорка вида
    «их сумма не является показателем настоящего ковенанта»
  * ковенант, применяемый лишь при выполнении условия, — is_conditional=true
    плюс condition_metric/condition_direction/condition_threshold
  * direction: max — показатель не должен ПРЕВЫШАТЬ порог; min — не должен
    опускаться НИЖЕ
  * threshold всегда положительное число; знак задаётся полем direction
  * операция без args бессмысленна: DIV, SUB, MAX, MIN, ADD обязаны
    ссылаться на дочерние узлы, иначе показатель вычислить нельзя
  * unit: amount — сумма в долларах; ratio — коэффициент или доля
  * period_start и period_end — период, за который проверяется ковенант,
    в формате YYYY-MM-DD, взятый из самого пункта или из раздела
  * carve_outs — оговорки и исключения, при которых превышение допускается

НАБЛЮДЁННЫЕ ФОРМЫ (список не исчерпывающий — встретишь иную, собери дерево сам):
{forms}

ЦИТАТА: поле quote обязано содержать ДОСЛОВНЫЙ фрагмент пункта из текста ниже, \
не короче одного предложения, без правок и сокращений. Цитата проверяется \
автоматически на вхождение в исходный текст.

ИЗВЛЕКИ ВСЕ пункты раздела. Пропущенный пункт стоит дороже лишнего.

ТЕКСТ ДОГОВОРА:
"""


def build_prompt(section_text: str) -> str:
    return _PROMPT.format(
        categories="\n".join(f"  {c}" for c in CATEGORIES),
        any_category=ANY_CATEGORY,
        scopes=", ".join(SCOPES),
        parties=", ".join(PARTY_KINDS),
        forms=_forms_block(),
    ) + section_text


# --------------------------------------------------------------------------- #
# Порог обязан встречаться в тексте
# --------------------------------------------------------------------------- #


def threshold_forms(value: float) -> list[str]:
    """Написания, в которых порог может стоять в договоре.

    Суммы пишутся как «$2,000,000.00», коэффициенты — как «0.08x» или
    «3.00x». Проверяются оба разделителя разрядов (запятая и пробел):
    в англоязычной редакции формат может отличаться.
    """
    forms: list[str] = []
    if value == int(value):
        integer = int(value)
        grouped = f"{integer:,}"
        forms += [grouped, grouped.replace(",", " "), grouped.replace(",", ""),
                  f"{grouped}.00", f"{integer}.00"]
    forms += [f"{value:.2f}", f"{value:g}", f"{value:,.2f}"]
    return sorted(set(forms))


def threshold_appears(text: str, value: float) -> bool:
    """Есть ли порог в тексте хоть в одном написании."""
    flat = re.sub(r"\s+", " ", text)
    return any(form in flat for form in threshold_forms(value))


def check_thresholds(payload: dict, section_text: str) -> list[str]:
    """Порог, которого нет в тексте, — выдуманный.

    ЗАЧЕМ ОТДЕЛЬНО ОТ ЦИТАТЫ. Цитата подтверждает, что пункт существует,
    но модель может привести верную цитату и ошибиться числом на порядок —
    цитата при этом останется дословной. Порог же входит в ответ напрямую:
    ошибка в нём переворачивает status, а это половина стоимости ячейки.
    Проверка бесплатна, потому что порог в договоре всегда написан явно.

    ГРАНИЦА ПРИМЕНИМОСТИ. Проверка не ловит ПЕРЕПУТАННЫЕ пороги: если
    в разделе есть и 0.08x, и 3.00x, оба «встречаются». Она ловит только
    выдуманные значения — но именно они дают самые дикие расхождения.
    """
    problems: list[str] = []
    for c in payload.get("covenants", []):
        value = c.get("threshold")
        if not isinstance(value, (int, float)):
            continue
        if not threshold_appears(section_text, float(value)):
            problems.append(
                f"пункт {c.get('point', '?')}: порог {value} не встречается "
                f"в тексте раздела ни в одном написании — перечитай пункт"
            )
    return problems


# --------------------------------------------------------------------------- #
# Извлечение одного договора
# --------------------------------------------------------------------------- #


@dataclass
class ScenarioCovenants:
    scenario_id: str
    doc_id: str | None = None
    covenants: list[dict] = field(default_factory=list)
    section_anchor: str | None = None
    problems: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def points(self) -> list[str]:
        return sorted(c.get("point", "?") for c in self.covenants)

    def to_dict(self) -> dict:
        return {
            "scenario_id": self.scenario_id,
            "doc_id": self.doc_id,
            "section_anchor": self.section_anchor,
            "points": self.points(),
            "covenants": self.covenants,
            "problems": self.problems,
            "notes": self.notes,
        }


def nest_condition(covenant: dict) -> dict:
    """Дерево условия springing-теста — тот же плоский формат."""
    from .schemas import nest_metric

    return nest_metric(covenant["condition_nodes"], str(covenant["condition_root"]))


def extract_one(
    scenario_id: str,
    doc_id: str,
    text: str,
    llm: LLMClient,
    expected_points: list[str] | None = None,
    model: str | None = None,
) -> ScenarioCovenants:
    """Извлекает ковенанты одного заёмщика.

    Валидатор объединяет три проверки в одну функцию сознательно: клиент
    отдаёт модели список замечаний и просит исправить ответ, поэтому чем
    полнее список за один проход, тем меньше кругов исправления.
    """
    out = ScenarioCovenants(scenario_id=scenario_id, doc_id=doc_id)
    section = find_section(text)
    out.section_anchor = section.anchor
    out.notes.extend(section.notes)

    quote_check = make_quote_validator(section.text)

    def validator(payload: dict) -> list[str]:
        problems = (validate_covenant_spec(payload) + quote_check(payload)
                    + check_thresholds(payload, section.text))
        if expected_points:
            got = {c.get("point") for c in payload.get("covenants", [])}
            missing = sorted(set(expected_points) - got)
            if missing:
                # Шаблон — данность, а не мнение: пункт из него обязан быть.
                problems.append(
                    f"не извлечены пункты {missing}, ожидаемые шаблоном; "
                    f"перечитай текст и найди их"
                )
        return problems

    kwargs: dict[str, Any] = {"model": model} if model else {}
    request = LLMRequest(
        prompt=build_prompt(section.text),
        schema=COVENANT_SPEC_SCHEMA,
        prompt_version=PROMPT_VERSION,
        max_tokens=8000,
        **kwargs,
    )

    try:
        payload = llm.extract(request, validator=validator).data
    except ValidationFailed as exc:
        # ЧАСТИЧНОЕ ИЗВЛЕЧЕНИЕ ЛУЧШЕ ПУСТОГО. Замечание валидатора относится
        # к ответу целиком: не извлечён один пункт — забракован весь ответ.
        # Но пункты независимы, и каждый оценивается своей ячейкой. Уронив
        # ответ целиком, мы теряем ТРИ ячейки там, где испорчена одна.
        # Поэтому берём последний ответ и отбираем из него пункты, которые
        # проходят проверку ПООДИНОЧКЕ.
        payload = exc.last_payload if isinstance(exc.last_payload, dict) else {}
        kept, dropped = [], []
        for c in payload.get("covenants", []):
            single = {"covenants": [c]}
            if (validate_covenant_spec(single) or quote_check(single)
                    or check_thresholds(single, section.text)):
                dropped.append(c.get("point", "?"))
            else:
                kept.append(c)
        payload = {"covenants": kept}
        out.problems.append(
            f"ответ не прошёл проверку ({'; '.join(exc.problems)[:300]}); "
            f"сохранены пункты {sorted(c.get('point', '?') for c in kept)}"
            + (f", отброшены {sorted(dropped)}" if dropped else "")
        )

    # Плоский список разворачивается в дерево ЗДЕСЬ: дальше по пайплайну
    # (шаг 12) живёт только вложенная форма, и знать о плоской ей незачем.
    covenants: list[dict] = []
    for c in payload.get("covenants", []):
        try:
            c = {**c, "metric": metric_from_payload(c)}
        except MetricGraphError as exc:
            out.problems.append(f"пункт {c.get('point', '?')}: {exc} — пункт отброшен")
            continue
        if c.get("condition_nodes") and c.get("condition_root"):
            try:
                c["condition_metric"] = nest_condition(c)
            except MetricGraphError as exc:
                out.notes.append(f"пункт {c.get('point', '?')}: условие нечитаемо ({exc})")
        covenants.append(c)
    out.covenants = covenants

    if expected_points:
        got = set(out.points())
        missing = sorted(set(expected_points) - got)
        extra = sorted(got - set(expected_points))
        if missing:
            out.problems.append(
                f"пункты {missing} не извлечены, хотя шаблон их требует — "
                f"эти ячейки будут заполнены запасным значением"
            )
        if extra:
            # Не ошибка сама по себе: в договоре бывают пункты, за которые
            # не начисляют очки. Но лишний пункт может означать, что модель
            # перепутала нумерацию, и тогда пострадают нужные.
            out.notes.append(f"извлечены пункты вне шаблона: {extra}")
    return out


# --------------------------------------------------------------------------- #
# Прогон
# --------------------------------------------------------------------------- #


@dataclass
class CovenantReport:
    scenarios: list[ScenarioCovenants] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    def by_scenario(self) -> dict[str, ScenarioCovenants]:
        return {s.scenario_id: s for s in self.scenarios}

    def alarms(self) -> list[str]:
        """Признаки, что извлечение непригодно для расчёта."""
        out: list[str] = []
        empty = [s.scenario_id for s in self.scenarios if not s.covenants]
        if empty:
            out.append(f"без ковенантов остались заёмщики {empty} — расчёт для них невозможен")
        untrimmed = [s.scenario_id for s in self.scenarios if s.section_anchor is None]
        if len(untrimmed) == len(self.scenarios) and self.scenarios:
            out.append(
                "раздел не опознан НИ В ОДНОМ договоре — заголовок в этом наборе "
                "называется иначе, добавьте якорь в SECTION_ANCHORS"
            )
        return out

    def to_dict(self) -> dict:
        return {
            "alarms": self.alarms(),
            "problems": self.problems,
            "scenarios": {s.scenario_id: s.to_dict() for s in self.scenarios},
        }


def _switch_model(llm: LLMClient, exhausted_model: str) -> tuple[str | None, list[str]]:
    """Ищет модель со свежей суточной квотой.

    Возвращает (имя или None, пояснения). Ничего не делает для провайдеров,
    у которых понятия «каталог моделей» нет.
    """
    provider = getattr(llm, "provider", None)
    if not hasattr(provider, "available_models"):
        return None, ["провайдер не умеет перечислять модели"]
    try:
        from .gemini import verify_model

        catalogue = [m for m in provider.available_models() if m != exhausted_model]
        return verify_model(provider, None, catalogue)
    except Exception as exc:  # noqa: BLE001 — не повод ронять прогон
        return None, [f"подобрать замену не удалось: {type(exc).__name__}"]


def expected_points(template: dict) -> dict[str, list[str]]:
    """Какие пункты ждёт шаблон от каждого заёмщика."""
    return {
        scenario: sorted(cells)
        for scenario, cells in (template.get("answers") or {}).items()
    }


def run(
    dataset: DatasetPaths,
    paths: RunPaths,
    llm: LLMClient,
    model: str | None = None,
    workers: int = 6,
) -> CovenantReport:
    from . import classify

    docs = classify.load(paths)
    texts_dir = paths.artifacts / A.TEXTS_DIR
    template = json.loads(dataset.template_json.read_text(encoding="utf-8"))
    wanted = expected_points(template)

    report = CovenantReport()

    # Действующий договор выбран шагом 4 по периоду действия. Если шаг 4
    # не отработал, здесь не будет ни одного LOAN_ACTIVE — это видно сразу,
    # а не превращается в тихо пустой расчёт.
    active: dict[str, str] = {}
    for doc_id, d in sorted(docs.items()):
        if d.type == DocType.LOAN_ACTIVE and d.scenario_id:
            if d.scenario_id in active:
                report.problems.append(
                    f"{d.scenario_id}: действующих договоров больше одного "
                    f"({active[d.scenario_id]}, {doc_id}) — взят первый по порядку"
                )
                continue
            active[d.scenario_id] = doc_id

    for scenario in sorted(wanted):
        if scenario not in active:
            report.problems.append(
                f"{scenario}: действующий договор не найден — ковенанты неизвестны"
            )
            report.scenarios.append(ScenarioCovenants(
                scenario_id=scenario,
                problems=["нет действующего договора"],
            ))

    jobs = [(s, active[s]) for s in sorted(wanted) if s in active]

    def work(job: tuple[str, str]) -> ScenarioCovenants:
        scenario, doc_id = job
        text = (texts_dir / f"{doc_id}.txt").read_text(encoding="utf-8")
        # model берётся из замыкания: при переходе на запасную модель
        # повторные вызовы обязаны идти уже с новым именем.
        return extract_one(scenario, doc_id, text, llm, wanted.get(scenario), model)

    results = LLMClient.map_parallel(work, jobs, workers=workers)

    # ПЕРЕХОД НА ДРУГУЮ МОДЕЛЬ ПРИ ИСЧЕРПАНИИ СУТОЧНОЙ КВОТЫ.
    #
    # Квота бесплатного тарифа считается на проект И НА МОДЕЛЬ. Значит
    # соседняя модель — это не «попробовать ещё раз то же самое», а
    # полностью свежий счётчик. Терять восемь заёмщиков из двенадцати
    # из-за исчерпанного счётчика, когда рядом лежит работающая модель,
    # бессмысленно: в боевом окне это две трети возможных очков.
    #
    # Ретраями это не лечится принципиально: суточная квота не
    # рассасывается за секунды, и ждать её в трёхчасовом окне нельзя.
    exhausted = [
        (job, res) for job, res in zip(jobs, results)
        if isinstance(res, Exception) and type(res).__name__ == "DailyQuotaExhausted"
    ]
    if exhausted and model:
        spare, notes = _switch_model(llm, model)
        for note in notes:
            report.problems.append(f"смена модели: {note}")
        if spare and spare != model:
            retry_jobs = [job for job, _ in exhausted]
            log.warning(
                "КОВЕНАНТЫ: квота модели %s исчерпана, повторяю %d заёмщиков на %s",
                model, len(retry_jobs), spare,
            )
            model = spare
            retried = LLMClient.map_parallel(work, retry_jobs, workers=workers)
            replacement = dict(zip((j for j, _ in exhausted), retried))
            results = [replacement.get(job, res) for job, res in zip(jobs, results)]

    for job, result in zip(jobs, results):
        scenario, doc_id = job
        if isinstance(result, Exception):
            # Падение одного заёмщика не должно обнулять остальных.
            report.problems.append(
                f"{scenario}: извлечение упало — {type(result).__name__}: {result}"
            )
            report.scenarios.append(ScenarioCovenants(
                scenario_id=scenario, doc_id=doc_id,
                problems=[f"извлечение упало: {type(result).__name__}"],
            ))
            continue
        report.scenarios.append(result)
        report.problems.extend(f"{scenario}: {p}" for p in result.problems)

    report.scenarios.sort(key=lambda s: s.scenario_id)
    out = paths.artifacts / A.COVENANTS
    out.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    for alarm in report.alarms():
        log.warning("КОВЕНАНТЫ: %s", alarm)
    for problem in report.problems:
        log.warning("КОВЕНАНТЫ: %s", problem)
    log.info(
        "Извлечено ковенантов: %d у %d заёмщиков",
        sum(len(s.covenants) for s in report.scenarios), len(report.scenarios),
    )
    return report


def load(paths: RunPaths) -> dict[str, ScenarioCovenants]:
    """Чтение артефакта следующими шагами."""
    data = json.loads((paths.artifacts / A.COVENANTS).read_text(encoding="utf-8"))
    out = {}
    for scenario, payload in data["scenarios"].items():
        out[scenario] = ScenarioCovenants(
            scenario_id=scenario,
            doc_id=payload.get("doc_id"),
            covenants=payload.get("covenants", []),
            section_anchor=payload.get("section_anchor"),
            problems=payload.get("problems", []),
            notes=payload.get("notes", []),
        )
    return out
