"""Шаг 7: аудиторские корректировки из финальных приложений.

Вход:  <run>/artifacts/01_texts/, <run>/artifacts/02_doc_index.json (AUDIT_FINAL)
Выход: <run>/artifacts/06_adjustments.json

ЗАЧЕМ ЭТОТ ШАГ

Реестр показывает, как операции были УЧТЕНЫ. Ковенант проверяется по тому,
как они должны учитываться ДЛЯ ЦЕЛЕЙ ДОГОВОРА, а это разные вещи. Разницу
описывает аудитор в приложении о соблюдении ковенантов: что переложить между
статьями, что вынести за период, по какому курсу пересчитать валюту, что
добавить к EBITDA сверх реестра.

Формы, встреченные в публичном наборе:

  переклассификация  сумма перекладывается из одной статьи в другую (P2, P10)
  отсечение          операция относится к другому периоду и исключается (B4, P1)
  валютный курс      раскрыта пара «счёт в валюте → платёж в долларах» (P3)
  разовые статьи     добавления к EBITDA сверх реестра, с порогом (P4)
  ничего             «переклассификаций не требовалось» (P5, P6, P7)

ГЛАВНАЯ ОПАСНОСТЬ ЗДЕСЬ — НЕ ПРОПУСК, А ЛИШНЕЕ ПРИМЕНЕНИЕ

Приложение упоминает суммы, которые применять НЕЛЬЗЯ, и выглядят они как
обычные корректировки — с суммой, контрагентом и словом «переклассификация»:

  * B1, п. 9.1: сумма «отобрана для проверки классификации», а вывод изложен
    в другом отчёте и «в настоящих примечаниях не повторяется». Вывода здесь
    нет — применять нечего;
  * P10, п. 7.2: операция «рассматривалась на предмет возможной
    переклассификации; по итогам разъяснений первоначальная классификация
    сохраняется».

Применить их — значит переложить сотни тысяч долларов между статьями без
основания. Расчёт не заметит: числа сойдутся, статусы будут выставлены,
и всё будет выглядеть нормально. Поэтому статус извлекается отдельным полем,
а применяются только примечания со статусом `applied`.

ПОРОГ СУЩЕСТВЕННОСТИ — ТОЖЕ ИЗ ДОКУМЕНТА

У P4 три разовые статьи, но применяются не все: «Разовыми для целей
ковенантов признаются статьи в сумме не менее $300,000.00; статьи меньшей
суммы к EBITDA не прибавляются». Сумма всех трёх дала бы 1,075,491.85
вместо 824,152.91 — расхождение в 30%. Сравнение с порогом делает КОД:
модель перечисляет все статьи, отбор — арифметика.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from typing import Any

from .classify import DocType
from . import artifacts as A
from .config import RunPaths
from .covenant_types import CATEGORIES
from .llm import LLMClient, LLMRequest, ValidationFailed
from .schemas import (
    AUDIT_ADJUSTMENTS_SCHEMA,
    make_quote_validator,
    validate_audit_adjustments,
)

log = logging.getLogger(__name__)

PROMPT_VERSION = "adjustments-v1"

#: Раздел, ради которого всё затевается. Примечания выше — общая учётная
#: политика, одинаковая у всех заёмщиков и для ковенантов бесполезная.
SECTION_ANCHORS = (
    "ДОПОЛНЕНИЕ О СОБЛЮДЕНИИ КОВЕНАНТОВ",
    "COVENANT COMPLIANCE SUPPLEMENT",
    "COVENANT COMPLIANCE SCHEDULE",
    "COVENANT COMPLIANCE ANNEX",
)


def find_section(text: str) -> tuple[str, str | None]:
    """Вырезает приложение о соблюдении ковенантов.

    Возвращает (текст, найденный якорь). Якорь не найден — отдаём документ
    целиком: он невелик (около 5 000 знаков), и потерять раздел дороже,
    чем прочитать лишнее.
    """
    for anchor in SECTION_ANCHORS:
        i = text.find(anchor)
        if i >= 0:
            return text[i:], anchor
    return text, None


_PROMPT = """Перед тобой приложение аудитора о соблюдении ковенантов. \
Документ может быть на русском или на английском языке.

ЗАДАЧА: выпиши все примечания этого раздела, влияющие на расчёт ковенантов.

САМОЕ ВАЖНОЕ — ПОЛЕ status. В приложении встречаются суммы, которые применять \
НЕЛЬЗЯ, и выглядят они как обычные корректировки:

  * «Сумма … была отобрана для проверки классификации. Вывод изложен в отчёте \
№ AR-… и в настоящих примечаниях не повторяется» → status=referred_elsewhere. \
Вывода здесь НЕТ, применять нечего.
  * «рассматривалась на предмет возможной переклассификации; по итогам \
разъяснений первоначальная классификация сохраняется» → \
status=considered_but_rejected. Ничего не изменилось.
  * «Сумма … переклассифицирована для целей соблюдения ковенантов как …» → \
status=applied. Вот это применяем.
  * Описание правила без конкретной суммы («Выручка признаётся в том периоде, \
в котором оказаны услуги») → status=informational.

Не додумывай вывод за аудитора. Если сказано, что заключение в другом \
документе, — так и отметь.

ПОРОГ СУЩЕСТВЕННОСТИ. Если сказано «Разовыми признаются статьи в сумме \
не менее $X», занеси X в materiality_threshold_usd и перечисли ВСЕ статьи, \
включая те, что ниже порога. Отбор сделает код — тебе считать не нужно.

ЕСЛИ КОРРЕКТИРОВОК НЕТ («Переклассификаций за ковенантный период \
не требовалось») — поставь no_adjustments_stated=true.

СТАТЬИ РАСХОДОВ для from_category и to_category бери ТОЛЬКО из списка:
{categories}
Названия из документа сопоставь с этим списком: «Страховые премии» → insurance, \
«Операционные расходы» → opex, «Капитальные затраты» → capex.

value_usd — всегда ПОЛОЖИТЕЛЬНОЕ число, направление задаётся полем sign.
ЦИТАТА обязана быть дословной: она проверяется автоматически.

ПРИЛОЖЕНИЕ:
"""


def build_prompt(section_text: str) -> str:
    return _PROMPT.format(
        categories="\n".join(f"  {c}" for c in CATEGORIES)
    ) + section_text


# --------------------------------------------------------------------------- #
# Результат
# --------------------------------------------------------------------------- #


@dataclass
class Note:
    note_id: str
    kind: str
    status: str
    description: str = ""
    quote: str = ""
    target_txn_id: str | None = None
    target_counterparty: str | None = None
    from_category: str | None = None
    to_category: str | None = None
    value_usd: float | None = None
    sign: str | None = None
    period_start: str | None = None
    period_end: str | None = None
    #: Проставляет КОД: прошло ли примечание порог существенности.
    material: bool = True
    #: Почему примечание не применяется, если не применяется.
    skipped_because: str | None = None

    @property
    def applies(self) -> bool:
        return self.status == "applied" and self.material


@dataclass
class ScenarioAdjustments:
    scenario_id: str
    doc_id: str | None = None
    section_anchor: str | None = None
    notes: list[Note] = field(default_factory=list)
    materiality_threshold_usd: float | None = None
    no_adjustments_stated: bool = False
    problems: list[str] = field(default_factory=list)
    remarks: list[str] = field(default_factory=list)

    def applied(self) -> list[Note]:
        return [n for n in self.notes if n.applies]

    def to_dict(self) -> dict:
        return {
            "scenario_id": self.scenario_id,
            "doc_id": self.doc_id,
            "section_anchor": self.section_anchor,
            "materiality_threshold_usd": self.materiality_threshold_usd,
            "no_adjustments_stated": self.no_adjustments_stated,
            "applied_ids": [n.note_id for n in self.applied()],
            "notes": [asdict(n) for n in self.notes],
            "problems": self.problems,
            "remarks": self.remarks,
        }


def apply_materiality(notes: list[Note], threshold: float | None) -> list[str]:
    """Отбирает разовые статьи по порогу существенности. Меняет notes на месте.

    ПОЧЕМУ ЭТО ДЕЛАЕТ КОД. Порог назван в документе, а отбор по нему —
    арифметика. У P4 из трёх разовых статей проходят две: сумма всех трёх
    дала бы 1,075,491.85 вместо 824,152.91.

    Порог применяется ТОЛЬКО к разовым статьям EBITDA. Переклассификация
    и отсечение существенности не знают: аудитор либо сделал корректировку,
    либо нет, и её размер тут ни при чём.
    """
    remarks: list[str] = []
    if threshold is None:
        return remarks
    for note in notes:
        if note.kind != "ebitda_adjustment" or note.status != "applied":
            continue
        if note.value_usd is None:
            note.material = False
            note.skipped_because = "сумма не указана, сравнить с порогом нечего"
            remarks.append(f"{note.note_id}: сумма не указана — статья не учтена")
            continue
        if note.value_usd < threshold:
            note.material = False
            note.skipped_because = (
                f"{note.value_usd:,.2f} ниже порога существенности {threshold:,.2f}"
            )
            remarks.append(
                f"{note.note_id}: {note.value_usd:,.2f} ниже порога {threshold:,.2f} "
                f"— к EBITDA не прибавляется"
            )
    return remarks


# --------------------------------------------------------------------------- #
# Извлечение
# --------------------------------------------------------------------------- #


def extract_one(
    scenario_id: str,
    doc_id: str,
    text: str,
    llm: LLMClient,
    model: str | None = None,
) -> ScenarioAdjustments:
    out = ScenarioAdjustments(scenario_id=scenario_id, doc_id=doc_id)
    section, anchor = find_section(text)
    out.section_anchor = anchor
    if anchor is None:
        out.remarks.append(
            "заголовок приложения не найден — документ отправлен целиком"
        )

    quote_check = make_quote_validator(section)

    def validator(payload: dict) -> list[str]:
        return validate_audit_adjustments(payload) + quote_check(payload)

    kwargs: dict[str, Any] = {"model": model} if model else {}
    request = LLMRequest(
        prompt=build_prompt(section),
        schema=AUDIT_ADJUSTMENTS_SCHEMA,
        prompt_version=PROMPT_VERSION,
        max_tokens=8000,
        **kwargs,
    )

    try:
        payload = llm.extract(request, validator=validator).data
    except ValidationFailed as exc:
        payload = exc.last_payload if isinstance(exc.last_payload, dict) else {}
        out.problems.append(
            f"ответ не прошёл проверку ({'; '.join(exc.problems)[:300]}) — "
            f"взят последний ответ как есть"
        )

    out.materiality_threshold_usd = payload.get("materiality_threshold_usd")
    out.no_adjustments_stated = bool(payload.get("no_adjustments_stated"))

    known = set(CATEGORIES)
    for raw in payload.get("notes", []):
        note = Note(
            note_id=str(raw.get("note_id", "?")),
            kind=str(raw.get("kind", "other")),
            status=str(raw.get("status", "informational")),
            description=raw.get("description", ""),
            quote=raw.get("quote", ""),
            target_txn_id=raw.get("target_txn_id"),
            target_counterparty=raw.get("target_counterparty"),
            from_category=raw.get("from_category"),
            to_category=raw.get("to_category"),
            value_usd=raw.get("value_usd"),
            sign=raw.get("sign"),
            period_start=raw.get("period_start"),
            period_end=raw.get("period_end"),
        )
        if note.status != "applied":
            note.skipped_because = f"статус {note.status}"
        # Статья вне словаря не упадёт — она молча промахнётся мимо агрегата,
        # поэтому проверяется здесь, а не при расчёте.
        for field_name in ("from_category", "to_category"):
            value = getattr(note, field_name)
            if value and value not in known:
                out.problems.append(
                    f"примечание {note.note_id}: {field_name}={value!r} вне словаря "
                    f"статей — переклассификация не попадёт в расчёт"
                )
        out.notes.append(note)

    out.remarks.extend(apply_materiality(out.notes, out.materiality_threshold_usd))

    if out.no_adjustments_stated and out.applied():
        out.problems.append(
            "сказано, что корректировок не требовалось, но применяемые примечания есть"
        )
    return out


# --------------------------------------------------------------------------- #
# Прогон
# --------------------------------------------------------------------------- #


@dataclass
class AdjustmentsReport:
    scenarios: list[ScenarioAdjustments] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    def by_scenario(self) -> dict[str, ScenarioAdjustments]:
        return {s.scenario_id: s for s in self.scenarios}

    def alarms(self) -> list[str]:
        out: list[str] = []
        if not self.scenarios:
            return ["ни одного финального аудиторского приложения не обработано"]
        no_anchor = [s.scenario_id for s in self.scenarios if s.section_anchor is None]
        if len(no_anchor) == len(self.scenarios):
            out.append(
                "заголовок приложения не найден НИ В ОДНОМ документе — "
                "формулировка в этом наборе, вероятно, иная"
            )
        silent = [
            s.scenario_id for s in self.scenarios
            if not s.notes and not s.no_adjustments_stated
        ]
        if silent:
            out.append(
                f"у заёмщиков {silent} нет ни примечаний, ни отметки об их "
                f"отсутствии — вероятно, раздел не разобран"
            )
        return out

    def to_dict(self) -> dict:
        return {
            "alarms": self.alarms(),
            "problems": self.problems,
            "scenarios": {s.scenario_id: s.to_dict() for s in self.scenarios},
        }


def run(
    paths: RunPaths,
    llm: LLMClient,
    model: str | None = None,
    workers: int = 6,
) -> AdjustmentsReport:
    from . import classify

    docs = classify.load(paths)
    texts_dir = paths.artifacts / A.TEXTS_DIR
    report = AdjustmentsReport()

    annexes: dict[str, str] = {}
    for doc_id, d in sorted(docs.items()):
        if d.type != DocType.AUDIT_FINAL or not d.scenario_id:
            continue
        if d.scenario_id in annexes:
            report.problems.append(
                f"{d.scenario_id}: финальных приложений больше одного — "
                f"взято первое по порядку"
            )
            continue
        annexes[d.scenario_id] = doc_id

    jobs = sorted(annexes.items())

    def work(job: tuple[str, str]) -> ScenarioAdjustments:
        scenario, doc_id = job
        text = (texts_dir / f"{doc_id}.txt").read_text(encoding="utf-8")
        return extract_one(scenario, doc_id, text, llm, model)

    for job, result in zip(jobs, LLMClient.map_parallel(work, jobs, workers=workers)):
        scenario, doc_id = job
        if isinstance(result, Exception):
            report.problems.append(
                f"{scenario}: извлечение упало — {type(result).__name__}: {result}"
            )
            report.scenarios.append(ScenarioAdjustments(
                scenario_id=scenario, doc_id=doc_id,
                problems=[f"извлечение упало: {type(result).__name__}"],
            ))
            continue
        report.scenarios.append(result)
        report.problems.extend(f"{scenario}: {p}" for p in result.problems)

    report.scenarios.sort(key=lambda s: s.scenario_id)
    out = paths.artifacts / A.AUDIT_ADJUSTMENTS
    out.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    for alarm in report.alarms():
        log.warning("КОРРЕКТИРОВКИ: %s", alarm)
    for problem in report.problems:
        log.warning("КОРРЕКТИРОВКИ: %s", problem)
    log.info(
        "Применяемых корректировок: %d у %d заёмщиков",
        sum(len(s.applied()) for s in report.scenarios), len(report.scenarios),
    )
    return report


def load(paths: RunPaths) -> dict[str, ScenarioAdjustments]:
    data = json.loads((paths.artifacts / A.AUDIT_ADJUSTMENTS).read_text(encoding="utf-8"))
    out: dict[str, ScenarioAdjustments] = {}
    for scenario, payload in data["scenarios"].items():
        out[scenario] = ScenarioAdjustments(
            scenario_id=scenario,
            doc_id=payload.get("doc_id"),
            section_anchor=payload.get("section_anchor"),
            notes=[Note(**n) for n in payload.get("notes", [])],
            materiality_threshold_usd=payload.get("materiality_threshold_usd"),
            no_adjustments_stated=payload.get("no_adjustments_stated", False),
            problems=payload.get("problems", []),
            remarks=payload.get("remarks", []),
        )
    return out
