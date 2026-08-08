"""Шаг 8: связанные стороны из досье KYC.

Вход:  <run>/artifacts/01_texts/, <run>/artifacts/02_doc_index.json (тип KYC)
Выход: <run>/artifacts/05_related_parties.json

ЗАЧЕМ ЭТОТ ШАГ СУЩЕСТВУЕТ

Ковенанты вида `AGG(any, party=related)` спрашивают: сколько заёмщик заплатил
СВЯЗАННЫМ СТОРОНАМ. Реестр про связанность ничего не знает — там только имена
контрагентов. Знание о том, кто связанная сторона, живёт исключительно
в досье KYC, и без этого шага все такие агрегаты вернут ноль.

Таких ковенантов в публичном наборе десять из тридцати шести.

ЧТО ЗДЕСЬ ГЛАВНОЕ

ПОРОГ УЧАСТИЯ РАЗНЫЙ У КАЖДОГО ЗАЁМЩИКА. Это выяснилось на данных, а не
предполагалось: в публичном наборе пороги идут от 20.0% до 40.0%, и каждый
записан в тексте своего досье. Если взять «40% как принято», у восьми
заёмщиков из двенадцати часть связанных сторон не будет опознана. Платежи
им выпадут из агрегата, `actual` окажется заниженным — и правдоподобным.
Ковенант «не более 0.04x выручки» при заниженном числителе спокойно
покажет COMPLIANT там, где на деле BREACH.

РЕШЕНИЕ ПРИНИМАЕТ КОД. Признание связанной стороной — это сравнение
`доля >= порог`. Арифметику делает код: она проверяема и воспроизводима.
Мнение модели запрашивается, но служит независимой сверкой — расхождение
означает, что одно из двух чисел прочитано неверно.

ОТСУТСТВИЕ РАЗДЕЛА И ОТСУТСТВИЕ СТОРОН — РАЗНЫЕ ВЕЩИ. У P2 в досье нет
раздела о бенефициарном владении вовсе. Это законно и означает «связанных
сторон не заявлено», то есть агрегат честно равен нулю. Но выглядит такой
случай ровно как неудачное извлечение, поэтому различие фиксируется явным
полем, а не угадывается по пустому списку.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .classify import DocType
from . import artifacts as A
from .config import RunPaths
from .entities import Entity, EntityIndex, normalize_entity_name
from .llm import LLMClient, LLMRequest, ValidationFailed
from .schemas import RELATED_PARTIES_SCHEMA, make_quote_validator, validate_related_parties

log = logging.getLogger(__name__)

PROMPT_VERSION = "related-v1"

_PROMPT = """Перед тобой досье «Знай своего клиента» (KYC), составленное банком \
на заёмщика. Документ может быть на русском или на английском языке.

ЗАДАЧА: выпиши организации, чьи доли участия перечислены в досье, и порог, \
при котором организация признаётся СВЯЗАННОЙ СТОРОНОЙ.

ПОРОГ. В досье обычно есть фраза вида «Организации, в которых Группа владеет \
20.0% и более голосующих прав, признаются связанными сторонами для целей \
Договора». Число из этой фразы и есть threshold_pct. Порог у разных заёмщиков \
РАЗНЫЙ — не подставляй привычное значение, возьми написанное. Если порога \
в тексте нет, верни null.

ЕСЛИ РАЗДЕЛА О ВЛАДЕНИИ НЕТ ВОВСЕ — верни has_ownership_section=false и пустой \
список. Это не ошибка: так бывает. Не путай с ситуацией, когда раздел есть, \
но ты его не разобрал.

НАЗВАНИЯ переписывай ТОЧНО как в документе, вместе с организационно-правовой \
формой (LLP, JSC, Ltd) и кавычками, если они есть. По этим названиям потом \
ищутся платежи в бухгалтерском реестре, и «примерно похожее» название их не \
найдёт.

ЦИТАТА для каждой организации — дословная строка досье, где стоит её название \
и доля. Цитаты проверяются автоматически.

Поле is_related заполни своим мнением, но знай: окончательное решение примет \
код сравнением доли с порогом. Если организация связана НЕ через долю участия \
(например, общий контроль или прямое указание в тексте) — опиши это в basis.

ДОСЬЕ:
"""


# --------------------------------------------------------------------------- #
# Результат
# --------------------------------------------------------------------------- #


@dataclass
class Party:
    name: str
    ownership_pct: float | None = None
    basis: str | None = None
    quote: str = ""
    #: Вердикт КОДА, а не модели.
    is_related: bool = False
    #: Мнение модели — хранится для разбора расхождений.
    model_said: bool | None = None


@dataclass
class ScenarioParties:
    scenario_id: str
    doc_id: str | None = None
    threshold_pct: float | None = None
    threshold_quote: str | None = None
    has_ownership_section: bool = False
    parties: list[Party] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def related_names(self) -> list[str]:
        return sorted(p.name for p in self.parties if p.is_related)

    def to_dict(self) -> dict:
        return {
            "scenario_id": self.scenario_id,
            "doc_id": self.doc_id,
            "threshold_pct": self.threshold_pct,
            "threshold_quote": self.threshold_quote,
            "has_ownership_section": self.has_ownership_section,
            "related_names": self.related_names(),
            "parties": [asdict(p) for p in self.parties],
            "problems": self.problems,
            "notes": self.notes,
        }


def decide_related(pct: float | None, threshold: float | None, basis: str | None) -> bool:
    """Единственное место, где решается вопрос о связанности.

    Отдельная функция, а не строчка внутри цикла, потому что это правило —
    предмет спора и проверки. Границу берём НЕСТРОГОЙ: в документах написано
    «20.0% И БОЛЕЕ», то есть ровно пороговое значение уже делает сторону
    связанной. Строгое сравнение потеряло бы участника с долей ровно 20.0%.
    """
    if basis:
        # Связанность по иному основанию (общий контроль, прямое указание)
        # не отменяется отсутствием доли.
        return True
    if pct is None or threshold is None:
        return False
    return pct >= threshold


# --------------------------------------------------------------------------- #
# Извлечение одного досье
# --------------------------------------------------------------------------- #


def extract_one(
    scenario_id: str,
    doc_id: str,
    text: str,
    llm: LLMClient,
    model: str | None = None,
) -> ScenarioParties:
    out = ScenarioParties(scenario_id=scenario_id, doc_id=doc_id)
    quote_check = make_quote_validator(text)

    def validator(payload: dict) -> list[str]:
        return validate_related_parties(payload) + quote_check(payload)

    kwargs: dict[str, Any] = {"model": model} if model else {}
    request = LLMRequest(
        prompt=_PROMPT + text,
        schema=RELATED_PARTIES_SCHEMA,
        prompt_version=PROMPT_VERSION,
        max_tokens=6000,
        **kwargs,
    )

    try:
        payload = llm.extract(request, validator=validator).data
    except ValidationFailed as exc:
        # Как и на шаге 5: частичный результат лучше пустого. Связанная
        # сторона, не попавшая в список, — это недосчитанные платежи.
        payload = exc.last_payload if isinstance(exc.last_payload, dict) else {}
        out.problems.append(
            f"ответ не прошёл проверку ({'; '.join(exc.problems)[:300]}) — "
            f"взят последний ответ как есть"
        )

    out.has_ownership_section = bool(payload.get("has_ownership_section"))
    out.threshold_pct = payload.get("threshold_pct")
    out.threshold_quote = payload.get("threshold_quote")

    for raw in payload.get("parties", []):
        pct = raw.get("ownership_pct")
        basis = raw.get("basis")
        decided = decide_related(pct, out.threshold_pct, basis)
        model_said = raw.get("is_related")
        party = Party(
            name=str(raw.get("name", "")).strip(),
            ownership_pct=pct,
            basis=basis,
            quote=raw.get("quote", ""),
            is_related=decided,
            model_said=model_said if isinstance(model_said, bool) else None,
        )
        if party.model_said is not None and party.model_said != decided:
            # Расхождение не разрешается в пользу модели: решает код.
            # Но само расхождение означает, что доля или порог прочитаны
            # неверно, и это надо увидеть.
            out.notes.append(
                f"{party.name}: доля {pct}% при пороге {out.threshold_pct}% даёт "
                f"{decided}, модель считала {party.model_said}"
            )
        out.parties.append(party)

    if out.has_ownership_section and out.threshold_pct is None:
        out.problems.append(
            "раздел о владении есть, но порог не извлечён — "
            "связанные стороны определить нечем"
        )
    if out.has_ownership_section and not out.parties:
        out.problems.append("раздел о владении есть, но ни одной организации не извлечено")
    return out


# --------------------------------------------------------------------------- #
# Связывание с реестром
# --------------------------------------------------------------------------- #


def build_index(parties: ScenarioParties) -> EntityIndex:
    """Индекс имён для поиска платежей в реестре.

    Названия в досье и в реестре совпадают не буквально: в реестре к имени
    прирастают уточнения вроде «(Turkistan point)» или «Holdings Company».
    Сопоставлением занимается EntityIndex — тот же, что и в графе связей,
    чтобы правила совпадения были одни на весь проект.
    """
    return EntityIndex([
        Entity(name=p.name, role="counterparty", is_related=p.is_related,
               ownership_pct=p.ownership_pct, basis=p.basis or "")
        for p in parties.parties
    ])


def match_against_ledger(
    parties: ScenarioParties, counterparties: list[str]
) -> tuple[dict[str, str], list[str]]:
    """Какие контрагенты реестра оказались связанными сторонами.

    Возвращает (контрагент -> имя из досье, замечания). Замечания нужны из-за
    асимметрии рисков: связанная сторона, которой в реестре не нашлось ни
    одного платежа, — это либо правда (платежей не было), либо промах
    сопоставления имён. Первое безобидно, второе занижает агрегат, поэтому
    сообщается всегда.
    """
    index = build_index(parties)
    matched: dict[str, str] = {}
    hit_names: set[str] = set()

    for counterparty in counterparties:
        entity, _how = index.match(counterparty)
        if entity is not None and entity.is_related:
            matched[counterparty] = entity.name
            hit_names.add(normalize_entity_name(entity.name))

    notes: list[str] = []
    for party in parties.parties:
        if not party.is_related:
            continue
        if normalize_entity_name(party.name) not in hit_names:
            notes.append(
                f"связанная сторона {party.name!r} не найдена среди контрагентов "
                f"реестра — либо платежей не было, либо название не совпало"
            )
    return matched, notes


# --------------------------------------------------------------------------- #
# Прогон
# --------------------------------------------------------------------------- #


@dataclass
class RelatedReport:
    scenarios: list[ScenarioParties] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    def by_scenario(self) -> dict[str, ScenarioParties]:
        return {s.scenario_id: s for s in self.scenarios}

    def alarms(self) -> list[str]:
        out: list[str] = []
        if not self.scenarios:
            return ["ни одного досье KYC не обработано"]

        no_section = [s.scenario_id for s in self.scenarios if not s.has_ownership_section]
        if len(no_section) == len(self.scenarios):
            out.append(
                "НИ В ОДНОМ досье не найден раздел о владении — формулировка "
                "в этом наборе, вероятно, иная; все агрегаты по связанным "
                "сторонам окажутся нулевыми"
            )
        thresholds = {s.threshold_pct for s in self.scenarios if s.threshold_pct is not None}
        if len(thresholds) == 1 and len(self.scenarios) > 3:
            # Не ошибка, но повод присмотреться: в публичном наборе пороги
            # различались у каждого заёмщика.
            out.append(
                f"у всех заёмщиков один и тот же порог {thresholds.pop()}% — "
                f"проверьте, не подставлено ли значение по умолчанию"
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
) -> RelatedReport:
    from . import classify

    docs = classify.load(paths)
    texts_dir = paths.artifacts / A.TEXTS_DIR
    report = RelatedReport()

    dossiers: dict[str, str] = {}
    for doc_id, d in sorted(docs.items()):
        if d.type != DocType.KYC or not d.scenario_id:
            continue
        if d.scenario_id in dossiers:
            report.problems.append(
                f"{d.scenario_id}: досье KYC больше одного — взято первое по порядку"
            )
            continue
        dossiers[d.scenario_id] = doc_id

    jobs = sorted(dossiers.items())

    def work(job: tuple[str, str]) -> ScenarioParties:
        scenario, doc_id = job
        text = (texts_dir / f"{doc_id}.txt").read_text(encoding="utf-8")
        return extract_one(scenario, doc_id, text, llm, model)

    for job, result in zip(jobs, LLMClient.map_parallel(work, jobs, workers=workers)):
        scenario, doc_id = job
        if isinstance(result, Exception):
            report.problems.append(
                f"{scenario}: извлечение упало — {type(result).__name__}: {result}"
            )
            report.scenarios.append(ScenarioParties(
                scenario_id=scenario, doc_id=doc_id,
                problems=[f"извлечение упало: {type(result).__name__}"],
            ))
            continue
        report.scenarios.append(result)
        report.problems.extend(f"{scenario}: {p}" for p in result.problems)

    report.scenarios.sort(key=lambda s: s.scenario_id)
    out = paths.artifacts / A.RELATED_PARTIES
    out.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    for alarm in report.alarms():
        log.warning("СВЯЗАННЫЕ СТОРОНЫ: %s", alarm)
    for problem in report.problems:
        log.warning("СВЯЗАННЫЕ СТОРОНЫ: %s", problem)
    log.info(
        "Связанных сторон: %d у %d заёмщиков",
        sum(len(s.related_names()) for s in report.scenarios), len(report.scenarios),
    )
    return report


def load(paths: RunPaths) -> dict[str, ScenarioParties]:
    data = json.loads(
        (paths.artifacts / A.RELATED_PARTIES).read_text(encoding="utf-8")
    )
    out: dict[str, ScenarioParties] = {}
    for scenario, payload in data["scenarios"].items():
        out[scenario] = ScenarioParties(
            scenario_id=scenario,
            doc_id=payload.get("doc_id"),
            threshold_pct=payload.get("threshold_pct"),
            threshold_quote=payload.get("threshold_quote"),
            has_ownership_section=payload.get("has_ownership_section", False),
            parties=[Party(**p) for p in payload.get("parties", [])],
            problems=payload.get("problems", []),
            notes=payload.get("notes", []),
        )
    return out
