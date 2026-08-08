"""Шаг 11: сборка итогового реестра для расчёта.

Вход:  06_ledger_clean.csv (шаг 9), 07_categories.json (шаг 10),
       05_related_parties.json (шаг 8), 06_adjustments.json (шаг 7)
Выход: 08_ledger_final.csv

ЧТО ЗДЕСЬ ПРОИСХОДИТ

Шаги 7–10 добывали сведения по отдельности: как операции классифицируются,
кто из контрагентов связанная сторона, что аудитор велел переложить или
исключить. Здесь всё это сводится в одну таблицу, по которой считает
движок. Своих суждений шаг не выносит — он применяет чужие решения,
и это его главное свойство.

ПОРЯДОК ПРИМЕНЕНИЯ ЗНАЧИМ

    категория из шага 10
      → переклассификация аудитора поверх неё
        → метка связанной стороны
          → отсечение по периоду
            → пересчёт валюты

Переклассификация идёт ПОСЛЕ категоризации, потому что она её отменяет:
аудитор видел документы, модель видела описание строки. Отсечение идёт
после всего: исключённая операция не должна попасть ни в один агрегат
независимо от того, как её разложили.

КАЖДОЕ ИЗМЕНЕНИЕ ОСТАВЛЯЕТ СЛЕД

У каждой строки есть поле `origin`: откуда взялась её категория и почему
она такая. Без следа итоговый реестр — таблица чисел неизвестного
происхождения, и разобрать спорную ячейку в боевом окне будет нечем.
Шаг 13 ищет по этому же следу доказательства, а шаг 15 — сомнительные
места.

ЧЕГО ЗДЕСЬ СОЗНАТЕЛЬНО НЕ ДЕЛАЕТСЯ

Не применяются корректировки со статусом, отличным от `applied`
(см. adjustments.py: приложение упоминает суммы, которые применять
нельзя). Не выбрасываются строки: исключение помечается флагом, а не
удалением, — удалённую строку невозможно предъявить как доказательство
и невозможно отличить от потерянной.
"""
from __future__ import annotations

import csv
import json
import logging
from dataclasses import asdict, dataclass, field
from typing import Iterable, Sequence

from . import artifacts as A
from .adjustments import Note, ScenarioAdjustments
from .categorize import TxnCategory
from .config import RunPaths
from .related import ScenarioParties, build_index

log = logging.getLogger(__name__)

#: Столбцы итогового реестра. Порядок и имена — контракт с шагом 12
#: (см. compute.load_rows), менять их в одиночку нельзя.
FINAL_COLUMNS = [
    "txn_id", "scenario_id", "date", "counterparty", "amount_usd",
    "category", "party", "scope", "excluded", "exclusion_reason",
    "origin", "confidence",
]


@dataclass
class FinalRow:
    txn_id: str
    scenario_id: str
    date: str
    counterparty: str
    amount_usd: float
    category: str = "other"
    party: str | None = None
    scope: str = "borrower"
    excluded: bool = False
    exclusion_reason: str | None = None
    #: Откуда взялась категория. Читается человеком при разборе спорной ячейки.
    origin: str = ""
    confidence: float = 0.0

    def as_csv(self) -> list:
        return [
            self.txn_id, self.scenario_id, self.date, self.counterparty,
            f"{self.amount_usd:.2f}", self.category, self.party or "",
            self.scope, int(self.excluded), self.exclusion_reason or "",
            self.origin, f"{self.confidence:.2f}",
        ]


@dataclass
class ApplyReport:
    rows: list[FinalRow] = field(default_factory=list)
    reclassified: list[str] = field(default_factory=list)
    excluded: list[str] = field(default_factory=list)
    related_tagged: list[str] = field(default_factory=list)
    skipped_notes: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    def alarms(self) -> list[str]:
        out: list[str] = []
        if not self.rows:
            return ["итоговый реестр пуст — считать нечего"]

        uncategorised = [r.txn_id for r in self.rows if r.category == "other"]
        if len(uncategorised) / len(self.rows) > 0.2:
            out.append(
                f"{len(uncategorised) / len(self.rows):.0%} операций осталось "
                f"в статье 'other' — агрегаты будут занижены"
            )
        if not self.related_tagged:
            out.append(
                "ни одна операция не отмечена как платёж связанной стороне — "
                "все ковенанты по связанным сторонам дадут ноль"
            )
        return out

    def to_dict(self) -> dict:
        return {
            "alarms": self.alarms(),
            "problems": self.problems,
            "counts": {
                "rows": len(self.rows),
                "reclassified": len(self.reclassified),
                "excluded": len(self.excluded),
                "related_tagged": len(self.related_tagged),
            },
            "reclassified": self.reclassified,
            "excluded": self.excluded,
            "related_tagged": self.related_tagged,
            "skipped_notes": self.skipped_notes,
        }


# --------------------------------------------------------------------------- #
# Отдельные применения — каждое проверяемо в одиночку
# --------------------------------------------------------------------------- #


def match_note_to_row(note: Note, rows: Sequence[FinalRow]) -> list[FinalRow]:
    """Какие строки затрагивает примечание аудитора.

    Три способа адресации, от точного к приблизительному. Порядок важен:
    примечание с номером операции адресует РОВНО её, и искать после этого
    по контрагенту значило бы задеть однофамильцев.

    Адресация по сумме нужна потому, что аудитор часто пишет «сумма
    в размере $1,104,663.28, выплаченная контрагенту X» без номера
    операции. Сумма плюс контрагент — достаточно узкое сочетание.
    """
    if note.target_txn_id:
        return [r for r in rows if r.txn_id == note.target_txn_id]

    if note.target_counterparty and note.value_usd is not None:
        index = build_index_for(note.target_counterparty)
        return [
            r for r in rows
            if index(r.counterparty) and abs(abs(r.amount_usd) - note.value_usd) < 0.01
        ]

    if note.target_counterparty:
        index = build_index_for(note.target_counterparty)
        return [r for r in rows if index(r.counterparty)]

    if note.value_usd is not None:
        # ТОЛЬКО СУММА. Так пришло примечание 7.1 у P8: аудитор назвал
        # величину, но ни номера операции, ни контрагента. Отбрасывать
        # такое значило бы терять настоящую корректировку.
        #
        # Строки здесь уже отобраны по заёмщику, поэтому совпадение суммы
        # с точностью до цента достаточно узко. Но если совпало НЕСКОЛЬКО
        # строк, адресат неоднозначен — применять наугад к одной из них
        # хуже, чем не применять вовсе: неверная переклассификация портит
        # два агрегата сразу.
        matched = [
            r for r in rows if abs(abs(r.amount_usd) - note.value_usd) < 0.01
        ]
        return matched if len(matched) == 1 else []

    return []


def build_index_for(name: str):
    """Сопоставление названия из документа с названием в реестре.

    Используется тот же EntityIndex, что и для связанных сторон: правила
    совпадения имён обязаны быть одни на весь проект, иначе одна и та же
    организация окажется связанной стороной в одном месте и посторонней
    в другом.
    """
    from .entities import Entity, EntityIndex

    index = EntityIndex([Entity(name=name, role="counterparty")])

    def match(counterparty: str) -> bool:
        entity, _how = index.match(counterparty)
        return entity is not None

    return match


def apply_categories(
    rows: list[FinalRow], categories: dict[str, TxnCategory]
) -> list[str]:
    """Раскладывает операции по статьям. Первый слой, поверх которого всё."""
    problems: list[str] = []
    for row in rows:
        marked = categories.get(row.txn_id)
        if marked is None:
            row.category = "other"
            row.origin = "категория не получена"
            problems.append(f"{row.txn_id}: категория отсутствует — отнесена к 'other'")
            continue
        row.category = marked.category
        row.confidence = marked.confidence
        row.origin = (
            f"шаг 10: {marked.category}"
            + (" (запасное значение)" if marked.fallback else "")
        )
    return problems


def apply_related(
    rows: list[FinalRow], parties: ScenarioParties
) -> list[str]:
    """Отмечает платежи связанным сторонам.

    Метка ставится на СТРОКУ, а не на контрагента: агрегат спрашивает
    «сколько заплачено связанным сторонам», и отвечать на это надо
    построчно.
    """
    tagged: list[str] = []
    index = build_index(parties)
    for row in rows:
        entity, _how = index.match(row.counterparty)
        if entity is not None and entity.is_related:
            row.party = "related"
            row.origin += f"; связанная сторона ({entity.name})"
            tagged.append(row.txn_id)
    return tagged


def apply_adjustments(
    rows: list[FinalRow], adjustments: ScenarioAdjustments
) -> tuple[list[str], list[str], list[str]]:
    """Переклассификации и отсечения аудитора.

    Возвращает (переклассифицированные, исключённые, пропущенные примечания).

    Пропущенные перечисляются поимённо и с причиной: примечание, которое
    не удалось применить, — это либо ловушка (и тогда всё правильно),
    либо промах адресации (и тогда в расчёте недостаёт корректировки).
    Различить их можно только глядя на список.
    """
    reclassified: list[str] = []
    excluded: list[str] = []
    skipped: list[str] = []

    for note in adjustments.notes:
        if not note.applies:
            skipped.append(
                f"{adjustments.scenario_id} п.{note.note_id}: "
                f"{note.skipped_because or 'не применяется'}"
            )
            continue

        targets = match_note_to_row(note, rows)
        if not targets:
            skipped.append(
                f"{adjustments.scenario_id} п.{note.note_id}: не найдена операция "
                f"(txn={note.target_txn_id}, контрагент={note.target_counterparty}, "
                f"сумма={note.value_usd})"
            )
            continue

        if note.kind == "cutoff":
            for row in targets:
                row.excluded = True
                row.exclusion_reason = (
                    f"аудитор, п.{note.note_id}: {note.description[:120]}"
                )
                excluded.append(row.txn_id)
            continue

        if note.kind == "reclassification" and note.to_category:
            for row in targets:
                was = row.category
                row.category = note.to_category
                row.origin += f"; переклассифицировано аудитором п.{note.note_id}: {was} → {note.to_category}"
                reclassified.append(row.txn_id)
            continue

        # Прочие виды примечаний (курсы, внебалансовые суммы) применяются
        # не к строкам реестра: курс учтён шагом 9, а раскрытые величины
        # движок берёт узлом DISCLOSED.
        skipped.append(
            f"{adjustments.scenario_id} п.{note.note_id}: вид {note.kind} "
            f"не меняет строк реестра"
        )

    return reclassified, excluded, skipped


# --------------------------------------------------------------------------- #
# Прогон
# --------------------------------------------------------------------------- #


def load_clean_rows(paths: RunPaths) -> list[FinalRow]:
    """Читает очищенный реестр шага 9 и переводит в форму итогового.

    Операции без суммы пропускаются с отметкой: считать по ним нечего,
    а подставлять ноль значило бы молча занизить агрегат. Восстановленные
    суммы к этому моменту уже проставлены шагом 9.
    """
    rows: list[FinalRow] = []
    path = paths.artifacts / A.LEDGER_CLEAN
    with path.open(newline="", encoding="utf-8") as fh:
        for raw in csv.DictReader(fh):
            amount = raw.get("amount_usd") or raw.get("amount") or ""
            try:
                value = float(amount)
            except (TypeError, ValueError):
                log.warning(
                    "РЕЕСТР: %s без суммы — строка не попадёт в расчёт", raw.get("txn_id")
                )
                continue
            rows.append(FinalRow(
                txn_id=raw["txn_id"],
                scenario_id=raw.get("scenario_id", ""),
                date=raw.get("date", ""),
                counterparty=raw.get("counterparty", ""),
                amount_usd=value,
            ))
    return rows


def run(
    paths: RunPaths,
    categories: dict[str, TxnCategory] | None = None,
    related: dict[str, ScenarioParties] | None = None,
    adjustments: dict[str, ScenarioAdjustments] | None = None,
) -> ApplyReport:
    from . import adjustments as adj_module
    from . import categorize, related as related_module

    categories = categories if categories is not None else categorize.load(paths)
    related = related if related is not None else related_module.load(paths)
    adjustments = adjustments if adjustments is not None else adj_module.load(paths)

    report = ApplyReport(rows=load_clean_rows(paths))

    # 1. Категории — основа, всё остальное её уточняет.
    report.problems.extend(apply_categories(report.rows, categories))

    # 2. Связанные стороны и 3. корректировки — по заёмщикам: и те и другие
    #    определены в документах КОНКРЕТНОГО заёмщика и на чужие строки
    #    распространяться не должны.
    by_scenario: dict[str, list[FinalRow]] = {}
    for row in report.rows:
        by_scenario.setdefault(row.scenario_id, []).append(row)

    for scenario, rows in sorted(by_scenario.items()):
        parties = related.get(scenario)
        if parties is not None:
            report.related_tagged.extend(apply_related(rows, parties))
        elif scenario:
            report.problems.append(
                f"{scenario}: нет сведений о связанных сторонах — "
                f"соответствующие агрегаты дадут ноль"
            )

        notes = adjustments.get(scenario)
        if notes is not None:
            reclassified, excluded, skipped = apply_adjustments(rows, notes)
            report.reclassified.extend(reclassified)
            report.excluded.extend(excluded)
            report.skipped_notes.extend(skipped)

    out_csv = paths.artifacts / A.LEDGER_FINAL
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(FINAL_COLUMNS)
        for row in report.rows:
            writer.writerow(row.as_csv())

    (paths.artifacts / A.APPLY_REPORT).write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    for alarm in report.alarms():
        log.warning("СБОРКА РЕЕСТРА: %s", alarm)
    log.info(
        "Итоговый реестр: %d строк, переклассифицировано %d, исключено %d, "
        "связанных сторон отмечено %d",
        len(report.rows), len(report.reclassified), len(report.excluded),
        len(report.related_tagged),
    )
    return report
