"""Шаг 12: расчётный движок.

Вход:  <run>/artifacts/03_covenants.json   — спецификации ковенантов (шаг 5/6)
       <run>/artifacts/08_ledger_final.csv — реестр с категориями (шаги 10/11)
       <run>/artifacts/05_related_parties.json — связанные стороны (шаг 8)
       <run>/artifacts/04_adjustments.json — внебалансовые величины (шаг 7)
Выход: <run>/artifacts/09_results.json

Движок — тонкий диспетчер поверх реестра форм из шага 6. Вся вариативность
ковенантов живёт там; здесь только доступ к данным и трассировка.

АРИФМЕТИКУ СЧИТАЕТ КОД, НЕ МОДЕЛЬ. Это не стилистика, а суть архитектуры:
модель заполняет структуру, число получается из структуры детерминированно.

ТРАССИРОВКА ОБЯЗАТЕЛЬНА. Каждая ячейка знает, какие именно операции вошли
в каждый агрегат. Без этого шаг 13 не сможет проверить доказательство
контрфактуально, а шаг 15 — объяснить расхождение.
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

from . import artifacts as A
from .config import RunPaths
from .covenant_types import CovenantTest, TestResult, run_test

log = logging.getLogger(__name__)

#: Категория, означающая «любая»: используется в ковенантах по связанным
#: сторонам, где важен получатель, а не статья расхода.
ANY_CATEGORY = "any"


@dataclass
class Row:
    """Строка реестра в том виде, в каком её видит движок."""

    txn_id: str
    scenario_id: str
    date: str
    counterparty: str
    amount_usd: float
    category: str
    party: str | None = None       # 'related', 'unrestricted_subsidiary', ...
    scope: str = "borrower"        # 'borrower' | 'group'
    excluded: bool = False         # исключена правилом отсечения (шаг 11)
    exclusion_reason: str | None = None


@dataclass
class AggregateTrace:
    category: str
    scope: str
    party: str | None
    period: tuple[str, str] | None
    txn_ids: list[str]
    value: float


class LedgerAggregateSource:
    """Реализация протокола AggregateSource поверх подготовленного реестра.

    Ведёт журнал обращений: какие операции попали в какой агрегат. Журнал —
    не отладочная роскошь, а вход для шагов 13 и 15.
    """

    def __init__(
        self,
        rows: Iterable[Row],
        disclosed: dict[str, float] | None = None,
        exclude: set[str] | None = None,
        overrides: dict[str, dict] | None = None,
    ):
        self.rows = list(rows)
        self._disclosed = disclosed or {}
        #: Идентификаторы, временно исключённые из расчёта. Используется
        #: шагом 13 для контрфактуальной проверки доказательства.
        self.exclude = set(exclude or ())
        #: Подмена полей строки на время контрфактуального прогона: например,
        #: возврат переклассифицированной операции в исходную категорию.
        #: Для переклассификации это точнее исключения — операция никуда
        #: не исчезала, она лишь была отнесена к другой статье.
        self.overrides = overrides or {}
        self.traces: list[AggregateTrace] = []
        self.missing_categories: set[str] = set()

    # ---------------- протокол ---------------- #

    def aggregate(self, category, scope="borrower", party=None, period=None) -> float:
        matched: list[Row] = []
        for r in self.rows:
            if r.excluded or r.txn_id in self.exclude:
                continue
            over = self.overrides.get(r.txn_id, {})
            r_category = over.get("category", r.category)
            r_party = over.get("party", r.party)
            r_scope = over.get("scope", r.scope)
            if r_scope != scope:
                continue
            if category != ANY_CATEGORY and r_category != category:
                continue
            if party is not None and r_party != party:
                continue
            if period and not (period[0] <= r.date <= period[1]):
                continue
            matched.append(r)

        value = sum(abs(r.amount_usd) for r in matched)
        if not matched:
            if party is not None:
                # Ни одна операция не помечена нужным типом контрагента.
                # Чаще это провал разметки на шаге 8, чем реальное отсутствие
                # платежей связанным сторонам.
                self.missing_categories.add(f"party={party} (ни одной операции)")
            elif category != ANY_CATEGORY:
                # Пустой агрегат почти всегда означает расхождение словарей
                # категорий между шагом 6 и шагом 10, а не отсутствие операций.
                self.missing_categories.add(f"{category}/{scope}")
        self.traces.append(AggregateTrace(
            category=category, scope=scope, party=party, period=period,
            txn_ids=[r.txn_id for r in matched], value=value,
        ))
        return value

    def disclosed(self, key: str) -> float:
        if key not in self._disclosed:
            self.missing_categories.add(f"disclosed:{key}")
        return self._disclosed.get(key, 0.0)

    # ---------------- помощники ---------------- #

    def reset_traces(self) -> None:
        self.traces = []

    def touched_txn_ids(self) -> list[str]:
        """Все операции, участвовавшие в последнем расчёте, без повторов."""
        seen: dict[str, None] = {}
        for t in self.traces:
            for tid in t.txn_ids:
                seen.setdefault(tid, None)
        return list(seen)


@dataclass
class CellResult:
    scenario_id: str
    point: str
    status: str
    actual: float
    evidence_txn_id: str | None = None
    unit: str = "amount"
    condition_met: bool = True
    threshold: float | None = None
    direction: str | None = None
    trace: list[dict] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    quote: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["actual"] = None if (self.actual is None or math.isnan(self.actual)) else round(self.actual, 2)
        return d


def compute_cell(
    scenario_id: str,
    test: CovenantTest,
    source: LedgerAggregateSource,
) -> CellResult:
    """Считает одну ячейку и записывает трассировку."""
    source.reset_traces()
    before = set(source.missing_categories)
    result: TestResult = run_test(test, source)
    new_missing = source.missing_categories - before

    problems = list(result.problems)
    for m in sorted(new_missing):
        problems.append(
            f"агрегат {m} пуст — вероятно расхождение словаря категорий "
            f"между спецификацией ковенанта и категоризацией транзакций"
        )
    if math.isnan(result.actual):
        problems.append("actual не вычислен — ячейка требует ручного разбора")

    return CellResult(
        scenario_id=scenario_id,
        point=test.point,
        status=result.status,
        actual=result.actual,
        unit=test.unit,
        condition_met=result.condition_met,
        threshold=test.threshold,
        direction=test.direction,
        quote=test.quote,
        trace=[
            {
                "category": t.category, "scope": t.scope, "party": t.party,
                "period": list(t.period) if t.period else None,
                "n_txns": len(t.txn_ids), "value": round(t.value, 2),
                "txn_ids": t.txn_ids,
            }
            for t in source.traces
        ],
        problems=problems,
    )


# --------------------------------------------------------------------------- #
# Загрузка артефактов
# --------------------------------------------------------------------------- #


def load_rows(csv_path: Path) -> list[Row]:
    import csv as _csv

    rows: list[Row] = []
    with csv_path.open(newline="", encoding="utf-8") as fh:
        for r in _csv.DictReader(fh):
            amount = r.get("amount_usd") or r.get("amount") or ""
            try:
                value = float(amount)
            except ValueError:
                continue
            rows.append(Row(
                txn_id=r["txn_id"],
                scenario_id=r["scenario_id"],
                date=r.get("date", ""),
                counterparty=r.get("counterparty", ""),
                amount_usd=value,
                category=r.get("category", ""),
                party=r.get("party") or None,
                scope=r.get("scope") or "borrower",
                excluded=str(r.get("excluded", "")).strip() in {"1", "true", "True"},
                exclusion_reason=r.get("exclusion_reason") or None,
            ))
    return rows


def load_tests(path: Path) -> dict[str, list[CovenantTest]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, list[CovenantTest]] = {}
    for scenario, cells in data.get("scenarios", {}).items():
        tests = []
        for point, spec in cells.items():
            period = spec.get("period")
            tests.append(CovenantTest(
                point=point,
                direction=spec["direction"],
                threshold=float(spec["threshold"]),
                metric=spec["metric"],
                unit=spec.get("unit", "amount"),
                period=tuple(period) if period else None,
                condition=spec.get("condition"),
                quote=spec.get("quote", ""),
                notes=spec.get("notes", []),
            ))
        out[scenario] = sorted(tests, key=lambda t: t.point)
    return out


def run(paths: RunPaths) -> dict[str, list[CellResult]]:
    covenants_path = paths.artifacts / A.COVENANTS
    ledger_path = paths.artifacts / A.LEDGER_FINAL
    if not ledger_path.exists():
        ledger_path = paths.artifacts / A.LEDGER_CLEAN
    disclosed_path = paths.artifacts / A.DISCLOSED

    tests = load_tests(covenants_path)
    rows = load_rows(ledger_path)
    disclosed_all: dict[str, dict[str, float]] = (
        json.loads(disclosed_path.read_text(encoding="utf-8"))
        if disclosed_path.exists() else {}
    )

    by_scenario: dict[str, list[Row]] = {}
    for r in rows:
        by_scenario.setdefault(r.scenario_id, []).append(r)

    results: dict[str, list[CellResult]] = {}
    for scenario, scenario_tests in tests.items():
        source = LedgerAggregateSource(
            by_scenario.get(scenario, []),
            disclosed=disclosed_all.get(scenario, {}),
        )
        if not by_scenario.get(scenario):
            log.warning("РАСЧЁТ: у сценария %s нет ни одной операции", scenario)
        results[scenario] = [compute_cell(scenario, t, source) for t in scenario_tests]

    out = {
        s: {c.point: c.to_dict() for c in cells} for s, cells in results.items()
    }
    (paths.artifacts / A.RESULTS).write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    flagged = [
        f"{s}/{c.point}: {c.problems[0]}"
        for s, cells in results.items() for c in cells if c.problems
    ]
    for f in flagged:
        log.warning("РАСЧЁТ: %s", f)
    log.info(
        "Посчитано ячеек: %d, из них с замечаниями: %d",
        sum(len(v) for v in results.values()), len(flagged),
    )
    return results
