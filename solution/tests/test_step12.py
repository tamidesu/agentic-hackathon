"""Тесты шага 12: расчётный движок.

Движок — мост между реестром и деревом выражений. Проверяется не «считает
ли он», а держит ли он контракты: фильтры, трассировку и громкость там,
где данные не сходятся.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline import artifacts as A  # noqa: E402
from pipeline.compute import (  # noqa: E402
    LedgerAggregateSource,
    Row,
    compute_cell,
    load_rows,
    load_tests,
)
from pipeline.compute import run as compute_run  # noqa: E402
from pipeline.config import RunPaths  # noqa: E402
from pipeline.covenant_types import BREACH, COMPLIANT, CovenantTest  # noqa: E402


def AGG(cat, **kw):
    return {"op": "AGG", "category": cat, **kw}


def OP(op, *args):
    return {"op": op, "args": list(args)}


def rows(*specs) -> list[Row]:
    out = []
    for i, s in enumerate(specs, 1):
        out.append(Row(
            txn_id=s.get("id", f"TXN-P1-{i:04d}"),
            scenario_id="P1",
            date=s.get("date", "2025-06-01"),
            counterparty=s.get("cp", "CP"),
            amount_usd=s["amount"],
            category=s.get("cat", "opex"),
            party=s.get("party"),
            scope=s.get("scope", "borrower"),
            excluded=s.get("excluded", False),
        ))
    return out


# --------------------------------------------------------------------------- #
# Фильтры агрегата
# --------------------------------------------------------------------------- #


def test_aggregate_sums_absolute_values_of_category():
    src = LedgerAggregateSource(rows(
        {"amount": -1000.0, "cat": "capex"},
        {"amount": -500.0, "cat": "capex"},
        {"amount": -9999.0, "cat": "opex"},
    ))
    assert src.aggregate("capex") == 1500.0


def test_any_category_matches_everything():
    src = LedgerAggregateSource(rows(
        {"amount": -100.0, "cat": "capex"}, {"amount": -200.0, "cat": "opex"},
    ))
    assert src.aggregate("any") == 300.0


def test_party_filter_selects_related_only():
    src = LedgerAggregateSource(rows(
        {"amount": -100.0, "cat": "opex", "party": "related"},
        {"amount": -900.0, "cat": "opex"},
    ))
    assert src.aggregate("any", party="related") == 100.0


def test_scope_filter_separates_group_from_borrower():
    src = LedgerAggregateSource(rows(
        {"amount": -1000.0, "cat": "capex"},
        {"amount": -9000.0, "cat": "capex", "scope": "group"},
    ))
    assert src.aggregate("capex") == 1000.0
    assert src.aggregate("capex", scope="group") == 9000.0


def test_period_filter_slices_by_date():
    src = LedgerAggregateSource(rows(
        {"amount": -100.0, "cat": "revenue", "date": "2025-03-15"},
        {"amount": -200.0, "cat": "revenue", "date": "2025-11-20"},
    ))
    assert src.aggregate("revenue", period=("2025-10-01", "2025-12-31")) == 200.0


def test_excluded_rows_are_not_counted():
    """Операции, отсечённые правилом периода на шаге 11, в расчёт не идут."""
    src = LedgerAggregateSource(rows(
        {"amount": -100.0, "cat": "revenue"},
        {"amount": -900.0, "cat": "revenue", "excluded": True},
    ))
    assert src.aggregate("revenue") == 100.0


def test_counterfactual_exclusion_for_evidence_search():
    """Шаг 13 убирает операцию и пересчитывает: механизм обязан быть здесь."""
    data = rows({"id": "TXN-P1-0001", "amount": -100.0, "cat": "capex"},
                {"id": "TXN-P1-0002", "amount": -400.0, "cat": "capex"})
    assert LedgerAggregateSource(data).aggregate("capex") == 500.0
    assert LedgerAggregateSource(data, exclude={"TXN-P1-0002"}).aggregate("capex") == 100.0


# --------------------------------------------------------------------------- #
# Трассировка
# --------------------------------------------------------------------------- #


def test_trace_records_which_transactions_entered_the_aggregate():
    src = LedgerAggregateSource(rows(
        {"id": "TXN-P1-0001", "amount": -100.0, "cat": "capex"},
        {"id": "TXN-P1-0002", "amount": -200.0, "cat": "capex"},
        {"id": "TXN-P1-0003", "amount": -300.0, "cat": "opex"},
    ))
    src.aggregate("capex")
    assert src.traces[-1].txn_ids == ["TXN-P1-0001", "TXN-P1-0002"]
    assert src.traces[-1].value == 300.0


def test_trace_covers_both_sides_of_a_ratio():
    src = LedgerAggregateSource(rows(
        {"id": "TXN-P1-0001", "amount": 8000.0, "cat": "revenue"},
        {"id": "TXN-P1-0002", "amount": -2000.0, "cat": "opex"},
    ))
    cell = compute_cell("P1", CovenantTest(
        "6.1", "min", 2.0, OP("DIV", AGG("revenue"), AGG("opex")), unit="ratio",
    ), src)
    assert cell.actual == pytest.approx(4.0) and cell.status == COMPLIANT
    assert {t["category"] for t in cell.trace} == {"revenue", "opex"}
    assert sum(t["n_txns"] for t in cell.trace) == 2


# --------------------------------------------------------------------------- #
# Громкость на расхождениях
# --------------------------------------------------------------------------- #


def test_empty_category_is_flagged_as_vocabulary_mismatch():
    """Главный стык проекта: словарь категорий шага 6 против шага 10.
    Молчаливый ноль здесь означает неверный actual и, вероятно, статус."""
    src = LedgerAggregateSource(rows({"amount": -100.0, "cat": "opex"}))
    cell = compute_cell("P1", CovenantTest("6.2", "max", 1000.0, AGG("capital_expenditure")), src)
    assert cell.actual == 0.0
    assert any("расхождение словаря категорий" in p for p in cell.problems)


def test_untagged_related_parties_are_flagged():
    """Ноль платежей связанным сторонам чаще означает провал разметки
    на шаге 8, чем реальное отсутствие таких платежей."""
    src = LedgerAggregateSource(rows({"amount": -100.0, "cat": "opex"}))
    cell = compute_cell("P1", CovenantTest(
        "6.3", "max", 450_000.0, AGG("any", party="related")), src)
    assert any("party=related" in p for p in cell.problems)


def test_missing_disclosed_value_is_flagged():
    src = LedgerAggregateSource(rows({"amount": -100.0, "cat": "payroll"}))
    cell = compute_cell("P1", CovenantTest(
        "6.1", "max", 1000.0,
        OP("ADD", AGG("payroll"), {"op": "DISCLOSED", "key": "severance"})), src)
    assert any("disclosed:severance" in p for p in cell.problems)


def test_unknown_form_does_not_stop_other_cells():
    src = LedgerAggregateSource(rows({"amount": -100.0, "cat": "opex"}))
    bad = compute_cell("P1", CovenantTest("6.1", "max", 1.0, {"op": "НЕЧТО"}), src)
    good = compute_cell("P1", CovenantTest("6.2", "max", 1000.0, AGG("opex")), src)
    assert math.isnan(bad.actual) and bad.problems
    assert good.actual == 100.0 and good.status == COMPLIANT


# --------------------------------------------------------------------------- #
# Сериализация
# --------------------------------------------------------------------------- #


def test_actual_is_rounded_to_two_decimals():
    src = LedgerAggregateSource(rows(
        {"amount": 10_000.0, "cat": "revenue"}, {"amount": -3_000.0, "cat": "opex"},
    ))
    cell = compute_cell("P1", CovenantTest(
        "6.1", "min", 1.0, OP("DIV", AGG("revenue"), AGG("opex")), unit="ratio"), src)
    assert cell.to_dict()["actual"] == 3.33


def test_nan_actual_serialises_as_null_not_nan():
    """NaN в JSON ломает разбор у принимающей стороны."""
    src = LedgerAggregateSource([])
    cell = compute_cell("P1", CovenantTest("6.1", "max", 1.0, {"op": "НЕЧТО"}), src)
    payload = json.dumps(cell.to_dict())
    assert '"actual": null' in payload and "NaN" not in payload


# --------------------------------------------------------------------------- #
# Сквозной прогон
# --------------------------------------------------------------------------- #


@pytest.fixture
def prepared(tmp_path):
    paths = RunPaths.create(tmp_path / "run")
    (paths.artifacts / A.LEDGER_CLEAN).write_text(
        "txn_id,scenario_id,date,counterparty,description,amount,currency,amount_usd,"
        "category,party,scope,excluded\n"
        "TXN-P1-0001,P1,2025-02-01,Alpha,rev,8000000,USD,8000000,revenue,,borrower,0\n"
        "TXN-P1-0002,P1,2025-03-01,Beta,opex,-5000000,USD,-5000000,opex,,borrower,0\n"
        "TXN-P1-0003,P1,2025-04-01,Gamma,rp,-500000,USD,-500000,opex,related,borrower,0\n"
        "TXN-P2-0001,P2,2025-05-01,Delta,cap,-3500000,USD,-3500000,capex,,borrower,0\n",
        encoding="utf-8",
    )
    (paths.artifacts / A.COVENANTS).write_text(json.dumps({
        "scenarios": {
            "P1": {
                "6.1": {"direction": "min", "threshold": 2.0, "unit": "ratio",
                        "metric": {"op": "DIV",
                                   "args": [{"op": "AGG", "category": "revenue"},
                                            {"op": "AGG", "category": "opex"}]},
                        "quote": "не ниже 2.00x"},
                "6.3": {"direction": "max", "threshold": 450000.0, "unit": "amount",
                        "metric": {"op": "AGG", "category": "any", "party": "related"},
                        "quote": "не более $450,000.00"},
            },
            "P2": {
                "6.2": {"direction": "max", "threshold": 3000000.0, "unit": "amount",
                        "metric": {"op": "AGG", "category": "capex"}},
            },
        }
    }), encoding="utf-8")
    return paths


def test_end_to_end_produces_results_artifact(prepared):
    results = compute_run(prepared)

    p1 = {c.point: c for c in results["P1"]}
    # revenue 8 000 000 / opex 5 500 000 = 1.4545…
    # Знаменатель включает и платёж связанной стороне: он тоже opex.
    # На этом я ошибся при написании теста — считал 5 000 000.
    assert p1["6.1"].actual == pytest.approx(8_000_000 / 5_500_000)
    assert p1["6.1"].status == BREACH
    # 500 000 > 450 000
    assert p1["6.3"].actual == pytest.approx(500_000.0)
    assert p1["6.3"].status == BREACH

    p2 = {c.point: c for c in results["P2"]}
    assert p2["6.2"].actual == pytest.approx(3_500_000.0)
    assert p2["6.2"].status == BREACH

    out = json.loads((prepared.artifacts / A.RESULTS).read_text(encoding="utf-8"))
    assert set(out) == {"P1", "P2"}
    assert out["P1"]["6.1"]["actual"] == 1.45
    assert out["P1"]["6.1"]["trace"], "трассировка обязана попасть в артефакт"


def test_scenarios_are_isolated_from_each_other(prepared):
    """Операции P2 не должны попасть в агрегаты P1 — иначе все суммы поедут."""
    results = compute_run(prepared)
    p1_ids = {tid for c in results["P1"] for t in c.trace for tid in t["txn_ids"]}
    assert all(tid.startswith("TXN-P1-") for tid in p1_ids)


def test_loaders_read_what_the_previous_steps_write(prepared):
    rows_ = load_rows(prepared.artifacts / A.LEDGER_CLEAN)
    assert len(rows_) == 4
    assert rows_[2].party == "related"

    tests = load_tests(prepared.artifacts / A.COVENANTS)
    assert sorted(tests) == ["P1", "P2"]
    assert [t.point for t in tests["P1"]] == ["6.1", "6.3"]
    assert tests["P1"][0].unit == "ratio" and tests["P1"][0].quote


def test_scenario_without_transactions_is_reported(prepared, caplog):
    data = json.loads((prepared.artifacts / A.COVENANTS).read_text(encoding="utf-8"))
    data["scenarios"]["P9"] = {
        "6.1": {"direction": "max", "threshold": 1.0, "metric": {"op": "AGG", "category": "capex"}}
    }
    (prepared.artifacts / A.COVENANTS).write_text(json.dumps(data), encoding="utf-8")

    import logging

    with caplog.at_level(logging.WARNING):
        compute_run(prepared)
    assert any("нет ни одной операции" in r.getMessage() for r in caplog.records)
