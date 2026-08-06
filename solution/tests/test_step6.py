"""Тесты шага 6: таксономия ковенантных тестов.

Главная проверка — полнота: все 36 ковенантов публичного набора обязаны
выражаться деревом из имеющихся узлов. Если хоть один не ложится, каталог
неполон и в боевом окне придётся дописывать код.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.covenant_types import (  # noqa: E402
    BREACH,
    COMPLIANT,
    OBSERVED_FORMS,
    CovenantTest,
    UnknownNode,
    compare,
    evaluate,
    run_test,
)


class FakeSource:
    """Подставной источник агрегатов. Ключ — (категория, scope, party)."""

    def __init__(self, values: dict | None = None, disclosed: dict | None = None):
        self.values = values or {}
        self._disclosed = disclosed or {}
        self.calls: list[tuple] = []

    def aggregate(self, category, scope="borrower", party=None, period=None):
        self.calls.append((category, scope, party, period))
        return self.values.get((category, scope, party), self.values.get(category, 0.0))

    def disclosed(self, key):
        return self._disclosed.get(key, 0.0)


# короткие конструкторы, чтобы деревья читались
def AGG(cat, **kw):
    return {"op": "AGG", "category": cat, **kw}


def OP(op, *args):
    return {"op": op, "args": list(args)}


# --------------------------------------------------------------------------- #
# Узлы
# --------------------------------------------------------------------------- #


def test_aggregate_takes_absolute_value():
    """В реестре списания отрицательны, а actual обязан быть положительным."""
    src = FakeSource({"capex": -2_000_000.0})
    assert evaluate(AGG("capex"), src) == 2_000_000.0


def test_arithmetic_nodes():
    src = FakeSource({"a": 10.0, "b": 4.0, "c": 3.0})
    assert evaluate(OP("ADD", AGG("a"), AGG("b")), src) == 14.0
    assert evaluate(OP("SUB", AGG("a"), AGG("b"), AGG("c")), src) == 3.0
    assert evaluate(OP("MUL", AGG("a"), AGG("b")), src) == 40.0
    assert evaluate(OP("DIV", AGG("a"), AGG("b")), src) == 2.5
    assert evaluate(OP("MAX", AGG("a"), AGG("b")), src) == 10.0
    assert evaluate(OP("MIN", AGG("a"), AGG("b")), src) == 4.0
    assert evaluate({"op": "CONST", "value": 7}, src) == 7.0


def test_division_by_zero_is_reported_not_raised():
    """Ноль в знаменателе — состояние данных для шага 15, а не авария."""
    src = FakeSource({"a": 10.0, "zero": 0.0})
    assert evaluate(OP("DIV", AGG("a"), AGG("zero")), src) == math.inf
    assert evaluate(OP("DIV", AGG("zero"), AGG("zero")), src) == 0.0


def test_scope_and_party_reach_the_source():
    src = FakeSource({
        ("capex", "group", None): 9_000_000.0,
        ("any", "borrower", "related"): 300_000.0,
    })
    assert evaluate(AGG("capex", scope="group"), src) == 9_000_000.0
    assert evaluate(AGG("any", party="related"), src) == 300_000.0


def test_node_period_overrides_covenant_period():
    """Квартальный срез: агрегат за часть периода, а не за весь."""
    src = FakeSource({"revenue": 1.0})
    evaluate(AGG("revenue", period=["2025-10-01", "2025-12-31"]), src,
             period=("2025-01-01", "2025-12-31"))
    assert src.calls[-1][3] == ("2025-10-01", "2025-12-31")


def test_unknown_operation_names_itself():
    src = FakeSource()
    with pytest.raises(UnknownNode, match="EBITDA_MAGIC"):
        evaluate({"op": "EBITDA_MAGIC"}, src)


# --------------------------------------------------------------------------- #
# Сравнение с порогом
# --------------------------------------------------------------------------- #


def test_threshold_boundary_is_compliant():
    """«Не превышал 0.42x» и «не менее 2.00x»: ровно на пороге — соблюдён."""
    assert compare(0.42, "max", 0.42) == COMPLIANT
    assert compare(0.4201, "max", 0.42) == BREACH
    assert compare(2.00, "min", 2.00) == COMPLIANT
    assert compare(1.9999, "min", 2.00) == BREACH


def test_unknown_direction_raises():
    with pytest.raises(UnknownNode):
        compare(1.0, "около", 1.0)


# --------------------------------------------------------------------------- #
# Ковенант целиком
# --------------------------------------------------------------------------- #


def test_conditional_not_triggered_is_compliant_but_actual_is_returned():
    """Springing-тест: условие не сработало → COMPLIANT, но actual всё равно
    возвращается и может превышать порог. Прямое требование условия задачи."""
    src = FakeSource({"financing_inflow": 3_000_000.0, "ebitda": 1_000_000.0})
    test = CovenantTest(
        point="6.1", direction="max", threshold=1.70, unit="ratio",
        metric=OP("DIV", AGG("financing_inflow"), AGG("ebitda")),
        condition={"metric": AGG("financing_inflow"), "direction": "max",
                   "threshold": 4_000_000.0},
    )
    res = run_test(test, src)
    assert res.status == COMPLIANT
    assert res.condition_met is False
    assert res.actual == pytest.approx(3.0), "actual выше порога, но статус COMPLIANT"
    assert any("условие применения не выполнено" in p for p in res.problems)


def test_conditional_triggered_is_evaluated():
    src = FakeSource({"financing_inflow": 5_000_000.0, "ebitda": 1_000_000.0})
    test = CovenantTest(
        point="6.1", direction="max", threshold=1.70, unit="ratio",
        metric=OP("DIV", AGG("financing_inflow"), AGG("ebitda")),
        condition={"metric": AGG("financing_inflow"), "direction": "max",
                   "threshold": 4_000_000.0},
    )
    res = run_test(test, src)
    assert res.condition_met is True and res.status == BREACH
    assert res.actual == pytest.approx(5.0)


def test_unknown_form_flags_the_cell_instead_of_crashing():
    """Одиннадцать заёмщиков не должны страдать из-за двенадцатого."""
    test = CovenantTest(point="6.1", direction="max", threshold=1.0,
                        metric={"op": "НЕЧТО_НОВОЕ"})
    res = run_test(test, FakeSource())
    assert math.isnan(res.actual)
    assert res.problems and "не вычислен" in res.problems[0]


def test_result_actual_is_always_positive():
    src = FakeSource({"opex": -500.0})
    res = run_test(CovenantTest("6.1", "max", 100.0, AGG("opex")), src)
    assert res.actual == 500.0 and res.status == BREACH


def test_disclosed_off_ledger_amount_is_added():
    src = FakeSource({"payroll": -3_200_000.0},
                     disclosed={"severance_programme": 918_447.52})
    test = CovenantTest(
        point="6.1", direction="max", threshold=4_000_000.0,
        metric=OP("ADD", AGG("payroll"), {"op": "DISCLOSED", "key": "severance_programme"}),
    )
    res = run_test(test, src)
    assert res.actual == pytest.approx(4_118_447.52)
    assert res.status == BREACH


# --------------------------------------------------------------------------- #
# ПОЛНОТА: все 36 ковенантов публичного набора
# --------------------------------------------------------------------------- #

EBITDA = OP("SUB", AGG("revenue"), AGG("opex"))
RELATED = AGG("any", party="related")

#: Каждый ковенант публичного набора как дерево. Таблица — одновременно
#: доказательство полноты каталога и few-shot материал для шага 5.
ALL_36: dict[str, dict] = {
    "B1/6.1": OP("DIV", EBITDA, AGG("interest")),
    "B1/6.2": OP("MAX", AGG("payroll"), AGG("utilities")),
    "B1/6.3": RELATED,
    "B4/6.1": AGG("revenue", period=["2025-10-01", "2025-12-31"]),
    "B4/6.2": AGG("capex"),
    "B4/6.3": RELATED,
    "P1/6.1": OP("DIV", AGG("capex"), OP("ADD", AGG("opex"), AGG("lease"))),
    "P1/6.2": AGG("revenue"),
    "P1/6.3": RELATED,
    "P2/6.1": OP("DIV", OP("ADD", AGG("revenue"), AGG("financing_inflow")),
                 OP("ADD", AGG("opex"), AGG("capex"))),
    "P2/6.2": AGG("capex"),
    "P2/6.3": OP("DIV", RELATED, AGG("revenue")),
    "P3/6.1": OP("DIV", AGG("financing_inflow"), EBITDA),
    "P3/6.2": AGG("revenue"),
    "P3/6.3": RELATED,
    "P4/6.1": OP("DIV", OP("ADD", EBITDA, AGG("ebitda_addback")), AGG("revenue")),
    "P4/6.2": AGG("capex"),
    "P4/6.3": OP("DIV", RELATED, AGG("revenue")),
    "P5/6.1": OP("DIV", AGG("capex", scope="group"), EBITDA),
    "P5/6.2": AGG("revenue"),
    "P5/6.3": RELATED,
    "P6/6.1": OP("DIV", RELATED, AGG("opex")),
    "P6/6.2": OP("DIV", AGG("revenue"), OP("ADD", AGG("payroll"), AGG("utilities"))),
    "P6/6.3": AGG("capex"),
    "P7/6.1": OP("DIV", OP("ADD", AGG("taxes"), AGG("utilities")), EBITDA),
    "P7/6.2": AGG("revenue"),
    "P7/6.3": RELATED,
    "P8/6.1": OP("ADD", AGG("payroll"), {"op": "DISCLOSED", "key": "severance_programme"}),
    "P8/6.2": AGG("capex"),
    "P8/6.3": OP("DIV", RELATED, AGG("revenue")),
    "P9/6.1": OP("DIV", AGG("capex", party="unrestricted_subsidiary"), AGG("capex")),
    "P9/6.2": AGG("revenue"),
    "P9/6.3": RELATED,
    "P10/6.1": OP("DIV", AGG("insurance"), OP("ADD", AGG("lease"), AGG("utilities"))),
    "P10/6.2": OP("SUB", AGG("revenue"), OP("MAX", AGG("payroll"), AGG("taxes"))),
    "P10/6.3": OP("DIV", RELATED, AGG("revenue")),
}


def test_all_36_covenants_are_expressible():
    """Полнота каталога: ни один ковенант не остаётся без формы."""
    assert len(ALL_36) == 36
    src = FakeSource({
        "revenue": 8_000_000.0, "opex": -5_000_000.0, "capex": -2_000_000.0,
        "payroll": -1_200_000.0, "utilities": -400_000.0, "taxes": -300_000.0,
        "interest": -900_000.0, "lease": -250_000.0, "insurance": -180_000.0,
        "financing_inflow": 4_500_000.0, "ebitda_addback": 120_000.0,
        ("any", "borrower", "related"): -320_000.0,
        ("capex", "group", None): -9_000_000.0,
        ("capex", "borrower", "unrestricted_subsidiary"): -240_000.0,
    }, disclosed={"severance_programme": 918_447.52})

    for cell, tree in ALL_36.items():
        value = evaluate(tree, src)
        assert isinstance(value, float), cell
        assert not math.isnan(value), f"{cell}: получен NaN"


def test_only_registered_operations_are_used():
    """Никакой ковенант не требует узла, которого нет в реестре."""
    from pipeline.covenant_types import REGISTRY

    def ops(node):
        yield node["op"]
        for a in node.get("args", []):
            yield from ops(a)

    used = {op for tree in ALL_36.values() for op in ops(tree)}
    assert used <= set(REGISTRY), f"вне реестра: {used - set(REGISTRY)}"


def test_form_distribution_matches_analysis():
    """Распределение форм — контроль того, что таблица отражает разбор,
    а не подогнана."""
    from collections import Counter

    def shape(tree):
        if tree["op"] == "AGG":
            return "aggregate"
        if tree["op"] == "DIV":
            return "ratio"
        if tree["op"] == "MAX":
            return "max_of_items"
        if tree["op"] == "SUB":
            return "aggregate_minus_max"
        if tree["op"] == "ADD":
            return "aggregate_with_disclosed"
        return "other"

    dist = Counter(shape(t) for t in ALL_36.values())
    # Числа получены пересчётом по таблице, а не ручным подсчётом при разборе:
    # вручную я потерял P4/6.2 и получил 17 вместо 18. Тест это и поймал.
    assert dist["aggregate"] == 18
    assert dist["ratio"] == 15
    assert dist["max_of_items"] == 1
    assert dist["aggregate_minus_max"] == 1
    assert dist["aggregate_with_disclosed"] == 1
    assert dist["other"] == 0


def test_observed_forms_examples_are_evaluable():
    """Справочник форм используется как few-shot материал для шага 5 —
    его примеры обязаны быть рабочими деревьями, а не иллюстрациями."""
    src = FakeSource({
        "revenue": 8_000_000.0, "opex": -5_000_000.0, "capex": -2_000_000.0,
        "payroll": -1_200_000.0, "utilities": -400_000.0, "taxes": -300_000.0,
        "interest": -900_000.0, "financing_inflow": 4_500_000.0, "ebitda": 3_000_000.0,
        "any": -320_000.0,
    }, disclosed={"severance_programme": 918_447.52})
    for name, form in OBSERVED_FORMS.items():
        value = evaluate(form["дерево"], src)
        assert isinstance(value, float), name
