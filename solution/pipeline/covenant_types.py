"""Шаг 6: таксономия ковенантных тестов.

Выход: реестр вычислимых форм, на который опирается расчётный движок (шаг 12).

ПОЧЕМУ ДЕРЕВО ВЫРАЖЕНИЙ, А НЕ СПИСОК ТИПОВ

Разбор всех 36 ковенантов публичного набора дал шесть форм: агрегат против
порога, отношение двух агрегатов, максимум из нескольких статей, агрегат за
вычетом максимума, условный (springing) тест и агрегат с добавлением
внебалансовой суммы. Соблазн — завести шесть типов и по типу диспетчеризовать.

Это ловушка. Приватный набор почти наверняка принесёт седьмую форму, и тогда
придётся дописывать код в боевом окне, где на это нет времени. Поэтому вместо
перечисления форм здесь введён один примитив — агрегат по категории — и
набор операций над ним. Любая из шести наблюдённых форм выражается деревом:

    агрегат            AGG(capex)
    отношение          DIV(SUB(AGG(revenue), AGG(opex)), AGG(interest))
    максимум из статей MAX(AGG(payroll), AGG(utilities))
    агрегат минус макс SUB(AGG(revenue), MAX(AGG(payroll), AGG(taxes)))

Новая форма — новая комбинация тех же узлов, без правки кода.

СЕМАНТИКА actual

`actual` — фактическое значение показателя, ВСЕГДА положительное. В реестре
списания отрицательны, поэтому агрегат берёт модуль. Для условных тестов
`actual` возвращается всегда, в том числе когда условие не сработало и
статус COMPLIANT: по условию задачи оно может превышать порог.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

# --------------------------------------------------------------------------- #
# Контекст: доступ к агрегатам
# --------------------------------------------------------------------------- #


class AggregateSource(Protocol):
    """Источник агрегатов. Реализуется шагом 11 поверх готового реестра."""

    def aggregate(
        self,
        category: str,
        scope: str = "borrower",
        party: str | None = None,
        period: tuple[str, str] | None = None,
    ) -> float:
        """Сумма модулей по категории за период.

        scope:  'borrower' — только заёмщик; 'group' — вся группа по
                консолидированной отчётности материнской компании.
        party:  None — все контрагенты; 'related' — только связанные стороны;
                'unrestricted_subsidiary' — только неограниченные дочерние.
        """
        ...

    def disclosed(self, key: str) -> float:
        """Внебалансовая величина, раскрытая в документах и отсутствующая
        отдельной операцией в реестре."""
        ...


class UnknownNode(ValueError):
    """Незнакомый узел выражения.

    Существует, чтобы неизвестная форма помечала ячейку флагом, а не роняла
    прогон: одиннадцать заёмщиков не должны страдать из-за двенадцатого.
    """


# --------------------------------------------------------------------------- #
# Узлы выражения
# --------------------------------------------------------------------------- #

#: Каждый узел — словарь с ключом "op". Словарь, а не класс, потому что
#: дерево приходит от модели в виде JSON и уходит в артефакт как JSON.
NodeEvaluator = Callable[[dict, AggregateSource, tuple[str, str] | None], float]

REGISTRY: dict[str, NodeEvaluator] = {}


def node(op: str) -> Callable[[NodeEvaluator], NodeEvaluator]:
    def register(fn: NodeEvaluator) -> NodeEvaluator:
        REGISTRY[op] = fn
        return fn

    return register


def evaluate(expr: dict, src: AggregateSource, period: tuple[str, str] | None = None) -> float:
    if not isinstance(expr, dict) or "op" not in expr:
        raise UnknownNode(f"узел без операции: {expr!r}")
    op = expr["op"]
    if op not in REGISTRY:
        raise UnknownNode(
            f"неизвестная операция {op!r}; известны: {sorted(REGISTRY)}. "
            f"Добавьте обработчик в covenant_types.REGISTRY"
        )
    return REGISTRY[op](expr, src, period)


def _args(expr: dict, src: AggregateSource, period) -> list[float]:
    return [evaluate(a, src, period) for a in expr.get("args", [])]


@node("AGG")
def _agg(expr, src, period) -> float:
    """Сумма по категории. Всегда неотрицательна — берётся модуль."""
    own_period = expr.get("period")
    return abs(
        src.aggregate(
            category=expr["category"],
            scope=expr.get("scope", "borrower"),
            party=expr.get("party"),
            period=tuple(own_period) if own_period else period,
        )
    )


@node("DISCLOSED")
def _disclosed(expr, src, period) -> float:
    """Внебалансовая величина из документов (программа выходных пособий и т.п.)."""
    return abs(src.disclosed(expr["key"]))


@node("CONST")
def _const(expr, src, period) -> float:
    return float(expr["value"])


@node("ADD")
def _add(expr, src, period) -> float:
    return sum(_args(expr, src, period))


@node("SUB")
def _sub(expr, src, period) -> float:
    vals = _args(expr, src, period)
    if not vals:
        raise UnknownNode("SUB без аргументов")
    return vals[0] - sum(vals[1:])


@node("MUL")
def _mul(expr, src, period) -> float:
    out = 1.0
    for v in _args(expr, src, period):
        out *= v
    return out


@node("DIV")
def _div(expr, src, period) -> float:
    vals = _args(expr, src, period)
    if len(vals) != 2:
        raise UnknownNode(f"DIV ожидает два аргумента, получено {len(vals)}")
    num, den = vals
    if den == 0:
        # Ноль в знаменателе — не повод для исключения: это состояние данных,
        # о котором обязан узнать шаг 15, а не аварийная ситуация.
        return math.inf if num > 0 else 0.0
    return num / den


@node("MAX")
def _max(expr, src, period) -> float:
    vals = _args(expr, src, period)
    if not vals:
        raise UnknownNode("MAX без аргументов")
    return max(vals)


@node("MIN")
def _min(expr, src, period) -> float:
    vals = _args(expr, src, period)
    if not vals:
        raise UnknownNode("MIN без аргументов")
    return min(vals)


@node("ABS")
def _abs(expr, src, period) -> float:
    vals = _args(expr, src, period)
    if len(vals) != 1:
        raise UnknownNode("ABS ожидает один аргумент")
    return abs(vals[0])


# --------------------------------------------------------------------------- #
# Ковенант целиком
# --------------------------------------------------------------------------- #

COMPLIANT = "COMPLIANT"
BREACH = "BREACH"


@dataclass
class CovenantTest:
    """Вычислимая форма ковенанта.

    metric     — дерево выражений, дающее фактическое значение показателя;
    direction  — 'max' (не выше порога) или 'min' (не ниже порога);
    threshold  — порог, всегда положительный;
    condition  — необязательное условие срабатывания (springing): ковенант
                 проверяется, только если оно истинно.
    """

    point: str
    direction: str
    threshold: float
    metric: dict
    unit: str = "amount"
    period: tuple[str, str] | None = None
    condition: dict | None = None
    quote: str = ""
    notes: list[str] = field(default_factory=list)


@dataclass
class TestResult:
    actual: float
    status: str
    condition_met: bool = True
    condition_value: float | None = None
    problems: list[str] = field(default_factory=list)


def compare(actual: float, direction: str, threshold: float) -> str:
    if direction == "max":
        return BREACH if actual > threshold else COMPLIANT
    if direction == "min":
        return BREACH if actual < threshold else COMPLIANT
    raise UnknownNode(f"неизвестное направление {direction!r}, ожидалось 'max' или 'min'")


def evaluate_condition(cond: dict, src: AggregateSource, period) -> tuple[bool, float]:
    """Условие springing-теста: {'metric': <дерево>, 'direction', 'threshold'}.

    direction здесь означает, при каком соотношении ковенант ВКЛЮЧАЕТСЯ.
    """
    value = evaluate(cond["metric"], src, period)
    threshold = float(cond["threshold"])
    d = cond.get("direction", "max")
    met = value > threshold if d == "max" else value < threshold
    return met, value


def run_test(test: CovenantTest, src: AggregateSource) -> TestResult:
    """Вычисляет ковенант. Исключения не пробрасываются — они становятся
    проблемами в результате, чтобы падение одной ячейки не рушило прогон."""
    problems: list[str] = []
    period = test.period

    condition_met, condition_value = True, None
    if test.condition:
        try:
            condition_met, condition_value = evaluate_condition(test.condition, src, period)
        except (UnknownNode, KeyError, TypeError) as exc:
            problems.append(f"условие не вычислено ({exc}); ковенант считается применимым")
            condition_met = True

    try:
        actual = evaluate(test.metric, src, period)
    except (UnknownNode, KeyError, TypeError) as exc:
        return TestResult(
            actual=float("nan"), status=COMPLIANT, condition_met=condition_met,
            condition_value=condition_value,
            problems=[f"показатель не вычислен: {exc}"],
        )

    if not math.isfinite(actual):
        problems.append(f"показатель не конечен ({actual}) — проверьте знаменатель")

    # Условие не сработало — ковенант не применяется, но actual возвращается
    # всё равно: по условию задачи он может превышать порог при COMPLIANT.
    status = compare(abs(actual), test.direction, test.threshold) if condition_met else COMPLIANT
    if not condition_met:
        problems.append(
            f"условие применения не выполнено (значение {condition_value}), "
            f"ковенант не проверяется"
        )

    return TestResult(
        actual=abs(actual), status=status,
        condition_met=condition_met, condition_value=condition_value,
        problems=problems,
    )


# --------------------------------------------------------------------------- #
# Справочник форм — документация и материал для промпта шага 5
# --------------------------------------------------------------------------- #

#: Формы, наблюдённые в публичном наборе, с примерами деревьев. Используются
#: как few-shot материал для извлечения и как проверочный список при разборе
#: незнакомых формулировок приватного набора.
OBSERVED_FORMS: dict[str, dict[str, Any]] = {
    "aggregate": {
        "описание": "Сумма по категории против порога. Самая частая форма.",
        "пример": "совокупные расходы по статье «Капитальные затраты» ≤ $2,000,000",
        "дерево": {"op": "AGG", "category": "capex"},
    },
    "ratio": {
        "описание": "Отношение двух агрегатов.",
        "пример": "EBITDA (Выручка − Операционные расходы) / Процентные расходы ≥ 2.00x",
        "дерево": {
            "op": "DIV",
            "args": [
                {"op": "SUB", "args": [{"op": "AGG", "category": "revenue"},
                                       {"op": "AGG", "category": "opex"}]},
                {"op": "AGG", "category": "interest"},
            ],
        },
    },
    "proportion": {
        "описание": "Доля одного агрегата от другого. Порог — коэффициент.",
        "пример": "платежи связанным сторонам ≤ 0.04x выручки",
        "дерево": {
            "op": "DIV",
            "args": [{"op": "AGG", "category": "any", "party": "related"},
                     {"op": "AGG", "category": "revenue"}],
        },
    },
    "max_of_items": {
        "описание": (
            "Наибольшая из нескольких статей, НЕ их сумма. Формулировка-признак: "
            "«их сумма не является показателем настоящего ковенанта»."
        ),
        "пример": "ни одна отдельная статья накладных расходов не выше $1,500,000",
        "дерево": {"op": "MAX", "args": [{"op": "AGG", "category": "payroll"},
                                         {"op": "AGG", "category": "utilities"}]},
    },
    "aggregate_minus_max": {
        "описание": "Агрегат за вычетом наибольшей из статей. «Меньшая в расчёт не принимается».",
        "пример": "Выручка − max(Расходы на оплату труда, Налоги) ≥ $5,000,000",
        "дерево": {
            "op": "SUB",
            "args": [{"op": "AGG", "category": "revenue"},
                     {"op": "MAX", "args": [{"op": "AGG", "category": "payroll"},
                                            {"op": "AGG", "category": "taxes"}]}],
        },
    },
    "conditional": {
        "описание": (
            "Springing-тест: применяется только при срабатывании условия. "
            "actual возвращается всегда и может превышать порог при COMPLIANT."
        ),
        "пример": "отношение ≤ 1.70x, но только если поступления по финансированию > $4,000,000",
        "дерево": {"op": "DIV", "args": [{"op": "AGG", "category": "financing_inflow"},
                                         {"op": "AGG", "category": "ebitda"}]},
        "условие": {"metric": {"op": "AGG", "category": "financing_inflow"},
                    "direction": "max", "threshold": 4000000.0},
    },
    "aggregate_with_disclosed": {
        "описание": (
            "Агрегат плюс внебалансовая величина, раскрытая в документах и "
            "отсутствующая отдельной операцией в реестре."
        ),
        "пример": "расходы на оплату труда + обязательство по программе выходных пособий",
        "дерево": {
            "op": "ADD",
            "args": [{"op": "AGG", "category": "payroll"},
                     {"op": "DISCLOSED", "key": "severance_programme"}],
        },
    },
    "group_scope": {
        "описание": (
            "Агрегат по всей группе, а не по заёмщику: берётся из "
            "консолидированной отчётности материнской компании."
        ),
        "пример": "капитальные затраты Группы / EBITDA Заёмщика ≤ 9.00x",
        "дерево": {
            "op": "DIV",
            "args": [{"op": "AGG", "category": "capex", "scope": "group"},
                     {"op": "SUB", "args": [{"op": "AGG", "category": "revenue"},
                                            {"op": "AGG", "category": "opex"}]}],
        },
    },
    "sub_period": {
        "описание": "Агрегат за часть периода (квартал), а не за весь период.",
        "пример": "Выручка за четвёртый квартал ≥ $3,500,000",
        "дерево": {"op": "AGG", "category": "revenue",
                   "period": ["2025-10-01", "2025-12-31"]},
    },
}
