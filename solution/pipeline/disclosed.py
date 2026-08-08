"""Шаг 7б: раскрытые аудитором величины для узла DISCLOSED.

Вход:  <run>/artifacts/06_adjustments.json (шаг 7),
       <run>/artifacts/04_covenants.json  (шаг 5),
       <run>/artifacts/06_ledger_clean.csv (шаг 9)
Выход: <run>/artifacts/06_disclosed.json — {"P8": {"ключ": сумма}, ...}

ЗАЧЕМ ЭТОТ ШАГ

Дерево ковенанта умеет спрашивать величину, которой нет в реестре:
узел DISCLOSED("severance_programme_obligations") — обязательство,
раскрытое аудитором без отдельной операции. compute.py и evidence.py
читают этот артефакт с самого начала, но НИКТО его не писал: данные
оставались в 06_adjustments.json, и узел всегда возвращал ноль.

ДВЕ ЛОВУШКИ, ИЗ-ЗА КОТОРЫХ ЭТО НЕ ПРОСТАЯ СУММА

1. Порог существенности. Разовые статьи (ebitda_adjustment) складываются
   только те, что прошли порог из документа: у P4 из трёх статей проходят
   две (824,152.91), сумма всех трёх дала бы на 30% больше. Отбор уже
   сделан шагом 7 (поле material) — здесь он только уважается.

2. Повторный учёт. «Внебалансовая» величина может оказаться операцией,
   которая УЖЕ в реестре: у P8 сумма $884,204.16 из примечания 8.1 — это
   восстановленная шагом 9 операция TXN-P8-0031. Прибавить её к DISCLOSED
   значит посчитать одни деньги дважды. Проверка: примечание, чья операция
   или сумма находится в реестре заёмщика, внебалансовым не считается.

ИМЕНА КЛЮЧЕЙ ПРИХОДЯТ ОТ МОДЕЛИ И В ПРИВАТНОМ НАБОРЕ БУДУТ ДРУГИМИ

Ключ узла DISCLOSED сочиняет шаг 5 по тексту договора, вид примечания —
шаг 7 по тексту аудита. Точного равенства строк не будет никогда, поэтому
сопоставление трёхступенчатое: по смыслу слов в ключе → по единственности
(один запрошенный ключ, одна непустая группа примечаний) → отказ с
предупреждением. Угадывать нельзя: неверное непустое число хуже честного
нуля, потому что ноль поднимает флаг «агрегат пуст», а число — нет.
"""
from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, field

from . import artifacts as A
from .adjustments import Note, ScenarioAdjustments
from .config import RunPaths

log = logging.getLogger(__name__)

#: Слова, по которым ключ узла DISCLOSED узнаётся как «разовые статьи EBITDA».
ADDBACK_HINTS = (
    "addback", "add_back", "one_off", "oneoff", "one_time", "onetime",
    "non_recurring", "nonrecurring", "exceptional", "ebitda",
)
#: Слова, по которым ключ узнаётся как «внебалансовое обязательство».
OFF_LEDGER_HINTS = (
    "off_ledger", "offledger", "off_balance", "offbalance", "severance",
    "obligation", "liabilit", "commitment", "programme", "program",
    "undisclosed", "unrecorded",
)


@dataclass
class DisclosedReport:
    #: Итог: сценарий → {ключ: сумма}. Пишется в артефакт как есть.
    values: dict[str, dict[str, float]] = field(default_factory=dict)
    #: Что и почему НЕ вошло: повторный учёт, порог, несопоставленный ключ.
    remarks: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    def alarms(self) -> list[str]:
        return list(self.problems)


# --------------------------------------------------------------------------- #
# Какие ключи запрашивают деревья ковенантов
# --------------------------------------------------------------------------- #


def requested_keys(covenants_payload: dict) -> dict[str, list[str]]:
    """Сценарий → ключи узлов DISCLOSED в его деревьях.

    Обходится ВЕСЬ JSON спецификации, а не конкретное поле: дерево лежит
    и в плоском metric_nodes, и во вложенном metric, и формы обе живые
    (см. compute.load_tests). Обход по структуре переживает обе.
    """
    out: dict[str, list[str]] = {}

    def walk(node, sink: list[str]) -> None:
        if isinstance(node, dict):
            if node.get("op") == "DISCLOSED" and isinstance(node.get("key"), str):
                if node["key"] not in sink:
                    sink.append(node["key"])
            for value in node.values():
                walk(value, sink)
        elif isinstance(node, list):
            for item in node:
                walk(item, sink)

    for scenario, payload in covenants_payload.get("scenarios", {}).items():
        keys: list[str] = []
        walk(payload, keys)
        if keys:
            out[scenario] = keys
    return out


# --------------------------------------------------------------------------- #
# Группы примечаний
# --------------------------------------------------------------------------- #


def _ledger_index(paths: RunPaths) -> dict[str, tuple[set[str], set[float]]]:
    """Сценарий → (номера операций, модули сумм с точностью до цента)."""
    out: dict[str, tuple[set[str], set[float]]] = {}
    path = paths.artifacts / A.LEDGER_CLEAN
    if not path.exists():
        return out
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            scenario = row.get("scenario_id", "")
            ids, amounts = out.setdefault(scenario, (set(), set()))
            ids.add(row.get("txn_id", ""))
            raw = row.get("amount_usd") or row.get("amount") or ""
            try:
                amounts.add(round(abs(float(raw)), 2))
            except (TypeError, ValueError):
                pass
    return out


def _on_ledger(note: Note, ids: set[str], amounts: set[float]) -> str | None:
    """Почему примечание НЕ внебалансовое, если оно уже в реестре."""
    if note.target_txn_id and note.target_txn_id in ids:
        return f"операция {note.target_txn_id} уже есть в реестре"
    if note.value_usd is not None and round(abs(note.value_usd), 2) in amounts:
        return f"сумма {note.value_usd:,.2f} совпадает с существующей операцией"
    return None


def collect_pools(
    adj: ScenarioAdjustments, ids: set[str], amounts: set[float]
) -> tuple[dict[str, list[Note]], list[str]]:
    """Группы применяемых примечаний по смыслу: addbacks и off_ledger.

    Порог существенности уважается через note.applies (шаг 7 уже проставил
    material). Повторный учёт отсеивается здесь: только off_ledger может
    прикидываться операцией реестра. Разовые статьи EBITDA НАРОЧНО не
    проверяются на совпадение с реестром — они и должны быть реальными
    операциями, добавляемыми обратно к EBITDA.
    """
    pools: dict[str, list[Note]] = {"addback": [], "off_ledger": []}
    remarks: list[str] = []
    for note in adj.notes:
        if not note.applies or note.value_usd is None:
            continue
        if note.kind == "ebitda_adjustment":
            pools["addback"].append(note)
        elif note.kind == "off_ledger":
            reason = _on_ledger(note, ids, amounts)
            if reason:
                remarks.append(
                    f"{adj.scenario_id} п.{note.note_id}: {reason} — "
                    f"к DISCLOSED не прибавляется (повторный учёт)"
                )
                continue
            pools["off_ledger"].append(note)
    return pools, remarks


def match_key(key: str, pools: dict[str, list[Note]]) -> tuple[str | None, str]:
    """Какой группе примечаний отвечает ключ. Возвращает (группа, как решили).

    Сначала слова в самом ключе; если они молчат — единственность:
    ровно одна непустая группа означает, что спутать не с чем.
    """
    lowered = key.lower()
    is_addback = any(h in lowered for h in ADDBACK_HINTS)
    is_off = any(h in lowered for h in OFF_LEDGER_HINTS)
    if is_addback and not is_off:
        return "addback", "по смыслу слов в ключе"
    if is_off and not is_addback:
        return "off_ledger", "по смыслу слов в ключе"

    non_empty = [name for name, notes in pools.items() if notes]
    if len(non_empty) == 1:
        return non_empty[0], "по единственности непустой группы"
    return None, (
        "слова ключа не узнаны" if not non_empty
        else f"ключ неоднозначен между группами {non_empty}"
    )


# --------------------------------------------------------------------------- #
# Прогон
# --------------------------------------------------------------------------- #


def run(paths: RunPaths) -> DisclosedReport:
    from . import adjustments as adj_module

    report = DisclosedReport()

    covenants_path = paths.artifacts / A.COVENANTS
    if not covenants_path.exists():
        report.problems.append("нет артефакта ковенантов — запрошенные ключи неизвестны")
        _write(paths, report)
        return report

    keys_by_scenario = requested_keys(
        json.loads(covenants_path.read_text(encoding="utf-8"))
    )
    adjustments = adj_module.load(paths)
    ledger = _ledger_index(paths)

    for scenario, keys in sorted(keys_by_scenario.items()):
        adj = adjustments.get(scenario)
        if adj is None:
            report.problems.append(
                f"{scenario}: дерево запрашивает DISCLOSED {keys}, "
                f"но корректировок шага 7 нет — узел останется пустым"
            )
            continue

        ids, amounts = ledger.get(scenario, (set(), set()))
        pools, pool_remarks = collect_pools(adj, ids, amounts)
        report.remarks.extend(pool_remarks)

        for key in keys:
            pool_name, how = match_key(key, pools)
            if pool_name is None or not pools[pool_name]:
                report.problems.append(
                    f"{scenario}: ключ {key!r} не сопоставлен с примечаниями "
                    f"({how if pool_name is None else 'группа пуста'}) — "
                    f"узел DISCLOSED останется пустым, ячейка требует разбора"
                )
                continue
            total = round(sum(n.value_usd for n in pools[pool_name]), 2)
            report.values.setdefault(scenario, {})[key] = total
            report.remarks.append(
                f"{scenario}: {key} = {total:,.2f} "
                f"({len(pools[pool_name])} примечаний, {how})"
            )

    _write(paths, report)
    for remark in report.remarks:
        log.info("РАСКРЫТЫЕ: %s", remark)
    for problem in report.problems:
        log.warning("РАСКРЫТЫЕ: %s", problem)
    return report


def _write(paths: RunPaths, report: DisclosedReport) -> None:
    (paths.artifacts / A.DISCLOSED).write_text(
        json.dumps(report.values, ensure_ascii=False, indent=2), encoding="utf-8"
    )
