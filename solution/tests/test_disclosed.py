"""Тесты шага 7б: раскрытые аудитором величины для узла DISCLOSED.

Здесь три вещи, которые нельзя сломать молча:

  * порог существенности — статьи ниже порога НЕ складываются (у P4
    сумма всех трёх дала бы на 30% больше правильной);
  * повторный учёт — «внебалансовая» сумма, уже восстановленная в реестр
    (TXN-P8-0031), не прибавляется второй раз;
  * запрет на угадывание — несопоставленный ключ остаётся пустым
    с предупреждением, а не получает первое попавшееся число.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline import artifacts as A  # noqa: E402
from pipeline import disclosed  # noqa: E402
from pipeline.adjustments import Note, ScenarioAdjustments  # noqa: E402
from pipeline.config import RunPaths  # noqa: E402
from pipeline.disclosed import collect_pools, match_key, requested_keys  # noqa: E402


# --------------------------------------------------------------------------- #
# Помощники
# --------------------------------------------------------------------------- #


def _note(note_id="7.1", kind="off_ledger", status="applied", value=100.0,
          txn_id=None, material=True):
    return Note(note_id=note_id, kind=kind, status=status, value_usd=value,
                target_txn_id=txn_id, material=material)


def _adj(scenario="P8", notes=()):
    return ScenarioAdjustments(scenario_id=scenario, notes=list(notes))


def _write_run(tmp_path, covenants: dict, adjustments: dict, ledger_rows: str):
    rp = RunPaths.create(tmp_path / "run")
    (rp.artifacts / A.COVENANTS).write_text(
        json.dumps(covenants, ensure_ascii=False), encoding="utf-8")
    (rp.artifacts / A.AUDIT_ADJUSTMENTS).write_text(
        json.dumps(adjustments, ensure_ascii=False), encoding="utf-8")
    (rp.artifacts / A.LEDGER_CLEAN).write_text(ledger_rows, encoding="utf-8")
    return rp


LEDGER = (
    "txn_id,scenario_id,date,counterparty,amount,currency,amount_usd\n"
    "TXN-P8-0031,P8,2025-12-04,Kyzylorda Drilling Personnel LLP,"
    "-884204.16,USD,-884204.16\n"
    "TXN-P4-0001,P4,2025-03-01,Aral Freight Arbitration Bureau,"
    "-342905.28,USD,-342905.28\n"
)


def _covenants(scenario: str, key: str) -> dict:
    return {"scenarios": {scenario: {"covenants": [{
        "point": "6.1", "direction": "max", "threshold": 1.0,
        "metric_nodes": [
            {"id": "root", "op": "ADD", "args": ["agg", "disc"]},
            {"id": "agg", "op": "AGG", "category": "payroll"},
            {"id": "disc", "op": "DISCLOSED", "key": key},
        ],
        "metric_root": "root",
        "metric": {"op": "ADD", "args": [
            {"op": "AGG", "category": "payroll"},
            {"op": "DISCLOSED", "key": key},
        ]},
    }]}}}


# --------------------------------------------------------------------------- #
# Какие ключи запрашивают деревья
# --------------------------------------------------------------------------- #


def test_requested_keys_are_found_in_both_tree_forms():
    """Дерево живёт и в плоском metric_nodes, и во вложенном metric —
    обе формы поддерживает compute.load_tests, значит обе обязан видеть
    и этот шаг."""
    flat_only = {"scenarios": {"X": {"covenants": [{
        "metric_nodes": [{"op": "DISCLOSED", "key": "a"}]}]}}}
    nested_only = {"scenarios": {"X": {"covenants": [{
        "metric": {"op": "ADD", "args": [{"op": "DISCLOSED", "key": "b"}]}}]}}}
    assert requested_keys(flat_only) == {"X": ["a"]}
    assert requested_keys(nested_only) == {"X": ["b"]}


def test_scenarios_without_disclosed_nodes_are_absent():
    payload = {"scenarios": {"X": {"covenants": [{
        "metric": {"op": "AGG", "category": "opex"}}]}}}
    assert requested_keys(payload) == {}


# --------------------------------------------------------------------------- #
# Порог существенности
# --------------------------------------------------------------------------- #


def test_immaterial_addbacks_are_not_summed():
    """Форма P4: три разовые статьи, порог прошли две. Поле material
    проставил шаг 7 — здесь оно уважается, а не пересчитывается."""
    adj = _adj("P4", [
        _note("8.1", "ebitda_adjustment", value=251338.94, material=False),
        _note("8.2", "ebitda_adjustment", value=342905.28),
        _note("8.3", "ebitda_adjustment", value=481247.63),
    ])
    pools, _ = collect_pools(adj, set(), set())
    assert round(sum(n.value_usd for n in pools["addback"]), 2) == 824152.91


def test_non_applied_notes_do_not_contribute():
    """Статусы referred_elsewhere и considered_but_rejected — ловушки
    набора: суммы с виду обычные, применять их нельзя."""
    adj = _adj(notes=[
        _note(status="referred_elsewhere"),
        _note(status="considered_but_rejected"),
        _note(status="informational"),
    ])
    pools, _ = collect_pools(adj, set(), set())
    assert pools["addback"] == [] and pools["off_ledger"] == []


# --------------------------------------------------------------------------- #
# Повторный учёт
# --------------------------------------------------------------------------- #


def test_off_ledger_note_pointing_at_a_ledger_txn_is_dropped():
    """Форма P8 п.8.1: сумма восстановлена шагом 9 и уже лежит в реестре
    строкой TXN-P8-0031. Прибавить её к DISCLOSED — посчитать дважды."""
    adj = _adj(notes=[
        _note("7.1", value=918447.52),
        _note("8.1", value=884204.16, txn_id="TXN-P8-0031"),
    ])
    pools, remarks = collect_pools(adj, {"TXN-P8-0031"}, {884204.16})
    assert [n.note_id for n in pools["off_ledger"]] == ["7.1"]
    assert any("повторный учёт" in r for r in remarks)


def test_off_ledger_note_matching_a_ledger_amount_is_dropped():
    """Номера операции может не быть — тогда выдаёт совпадение суммы
    с точностью до цента среди операций ТОГО ЖЕ заёмщика."""
    adj = _adj(notes=[_note(value=884204.16)])
    pools, remarks = collect_pools(adj, set(), {884204.16})
    assert pools["off_ledger"] == []
    assert any("совпадает" in r for r in remarks)


def test_addbacks_are_exempt_from_the_double_count_check():
    """Разовые статьи EBITDA — реальные операции реестра, добавляемые
    обратно. Совпадение с реестром для них норма, а не повторный учёт."""
    adj = _adj("P4", [_note("8.2", "ebitda_adjustment", value=342905.28)])
    pools, _ = collect_pools(adj, {"TXN-P4-0001"}, {342905.28})
    assert len(pools["addback"]) == 1


# --------------------------------------------------------------------------- #
# Сопоставление ключа с группой
# --------------------------------------------------------------------------- #


def test_key_names_are_matched_by_meaning_not_by_exact_string():
    """Имена ключей сочиняет модель, в приватном наборе они будут другими."""
    pools = {"addback": [_note(kind="ebitda_adjustment")], "off_ledger": [_note()]}
    assert match_key("auditor_one_off_addbacks", pools)[0] == "addback"
    assert match_key("severance_programme_obligations", pools)[0] == "off_ledger"
    assert match_key("one_time_items_2026", pools)[0] == "addback"
    assert match_key("off_balance_sheet_liabilities", pools)[0] == "off_ledger"


def test_unrecognised_key_falls_back_to_the_only_non_empty_pool():
    pools = {"addback": [], "off_ledger": [_note()]}
    pool, how = match_key("something_the_model_invented", pools)
    assert pool == "off_ledger"
    assert "единственности" in how


def test_ambiguous_key_is_refused_not_guessed():
    """Неверное непустое число хуже честного нуля: ноль поднимает флаг
    «агрегат пуст», а число проходит молча."""
    pools = {"addback": [_note(kind="ebitda_adjustment")], "off_ledger": [_note()]}
    pool, _ = match_key("something_the_model_invented", pools)
    assert pool is None


# --------------------------------------------------------------------------- #
# Сквозной прогон
# --------------------------------------------------------------------------- #


def test_artifact_is_built_in_the_contract_shape(tmp_path):
    adjustments = {"scenarios": {"P8": _adj("P8", [
        _note("7.1", value=918447.52),
        _note("8.1", value=884204.16, txn_id="TXN-P8-0031"),
    ]).to_dict()}}
    rp = _write_run(tmp_path,
                    _covenants("P8", "severance_programme_obligations"),
                    adjustments, LEDGER)

    report = disclosed.run(rp)

    written = json.loads((rp.artifacts / A.DISCLOSED).read_text(encoding="utf-8"))
    assert written == {"P8": {"severance_programme_obligations": 918447.52}}
    assert report.problems == []


def test_unmatched_key_stays_empty_and_is_reported(tmp_path):
    """Нет подходящих примечаний — ключ не пишется вовсе: пустой агрегат
    поднимет флаг в расчёте, и ячейка попадёт в разбор."""
    adjustments = {"scenarios": {"P8": _adj("P8", []).to_dict()}}
    rp = _write_run(tmp_path,
                    _covenants("P8", "severance_programme_obligations"),
                    adjustments, LEDGER)

    report = disclosed.run(rp)

    written = json.loads((rp.artifacts / A.DISCLOSED).read_text(encoding="utf-8"))
    assert written == {}
    assert any("не сопоставлен" in p for p in report.problems)


def test_compute_reads_the_artifact_end_to_end(tmp_path):
    """Контракт с шагом 12: форма {сценарий: {ключ: сумма}} читается
    расчётом, и узел DISCLOSED перестаёт возвращать ноль."""
    from pipeline import compute

    adjustments = {"scenarios": {"P8": _adj("P8", [
        _note("7.1", value=918447.52),
    ]).to_dict()}}
    rp = _write_run(tmp_path,
                    _covenants("P8", "severance_programme_obligations"),
                    adjustments, LEDGER)
    disclosed.run(rp)

    results = compute.run(rp)
    cell = results["P8"][0]
    # payroll-агрегат пуст (в реестре нет payroll-строк), значит вся
    # величина — из DISCLOSED.
    assert cell.actual == 918447.52
    assert not any("disclosed" in p for p in cell.problems)
