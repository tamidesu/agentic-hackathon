"""Тесты шага 13: определение evidence_txn_id.

Главное здесь — что контрфактуальной проверки НЕДОСТАТОЧНО. Условие задачи
прямо запрещает считать доказательством строку, которая лишь вносит вклад
в сумму, даже если её удаление переворачивает вердикт.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline import artifacts as A  # noqa: E402
from pipeline.compute import Row  # noqa: E402
from pipeline.config import RunPaths  # noqa: E402
from pipeline.covenant_types import CovenantTest  # noqa: E402
from pipeline.evidence import (  # noqa: E402
    Candidate,
    candidates_from_artifacts,
    find_evidence,
)
from pipeline.evidence import run as evidence_run  # noqa: E402


def AGG(cat, **kw):
    return {"op": "AGG", "category": cat, **kw}


def OP(op, *args):
    return {"op": op, "args": list(args)}


def row(txn_id, amount, cat="opex", party=None, date="2025-06-01", cp="CP"):
    return Row(txn_id=txn_id, scenario_id="P1", date=date, counterparty=cp,
               amount_usd=amount, category=cat, party=party)


# --------------------------------------------------------------------------- #
# Контрфактуальной проверки недостаточно
# --------------------------------------------------------------------------- #


def test_mere_contributor_is_not_evidence():
    """Десять обычных платежей превысили порог. Удаление любого крупного
    переворачивает вердикт — но доказательством не является ни один:
    их трактовка ничем не продиктована, они просто есть."""
    rows = [row(f"TXN-P1-{i:04d}", -60_000.0, "capex") for i in range(1, 11)]
    test = CovenantTest("6.2", "max", 550_000.0, AGG("capex"))

    res = find_evidence(test, rows, candidates=[])
    assert res.txn_id is None, "вклад в сумму — не доказательство"


def test_largest_row_is_not_evidence_by_itself():
    """Условие прямо называет «самую крупную строку» неверным ответом."""
    rows = [row("TXN-P1-0001", -500_000.0, "capex"),
            row("TXN-P1-0002", -60_000.0, "capex")]
    test = CovenantTest("6.2", "max", 400_000.0, AGG("capex"))
    assert find_evidence(test, rows, candidates=[]).txn_id is None


# --------------------------------------------------------------------------- #
# Документально обусловленная операция
# --------------------------------------------------------------------------- #


def test_reclassified_transaction_is_evidence():
    """Аудитор перенёс операцию в проверяемую статью — без неё нарушения нет."""
    rows = [row("TXN-P1-0001", -300_000.0, "capex"),
            row("TXN-P1-0002", -200_000.0, "capex")]
    test = CovenantTest("6.2", "max", 400_000.0, AGG("capex"))
    cand = Candidate("TXN-P1-0002", "reclassified", revert={"category": "opex"})

    res = find_evidence(test, rows, [cand])
    assert res.txn_id == "TXN-P1-0002"
    assert res.basis == "reclassified"


def test_reclassification_reverts_category_rather_than_deleting():
    """Переклассифицированная операция никуда не исчезала — она была
    отнесена к другой статье. В отношении, где одна и та же операция стоит
    и в числителе, и в знаменателе, разница между возвратом и удалением
    меняет результат."""
    rows = [row("TXN-P1-0001", -100_000.0, "capex"),
            row("TXN-P1-0002", -400_000.0, "capex"),
            row("TXN-P1-0003", -500_000.0, "opex")]
    # capex / opex; возврат операции 0002 в opex меняет обе части сразу
    test = CovenantTest("6.1", "max", 0.5, OP("DIV", AGG("capex"), AGG("opex")),
                        unit="ratio")
    cand = Candidate("TXN-P1-0002", "reclassified", revert={"category": "opex"})

    res = find_evidence(test, rows, [cand])
    assert res.txn_id == "TXN-P1-0002"


def test_recovered_amount_is_evidence():
    """Сумма отсутствовала в реестре и восстановлена из документа."""
    rows = [row("TXN-P1-0001", -300_000.0, "payroll"),
            row("TXN-P1-0002", -884_204.16, "payroll")]
    test = CovenantTest("6.1", "max", 1_000_000.0, AGG("payroll"))
    cand = Candidate("TXN-P1-0002", "recovered_amount")

    assert find_evidence(test, rows, [cand]).txn_id == "TXN-P1-0002"


def test_related_party_inclusion_is_evidence():
    """Связь установлена по досье KYC, а не по назначению платежа."""
    rows = [row("TXN-P1-0001", -200_000.0, party="related", cp="Holding LLP"),
            row("TXN-P1-0002", -300_000.0, party="related", cp="Kiln LLP")]
    test = CovenantTest("6.3", "max", 400_000.0, AGG("any", party="related"))
    cands = [Candidate("TXN-P1-0001", "related_party", revert={"party": None}),
             Candidate("TXN-P1-0002", "related_party", revert={"party": None})]

    res = find_evidence(test, rows, cands)
    assert res.txn_id in {"TXN-P1-0001", "TXN-P1-0002"}
    assert res.problems and "решающих операций несколько" in res.problems[0]


def test_candidate_that_does_not_change_verdict_is_rejected():
    """Документально обусловленная, но не решающая — тоже не доказательство."""
    rows = [row("TXN-P1-0001", -900_000.0, "capex"),
            row("TXN-P1-0002", -10_000.0, "capex")]
    test = CovenantTest("6.2", "max", 400_000.0, AGG("capex"))
    cand = Candidate("TXN-P1-0002", "reclassified", revert={"category": "opex"})

    assert find_evidence(test, rows, [cand]).txn_id is None


def test_compliant_cell_can_also_have_evidence():
    """Ковенант соблюдён, но операция удерживает его от нарушения —
    её изъятие тоже меняет вердикт."""
    rows = [row("TXN-P1-0001", 900_000.0, "revenue"),
            row("TXN-P1-0002", 300_000.0, "revenue")]
    test = CovenantTest("6.2", "min", 1_000_000.0, AGG("revenue"))
    cand = Candidate("TXN-P1-0002", "reclassified", revert={"category": "other"})

    res = find_evidence(test, rows, [cand])
    assert res.txn_id == "TXN-P1-0002"


def test_reverting_party_does_not_affect_a_category_aggregate():
    """Признак связанной стороны значим только для ковенантов с фильтром
    по контрагенту. В агрегате по статье такая операция — обычная строка,
    и основанием быть не может."""
    rows = [row("TXN-P1-0001", -300_000.0, "capex", party="related"),
            row("TXN-P1-0002", -200_000.0, "capex")]
    test = CovenantTest("6.2", "max", 400_000.0, AGG("capex"))
    cand = Candidate("TXN-P1-0001", "related_party", revert={"party": None})
    assert find_evidence(test, rows, [cand]).txn_id is None


def test_stronger_basis_wins_when_several_are_decisive():
    rows = [row("TXN-P1-0001", -300_000.0, "capex"),
            row("TXN-P1-0002", -300_000.0, "capex")]
    test = CovenantTest("6.2", "max", 400_000.0, AGG("capex"))
    cands = [
        Candidate("TXN-P1-0001", "cutoff"),                    # исключение
        Candidate("TXN-P1-0002", "reclassified", revert={"category": "opex"}),
    ]
    res = find_evidence(test, rows, cands)
    assert res.txn_id == "TXN-P1-0002" and res.basis == "reclassified"
    assert res.problems and "сильнейшим основанием" in res.problems[0]


# --------------------------------------------------------------------------- #
# Сбор кандидатов
# --------------------------------------------------------------------------- #


def test_candidates_come_from_adjustments_and_kyc():
    rows = [row("TXN-P1-0001", -100.0), row("TXN-P1-0002", -200.0, party="related"),
            row("TXN-P1-0003", -300.0)]
    adjustments = [
        {"target_txn_id": "TXN-P1-0001", "kind": "reclassification",
         "from_category": "opex", "description": "перенесено в капзатраты"},
        {"target_txn_id": "TXN-P1-0003", "kind": "cutoff", "description": "услуги 2026 года"},
        {"target_txn_id": "TXN-P1-0009", "kind": "reclassification"},  # нет в реестре
        {"target_txn_id": "TXN-P1-0001", "kind": "fx_translation"},    # не основание
    ]
    cands = candidates_from_artifacts(rows, adjustments)["P1"]
    by_id = {c.txn_id: c for c in cands}

    assert set(by_id) == {"TXN-P1-0001", "TXN-P1-0002", "TXN-P1-0003"}
    assert by_id["TXN-P1-0001"].basis == "reclassified"
    assert by_id["TXN-P1-0001"].revert == {"category": "opex"}
    assert by_id["TXN-P1-0002"].basis == "related_party"
    assert by_id["TXN-P1-0003"].basis == "cutoff"


def test_candidates_are_deduplicated_by_strongest_basis():
    rows = [row("TXN-P1-0001", -100.0, party="related")]
    adjustments = [{"target_txn_id": "TXN-P1-0001", "kind": "reclassification",
                    "from_category": "opex"}]
    cands = candidates_from_artifacts(rows, adjustments)["P1"]
    assert len(cands) == 1 and cands[0].basis == "reclassified"


def test_candidates_are_grouped_per_scenario():
    rows = [row("TXN-P1-0001", -100.0, party="related")]
    rows.append(Row("TXN-P2-0001", "P2", "2025-06-01", "CP", -100.0, "opex", party="related"))
    cands = candidates_from_artifacts(rows, [])
    assert set(cands) == {"P1", "P2"}
    assert [c.txn_id for c in cands["P2"]] == ["TXN-P2-0001"]


def test_no_candidates_means_no_evidence():
    assert candidates_from_artifacts([row("TXN-P1-0001", -100.0)], []) == {}


# --------------------------------------------------------------------------- #
# Сквозной прогон
# --------------------------------------------------------------------------- #


@pytest.fixture
def prepared(tmp_path):
    paths = RunPaths.create(tmp_path / "run")
    (paths.artifacts / A.LEDGER_CLEAN).write_text(
        "txn_id,scenario_id,date,counterparty,description,amount,currency,amount_usd,"
        "category,party,scope,excluded\n"
        "TXN-P1-0001,P1,2025-02-01,Alpha,x,-300000,USD,-300000,capex,,borrower,0\n"
        "TXN-P1-0002,P1,2025-03-01,Beta,x,-200000,USD,-200000,capex,,borrower,0\n"
        "TXN-P1-0003,P1,2025-04-01,Holding,x,-500000,USD,-500000,opex,related,borrower,0\n",
        encoding="utf-8",
    )
    (paths.artifacts / A.COVENANTS).write_text(json.dumps({
        "scenarios": {"P1": {
            "6.2": {"direction": "max", "threshold": 400000.0, "unit": "amount",
                    "metric": {"op": "AGG", "category": "capex"}},
            "6.3": {"direction": "max", "threshold": 400000.0, "unit": "amount",
                    "metric": {"op": "AGG", "category": "any", "party": "related"}},
        }}
    }), encoding="utf-8")
    # Форма — та же, что пишет шаг 7: сверху alarms/problems/scenarios.
    # Подделывать здесь «удобную» структуру значит проверять несуществующий
    # контракт: ровно так расхождение формы и доехало до расчёта.
    (paths.artifacts / A.AUDIT_ADJUSTMENTS).write_text(json.dumps({
        "alarms": [], "problems": [],
        "scenarios": {"P1": {"scenario_id": "P1", "notes": [
            {"note_id": "7.1", "kind": "reclassification", "status": "applied",
             "target_txn_id": "TXN-P1-0002", "from_category": "opex",
             "to_category": "capex", "description": "перенесено аудитором"}]}}
    }), encoding="utf-8")
    from pipeline.compute import run as compute_run

    compute_run(paths)
    return paths


def test_end_to_end_fills_evidence_field(prepared):
    results = evidence_run(prepared)

    cell62 = results["P1"]["6.2"]
    assert cell62["status"] == "BREACH"
    assert cell62["evidence_txn_id"] == "TXN-P1-0002"
    assert cell62["evidence_basis"] == "reclassified"

    cell63 = results["P1"]["6.3"]
    assert cell63["status"] == "BREACH"
    assert cell63["evidence_txn_id"] == "TXN-P1-0003"

    on_disk = json.loads((prepared.artifacts / A.RESULTS).read_text(encoding="utf-8"))
    assert on_disk["P1"]["6.2"]["evidence_txn_id"] == "TXN-P1-0002"


def test_evidence_run_is_idempotent(prepared):
    first = evidence_run(prepared)
    second = evidence_run(prepared)
    assert first["P1"]["6.2"]["evidence_txn_id"] == second["P1"]["6.2"]["evidence_txn_id"]
