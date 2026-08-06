"""Тесты шага 14: сборка submission.json.

Структуру задаёт шаблон. Проверяется, что ни один ключ не потерян, не
добавлен и не переименован, а формат значений не даёт ячейке обнулиться
по формальной причине.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.assemble import build  # noqa: E402
from pipeline.assemble import run as assemble_run  # noqa: E402
from pipeline.config import RunPaths, discover_dataset  # noqa: E402
from pipeline.validate import validate  # noqa: E402

PUBLIC = ROOT.parent / "agentic-bank-public"
needs_public = pytest.mark.skipif(not PUBLIC.exists(), reason="нет публичного датасета")

TEMPLATE = {
    "team": "", "contact_email": "", "model": "",
    "answers": {
        "P1": {p: {"status": None, "actual": None, "evidence_txn_id": None}
               for p in ("6.1", "6.2", "6.3")},
        "P2": {p: {"status": None, "actual": None, "evidence_txn_id": None}
               for p in ("6.1", "6.2")},
    },
}


def results(**cells) -> dict:
    out: dict = {}
    for key, value in cells.items():
        scenario, point = key.split("__")
        out.setdefault(scenario, {})[point.replace("_", ".")] = value
    return out


def cell(status="BREACH", actual=1234.5678, evidence=None, **extra):
    return {"status": status, "actual": actual, "evidence_txn_id": evidence, **extra}


# --------------------------------------------------------------------------- #
# Структура задаётся шаблоном
# --------------------------------------------------------------------------- #


def test_keys_come_from_template_exactly():
    sub, rep = build(TEMPLATE, {}, "t", "a@b.c", "claude-opus-5")
    assert set(sub["answers"]) == {"P1", "P2"}
    assert set(sub["answers"]["P1"]) == {"6.1", "6.2", "6.3"}
    assert set(sub["answers"]["P2"]) == {"6.1", "6.2"}
    assert rep.cells_total == 5


def test_extra_scenario_in_results_is_ignored_and_reported():
    sub, rep = build(TEMPLATE, results(P9__6_1=cell()), "t", "a@b.c", "m")
    assert "P9" not in sub["answers"]
    assert any("нет в шаблоне" in p for p in rep.problems)


def test_every_cell_is_filled_even_without_results():
    """Пустая и неверная ячейка стоят одинаково — пустых быть не должно."""
    sub, rep = build(TEMPLATE, {}, "t", "a@b.c", "m")
    for scenario, cells in sub["answers"].items():
        for point, c in cells.items():
            assert c["status"] in ("COMPLIANT", "BREACH"), f"{scenario}/{point}"
            assert isinstance(c["actual"], float)
    assert len(rep.cells_fallback) == 5
    assert rep.cells_computed == 0


def test_top_level_fields_are_filled():
    sub, _ = build(TEMPLATE, {}, "команда", "me@example.com", "claude-sonnet-5")
    assert sub["team"] == "команда"
    assert sub["contact_email"] == "me@example.com"
    assert sub["model"] == "claude-sonnet-5"


# --------------------------------------------------------------------------- #
# Формат значений
# --------------------------------------------------------------------------- #


def test_actual_is_rounded_to_two_decimals():
    sub, _ = build(TEMPLATE, results(P1__6_1=cell(actual=1234.5678)), "t", "a@b.c", "m")
    assert sub["answers"]["P1"]["6.1"]["actual"] == 1234.57


def test_actual_is_made_positive():
    """В реестре списания отрицательны, а actual обязан быть модулем."""
    sub, _ = build(TEMPLATE, results(P1__6_1=cell(actual=-450_000.0)), "t", "a@b.c", "m")
    assert sub["answers"]["P1"]["6.1"]["actual"] == 450_000.0


def test_non_finite_actual_falls_back_and_is_reported():
    sub, rep = build(TEMPLATE, results(P1__6_1=cell(actual=float("nan"))), "t", "a@b.c", "m")
    assert sub["answers"]["P1"]["6.1"]["actual"] == 0.0
    assert "P1/6.1" in rep.cells_fallback
    assert any("actual не вычислен" in p for p in rep.problems)


def test_string_actual_falls_back():
    sub, rep = build(TEMPLATE, results(P1__6_1=cell(actual="1000.00")), "t", "a@b.c", "m")
    assert sub["answers"]["P1"]["6.1"]["actual"] == 0.0
    assert "P1/6.1" in rep.cells_fallback


def test_invalid_status_falls_back_and_is_reported():
    sub, rep = build(TEMPLATE, results(P1__6_1=cell(status="breach")), "t", "a@b.c", "m")
    assert sub["answers"]["P1"]["6.1"]["status"] == "COMPLIANT"
    assert any("невалиден" in p for p in rep.problems)


def test_non_string_evidence_is_nulled():
    sub, rep = build(TEMPLATE, results(P1__6_1=cell(evidence=42)), "t", "a@b.c", "m")
    assert sub["answers"]["P1"]["6.1"]["evidence_txn_id"] is None
    assert any("evidence не строка" in p for p in rep.problems)


def test_extra_fields_from_results_do_not_leak_into_submission():
    """Условие запрещает добавлять ключи; трассировка и флаги остаются
    в артефактах, а в ответ не попадают."""
    sub, _ = build(
        TEMPLATE,
        results(P1__6_1=cell(trace=[{"category": "capex"}], problems=["x"],
                             evidence_basis="reclassified")),
        "t", "a@b.c", "m",
    )
    assert set(sub["answers"]["P1"]["6.1"]) == {"status", "actual", "evidence_txn_id"}


def test_evidence_is_counted():
    sub, rep = build(
        TEMPLATE,
        results(P1__6_1=cell(evidence="TXN-P1-0020"), P1__6_2=cell()),
        "t", "a@b.c", "m",
    )
    assert rep.evidence_filled == 1
    assert sub["answers"]["P1"]["6.1"]["evidence_txn_id"] == "TXN-P1-0020"


def test_submission_is_serialisable_without_nan():
    sub, _ = build(TEMPLATE, results(P1__6_1=cell(actual=math.inf)), "t", "a@b.c", "m")
    payload = json.dumps(sub)
    assert "NaN" not in payload and "Infinity" not in payload


# --------------------------------------------------------------------------- #
# Сквозной прогон и стык с валидатором
# --------------------------------------------------------------------------- #


@needs_public
def test_end_to_end_output_passes_the_validator(tmp_path):
    """Замыкание цепочки: собранный ответ обязан проходить валидатор
    шага 1 — тот самый, что поедет в боевое окно."""
    ds = discover_dataset(PUBLIC)
    paths = RunPaths.create(tmp_path / "run")

    template = json.loads(ds.template_json.read_text(encoding="utf-8"))
    fake_results = {
        s: {p: {"status": "BREACH", "actual": 1000.0, "evidence_txn_id": None}
            for p in cells}
        for s, cells in template["answers"].items()
    }
    (paths.artifacts / "09_results.json").write_text(
        json.dumps(fake_results), encoding="utf-8")

    sub, rep = assemble_run(ds, paths, team="tbd", contact_email="me@example.com",
                            model="claude-opus-5")

    assert rep.cells_total == 36 and rep.cells_computed == 36
    assert rep.cells_fallback == []

    report = validate(sub, template)
    assert report.ok, [str(i) for i in report.errors]
    assert (paths.root / "submission.json").exists()


@needs_public
def test_empty_results_still_produce_a_valid_submission(tmp_path):
    """Худший случай боевого окна: расчёт не дал ничего. Файл всё равно
    обязан быть валидным — пустой сабмит стоит столько же, сколько неверный,
    но битый стоит дороже: он неоцениваем."""
    ds = discover_dataset(PUBLIC)
    paths = RunPaths.create(tmp_path / "run")

    sub, rep = assemble_run(ds, paths, team="t", contact_email="a@b.c",
                            model="claude-opus-5")
    assert len(rep.cells_fallback) == 36

    template = json.loads(ds.template_json.read_text(encoding="utf-8"))
    report = validate(sub, template)
    assert report.ok, [str(i) for i in report.errors]


def test_full_chain_from_ledger_to_score(tmp_path):
    """Сквозное замыкание: реестр → расчёт → доказательство → сборка →
    скорер. Проверяет, что артефакты шагов стыкуются без ручной правки."""
    from eval import score as scorer
    from pipeline.compute import run as compute_run
    from pipeline.evidence import run as evidence_run

    paths = RunPaths.create(tmp_path / "run")
    (paths.artifacts / "06_ledger_clean.csv").write_text(
        "txn_id,scenario_id,date,counterparty,description,amount,currency,amount_usd,"
        "category,party,scope,excluded\n"
        "TXN-P1-0001,P1,2025-02-01,Alpha,x,-300000,USD,-300000,capex,,borrower,0\n"
        "TXN-P1-0002,P1,2025-03-01,Beta,x,-200000,USD,-200000,capex,,borrower,0\n"
        "TXN-P1-0003,P1,2025-04-01,Holding,x,-500000,USD,-500000,opex,related,borrower,0\n"
        "TXN-P1-0004,P1,2025-05-01,Client,x,900000,USD,900000,revenue,,borrower,0\n",
        encoding="utf-8",
    )
    (paths.artifacts / "03_covenants.json").write_text(json.dumps({
        "scenarios": {"P1": {
            "6.1": {"direction": "min", "threshold": 1000000.0, "unit": "amount",
                    "metric": {"op": "AGG", "category": "revenue"}},
            "6.2": {"direction": "max", "threshold": 400000.0, "unit": "amount",
                    "metric": {"op": "AGG", "category": "capex"}},
            "6.3": {"direction": "max", "threshold": 400000.0, "unit": "amount",
                    "metric": {"op": "AGG", "category": "any", "party": "related"}},
        }}
    }), encoding="utf-8")
    (paths.artifacts / "04_adjustments.json").write_text(json.dumps({
        "P1": {"notes": [{"note_id": "7.1", "kind": "reclassification",
                          "target_txn_id": "TXN-P1-0002", "from_category": "opex",
                          "to_category": "capex", "description": "перенесено аудитором"}]}
    }), encoding="utf-8")

    compute_run(paths)
    evidence_run(paths)

    template = {"team": "", "contact_email": "", "model": "", "answers": {
        "P1": {p: {"status": None, "actual": None, "evidence_txn_id": None}
               for p in ("6.1", "6.2", "6.3")}}}
    (paths.root / "tpl.json").write_text(json.dumps(template), encoding="utf-8")

    results = json.loads((paths.artifacts / "09_results.json").read_text(encoding="utf-8"))
    sub, rep = build(template, results, "t", "a@b.c", "claude-opus-5")

    assert rep.cells_fallback == []
    assert validate(sub, template).ok

    key = {"scenarios": {"P1": {"covenants": {
        "6.1": {"status": "BREACH", "actual": 900000.0, "evidence_txn_id": None},
        "6.2": {"status": "BREACH", "actual": 500000.0, "evidence_txn_id": "TXN-P1-0002"},
        "6.3": {"status": "BREACH", "actual": 500000.0, "evidence_txn_id": "TXN-P1-0003"},
    }}}}
    report = scorer.score(sub, key)
    assert report.total == pytest.approx(1.0), [
        (c.scenario, c.point, c.points, c.notes) for c in report.losses()
    ]
