"""Тесты шага 1: скорер и валидатор.

Граничные прогоны из плана: пустой submission → 0.00, идеальный → 1.00,
сдвиг actual на 2.5% → ровно половина его веса.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval import score as scorer  # noqa: E402
from pipeline.validate import validate  # noqa: E402

KEY_PATH = ROOT / "eval" / "ground_truth.json"
TEMPLATE_PATH = ROOT.parent / "agentic-bank-public" / "submission_template.json"
LEDGER_PATH = ROOT.parent / "agentic-bank-public" / "master_ledger_2025.csv"

needs_key = pytest.mark.skipif(not KEY_PATH.exists(), reason="нет ground_truth")
needs_tpl = pytest.mark.skipif(not TEMPLATE_PATH.exists(), reason="нет шаблона")


@pytest.fixture
def key() -> dict:
    return json.loads(KEY_PATH.read_text(encoding="utf-8"))


@pytest.fixture
def template() -> dict:
    return json.loads(TEMPLATE_PATH.read_text(encoding="utf-8"))


def submission_from_key(key: dict) -> dict:
    return {
        "team": "t", "contact_email": "a@b.c", "model": "claude-opus-5",
        "answers": {
            s: {p: dict(c) for p, c in d["covenants"].items()}
            for s, d in key["scenarios"].items()
        },
    }


def empty_submission(template: dict) -> dict:
    return json.loads(json.dumps(template))


# --------------------------------------------------------------------------- #
# Скорер: граничные прогоны
# --------------------------------------------------------------------------- #


@needs_key
def test_perfect_submission_scores_one(key):
    rep = scorer.score(submission_from_key(key), key)
    assert rep.total == pytest.approx(1.0)
    assert rep.losses() == []
    assert len(rep.cells) == 36


@needs_key
@needs_tpl
def test_empty_submission_scores_zero(key, template):
    rep = scorer.score(empty_submission(template), key)
    assert rep.total == pytest.approx(0.0)


@needs_key
def test_actual_off_by_2_5_percent_gives_half(key):
    """Проверяет реализацию шкалы, а не только её края."""
    sub = submission_from_key(key)
    cell = sub["answers"]["B1"]["6.2"]      # evidence в ключе = null
    key_actual = key["scenarios"]["B1"]["covenants"]["6.2"]["actual"]
    cell["actual"] = key_actual * 1.025

    rep = scorer.score(sub, key)
    c = next(c for c in rep.cells if c.scenario == "B1" and c.point == "6.2")
    assert c.status_pts == pytest.approx(0.50)
    assert c.actual_pts == pytest.approx(0.15)     # половина от 0.30
    assert c.evidence_pts == pytest.approx(0.10)   # половина от 0.20 — ключ null
    assert c.points == pytest.approx(0.75)


@needs_key
def test_actual_off_by_5_percent_gives_zero_for_both(key):
    sub = submission_from_key(key)
    key_actual = key["scenarios"]["B1"]["covenants"]["6.2"]["actual"]
    sub["answers"]["B1"]["6.2"]["actual"] = key_actual * 1.05

    c = next(c for c in scorer.score(sub, key).cells if (c.scenario, c.point) == ("B1", "6.2"))
    assert c.actual_pts == 0.0 and c.evidence_pts == 0.0
    assert c.points == pytest.approx(0.50)


@needs_key
def test_wrong_status_zeroes_whole_cell(key):
    sub = submission_from_key(key)
    cur = sub["answers"]["B1"]["6.1"]["status"]
    sub["answers"]["B1"]["6.1"]["status"] = "COMPLIANT" if cur == "BREACH" else "BREACH"

    c = next(c for c in scorer.score(sub, key).cells if (c.scenario, c.point) == ("B1", "6.1"))
    assert c.points == 0.0, "неверный status обязан обнулить ячейку целиком"
    assert not c.status_ok


@needs_key
def test_lowercase_status_zeroes_cell(key):
    sub = submission_from_key(key)
    sub["answers"]["B1"]["6.1"]["status"] = "breach"
    c = next(c for c in scorer.score(sub, key).cells if (c.scenario, c.point) == ("B1", "6.1"))
    assert c.points == 0.0


@needs_key
def test_evidence_with_non_null_key_is_all_or_nothing(key):
    """B1/6.1 в ключе имеет evidence — там шкала не применяется."""
    sub = submission_from_key(key)
    assert key["scenarios"]["B1"]["covenants"]["6.1"]["evidence_txn_id"] is not None
    sub["answers"]["B1"]["6.1"]["evidence_txn_id"] = "TXN-B1-9999"

    c = next(c for c in scorer.score(sub, key).cells if (c.scenario, c.point) == ("B1", "6.1"))
    assert c.evidence_pts == 0.0
    assert c.actual_pts == pytest.approx(0.30), "точный actual не должен пострадать"


@needs_key
def test_string_actual_loses_actual_and_evidence(key):
    sub = submission_from_key(key)
    sub["answers"]["B1"]["6.2"]["actual"] = "1284663.42"
    c = next(c for c in scorer.score(sub, key).cells if (c.scenario, c.point) == ("B1", "6.2"))
    assert c.points == pytest.approx(0.50)
    assert any("не число" in n for n in c.notes)


@needs_key
def test_missing_scenario_is_reported_and_zeroed(key):
    sub = submission_from_key(key)
    del sub["answers"]["P5"]
    rep = scorer.score(sub, key)
    assert any("P5" in s for s in rep.structural)
    assert all(c.points == 0.0 for c in rep.cells if c.scenario == "P5")


@needs_key
def test_extra_keys_are_reported(key):
    sub = submission_from_key(key)
    sub["answers"]["P1"]["6.9"] = {"status": "BREACH", "actual": 1.0, "evidence_txn_id": None}
    rep = scorer.score(sub, key)
    assert any("6.9" in s for s in rep.structural)
    assert rep.total == pytest.approx(1.0), "лишний ключ не должен портить существующие ячейки"


def test_broken_json_yields_zero(tmp_path):
    p = tmp_path / "s.json"
    p.write_text("{не json", encoding="utf-8")
    data, problems = scorer.load_submission(p)
    assert data == {} and problems and "битый JSON" in problems[0]


def test_actual_scale_edges():
    assert scorer.actual_scale(100.0, 100.0)[0] == pytest.approx(1.0)
    assert scorer.actual_scale(102.5, 100.0)[0] == pytest.approx(0.5)
    assert scorer.actual_scale(105.0, 100.0)[0] == 0.0
    assert scorer.actual_scale(None, 100.0)[0] == 0.0
    assert scorer.actual_scale(True, 100.0)[0] == 0.0, "bool не должен считаться числом"
    assert scorer.actual_scale(float("nan"), 100.0)[0] == 0.0
    assert scorer.actual_scale(0.0, 0.0)[0] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Валидатор: работает без ключа
# --------------------------------------------------------------------------- #


@needs_tpl
def test_validator_accepts_well_formed(template):
    sub = {
        "team": "t", "contact_email": "a@b.c", "model": "claude-opus-5",
        "answers": {
            s: {p: {"status": "COMPLIANT", "actual": 1.23, "evidence_txn_id": None}
                for p in cells}
            for s, cells in template["answers"].items()
        },
    }
    rep = validate(sub, template)
    assert rep.ok, [str(i) for i in rep.errors]
    assert rep.cells_checked == 36


@needs_tpl
def test_validator_catches_format_killers(template):
    sub = {
        "team": "", "model": "m",
        "answers": {
            s: {p: {"status": "COMPLIANT", "actual": 1.0, "evidence_txn_id": None}
                for p in cells}
            for s, cells in template["answers"].items()
        },
    }
    sub["answers"]["P1"]["6.1"]["status"] = "compliant"       # регистр
    sub["answers"]["P1"]["6.2"]["actual"] = "1000.00"         # строка
    sub["answers"]["P1"]["6.3"]["actual"] = -5.0              # отрицательный
    sub["answers"]["P2"]["6.1"]["evidence_txn_id"] = "какая-то строка"
    del sub["answers"]["P3"]["6.1"]

    rep = validate(sub, template)
    msgs = " | ".join(str(i) for i in rep.errors)
    assert "неверном регистре" in msgs
    assert "не число" in msgs
    assert "отрицателен" in msgs
    assert "не похож на идентификатор" in msgs
    assert "пункт отсутствует" in msgs
    assert "contact_email" in msgs      # поле отсутствует
    assert "team" in msgs               # поле пустое
    assert not rep.ok


@needs_tpl
def test_validator_rejects_unfilled_template(template):
    """Шаблон как есть — это 36 незаполненных ячеек, а не готовый сабмит."""
    rep = validate(json.loads(json.dumps(template)), template)
    assert not rep.ok
    assert len([i for i in rep.errors if "status не заполнен" in i.message]) == 36


@needs_tpl
@pytest.mark.skipif(not LEDGER_PATH.exists(), reason="нет реестра")
def test_validator_checks_evidence_ownership(template):
    """Сильная проверка: evidence обязан существовать и принадлежать своему сценарию."""
    from pipeline.validate import _load_ledger_index

    idx = _load_ledger_index(LEDGER_PATH)
    sub = {
        "team": "t", "contact_email": "a@b.c", "model": "m",
        "answers": {
            s: {p: {"status": "BREACH", "actual": 1.0, "evidence_txn_id": None} for p in cells}
            for s, cells in template["answers"].items()
        },
    }
    sub["answers"]["P1"]["6.1"]["evidence_txn_id"] = "TXN-P2-0040"   # чужой сценарий
    sub["answers"]["P1"]["6.2"]["evidence_txn_id"] = "TXN-P1-9999"   # не существует

    rep = validate(sub, template, idx)
    msgs = " | ".join(str(i) for i in rep.errors)
    assert "принадлежит сценарию P2" in msgs
    assert "отсутствует в реестре" in msgs


@needs_tpl
def test_validator_warns_on_zero_and_extra_decimals(template):
    sub = {
        "team": "t", "contact_email": "a@b.c", "model": "m",
        "answers": {
            s: {p: {"status": "BREACH", "actual": 1.0, "evidence_txn_id": None} for p in cells}
            for s, cells in template["answers"].items()
        },
    }
    sub["answers"]["P1"]["6.1"]["actual"] = 0.0
    sub["answers"]["P1"]["6.2"]["actual"] = 1234.56789
    rep = validate(sub, template)
    warns = " | ".join(str(i) for i in rep.warnings)
    assert "actual = 0" in warns and "знаками после запятой" in warns
    assert rep.ok, "это предупреждения, а не ошибки"


@needs_tpl
def test_validator_warns_on_unknown_model_and_extra_cell_fields(template):
    sub = {
        "team": "t", "contact_email": "a@b.c", "model": "gpt-хз-какая",
        "answers": {
            s: {p: {"status": "BREACH", "actual": 1.0, "evidence_txn_id": None} for p in cells}
            for s, cells in template["answers"].items()
        },
    }
    sub["answers"]["P1"]["6.1"]["debug_trace"] = ["отладочный мусор"]
    rep = validate(sub, template)
    warns = " | ".join(str(i) for i in rep.warnings)
    assert "не входит в список известных" in warns
    assert "лишние поля в ячейке" in warns
    assert rep.ok
