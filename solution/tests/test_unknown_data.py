"""Тесты задачи «незнакомое не должно теряться молча».

Неопознанная валюта, организационно-правовая форма или статья на
приватном наборе почти наверняка встретятся. Поведение остаётся прежним
(ничего не применяется по догадке) — но каждый случай обязан оставить
строку в отчёте шага и попасть во флаги уверенности. Тихая потеря
данных не даёт ни одного шанса себя заметить; громкая — даёт.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline import artifacts as A  # noqa: E402
from pipeline.confidence import assess, _scenario_report_problems  # noqa: E402
from pipeline.config import RunPaths  # noqa: E402
from pipeline.entities import build_graph, suspect_unknown_legal_form  # noqa: E402
from pipeline.ledger import find_fx_rates  # noqa: E402


# --------------------------------------------------------------------------- #
# Валюта вне словаря
# --------------------------------------------------------------------------- #

FX_TEXT = "счёт на сумму 1,000.00 QQQ урегулирован платежом в размере $1,160.00"


def test_unknown_currency_rate_is_dropped_but_reported():
    problems: list[str] = []
    rates = find_fx_rates(FX_TEXT, problems=problems)
    assert "QQQ" not in rates, "поведение не меняется: мусорный курс не применяется"
    assert problems and "вне словаря" in problems[0] and "QQQ" in problems[0]


def test_find_fx_rates_still_works_without_a_sink():
    """Старая сигнатура жива: вызовы без стока не падают."""
    assert find_fx_rates(FX_TEXT) == {}


# --------------------------------------------------------------------------- #
# Организационно-правовая форма вне словаря
# --------------------------------------------------------------------------- #


def test_unknown_legal_form_is_suspected():
    assert suspect_unknown_legal_form("Alpha Beta OOO") == "OOO"
    assert suspect_unknown_legal_form("Alpha Chemie S.A.") == "SA"


def test_known_forms_and_ordinary_words_are_not_suspected():
    assert suspect_unknown_legal_form("Ertis Capital, LLP") is None
    assert suspect_unknown_legal_form("Kazyna Capital LLP.") is None
    assert suspect_unknown_legal_form("Irtysh Advisory Bureau") is None
    # Длинное слово заглавными — часть названия, а не форма.
    assert suspect_unknown_legal_form("ALPHA GAMMA") is None


def test_build_graph_reports_the_suspicious_tail():
    g = build_graph("P1", "X", {"threshold_pct": 40.0, "parties": [
        {"name": "Vostok Trading OOO", "ownership_pct": 51.0},
    ]}, {})
    assert g.related_names() == ["Vostok Trading OOO"], "поведение не меняется"
    assert any("вне словаря LEGAL_FORMS" in p for p in g.problems)


# --------------------------------------------------------------------------- #
# Дорога в флаги уверенности
# --------------------------------------------------------------------------- #


def _write(paths: RunPaths, name: str, payload: dict) -> None:
    (paths.artifacts / name).write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_report_lines_reach_the_right_scenario(tmp_path):
    paths = RunPaths.create(tmp_path / "run")
    _write(paths, A.TXN_CATEGORIES, {"problems": [
        "TXN-P9-0025: статья 'blockchain' вне словаря — строка не размечена"]})
    _write(paths, A.LEDGER_REPORT, {"problems": [], "unresolved": [
        "TXN-P3-0001: 100.00 QQQ без раскрытого курса (валюта вне словаря KNOWN_CURRENCIES)"]})
    _write(paths, A.ENTITY_GRAPH, {"P6": {"problems": [
        "Vostok OOO: хвост 'OOO' похож на форму вне словаря LEGAL_FORMS — ..."]}})

    collected = _scenario_report_problems(paths)

    assert any("blockchain" in line for line in collected["P9"])
    assert any("QQQ" in line for line in collected["P3"])
    assert any("OOO" in line for line in collected["P6"])


def test_unattributed_loss_reaches_every_cell(tmp_path):
    """Потеря, у которой не определить заёмщика, — повод смотреть везде,
    а не нигде."""
    paths = RunPaths.create(tmp_path / "run")
    _write(paths, A.LEDGER_REPORT, {"problems": [
        "валюта 'QQQ' вне словаря KNOWN_CURRENCIES — кандидат на курс отброшен"]})

    collected = _scenario_report_problems(paths)
    assert any("QQQ" in line for line in collected["*"])

    cell = {"status": "COMPLIANT", "actual": 1.0, "threshold": 2.0,
            "unit": "amount", "trace": [], "problems": []}
    risks = assess({"X": {"6.1": cell}}, scenario_problems=collected)
    assert risks[0].flagged
    assert any("неопознанные данные" in s for s in risks[0].signals)


def test_scenario_problems_flag_the_scenarios_cells():
    cell = {"status": "COMPLIANT", "actual": 1.0, "threshold": 2.0,
            "unit": "amount", "trace": [], "problems": []}
    risks = assess(
        {"P9": {"6.1": dict(cell)}, "P1": {"6.1": dict(cell)}},
        scenario_problems={"P9": ["TXN-P9-0025: статья вне словаря"]},
    )
    by = {r.where: r for r in risks}
    assert by["P9/6.1"].flagged
    assert not by["P1/6.1"].flagged