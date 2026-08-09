"""Тест переносимости: что сломается, когда приватный набор назовёт всё иначе.

Цель — не «пройти», а УЗНАТЬ до боевого окна. Механически меняется то,
что почти наверняка отличается в приватном наборе:

  * идентификаторы сценариев (P1 → CASE01) — во всех артефактах разом;
  * валюта и язык раскрытия курса;
  * регистр, кавычки и пунктуация в названиях организаций;
  * язык заголовков разделов (аудиторское приложение, досье, отчётность).

Привязка документов при переименованных счетах уже проверяется медленным
test_attribution_survives_renamed_identifiers; здесь — расчётная половина
пайплайна (шаги 7б, 11–13) и детерминированные разборщики текста.
"""
from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline import apply, artifacts as A, compute, disclosed, evidence  # noqa: E402
from pipeline.adjustments import SECTION_ANCHORS, find_section  # noqa: E402
from pipeline.config import RunPaths  # noqa: E402
from pipeline.entities import (  # noqa: E402
    EntityIndex,
    Entity,
    derive_group_capex,
    parse_subsidiary_pledges,
)
from pipeline.ledger import find_fx_rates  # noqa: E402

SNAPSHOT = ROOT / "fixtures" / "baseline" / "artifacts"

#: Приватный набор назовёт сценарии иначе. Длинные — первыми: P1 — префикс P10.
SCENARIO_MAP = {
    "P10": "CASE10", "P1": "CASE01", "P2": "CASE02", "P3": "CASE03",
    "P4": "CASE04", "P5": "CASE05", "P6": "CASE06", "P7": "CASE07",
    "P8": "CASE08", "P9": "CASE09", "B1": "CASE91", "B4": "CASE94",
}
_ORDERED = sorted(SCENARIO_MAP, key=len, reverse=True)


def rename_ids(text: str) -> str:
    for old in _ORDERED:
        text = re.sub(rf"\b{old}\b", SCENARIO_MAP[old], text)
    return text


def _run_offline(artifacts_from: Path, tmp_path: Path, transform=None) -> dict:
    """Шаги 7б и 11–13 поверх копии артефактов, с необязательной правкой."""
    rp = RunPaths.create(tmp_path)
    for src in artifacts_from.iterdir():
        if src.is_dir():
            continue
        if transform is None:
            shutil.copy(src, rp.artifacts / src.name)
        else:
            (rp.artifacts / src.name).write_text(
                transform(src.read_text(encoding="utf-8")), encoding="utf-8")
    disclosed.run(rp)
    apply.run(rp)
    compute.run(rp)
    evidence.run(rp)
    return json.loads((rp.artifacts / A.RESULTS).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def baseline_results(tmp_path_factory):
    return _run_offline(SNAPSHOT, tmp_path_factory.mktemp("base"))


@pytest.fixture(scope="module")
def renamed_results(tmp_path_factory):
    return _run_offline(SNAPSHOT, tmp_path_factory.mktemp("renamed"),
                        transform=rename_ids)


# --------------------------------------------------------------------------- #
# Идентификаторы сценариев
# --------------------------------------------------------------------------- #


def test_every_cell_survives_scenario_renaming(baseline_results, renamed_results):
    """Ни одно значение не смеет зависеть от того, как называется сценарий.
    Зависимость означала бы хардкод, который приватный набор молча обнулит."""
    assert set(renamed_results) == {SCENARIO_MAP[s] for s in baseline_results}

    for scenario, cells in baseline_results.items():
        renamed = renamed_results[SCENARIO_MAP[scenario]]
        assert set(renamed) == set(cells), scenario
        for point, cell in cells.items():
            other = renamed[point]
            where = f"{scenario}/{point}"
            assert other["status"] == cell["status"], where
            if cell["actual"] is None:
                assert other["actual"] is None, where
            else:
                assert other["actual"] == pytest.approx(cell["actual"], abs=0.01), where
            expected_ev = cell.get("evidence_txn_id")
            assert other.get("evidence_txn_id") == (
                rename_ids(expected_ev) if expected_ev else expected_ev
            ), where


def test_submission_assembles_under_renamed_ids(renamed_results):
    """Сборка идёт от шаблона: переименованный шаблон должен собраться
    без единого запасного значения."""
    from pipeline.assemble import build

    template = {"answers": {
        s: {p: {"status": None, "actual": None, "evidence_txn_id": None}
            for p in cells}
        for s, cells in renamed_results.items()
    }}
    submission, report = build(template, renamed_results,
                               team="t", contact_email="e", model="m")
    assert report.cells_fallback == []
    assert report.cells_total == 36


# --------------------------------------------------------------------------- #
# Валюта и язык раскрытия курса
# --------------------------------------------------------------------------- #


def test_fx_pair_is_found_in_english_and_other_currency():
    text = ("Settlements with Rheinland Katalyse Service GmbH: an invoice of "
            "72,146.75 CHF was settled by a USD payment of $83,690.23.")
    rates = find_fx_rates(text)
    assert "CHF" in rates
    assert rates["CHF"][0] == pytest.approx(83690.23 / 72146.75, rel=1e-6)


def test_fx_pair_survives_the_russian_original():
    text = ("Расчёты: счёт на сумму 72,146.75 EUR урегулирован платежом "
            "в долларах США в размере $83,690.23.")
    assert find_fx_rates(text)["EUR"][0] == pytest.approx(1.16, abs=0.01)


# --------------------------------------------------------------------------- #
# Регистр и пунктуация названий
# --------------------------------------------------------------------------- #

MUTATIONS = [
    lambda s: s.upper(),
    lambda s: s.lower(),
    lambda s: f"«{s}»",
    lambda s: s.replace("LLP", "L.L.P."),
    lambda s: s + " (Ekibastuz block B)",
    lambda s: s.replace(" ", "  ") + " ,",
]


@pytest.mark.parametrize("mutate", MUTATIONS)
def test_entity_matching_survives_case_and_punctuation(mutate):
    dossier = Entity(name="Taraz Holding Group LLP", role="counterparty",
                     is_related=True)
    index = EntityIndex([dossier])
    entity, _how = index.match(mutate("Taraz Holding Group LLP"))
    assert entity is dossier, mutate("Taraz Holding Group LLP")


# --------------------------------------------------------------------------- #
# Язык заголовков разделов
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("anchor", SECTION_ANCHORS)
def test_audit_annex_is_found_under_every_supported_heading(anchor):
    text = f"шапка документа\n{anchor}\nсодержимое раздела"
    section, found = find_section(text)
    assert found == anchor
    assert section.startswith(anchor)


def test_audit_annex_with_unknown_heading_degrades_loudly():
    """Неизвестный заголовок — документ уходит целиком, якорь None:
    это видно в отчёте шага 7 (remarks), а не теряется молча."""
    section, found = find_section("COVENANT NOTES\nтело")
    assert found is None and "тело" in section


def test_pledge_table_is_parsed_in_english():
    text = """
Security coverage of subsidiaries

Share of assets pledged is presented below as of the review date.

Subsidiary Share of assets pledged
Zhezkazgan Conveyor Assets LLP 87.6%
Zhezkazgan Processing Holdings LLP 11.4%

Subsidiaries whose pledged asset share is below 50.0% are outside the
security perimeter and are treated as unrestricted for the purposes of
the Agreement.
"""
    rows, below, problems = parse_subsidiary_pledges(text)
    assert [n for n, _ in rows] == ["Zhezkazgan Conveyor Assets LLP",
                                    "Zhezkazgan Processing Holdings LLP"]
    assert below == 50.0 and problems == []


def test_group_capex_is_derived_from_the_russian_wording():
    text = ("Балансовая стоимость на начало периода 148,028,989.69 "
            "Амортизационные отчисления за период 15,826,229.43 "
            "Балансовая стоимость на конец периода 154,050,122.81 "
            "Выбытий не было.")
    value, why = derive_group_capex(text)
    assert value == pytest.approx(21_847_362.55, abs=0.01)
