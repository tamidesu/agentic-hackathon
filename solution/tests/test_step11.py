"""Тесты шага 11: сборка итогового реестра.

Шаг не выносит своих суждений — он применяет чужие. Поэтому проверяется
не «правильно ли решено», а «правильно ли применено»: порядок слоёв,
адресация примечаний к строкам, и то, что ничего не применяется сверх
разрешённого.

Самая дорогая ошибка здесь — применить корректировку, которую аудитор
не делал. Числа сойдутся, статусы выставятся, и результат будет выглядеть
совершенно нормально.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline import apply  # noqa: E402
from pipeline import artifacts as A  # noqa: E402
from pipeline.adjustments import Note, ScenarioAdjustments  # noqa: E402
from pipeline.apply import FinalRow, apply_adjustments, apply_categories  # noqa: E402
from pipeline.apply import apply_related, match_note_to_row  # noqa: E402
from pipeline.categorize import TxnCategory  # noqa: E402
from pipeline.config import RunPaths  # noqa: E402
from pipeline.related import Party, ScenarioParties  # noqa: E402


def _row(txn_id="TXN-P2-0001", counterparty="Tien Shan Advisory Bureau",
         amount=-1104663.28, scenario="P2", date="2025-03-01"):
    return FinalRow(txn_id=txn_id, scenario_id=scenario, date=date,
                    counterparty=counterparty, amount_usd=amount)


# --------------------------------------------------------------------------- #
# Категории — основа
# --------------------------------------------------------------------------- #


def test_category_and_confidence_are_carried_over():
    rows = [_row()]
    apply_categories(rows, {"TXN-P2-0001": TxnCategory("TXN-P2-0001", "opex",
                                                       confidence=0.9)})
    assert rows[0].category == "opex"
    assert rows[0].confidence == 0.9
    assert "шаг 10" in rows[0].origin


def test_a_row_without_a_category_is_reported_not_guessed():
    """Угадать статью значило бы придумать число из воздуха. 'other'
    не входит ни в один ковенант, то есть безопасен."""
    rows = [_row()]
    problems = apply_categories(rows, {})
    assert rows[0].category == "other"
    assert problems and "отнесена к 'other'" in problems[0]


def test_a_fallback_category_says_so_in_the_origin():
    rows = [_row()]
    apply_categories(rows, {"TXN-P2-0001": TxnCategory("TXN-P2-0001", "other",
                                                       fallback=True)})
    assert "запасное значение" in rows[0].origin


# --------------------------------------------------------------------------- #
# Адресация примечаний
# --------------------------------------------------------------------------- #


def test_a_note_with_a_txn_id_hits_exactly_that_row():
    rows = [_row("TXN-P2-0001"), _row("TXN-P2-0002")]
    note = Note("9.1", "cutoff", "applied", target_txn_id="TXN-P2-0002")
    assert [r.txn_id for r in match_note_to_row(note, rows)] == ["TXN-P2-0002"]


def test_a_txn_id_wins_over_the_counterparty():
    """Примечание с номером операции адресует РОВНО её. Искать после
    этого по контрагенту значило бы задеть однофамильцев."""
    rows = [_row("TXN-P2-0001"), _row("TXN-P2-0002")]
    note = Note("9.1", "cutoff", "applied", target_txn_id="TXN-P2-0002",
                target_counterparty="Tien Shan Advisory Bureau")
    assert [r.txn_id for r in match_note_to_row(note, rows)] == ["TXN-P2-0002"]


def test_amount_plus_counterparty_addresses_one_row():
    """Настоящая форма из набора: «сумма в размере $1,104,663.28,
    выплаченная контрагенту X» — без номера операции."""
    rows = [_row("TXN-P2-0001", amount=-1104663.28),
            _row("TXN-P2-0002", amount=-500.0)]
    note = Note("9.1", "reclassification", "applied",
                target_counterparty="Tien Shan Advisory Bureau",
                value_usd=1104663.28, to_category="opex")
    assert [r.txn_id for r in match_note_to_row(note, rows)] == ["TXN-P2-0001"]


def test_a_ledger_name_with_a_suffix_still_matches():
    rows = [_row(counterparty="Tien Shan Advisory Bureau (Almaty office)")]
    note = Note("9.1", "reclassification", "applied",
                target_counterparty="Tien Shan Advisory Bureau",
                value_usd=1104663.28, to_category="opex")
    assert match_note_to_row(note, rows)


def test_a_note_addressing_nothing_returns_nothing():
    note = Note("9.1", "reclassification", "applied", to_category="opex")
    assert match_note_to_row(note, [_row()]) == []


# --------------------------------------------------------------------------- #
# Применение корректировок
# --------------------------------------------------------------------------- #


def _adjustments(*notes, scenario="P2", **kw):
    return ScenarioAdjustments(scenario_id=scenario, notes=list(notes), **kw)


def test_a_reclassification_replaces_the_category_and_leaves_a_trace():
    rows = [_row()]
    apply_categories(rows, {"TXN-P2-0001": TxnCategory("TXN-P2-0001", "opex")})
    note = Note("9.1", "reclassification", "applied", target_txn_id="TXN-P2-0001",
                to_category="insurance")
    reclassified, _, _ = apply_adjustments(rows, _adjustments(note))

    assert rows[0].category == "insurance"
    assert reclassified == ["TXN-P2-0001"]
    assert "opex → insurance" in rows[0].origin, "изменение обязано оставить след"


def test_a_rejected_note_changes_nothing():
    """Реальная ловушка P10 7.2: «первоначальная классификация
    сохраняется». Применить её значило бы переложить деньги без основания."""
    rows = [_row()]
    apply_categories(rows, {"TXN-P2-0001": TxnCategory("TXN-P2-0001", "opex")})
    note = Note("7.2", "reclassification", "considered_but_rejected",
                target_txn_id="TXN-P2-0001", to_category="insurance",
                skipped_because="статус considered_but_rejected")
    reclassified, _, skipped = apply_adjustments(rows, _adjustments(note))

    assert rows[0].category == "opex"
    assert reclassified == []
    assert skipped and "considered_but_rejected" in skipped[0]


def test_a_note_referred_elsewhere_changes_nothing():
    """Реальная ловушка B1 9.1: вывод изложен в другом отчёте."""
    rows = [_row()]
    apply_categories(rows, {"TXN-P2-0001": TxnCategory("TXN-P2-0001", "opex")})
    note = Note("9.1", "reclassification", "referred_elsewhere",
                target_txn_id="TXN-P2-0001", to_category="capex",
                skipped_because="статус referred_elsewhere")
    apply_adjustments(rows, _adjustments(note))
    assert rows[0].category == "opex"


def test_a_cutoff_excludes_the_row_without_deleting_it():
    """Удалённую строку невозможно предъявить как доказательство
    и невозможно отличить от потерянной."""
    rows = [_row("TXN-B4-0026")]
    note = Note("9.1", "cutoff", "applied", target_txn_id="TXN-B4-0026",
                description="переход рисков в январе 2026")
    _, excluded, _ = apply_adjustments(rows, _adjustments(note))

    assert len(rows) == 1, "строка не должна исчезать"
    assert rows[0].excluded and excluded == ["TXN-B4-0026"]
    assert "аудитор, п.9.1" in rows[0].exclusion_reason


def test_an_unmatched_note_is_reported_by_name():
    """Непримененное примечание — это либо ловушка, либо промах адресации.
    Различить их можно только глядя на список."""
    note = Note("9.1", "reclassification", "applied",
                target_txn_id="TXN-НЕТ-ТАКОЙ", to_category="opex")
    _, _, skipped = apply_adjustments([_row()], _adjustments(note))
    assert skipped and "не найдена операция" in skipped[0]


def test_a_reclassification_without_a_target_category_is_skipped():
    rows = [_row()]
    apply_categories(rows, {"TXN-P2-0001": TxnCategory("TXN-P2-0001", "opex")})
    note = Note("9.1", "reclassification", "applied", target_txn_id="TXN-P2-0001")
    reclassified, _, skipped = apply_adjustments(rows, _adjustments(note))
    assert reclassified == [] and rows[0].category == "opex"
    assert skipped


# --------------------------------------------------------------------------- #
# Связанные стороны
# --------------------------------------------------------------------------- #


def test_a_payment_to_a_related_party_is_tagged():
    rows = [_row(counterparty="Taraz Holding Group LLP (Taraz yard)")]
    parties = ScenarioParties("P6", parties=[
        Party("Taraz Holding Group LLP", 46.8, is_related=True)])
    tagged = apply_related(rows, parties)

    assert rows[0].party == "related"
    assert tagged == [rows[0].txn_id]
    assert "связанная сторона" in rows[0].origin


def test_a_payment_to_a_non_related_party_is_not_tagged():
    rows = [_row(counterparty="Ural Grinding Works LLP")]
    parties = ScenarioParties("P6", parties=[
        Party("Ural Grinding Works LLP", 11.5, is_related=False)])
    assert apply_related(rows, parties) == []
    assert rows[0].party is None


# --------------------------------------------------------------------------- #
# Полный прогон
# --------------------------------------------------------------------------- #


@pytest.fixture
def prepared(tmp_path):
    paths = RunPaths.create(tmp_path / "run")
    with (paths.artifacts / A.LEDGER_CLEAN).open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["txn_id", "scenario_id", "date", "counterparty",
                         "description", "amount", "currency", "amount_usd"])
        writer.writerow(["TXN-P2-0001", "P2", "2025-03-01", "Tien Shan Advisory Bureau",
                         "Consulting", "-1104663.28", "USD", "-1104663.28"])
        writer.writerow(["TXN-P2-0002", "P2", "2025-04-01", "Almaty Holdings LLP",
                         "Payment", "-5000.00", "USD", "-5000.00"])
        writer.writerow(["TXN-P2-0003", "P2", "2025-11-20", "Прочие",
                         "Advance", "900.00", "USD", "900.00"])
        writer.writerow(["TXN-P2-0004", "P2", "2025-05-01", "Без суммы",
                         "Broken", "", "USD", ""])
    return paths


def test_full_assembly_writes_the_contract_columns(prepared):
    report = apply.run(
        prepared,
        categories={
            "TXN-P2-0001": TxnCategory("TXN-P2-0001", "opex", confidence=0.9),
            "TXN-P2-0002": TxnCategory("TXN-P2-0002", "capex", confidence=0.8),
            "TXN-P2-0003": TxnCategory("TXN-P2-0003", "revenue", confidence=0.7),
        },
        related={"P2": ScenarioParties("P2", parties=[
            Party("Almaty Holdings LLP", 50.0, is_related=True)])},
        adjustments={"P2": ScenarioAdjustments("P2", notes=[
            Note("9.1", "reclassification", "applied",
                 target_txn_id="TXN-P2-0001", to_category="insurance"),
            Note("9.2", "cutoff", "applied", target_txn_id="TXN-P2-0003",
                 description="переход рисков в 2026"),
        ])},
    )

    with (prepared.artifacts / A.LEDGER_FINAL).open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    assert list(rows[0]) == apply.FINAL_COLUMNS
    by_id = {r["txn_id"]: r for r in rows}

    assert by_id["TXN-P2-0001"]["category"] == "insurance", "переклассификация"
    assert by_id["TXN-P2-0002"]["party"] == "related", "связанная сторона"
    assert by_id["TXN-P2-0003"]["excluded"] == "1", "отсечение"
    assert "TXN-P2-0004" not in by_id, "строка без суммы не попадает в расчёт"
    assert report.problems == [] or all("TXN-P2-0004" not in p for p in report.problems)


def test_the_final_ledger_is_readable_by_the_compute_step(prepared):
    """Круговая проверка контракта: шаг 12 обязан прочитать то, что
    написал шаг 11. Расхождение столбцов ломается здесь, а не в расчёте."""
    from pipeline.compute import load_rows

    apply.run(
        prepared,
        categories={f"TXN-P2-000{i}": TxnCategory(f"TXN-P2-000{i}", "opex")
                    for i in (1, 2, 3)},
        related={}, adjustments={},
    )
    rows = load_rows(prepared.artifacts / A.LEDGER_FINAL)
    assert len(rows) == 3
    assert rows[0].category == "opex"
    assert rows[0].scenario_id == "P2"


def test_alarm_when_nothing_is_tagged_as_related(prepared):
    report = apply.run(
        prepared,
        categories={f"TXN-P2-000{i}": TxnCategory(f"TXN-P2-000{i}", "opex")
                    for i in (1, 2, 3)},
        related={}, adjustments={},
    )
    assert any("связанной стороне" in a for a in report.alarms())


def test_alarm_when_too_much_stayed_uncategorised(prepared):
    report = apply.run(prepared, categories={}, related={}, adjustments={})
    assert any("'other'" in a for a in report.alarms())


def test_missing_related_data_for_a_scenario_is_reported(prepared):
    report = apply.run(
        prepared,
        categories={f"TXN-P2-000{i}": TxnCategory(f"TXN-P2-000{i}", "opex")
                    for i in (1, 2, 3)},
        related={}, adjustments={},
    )
    assert any("связанных сторонах" in p for p in report.problems)


def test_the_report_artifact_records_what_was_applied(prepared):
    apply.run(
        prepared,
        categories={f"TXN-P2-000{i}": TxnCategory(f"TXN-P2-000{i}", "opex")
                    for i in (1, 2, 3)},
        related={"P2": ScenarioParties("P2", parties=[
            Party("Almaty Holdings LLP", 50.0, is_related=True)])},
        adjustments={"P2": ScenarioAdjustments("P2", notes=[
            Note("9.1", "reclassification", "applied",
                 target_txn_id="TXN-P2-0001", to_category="insurance")])},
    )
    data = json.loads((prepared.artifacts / A.APPLY_REPORT).read_text(encoding="utf-8"))
    assert data["counts"]["reclassified"] == 1
    assert data["counts"]["related_tagged"] == 1
    assert data["reclassified"] == ["TXN-P2-0001"]


def test_a_note_with_only_an_amount_finds_its_row():
    """Реальный случай P8 7.1: аудитор назвал сумму, но ни номера
    операции, ни контрагента. Отбрасывать такое значит терять настоящую
    корректировку."""
    rows = [_row("TXN-P8-0001", amount=-918447.52),
            _row("TXN-P8-0002", amount=-5000.0)]
    note = Note("7.1", "reclassification", "applied", value_usd=918447.52,
                to_category="insurance")
    assert [r.txn_id for r in match_note_to_row(note, rows)] == ["TXN-P8-0001"]


def test_an_ambiguous_amount_is_not_applied_at_all():
    """Применить наугад к одной из совпавших строк хуже, чем не применять:
    неверная переклассификация портит два агрегата сразу."""
    rows = [_row("TXN-P8-0001", amount=-1000.0), _row("TXN-P8-0002", amount=-1000.0)]
    note = Note("7.1", "reclassification", "applied", value_usd=1000.0,
                to_category="insurance")
    assert match_note_to_row(note, rows) == []


def test_amount_matching_ignores_the_sign():
    """В реестре расходы отрицательны, аудитор пишет величину."""
    rows = [_row("TXN-P8-0001", amount=-918447.52)]
    note = Note("7.1", "reclassification", "applied", value_usd=918447.52,
                to_category="insurance")
    assert match_note_to_row(note, rows)
