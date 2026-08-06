"""Тесты шага 9: подготовка реестра транзакций."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline import ledger  # noqa: E402
from pipeline.ledger import find_disclosed_amount, find_fx_rates  # noqa: E402


# --------------------------------------------------------------------------- #
# Восстановление пропущенных сумм
# --------------------------------------------------------------------------- #

AUDIT_NOTE = (
    "Примечание 8 — Суммы, не отражённые в выгрузке реестра "
    "(8.1) Операция TXN-P8-0031 (Kyzylorda Drilling Personnel LLP): сумма не "
    "отражена в выгрузке реестра; фактическая сумма операции составляет "
    "$884,204.16 (расход)."
)

TREASURY_NOTE = (
    "4. Позиции, не выгруженные в бухгалтерскую выгрузку "
    "(1) Операция TXN-P7-0033 (State Revenue Committee): сумма не отражена "
    "в выгрузке реестра; фактическая сумма операции составляет "
    "$486,204.19 (расход)."
)


def test_recovers_amount_from_audit_note():
    got = find_disclosed_amount("TXN-P8-0031", AUDIT_NOTE)
    assert got is not None
    value, evidence, direction_known = got
    assert direction_known
    assert value == pytest.approx(-884204.16), "расход обязан быть отрицательным"
    assert "884,204.16" in evidence


def test_recovers_amount_from_treasury_memo():
    """Второй пропуск лежит в единственной казначейской записке корпуса —
    документе, который любой ретривер отранжировал бы низко."""
    got = find_disclosed_amount("TXN-P7-0033", TREASURY_NOTE)
    assert got is not None and got[0] == pytest.approx(-486204.19) and got[2]


def test_income_keeps_positive_sign():
    text = "Операция TXN-X-1: фактическая сумма операции составляет $1,000.00 (поступление)."
    assert find_disclosed_amount("TXN-X-1", text)[0] == pytest.approx(1000.00)


def test_prefers_actual_amount_over_other_numbers():
    """В тексте может стоять и ошибочно отражённая сумма — нужна фактическая."""
    text = (
        "Операция TXN-X-1: в выгрузке ошибочно указано $100.00; "
        "фактическая сумма операции составляет $250.00 (расход)."
    )
    assert find_disclosed_amount("TXN-X-1", text)[0] == pytest.approx(-250.00)


def test_returns_none_when_transaction_not_mentioned():
    assert find_disclosed_amount("TXN-X-9", AUDIT_NOTE) is None


def test_returns_none_when_mentioned_without_amount():
    assert find_disclosed_amount("TXN-X-1", "Операция TXN-X-1 рассмотрена отдельно.") is None


def test_recovery_survives_line_breaks():
    broken = AUDIT_NOTE.replace("составляет ", "составляет\n")
    assert find_disclosed_amount("TXN-P8-0031", broken)[0] == pytest.approx(-884204.16)


# --------------------------------------------------------------------------- #
# Курсы валют
# --------------------------------------------------------------------------- #

FX_NOTE = (
    "Примечание 9 — Валютные курсы. Суммы в валютах, отличных от доллара США, "
    "пересчитываются по курсу фактического расчёта; отдельная таблица курсов "
    "не ведётся. (9.1) Расчёты с контрагентом «Rheinland Katalyse Service GmbH»: "
    "счёт на сумму 72,146.75 EUR урегулирован платежом в долларах США "
    "в размере $83,690.23."
)


def test_derives_rate_from_disclosed_settlement_pair():
    rates = find_fx_rates(FX_NOTE)
    assert "EUR" in rates
    assert rates["EUR"][0] == pytest.approx(1.16, abs=1e-6)


def test_rate_is_not_invented_when_nothing_disclosed():
    """Внешние котировки не используются: расхождение с курсом расчёта
    легко превышает 5% и обнуляет и actual, и evidence."""
    assert find_fx_rates("Операции в евро пересчитаны по курсу на дату сделки.") == {}


def test_implausible_rate_is_rejected():
    bad = "счёт на сумму 1.00 EUR урегулирован платежом в размере $500,000.00."
    assert "EUR" not in find_fx_rates(bad)


def test_usd_pair_is_not_treated_as_conversion():
    assert find_fx_rates("сумма 100.00 USD оплачена платежом $100.00.") == {}


# --------------------------------------------------------------------------- #
# Полный прогон
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="session")
def prepared_ledger(attributed, public_dataset, corpus_report):
    _, rp = corpus_report
    return ledger.run(public_dataset, rp)


@pytest.mark.slow
def test_noise_is_dropped_by_template_membership(prepared_ledger):
    txns, rep = prepared_ledger
    assert rep.total_rows == 1473
    assert rep.kept_rows == 673
    assert rep.dropped_noise == 800
    assert {t.scenario_id for t in txns} == {
        "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "P10", "B1", "B4"
    }


@pytest.mark.slow
def test_both_missing_amounts_are_recovered(prepared_ledger):
    """Один пропуск лежит в аудите, другой — в казначейской записке.
    Потерять любой значит занизить агрегат на сотни тысяч."""
    txns, rep = prepared_ledger
    recovered = {t.txn_id: t for t in txns if t.recovered}
    assert set(recovered) == {"TXN-P7-0033", "TXN-P8-0031"}
    assert recovered["TXN-P7-0033"].amount == pytest.approx(-486204.19)
    assert recovered["TXN-P8-0031"].amount == pytest.approx(-884204.16)
    assert all(t.evidence for t in recovered.values()), "восстановление без источника"


@pytest.mark.slow
def test_no_row_is_left_without_amount(prepared_ledger):
    txns, rep = prepared_ledger
    assert [t.txn_id for t in txns if t.amount is None] == []
    assert rep.unresolved == []


@pytest.mark.slow
def test_currency_is_converted_from_disclosed_rate(prepared_ledger):
    txns, rep = prepared_ledger
    assert rep.fx_rates == {"EUR": 1.16}
    assert "Rheinland" in rep.fx_evidence["EUR"]

    eur = [t for t in txns if t.currency == "EUR"]
    assert len(eur) == 15
    assert all(t.amount_usd is not None for t in eur)

    p3 = next(t for t in eur if t.txn_id == "TXN-P3-0024")
    assert p3.amount_usd == pytest.approx(-710945.73, abs=0.01)


@pytest.mark.slow
def test_borrowed_rate_is_reported_not_hidden(prepared_ledger):
    """Курс раскрыт у одного заёмщика и применён к остальным — решение
    допустимое, но обязано быть видимым."""
    _, rep = prepared_ledger
    assert any("применён к" in p for p in rep.problems)


@pytest.mark.slow
def test_usd_rows_pass_through_unchanged(prepared_ledger):
    txns, _ = prepared_ledger
    usd = [t for t in txns if t.currency == "USD"]
    assert len(usd) == 658
    assert all(t.amount_usd == t.amount for t in usd)


@pytest.mark.slow
def test_no_duplicate_transaction_ids(prepared_ledger):
    _, rep = prepared_ledger
    assert not [p for p in rep.problems if "дубль" in p]


@pytest.mark.slow
def test_artifact_is_written_for_next_step(prepared_ledger, corpus_report):
    _, rp = corpus_report
    out = rp.artifacts / "06_ledger_clean.csv"
    assert out.exists()
    import csv

    rows = list(csv.DictReader(out.open(encoding="utf-8")))
    assert len(rows) == 673
    assert {"txn_id", "scenario_id", "amount_usd", "evidence"} <= set(rows[0])


def test_unknown_direction_is_reported_not_assumed():
    """Ошибка в знаке крупной суммы искажает агрегат вдвое —
    молчаливое допущение здесь недопустимо."""
    text = "Операция TXN-X-1: фактическая сумма операции составляет $500.00."
    value, _, direction_known = find_disclosed_amount("TXN-X-1", text)
    assert value == pytest.approx(500.00)
    assert direction_known is False
