"""Переносимость на англоязычные документы.

Организаторы предупредили: документы приватного набора «могут быть на
английском, но в основном на русском». Публичный набор целиком русский,
поэтому английский путь НЕ проверяется ни одним тестом на реальных файлах —
эти тесты закрывают пробел на переводных образцах.

Проверяется не «модуль умеет английский», а конкретные места, где промах
языка ведёт к НЕВЕРНОМУ ЧИСЛУ, а не к падению: тип документа и период
действия договора.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline import attribute, classify  # noqa: E402
from pipeline.classify import DocType  # noqa: E402
from pipeline.llm import LLMClient, MockProvider  # noqa: E402


# --------------------------------------------------------------------------- #
# Классификация
# --------------------------------------------------------------------------- #

ENGLISH_DOCS = {
    "Article 6 — FINANCIAL COVENANTS. The Borrower undertakes...": DocType.LOAN,
    "DRAFT — INTERIM SCHEDULE of adjustments": DocType.AUDIT_DRAFT,
    "These figures are NOT THE FINAL POSITION of the auditor": DocType.AUDIT_DRAFT,
    "COVENANT COMPLIANCE SUPPLEMENT to the audited statements": DocType.AUDIT_FINAL,
    "Customer file KYC-ACC-7801-2025": DocType.KYC,
    "TREASURY MEMORANDUM regarding settlement of invoice": DocType.TREASURY_MEMO,
    "Quarterly market outlook for the logistics sector": DocType.BACKGROUND,
}


@pytest.mark.parametrize("text,expected", list(ENGLISH_DOCS.items()))
def test_english_markers_are_recognised(text, expected):
    assert classify.classify_text(text).type == expected


def test_draft_still_beats_final_in_english():
    """Порядок правил обязан сохраняться и на английском: черновик,
    содержащий слова финального приложения, остаётся черновиком."""
    text = ("DRAFT — INTERIM SCHEDULE\n"
            "COVENANT COMPLIANCE SUPPLEMENT\n"
            "adjustments below are provisional")
    assert classify.classify_text(text).type == DocType.AUDIT_DRAFT


def test_bare_word_draft_does_not_create_a_false_draft():
    """«DRAFT» в колонтитуле — не признак черновика. Ложный AUDIT_DRAFT
    выбросил бы настоящие корректировки, что дороже пропуска."""
    text = "COVENANT COMPLIANCE SUPPLEMENT\nfooter: draft printed on recycled paper"
    assert classify.classify_text(text).type == DocType.AUDIT_FINAL


def test_russian_corpus_is_unaffected():
    """Английские альтернативы не должны переигрывать русские правила."""
    assert classify.classify_text(
        "Статья 6 — Финансовые ковенанты").type == DocType.LOAN
    assert classify.classify_text(
        "ДОПОЛНЕНИЕ О СОБЛЮДЕНИИ КОВЕНАНТОВ").type == DocType.AUDIT_FINAL


def test_english_supersession_mark_is_noted():
    d = classify.classify_text(
        "Article 6 — FINANCIAL COVENANTS\nThis edition is NO LONGER IN FORCE")
    assert d.type == DocType.LOAN
    assert any("отмене редакции" in n for n in d.notes)


# --------------------------------------------------------------------------- #
# Период действия — от него зависит выбор действующего договора
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("text,expected", [
    ("FINANCIAL COVENANTS ... for the period from 2025-01-01 to 2025-12-31 ...",
     ("2025-01-01", "2025-12-31")),
    ("FINANCIAL COVENANTS ... from 1 January 2025 through 31 December 2025 ...",
     ("2025-01-01", "2025-12-31")),
    ("FINANCIAL COVENANTS ... between 2025-01-01 and 2025-06-30 ...",
     ("2025-01-01", "2025-06-30")),
    ("Финансовые ковенанты ... с 2025-01-01 по 2025-12-31 ...",
     ("2025-01-01", "2025-12-31")),
    ("Финансовые ковенанты ... с 1 января 2025 по 31 декабря 2025 ...",
     ("2025-01-01", "2025-12-31")),
])
def test_covenant_period_is_read_in_both_languages(text, expected):
    assert attribute.covenant_period(text) == expected


def test_two_adjacent_dates_are_not_mistaken_for_a_period():
    """Без предлога это шапка документа, а не период действия."""
    assert attribute.covenant_period("Reference 2025-01-01 2025-12-31 page 1") is None


def test_english_section_anchor_is_used_before_free_search():
    """Первый период в договоре — период выборки кредита. Если якорь
    раздела не найден, вернётся именно он, и договор будет признан
    недействующим по чужим датам."""
    text = ("Drawdown available from 2024-02-01 to 2024-05-31. "
            "Article 6 — FINANCIAL COVENANTS. "
            "Ratios are tested for the period from 2025-01-01 to 2025-12-31.")
    assert attribute.covenant_period(text) == ("2025-01-01", "2025-12-31")


@pytest.mark.parametrize("text,expected", [
    ("Dated December 31, 2025", ["2025-12-31"]),
    ("Dated 31 December 2025", ["2025-12-31"]),
    ("Signed 15 марта 2026", ["2026-03-15"]),
])
def test_document_dates_parse_in_both_languages(text, expected):
    assert attribute.all_dates(text) == expected


# --------------------------------------------------------------------------- #
# Запасная классификация моделью
# --------------------------------------------------------------------------- #

def _report(pairs):
    rep = classify.ClassifyReport()
    for doc_id, doc_type in pairs:
        rep.docs.append(classify.DocClass(doc_id=doc_id, type=doc_type, confidence=1.0))
    return rep


def _client(payload, tmp_path):
    mock = MockProvider()
    mock.register_rule(lambda r: True, lambda r: payload)
    return LLMClient(cache_dir=tmp_path / "c", provider=mock), mock


def test_fallback_does_not_spend_when_rules_worked(tmp_path):
    """Главная защита от расхода: тревог нет — вызовов нет."""
    rep = _report([("a", DocType.LOAN), ("b", DocType.AUDIT_FINAL),
                   ("c", DocType.KYC), ("d", DocType.TREASURY_MEMO)])
    assert rep.alarms() == []
    client, mock = _client({"doc_type": "KYC", "evidence_quote": "", "confidence": 1.0}, tmp_path)
    assert classify.llm_fallback(rep, {}, client) == []
    assert mock.calls == []


def test_fallback_recovers_a_document_the_rules_missed(tmp_path):
    text = "Counterparty due diligence file, reference number 7801"
    rep = _report([("a", DocType.LOAN), ("b", DocType.AUDIT_FINAL), ("x", DocType.BACKGROUND)])
    assert rep.alarms(), "KYC отсутствует — тревога обязана сработать"
    client, mock = _client(
        {"doc_type": "KYC", "evidence_quote": "due diligence file", "confidence": 0.9}, tmp_path)
    classify.llm_fallback(rep, {"x": text}, client)
    doc = next(d for d in rep.docs if d.doc_id == "x")
    assert doc.type == DocType.KYC and doc.rule == "llm_fallback"
    assert len(mock.calls) == 1, "вызов только для нераспознанного документа"


def test_fallback_rejects_a_quote_that_is_not_in_the_document(tmp_path):
    """Цитата, которой нет в тексте, — признак выдумки. Такой вердикт
    не применяется, но и не замалчивается."""
    rep = _report([("a", DocType.LOAN), ("b", DocType.AUDIT_FINAL), ("x", DocType.BACKGROUND)])
    client, _ = _client(
        {"doc_type": "AUDIT_FINAL", "evidence_quote": "COVENANT COMPLIANCE SUPPLEMENT",
         "confidence": 0.99}, tmp_path)
    classify.llm_fallback(rep, {"x": "unrelated newsletter text"}, client)
    doc = next(d for d in rep.docs if d.doc_id == "x")
    assert doc.type == DocType.BACKGROUND
    assert any("цитата не найдена" in n for n in doc.notes)


def test_fallback_rejects_low_confidence(tmp_path):
    rep = _report([("a", DocType.LOAN), ("b", DocType.AUDIT_FINAL), ("x", DocType.BACKGROUND)])
    client, _ = _client(
        {"doc_type": "KYC", "evidence_quote": "file", "confidence": 0.3}, tmp_path)
    classify.llm_fallback(rep, {"x": "file"}, client)
    doc = next(d for d in rep.docs if d.doc_id == "x")
    assert doc.type == DocType.BACKGROUND
    assert any("уверенность" in n for n in doc.notes)


def test_fallback_cannot_override_a_rule_verdict(tmp_path):
    """Правила остаются источником истины: модель трогает только фон."""
    rep = _report([("a", DocType.LOAN), ("b", DocType.AUDIT_DRAFT), ("x", DocType.BACKGROUND)])
    client, mock = _client(
        {"doc_type": "AUDIT_FINAL", "evidence_quote": "x", "confidence": 1.0}, tmp_path)
    classify.llm_fallback(rep, {"x": "x"}, client)
    assert next(d for d in rep.docs if d.doc_id == "b").type == DocType.AUDIT_DRAFT
    assert len(mock.calls) == 1


def test_fallback_survives_a_provider_failure(tmp_path):
    """Один сбойный документ не должен ронять шаг в боевом окне."""
    rep = _report([("a", DocType.LOAN), ("x", DocType.BACKGROUND)])

    class Boom:
        def extract(self, req):
            raise RuntimeError("нет сети")

    notes = classify.llm_fallback(rep, {"x": "текст"}, Boom())
    assert notes
    doc = next(d for d in rep.docs if d.doc_id == "x")
    assert doc.type == DocType.BACKGROUND
    assert any("не удалась" in n for n in doc.notes)


def test_fallback_reports_when_no_model_is_available(tmp_path):
    rep = _report([("a", DocType.BACKGROUND)])
    notes = classify.llm_fallback(rep, {"a": "x"}, None)
    assert notes and "модель недоступна" in notes[0]
