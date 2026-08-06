"""Тесты шага 3: классификация типа документа.

Основное здесь — не «правила срабатывают», а «правила срабатывают
в правильном порядке». Именно порядок отделяет действующий договор
от отменённого и финальный аудит от черновика.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline import classify  # noqa: E402
from pipeline.classify import DocType, classify_text, normalize  # noqa: E402
from pipeline.config import RunPaths  # noqa: E402

PUBLIC = ROOT.parent / "agentic-bank-public"
needs_public = pytest.mark.skipif(not PUBLIC.exists(), reason="нет публичного датасета")

COVENANT_HEADER = "Статья 6 — Финансовые ковенанты"


# --------------------------------------------------------------------------- #
# Приоритет правил
# --------------------------------------------------------------------------- #


def test_both_editions_are_classified_as_loan():
    """Шаг 3 НЕ решает, какая редакция действует. По разъяснению
    организаторов критерий — период действия из самого договора, а пометка
    решающей не является. Разделение делает шаг 4."""
    active = f"{COVENANT_HEADER}\nПункт 6.1 ..."
    stale = f"НЕДЕЙСТВУЮЩАЯ РЕДАКЦИЯ (2024 г.). Заменена.\n{COVENANT_HEADER}\nПункт 6.1 ..."
    assert classify_text(active).type == DocType.LOAN
    assert classify_text(stale).type == DocType.LOAN


def test_superseded_mark_is_recorded_as_a_hint_only():
    stale = f"НЕДЕЙСТВУЮЩАЯ РЕДАКЦИЯ (2024 г.).\n{COVENANT_HEADER}"
    d = classify_text(stale)
    assert any("пометка об отмене" in n for n in d.notes)
    assert classify_text(f"{COVENANT_HEADER}").notes == []


def test_kyc_phrase_inside_loan_does_not_win():
    """Реальный случай: договор ссылается на KYC в пункте 6.3.
    На этом сломался мой первый, наивный классификатор."""
    text = (
        f"{COVENANT_HEADER}\n"
        "Пункт 6.3. Связанные стороны определяются в соответствии с МСФО (IAS) 24 "
        "и сведениями, раскрытыми в досье «Знай своего клиента» (KYC)."
    )
    assert classify_text(text).type == DocType.LOAN


def test_draft_beats_final_audit():
    text = (
        "ПРОЕКТ — ПРОМЕЖУТОЧНАЯ ВЕДОМОСТЬ. НЕ ЯВЛЯЕТСЯ ОКОНЧАТЕЛЬНОЙ ПОЗИЦИЕЙ АУДИТОРА.\n"
        "ДОПОЛНЕНИЕ О СОБЛЮДЕНИИ КОВЕНАНТОВ\nПримечание 8.1 ..."
    )
    assert classify_text(text).type == DocType.AUDIT_DRAFT, (
        "черновик, содержащий приложение, обязан остаться черновиком"
    )


def test_rule_order_is_the_contract():
    """Регрессия на случай перестановки правил при рефакторинге."""
    order = [r.name for r in classify.RULES]
    assert order.index("audit_draft") < order.index("audit_final")
    assert order.index("loan") < order.index("kyc_phrase")
    assert order.index("kyc_code") < order.index("kyc_phrase")


def test_executed_copy_is_never_used_as_a_marker():
    """Пометка «ИСПОЛНИТЕЛЬНЫЙ ЭКЗЕМПЛЯР» стоит на ОБЕИХ редакциях договора.
    Я один раз уже принял её за признак актуальности — этот тест не даст
    повторить."""
    joined = " ".join(r.pattern for r in classify.RULES).upper()
    assert "ИСПОЛНИТЕЛЬНЫЙ" not in joined
    assert "EXECUTION COPY" not in joined


# --------------------------------------------------------------------------- #
# Устойчивость маркеров
# --------------------------------------------------------------------------- #


def test_normalization_survives_line_breaks():
    """В трёх документах корпуса фраза разорвана переносом строки."""
    assert "Знай своего клиента" in normalize("Знай  своего\nклиента")
    text = "досье «Знай\nсвоего клиента»\nKYC-ACC-7801-2025"
    assert classify_text(text).type == DocType.KYC


def test_normalization_handles_dash_variants():
    for dash in "—–-":
        assert classify_text(f"Статья 6 {dash} Финансовые ковенанты").type == DocType.LOAN


def test_kyc_code_survives_lost_cyrillic():
    """Скан корпуса потерял всю кириллицу, но латинский код уцелел.
    Классификация обязана держаться на нём."""
    garbled = "Halyk Bank of Kazakhstan JSC Jjocbe «3HaH CBOero K1HeHTa» KYC-ACC-7806-2025 Cuёт"
    d = classify_text(garbled)
    assert d.type == DocType.KYC
    assert d.rule == "kyc_code"


def test_background_is_the_default():
    d = classify_text("Стандартная операционная процедура — доступ подрядчиков")
    assert d.type == DocType.BACKGROUND
    assert d.rule is None


def test_empty_text_is_flagged_not_guessed(tmp_path):
    paths = RunPaths.create(tmp_path / "run")
    texts = paths.artifacts / "01_texts"
    texts.mkdir(parents=True)
    (texts / "empty.txt").write_text("", encoding="utf-8")
    rep = classify.run(paths)
    d = rep.docs[0]
    assert d.type == DocType.BACKGROUND and d.confidence == 0.0
    assert any("недостоверна" in n for n in d.notes)


def test_confidence_is_lower_for_secondary_marker():
    by_name = {r.name: r for r in classify.RULES}
    assert by_name["kyc_phrase"].confidence < by_name["kyc_code"].confidence


# --------------------------------------------------------------------------- #
# Реальный корпус
# --------------------------------------------------------------------------- #


# corpus_report — сессионная фикстура из conftest.py


@pytest.mark.slow
def test_corpus_counts_are_exact(corpus_report):
    rep, _ = corpus_report
    assert rep.counts() == {
        "LOAN": 24,
        "AUDIT_FINAL": 12,
        "AUDIT_DRAFT": 5,
        "KYC": 12,
        "TREASURY_MEMO": 1,
        "BACKGROUND": 148,
    }


@pytest.mark.slow
def test_loans_are_left_unresolved_for_step_four(corpus_report):
    """Все 24 договора выходят из шага 3 нерешёнными: выбор действующего —
    задача шага 4, где известен отчётный период."""
    rep, _ = corpus_report
    assert len(rep.of_type(DocType.LOAN)) == 24
    assert rep.of_type(DocType.LOAN_ACTIVE) == []
    marked = [d for d in rep.of_type(DocType.LOAN)
              if any("пометка об отмене" in n for n in d.notes)]
    assert len(marked) == 12, "пометка фиксируется как справочный признак"


@pytest.mark.slow
def test_scan_is_classified_despite_broken_ocr(corpus_report):
    rep, _ = corpus_report
    scan = next(d for d in rep.docs if d.doc_id == "f3fa6d20c8a1")
    assert scan.type == DocType.KYC
    assert scan.rule == "kyc_code"


@pytest.mark.slow
def test_collisions_are_recorded_for_diagnosis(corpus_report):
    rep, _ = corpus_report
    assert rep.collisions, "коллизии маркеров есть по устройству корпуса — их надо видеть"
    kyc_collisions = [c for c in rep.collisions if "kyc_code" in c and "kyc_phrase" in c]
    assert kyc_collisions, "код и формулировка KYC пересекаются по устройству корпуса"


@pytest.mark.slow
def test_artifact_is_readable_by_next_step(corpus_report):
    _, rp = corpus_report
    loaded = classify.load(rp)
    assert len(loaded) == 202
    assert all(d.scenario_id is None for d in loaded.values()), "scenario_id заполняет шаг 4"


def test_alarms_fire_when_markers_do_not_match():
    """Если формулировки приватного набора окажутся другими, всё уедет
    в фон. Это надо увидеть в первые минуты, а не в конце."""
    rep = classify.ClassifyReport(docs=[
        classify.DocClass(doc_id=f"d{i}", type=DocType.BACKGROUND, confidence=1.0)
        for i in range(20)
    ])
    alarms = " | ".join(rep.alarms())
    assert "LOAN" in alarms and "AUDIT_FINAL" in alarms and "KYC" in alarms
    assert "ушло 100%" in alarms


def test_alarms_catch_loan_audit_imbalance():
    docs = [classify.DocClass(f"l{i}", DocType.LOAN) for i in range(4)]
    docs += [classify.DocClass(f"a{i}", DocType.AUDIT_FINAL) for i in range(12)]
    docs += [classify.DocClass(f"k{i}", DocType.KYC) for i in range(12)]
    docs += [classify.DocClass("t", DocType.TREASURY_MEMO)]
    rep = classify.ClassifyReport(docs=docs)
    assert any("не может быть меньше" in a for a in rep.alarms())


@pytest.mark.slow
def test_no_alarms_on_public_corpus(corpus_report):
    rep, _ = corpus_report
    assert rep.alarms() == [], f"неожиданные тревоги: {rep.alarms()}"


def test_kyc_code_does_not_assume_identifier_format():
    """Регрессия: первая версия правила требовала «буквы, затем цифры»
    и разваливалась на счетах вида BANK-X6. Поймал тест переносимости."""
    for code in ["KYC-ACC-7806-2025", "KYC-BANK-X6-2025", "KYC CASE12-0001"]:
        assert classify_text(code).type == DocType.KYC, code


def test_kyc_code_does_not_fire_on_bare_mentions():
    """Договор упоминает KYC в пункте 6.3 — это не делает его досье."""
    for text in ["досье «Знай своего клиента» (KYC)", "процедуры KYC и AML", "KYC-досье клиента"]:
        rule = classify_text(text).rule
        assert rule != "kyc_code", f"{text!r} ошибочно принят за код досье"
