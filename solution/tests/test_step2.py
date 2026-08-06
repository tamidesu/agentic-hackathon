"""Тесты шага 2: извлечение текста и распознавание сканов."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline import extract  # noqa: E402
from pipeline.config import RunPaths, discover_dataset  # noqa: E402
from pipeline.llm import LLMClient, MockProvider  # noqa: E402

PUBLIC = ROOT.parent / "agentic-bank-public"
SCAN_PDF = PUBLIC / "documents" / "f3fa6d20c8a1.pdf"

needs_public = pytest.mark.skipif(not PUBLIC.exists(), reason="нет публичного датасета")
needs_scan = pytest.mark.skipif(not SCAN_PDF.exists(), reason="нет скана")


# --------------------------------------------------------------------------- #
# Метрики качества
# --------------------------------------------------------------------------- #


def test_cyrillic_ratio_ignores_digits_and_punctuation():
    assert extract._cyrillic_ratio("Счёт ACC-7806: 46.8%") == pytest.approx(4 / 7)
    assert extract._cyrillic_ratio("12345 !!! ---") == 0.0
    assert extract._cyrillic_ratio("") == 0.0


def test_script_anomaly_catches_fully_latinised_ocr():
    """Реальный провал, на котором сломалась первая версия метрики:
    кириллица распознана целиком латиницей, стыков нет, garble_ratio = 0."""
    docs = [
        extract.DocExtract(f"ok{i}", f"ok{i}.pdf", "pdfplumber", chars=2000, cyrillic_ratio=0.95)
        for i in range(10)
    ]
    bad = extract.DocExtract("scan", "scan.pdf", "tesseract", chars=5000, cyrillic_ratio=0.0)
    docs.append(bad)

    extract._flag_script_anomalies(docs)
    assert bad.needs_review
    assert any("письменность не совпадает" in w for w in bad.warnings)
    assert not any(d.needs_review for d in docs[:10])


def test_script_anomaly_silent_on_latin_corpus():
    """На английском корпусе проверка обязана отключиться, а не сыпать ложными."""
    docs = [
        extract.DocExtract(f"ok{i}", "x.pdf", "pdfplumber", chars=2000, cyrillic_ratio=0.0)
        for i in range(10)
    ]
    scan = extract.DocExtract("scan", "s.pdf", "vision", chars=5000, cyrillic_ratio=0.0)
    docs.append(scan)
    extract._flag_script_anomalies(docs)
    assert not scan.needs_review


def test_script_anomaly_needs_enough_evidence():
    docs = [extract.DocExtract("a", "a.pdf", "pdfplumber", chars=2000, cyrillic_ratio=0.95)]
    scan = extract.DocExtract("s", "s.pdf", "tesseract", chars=5000, cyrillic_ratio=0.0)
    docs.append(scan)
    extract._flag_script_anomalies(docs)
    assert not scan.needs_review, "на одном образце медиана не показательна"


def test_garble_ratio_catches_intra_word_mixing():
    assert extract._garble_ratio("Cчёт") > 0
    assert extract._garble_ratio("Счёт нормальный") == 0.0


# --------------------------------------------------------------------------- #
# Preflight
# --------------------------------------------------------------------------- #


def test_preflight_warns_about_missing_language_packs(monkeypatch):
    monkeypatch.setattr(extract, "tesseract_languages", lambda: ["eng", "osd"])
    problems = extract.preflight(None)
    assert any("rus" in p and "молча и неверно" in p for p in problems)


def test_preflight_quiet_when_llm_available(monkeypatch):
    monkeypatch.setattr(extract, "tesseract_languages", lambda: [])
    monkeypatch.setattr(extract.shutil, "which", lambda n: "/usr/bin/" + n)
    fake = object()
    assert extract.preflight(fake) == []  # type: ignore[arg-type]


def test_preflight_reports_no_ocr_at_all(monkeypatch):
    monkeypatch.setattr(extract, "tesseract_languages", lambda: [])
    monkeypatch.setattr(extract.shutil, "which", lambda n: None)
    problems = extract.preflight(None)
    assert any("распознать нечем" in p for p in problems)
    assert any("pdftoppm" in p for p in problems)


# --------------------------------------------------------------------------- #
# Извлечение отдельных файлов
# --------------------------------------------------------------------------- #


def test_extracts_plain_text_file(tmp_path):
    p = tmp_path / "doc.txt"
    p.write_text("Договор банковского займа", encoding="utf-8")
    d = extract.extract_one(p, None, [])
    assert d.method == "text"
    assert getattr(d, "_text") == "Договор банковского займа"


def test_unknown_extension_fails_loudly(tmp_path):
    p = tmp_path / "x.exe"
    p.write_bytes(b"\x00\x01")
    d = extract.extract_one(p, None, [])
    assert d.method == "failed"
    assert any("неизвестное расширение" in w for w in d.warnings)


def test_broken_pdf_does_not_crash(tmp_path):
    p = tmp_path / "broken.pdf"
    p.write_bytes("%PDF-1.4 это не настоящий pdf".encode("utf-8"))
    d = extract.extract_one(p, None, [])
    assert d.method == "failed" and d.warnings


@needs_scan
def test_scan_is_detected_and_routed_to_vision():
    """Скан обязан уйти на распознавание, а не молча дать пустой текст."""
    mock = MockProvider()
    mock.register_rule(
        lambda r: True,
        lambda r: {"text": "Досье «Знай своего клиента» KYC-ACC-7806-2025", "uncertain": []},
    )
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        client = LLMClient(cache_dir=Path(tmp) / "c", provider=mock)
        d = extract.extract_one(SCAN_PDF, client, [])

    assert d.method == "vision"
    assert d.pages == 3
    assert len(mock.calls) == 3, "должен быть один вызов на страницу"
    assert all(c.images for c in mock.calls), "изображения не доехали до модели"
    assert "ACC-7806" in getattr(d, "_text")


@needs_scan
def test_vision_request_images_change_cache_key():
    """Разные страницы обязаны иметь разные ключи кэша, иначе стр. 2 и 3
    получат транскрипцию первой."""
    mock = MockProvider()
    mock.register_rule(lambda r: True, lambda r: {"text": "x" * 20, "uncertain": []})
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        client = LLMClient(cache_dir=Path(tmp) / "c", provider=mock)
        extract.extract_one(SCAN_PDF, client, [])
    keys = {c.cache_key() for c in mock.calls}
    assert len(keys) == 3, "ключи кэша страниц совпали — изображения не входят в ключ"


# --------------------------------------------------------------------------- #
# Полный прогон
# --------------------------------------------------------------------------- #


# full_run — сессионная фикстура из conftest.py, общая для всех медленных тестов


@pytest.fixture
def small_corpus(tmp_path):
    """Малый корпус для тестов инкрементальности — быстро и по той же логике."""
    if not PUBLIC.exists():
        pytest.skip("нет публичного датасета")
    root = tmp_path / "ds"
    docs = root / "documents"
    docs.mkdir(parents=True)
    for p in sorted((PUBLIC / "documents").glob("*.pdf"))[:8]:
        shutil.copy(p, docs / p.name)
    shutil.copy(PUBLIC / "master_ledger_2025.csv", root / "ledger.csv")
    shutil.copy(PUBLIC / "submission_template.json", root / "tpl.json")
    return discover_dataset(root)


@pytest.mark.slow
def test_full_run_covers_corpus_and_skips_non_documents(full_run):
    rep, rp = full_run

    assert len(rep.docs) == 202, "200 PDF + csv + txt"
    assert "Thumbs.db" in rep.skipped
    assert rep.failed == [], f"провалы: {[d.doc_id for d in rep.failed]}"

    by_method = rep.to_dict()["by_method"]
    assert by_method["pdfplumber"] == 199
    assert by_method.get("tesseract", 0) + by_method.get("vision", 0) == 1

    assert len(list((rp.artifacts / "01_texts").glob("*.txt"))) == 202
    assert (rp.artifacts / "01_extract_report.json").exists()


@pytest.mark.slow
def test_extracted_text_matches_known_content(full_run):
    """Содержательная проверка: договор P1 извлёкся целиком, с порогом 6.3."""
    _, rp = full_run
    loan = (rp.artifacts / "01_texts" / "8d878af064f2.txt").read_text(encoding="utf-8")
    assert "Статья 6 — Финансовые ковенанты" in loan
    assert "ACC-7801" in loan
    assert "$450,000.00" in loan
    assert len(loan) > 40000


@pytest.mark.slow
def test_only_the_scan_needs_review(full_run):
    rep, _ = full_run
    assert [d.doc_id for d in rep.review] == ["f3fa6d20c8a1"]


def test_rerun_is_incremental(small_corpus, tmp_path):
    """Критично для боевого окна: повторный прогон не переизвлекает всё заново."""
    rp = RunPaths.create(tmp_path / "run")
    first = extract.run(small_corpus, rp, llm=None, workers=4)
    second = extract.run(small_corpus, rp, llm=None, workers=4)

    assert len(second.docs) == len(first.docs) == 8
    assert first.extracted == 8 and first.reused == 0
    assert second.extracted == 0 and second.reused == 8, (
        "повторный прогон переизвлёк документы заново"
    )


def test_force_ignores_cache(small_corpus, tmp_path):
    rp = RunPaths.create(tmp_path / "run")
    extract.run(small_corpus, rp, llm=None, workers=4)
    cached = extract.run(small_corpus, rp, llm=None, workers=4)
    forced = extract.run(small_corpus, rp, llm=None, workers=4, force=True)
    assert cached.extracted == 0 and cached.reused == 8
    assert forced.extracted == 8 and forced.reused == 0


def test_changed_source_is_reextracted(small_corpus, tmp_path):
    """Отпечаток обязан реагировать на изменение файла, иначе правка
    исходника молча не доедет до артефакта."""
    rp = RunPaths.create(tmp_path / "run")
    extract.run(small_corpus, rp, llm=None, workers=4)

    victim = sorted(small_corpus.documents_dir.glob("*.pdf"))[0]
    victim.write_bytes(victim.read_bytes() + b"\n% touched")

    rep = extract.run(small_corpus, rp, llm=None, workers=4)
    changed = next(d for d in rep.docs if d.doc_id == victim.stem)
    assert changed.method != "cached"


@needs_scan
def test_long_scan_truncation_is_reported(monkeypatch):
    """Тихое усечение длинного скана — потеря данных. Обязано быть громким."""
    monkeypatch.setattr(extract, "MAX_RENDER_PAGES", 2)
    imgs, notes = extract._render_pages(SCAN_PDF, extract.VISION_DPI, total_pages=3, max_pages=2)
    assert len(imgs) == 2
    assert notes and "НЕ прочитаны" in notes[0]


@needs_scan
def test_vision_uses_lower_dpi_than_tesseract():
    """Модель ужимает изображение — гнать 300 DPI значит платить втрое ни за что."""
    assert extract.VISION_DPI < extract.TESSERACT_DPI
    vis, _ = extract._render_pages(SCAN_PDF, extract.VISION_DPI, 3, max_pages=1)
    tes, _ = extract._render_pages(SCAN_PDF, extract.TESSERACT_DPI, 3, max_pages=1)
    assert len(vis[0]) < len(tes[0]) / 2
