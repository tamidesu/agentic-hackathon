"""Тесты шага 2: извлечение текста и распознавание сканов."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline import artifacts as A  # noqa: E402
from pipeline import extract  # noqa: E402
from pipeline.config import RunPaths, discover_dataset  # noqa: E402
from pipeline.llm import LLMClient, MockProvider  # noqa: E402

PUBLIC = ROOT.parent / "agentic-bank-public"
SCAN_PDF = PUBLIC / "documents" / "f3fa6d20c8a1.pdf"

needs_public = pytest.mark.skipif(not PUBLIC.exists(), reason="нет публичного датасета")
needs_scan = pytest.mark.skipif(not SCAN_PDF.exists(), reason="нет скана")
#: Рендер страниц требует poppler-utils. На Windows его обычно нет, и тесты
#: распознавания должны ПРОПУСКАТЬСЯ, а не падать: отсутствие системного
#: пакета — состояние окружения, а не дефект кода. Само отсутствие ловит
#: preflight шага 2, у которого есть отдельный тест.
needs_poppler = pytest.mark.skipif(
    shutil.which("pdftoppm") is None,
    reason="pdftoppm (poppler-utils) не установлен — сканы отрендерить нечем",
)


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
@needs_poppler
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
@needs_poppler
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
    # 196 чисто текстовых + 3 смешанных (текст плюс нарисованная страница)
    # + 1 полный скан = 200 PDF.
    mixed = sum(v for k, v in by_method.items() if k.startswith("pdfplumber+"))
    scanned = by_method.get("tesseract", 0) + by_method.get("vision", 0)
    assert by_method["pdfplumber"] == 196
    assert mixed == 3, (
        f"смешанных документов {mixed}, ожидалось 3: досье KYC у P2 и P9 "
        f"и аудиторское приложение P4 несут нарисованные страницы"
    )
    assert scanned == 1

    assert len(list((rp.artifacts / "01_texts").glob("*.txt"))) == 202
    assert (rp.artifacts / A.EXTRACT_REPORT).exists()


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
@needs_poppler
def test_long_scan_truncation_is_reported(monkeypatch):
    """Тихое усечение длинного скана — потеря данных. Обязано быть громким."""
    monkeypatch.setattr(extract, "MAX_RENDER_PAGES", 2)
    imgs, notes = extract._render_pages(SCAN_PDF, extract.VISION_DPI, total_pages=3, max_pages=2)
    assert len(imgs) == 2
    assert notes and "НЕ прочитаны" in notes[0]


@needs_scan
@needs_poppler
def test_vision_uses_lower_dpi_than_tesseract():
    """Модель ужимает изображение — гнать 300 DPI значит платить втрое ни за что."""
    assert extract.VISION_DPI < extract.TESSERACT_DPI
    vis, _ = extract._render_pages(SCAN_PDF, extract.VISION_DPI, 3, max_pages=1)
    tes, _ = extract._render_pages(SCAN_PDF, extract.TESSERACT_DPI, 3, max_pages=1)
    assert len(vis[0]) < len(tes[0]) / 2


# --------------------------------------------------------------------------- #
# Провал извлечения не должен кэшироваться
#
# РЕАЛЬНЫЙ СЛУЧАЙ. Скан KYC заёмщика P6 не читался: не был установлен
# poppler. Отпечаток файла при этом записывался наравне с успешными, рядом
# ложился ПУСТОЙ .txt — и следующий прогон считал документ обработанным.
# Пользователь установил poppler, перезагрузил машину, перезапустил — и
# увидел ту же ошибку, приехавшую из отчёта прошлого раза.
#
# Причина неудачи лежит ВНЕ входного файла, поэтому отпечаток файла о ней
# ничего не знает и знать не может. Кэшировать имеет смысл результат,
# а не неудачу.
# --------------------------------------------------------------------------- #


@pytest.fixture
def corpus_with_a_broken_file(tmp_path):
    if not PUBLIC.exists():
        pytest.skip("нет публичного датасета")
    root = tmp_path / "ds"
    docs = root / "documents"
    docs.mkdir(parents=True)
    # >=5 файлов: discover_dataset ищет каталог документов по их количеству.
    for p in sorted((PUBLIC / "documents").glob("*.pdf"))[:5]:
        shutil.copy(p, docs / p.name)
    broken = docs / "broken.pdf"
    broken.write_bytes(b"%PDF-1.4 not really a pdf")
    shutil.copy(PUBLIC / "master_ledger_2025.csv", root / "ledger.csv")
    shutil.copy(PUBLIC / "submission_template.json", root / "tpl.json")
    return discover_dataset(root), broken


def test_a_failure_is_retried_on_the_next_run(corpus_with_a_broken_file, tmp_path):
    """Главная проверка: починка окружения обязана иметь эффект."""
    dataset, _ = corpus_with_a_broken_file
    rp = RunPaths.create(tmp_path / "run")

    first = extract.run(dataset, rp, llm=None, workers=2)
    assert [d.doc_id for d in first.failed] == ["broken"]

    second = extract.run(dataset, rp, llm=None, workers=2)
    assert second.reused == 5, "исправные документы обязаны переиспользоваться"
    assert second.extracted == 1, "провалившийся обязан быть перечитан заново"


def test_a_failure_leaves_no_fingerprint(corpus_with_a_broken_file, tmp_path):
    dataset, _ = corpus_with_a_broken_file
    rp = RunPaths.create(tmp_path / "run")
    extract.run(dataset, rp, llm=None, workers=2)

    fingerprints = json.loads(
        (rp.artifacts / A.FINGERPRINTS).read_text(encoding="utf-8"))
    assert "broken" not in fingerprints


def test_a_failure_leaves_no_empty_text_file(corpus_with_a_broken_file, tmp_path):
    """Пустой .txt неотличим от документа без содержания и молча уезжает
    в классификацию как BACKGROUND."""
    dataset, _ = corpus_with_a_broken_file
    rp = RunPaths.create(tmp_path / "run")
    extract.run(dataset, rp, llm=None, workers=2)
    assert not (rp.artifacts / "01_texts" / "broken.txt").exists()


def test_a_fixed_file_is_picked_up_without_force(corpus_with_a_broken_file, tmp_path):
    """Сквозная проверка сценария: сломалось, починили, заработало."""
    dataset, broken = corpus_with_a_broken_file
    rp = RunPaths.create(tmp_path / "run")
    extract.run(dataset, rp, llm=None, workers=2)

    good = sorted((PUBLIC / "documents").glob("*.pdf"))[7]
    shutil.copy(good, broken)

    report = extract.run(dataset, rp, llm=None, workers=2)
    assert report.failed == []
    recovered = next(d for d in report.docs if d.doc_id == "broken")
    assert recovered.method == "pdfplumber"
    assert (rp.artifacts / "01_texts" / "broken.txt").read_text(encoding="utf-8").strip()


# --------------------------------------------------------------------------- #
# Страницы-картинки внутри текстового документа
#
# РЕАЛЬНАЯ ПОТЕРЯ ДАННЫХ. Решение «скан или текст» принималось для ДОКУМЕНТА
# целиком: символы всех страниц складывались и делились на их число.
# Документ из четырёх плотных текстовых страниц и одной страницы-картинки
# уверенно проходил как текстовый, а картинка молча пропадала.
#
# В публичном наборе так терялись:
#   * досье KYC заёмщиков P2 и P9 — там нарисован раздел о бенефициарном
#     владении, то есть ВЕСЬ список связанных сторон;
#   * аудиторское приложение P4 — там «Примечание 8 — Корректировки EBITDA»,
#     величина, на которую прямо ссылается ковенант 6.1.
#
# Отказ был идеально тихим: документ прочитан, текста много, ошибок нет.
# Просто у P2 «не оказалось» связанных сторон, а у P4 — корректировок.
# --------------------------------------------------------------------------- #

MIXED_PDF = PUBLIC / "documents" / "2ed0b2ee4b57.pdf"  # аудит P4, стр. 4 — картинка
needs_mixed = pytest.mark.skipif(not MIXED_PDF.exists(), reason="нет документа")


def test_image_only_pages_are_detected():
    pages = [("плотный текст " * 40, False),
             ("", True),
             ("3", True),
             ("ещё текст " * 40, False)]
    assert extract.image_only_pages(pages) == [2, 3]


def test_a_short_page_without_an_image_is_not_flagged():
    """Титул и страница с подписью законно короткие."""
    pages = [("За аудитора и от его имени", False), ("текст " * 50, False)]
    assert extract.image_only_pages(pages) == []


def test_a_page_with_an_image_and_real_text_is_not_flagged():
    """Логотип на текстовой странице — не повод её распознавать."""
    pages = [("Полноценный текст страницы, здесь много букв. " * 5, True)]
    assert extract.image_only_pages(pages) == []


@needs_mixed
@needs_poppler
def test_drawn_page_is_recognised_and_put_in_its_place(tmp_path):
    """Порядок страниц несёт смысл: распознанное обязано встать туда,
    откуда оно взято, а не в конец."""
    mock = MockProvider()
    mock.register_rule(lambda r: True,
                       lambda r: {"text": "МАРКЕР-СТРАНИЦЫ-4", "uncertain": []})
    client = LLMClient(cache_dir=tmp_path / "c", provider=mock)

    d = extract.extract_one(MIXED_PDF, client, [])
    text = getattr(d, "_text")

    assert d.method == "pdfplumber+vision"
    assert len(mock.calls) == 1, "распознаваться должна ТОЛЬКО нарисованная страница"
    assert "МАРКЕР-СТРАНИЦЫ-4" in text
    assert text.index("Примечание 8") < text.index("МАРКЕР-СТРАНИЦЫ-4")
    assert text.index("МАРКЕР-СТРАНИЦЫ-4") < text.index("За аудитора и от его имени")


@needs_mixed
@needs_poppler
def test_text_pages_are_not_sent_to_the_model(tmp_path):
    """Текст остальных страниц уже верен. Гонять его через распознавание
    значило бы платить за ухудшение."""
    mock = MockProvider()
    mock.register_rule(lambda r: True, lambda r: {"text": "x", "uncertain": []})
    client = LLMClient(cache_dir=tmp_path / "c", provider=mock)

    d = extract.extract_one(MIXED_PDF, client, [])
    text = getattr(d, "_text")
    assert "Примечание 4 — Основные средства" in text, "текстовая страница потеряна"
    assert "Sary-Arka Assurance LLP" in text


@needs_mixed
def test_an_unreadable_drawn_page_leaves_a_loud_marker():
    """Пустое место в тексте неотличимо от отсутствия сведений — именно
    так и теряются связанные стороны и аудиторские корректировки."""
    d = extract.extract_one(MIXED_PDF, None, [])
    text = getattr(d, "_text")
    assert "НЕ ПРОЧИТАНА" in text
    assert d.needs_review
    assert any("нарисовано" in w for w in d.warnings)


@pytest.mark.slow
def test_the_corpus_has_exactly_the_known_mixed_documents(full_run):
    """Сторожевой тест: если в наборе появится ещё один документ
    с нарисованной страницей, это надо заметить, а не узнать по пустому
    агрегату.

    Ответ берётся из отчёта уже выполненного прогона, а не пересчитывается
    открытием двухсот PDF заново: та же проверка стоила 58 секунд там,
    где данные уже лежат готовыми.
    """
    rep, _ = full_run
    mixed = sorted(d.doc_id for d in rep.docs if d.method.startswith("pdfplumber+"))
    assert mixed == ["2ed0b2ee4b57", "63e162bd710b", "aaf665cbc612"], (
        f"состав смешанных документов изменился: {mixed}"
    )
    for d in rep.docs:
        if d.method.startswith("pdfplumber+"):
            assert any("нарисовано" in w for w in d.warnings), (
                f"{d.doc_id}: страница распознана, но об этом не сказано"
            )


def test_changing_the_extractor_version_invalidates_the_cache(small_corpus, tmp_path,
                                                              monkeypatch):
    """Отпечаток отвечал на вопрос «изменился ли файл» и использовался как
    ответ на вопрос «нужно ли извлекать заново». Это разные вопросы:
    результат зависит и от КОДА, который читает файл.

    Реальный случай: постраничное распознавание научилось дочитывать
    нарисованные страницы, но документы остались «переиспользованными» —
    файлы-то не менялись, и исправление до данных не дошло."""
    rp = RunPaths.create(tmp_path / "run")
    first = extract.run(small_corpus, rp, llm=None, workers=4)
    assert first.extracted == 8

    cached = extract.run(small_corpus, rp, llm=None, workers=4)
    assert cached.reused == 8

    monkeypatch.setattr(extract, "EXTRACTOR_VERSION", "3-нечто-новое")
    after_change = extract.run(small_corpus, rp, llm=None, workers=4)
    assert after_change.extracted == 8, "правка логики не дошла до данных"
    assert after_change.reused == 0
