"""Шаг 2: извлечение текста из документов, включая сканы.

Вход:  <dataset>/documents/
Выход: <run>/artifacts/01_texts/{doc_id}.txt
       <run>/artifacts/01_extract_report.json

Стратегия для каждого файла:
  1. текстовый слой через pdfplumber — покрывает подавляющее большинство;
  2. если текста мало, а изображения есть — это скан, включается распознавание;
  3. распознавание: сначала LLM-зрение (не требует языковых пакетов ОС
     и работает на любом языке), при отсутствии клиента — tesseract;
  4. если ничего не вышло — документ помечается FAILED и попадает в отчёт.

Почему зрение приоритетнее tesseract. В сборке этого окружения установлен
только eng.traineddata, и `tesseract -l rus` молча распознаёт кириллицу
латиницей, выдавая правдоподобный мусор («Cuër ACC-7806»). Тихая деградация
хуже отказа: она проходит дальше по пайплайну и портит расчёт. Поэтому
языковые пакеты проверяются заранее, а основной путь для сканов не зависит
от пакетов ОС вообще.
"""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import re
import shutil
import subprocess
import tempfile
import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .config import DatasetPaths, RunPaths
from .llm import LLMClient, LLMRequest
from .schemas import PAGE_TRANSCRIPTION_SCHEMA

log = logging.getLogger(__name__)

# Файлы, которые не являются документами корпуса.
IGNORED_NAMES = {"thumbs.db", ".ds_store", "desktop.ini"}
IGNORED_SUFFIXES = {".db", ".ini", ".lnk"}

TEXT_SUFFIXES = {".txt", ".md", ".csv", ".tsv", ".json"}
PDF_SUFFIXES = {".pdf"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

# Ниже этого числа символов на страницу считаем, что текстового слоя нет.
SCAN_CHARS_PER_PAGE = 120
# tesseract выигрывает от высокого разрешения.
TESSERACT_DPI = 300
# Модель ужимает изображение до ~1568 px по длинной стороне, поэтому A4
# при 150 DPI (1241×1754) — потолок полезного размера. 300 DPI даёт втрое
# больше байт и ни одного лишнего распознанного символа.
VISION_DPI = 150
# Больше этого числа страниц в одном скане не рендерим, но и не молчим.
MAX_RENDER_PAGES = 40
# Порог доли «подозрительных» символов, при котором качество распознавания
# помечается как сомнительное и документ идёт в ручную проверку.
GARBLE_RATIO_WARN = 0.04

PREFERRED_OCR_LANGS = ("rus", "kaz", "eng")


@dataclass
class DocExtract:
    doc_id: str
    source: str
    method: str          # pdfplumber | text | vision | tesseract | failed
    pages: int = 0
    chars: int = 0
    chars_per_page: float = 0.0
    garble_ratio: float = 0.0
    cyrillic_ratio: float = 0.0
    needs_review: bool = False
    warnings: list[str] = field(default_factory=list)
    duration_s: float = 0.0


@dataclass
class ExtractReport:
    docs: list[DocExtract] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    ocr_langs_available: list[str] = field(default_factory=list)
    total_duration_s: float = 0.0
    #: Сколько документов переиспользовано из кэша, а сколько извлечено заново.
    #: Наблюдаемый признак инкрементальности — надёжнее замера времени.
    reused: int = 0
    extracted: int = 0

    @property
    def failed(self) -> list[DocExtract]:
        return [d for d in self.docs if d.method == "failed"]

    @property
    def review(self) -> list[DocExtract]:
        return [d for d in self.docs if d.needs_review]

    def to_dict(self) -> dict:
        return {
            "total_documents": len(self.docs),
            "total_duration_s": round(self.total_duration_s, 2),
            "ocr_langs_available": self.ocr_langs_available,
            "reused": self.reused,
            "extracted": self.extracted,
            "by_method": {
                m: sum(1 for d in self.docs if d.method == m)
                for m in sorted({d.method for d in self.docs})
            },
            "failed": [d.doc_id for d in self.failed],
            "needs_review": [d.doc_id for d in self.review],
            "skipped": self.skipped,
            "documents": [asdict(d) for d in sorted(self.docs, key=lambda d: d.doc_id)],
        }


# --------------------------------------------------------------------------- #
# Диагностика окружения
# --------------------------------------------------------------------------- #


def tesseract_languages() -> list[str]:
    if not shutil.which("tesseract"):
        return []
    try:
        out = subprocess.run(
            ["tesseract", "--list-langs"], capture_output=True, text=True, timeout=20
        )
    except (OSError, subprocess.SubprocessError):
        return []
    lines = (out.stdout or "").strip().splitlines()
    return [ln.strip() for ln in lines[1:] if ln.strip()]


def preflight(llm: LLMClient | None) -> list[str]:
    """Проблемы окружения, которые надо знать ДО прогона, а не после."""
    problems: list[str] = []
    langs = tesseract_languages()
    missing = [l for l in PREFERRED_OCR_LANGS if l not in langs]

    if llm is None:
        if not langs:
            problems.append(
                "Сканы распознать нечем: LLM-клиент не передан и tesseract недоступен. "
                "Передайте клиент или установите tesseract."
            )
        elif missing:
            problems.append(
                f"tesseract без языковых пакетов {missing} (есть: {langs or '—'}). "
                f"Кириллица будет распознана латиницей — молча и неверно. "
                f"Установите: apt-get install tesseract-ocr-rus tesseract-ocr-kaz, "
                f"либо передайте LLM-клиент для распознавания зрением."
            )
    if not shutil.which("pdftoppm"):
        problems.append(
            "pdftoppm не найден (пакет poppler-utils) — сканы отрендерить не получится."
        )
    return problems


# --------------------------------------------------------------------------- #
# Извлечение
# --------------------------------------------------------------------------- #


def _garble_ratio(text: str) -> float:
    """Доля стыков кириллица↔латиница внутри слов.

    Ловит случай, когда распознавание подменяет отдельные буквы: «Cuёт».
    НЕ ловит случай, когда кириллица целиком распознана латиницей
    («Ynpapnenue» вместо «Управление») — там стыков нет вообще.
    Для этого служит корпусная проверка _flag_script_anomalies.
    """
    if not text:
        return 0.0
    mixed = len(re.findall(r"[А-Яа-яЁё][A-Za-z]|[A-Za-z][А-Яа-яЁё]", text))
    return mixed / max(len(text), 1)


def _cyrillic_ratio(text: str) -> float:
    """Доля кириллицы среди букв. Метрика письменности, а не языка."""
    letters = re.findall(r"[^\W\d_]", text, flags=re.UNICODE)
    if not letters:
        return 0.0
    cyr = sum(1 for ch in letters if "Ѐ" <= ch <= "ӿ")
    return cyr / len(letters)


def _flag_script_anomalies(docs: list["DocExtract"], min_chars: int = 500) -> None:
    """Корпусная проверка письменности.

    Сравнивает каждый распознанный документ с медианой по документам,
    у которых есть родной текстовый слой. Если корпус кириллический,
    а распознанный документ — нет, распознавание почти наверняка провалилось.

    Работает без словарей и без привязки к конкретному языку: на казахском
    или английском корпусе медиана сместится сама.
    """
    trusted = [
        d.cyrillic_ratio for d in docs
        if d.method in {"pdfplumber", "text"} and d.chars >= min_chars
    ]
    if len(trusted) < 5:
        return
    trusted.sort()
    median = trusted[len(trusted) // 2]
    if median < 0.2:
        return  # корпус не кириллический — проверка неприменима

    for d in docs:
        if d.method not in {"vision", "tesseract"} or d.chars < min_chars:
            continue
        if d.cyrillic_ratio < median / 3:
            d.warnings.append(
                f"письменность не совпадает с корпусом: кириллицы {d.cyrillic_ratio:.0%} "
                f"против медианных {median:.0%} — распознавание, вероятно, провалилось"
            )
            d.needs_review = True


def _pdf_text(path: Path) -> tuple[str, int, bool]:
    """Возвращает (текст, число страниц, есть ли изображения)."""
    import pdfplumber

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pdfplumber.open(str(path)) as pdf:
            parts, has_images = [], False
            for page in pdf.pages:
                parts.append(page.extract_text() or "")
                if page.images:
                    has_images = True
            return "\n".join(parts), len(pdf.pages), has_images


def _render_pages(
    path: Path, dpi: int, total_pages: int, max_pages: int = MAX_RENDER_PAGES
) -> tuple[list[bytes], list[str]]:
    """Рендер страниц в PNG. Возвращает (изображения, предупреждения)."""
    if not shutil.which("pdftoppm"):
        raise RuntimeError("pdftoppm недоступен (пакет poppler-utils)")
    notes: list[str] = []
    if total_pages > max_pages:
        # Раньше здесь было тихое усечение — прямая потеря данных.
        notes.append(
            f"скан из {total_pages} стр., распознаны первые {max_pages}; "
            f"остальные НЕ прочитаны — поднимите MAX_RENDER_PAGES"
        )
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(
            ["pdftoppm", "-r", str(dpi), "-png", "-l", str(max_pages), str(path), f"{tmp}/p"],
            check=True, capture_output=True, timeout=900,
        )
        return [p.read_bytes() for p in sorted(Path(tmp).glob("p*.png"))], notes


def _tesseract_pages(images: list[bytes], langs: list[str]) -> str:
    lang = "+".join(l for l in PREFERRED_OCR_LANGS if l in langs) or "eng"
    out = []
    for img in images:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as fh:
            fh.write(img)
            tmp = fh.name
        try:
            r = subprocess.run(
                ["tesseract", tmp, "-", "-l", lang],
                capture_output=True, text=True, timeout=300,
            )
            out.append(r.stdout or "")
        finally:
            Path(tmp).unlink(missing_ok=True)
    return "\n".join(out)


VISION_PROMPT = (
    "Это страница отсканированного финансового документа. Перепиши ВЕСЬ видимый "
    "текст дословно, сохраняя порядок и структуру.\n\n"
    "Требования:\n"
    "- Числа, проценты, номера счетов и идентификаторы переноси СИМВОЛ В СИМВОЛ. "
    "Ошибка в цифре меняет смысл документа.\n"
    "- Таблицы передавай построчно, разделяя колонки символом |.\n"
    "- Ничего не переводи, не сокращай и не резюмируй.\n"
    "- Если фрагмент неразборчив, поставь [нрзб] вместо догадки.\n"
    "- В поле uncertain перечисли места, где ты не уверен."
)


def _vision_pages(images: list[bytes], llm: LLMClient, doc_id: str) -> tuple[str, list[str]]:
    texts: list[str] = []
    notes: list[str] = []
    for i, img in enumerate(images, 1):
        req = LLMRequest(
            prompt=VISION_PROMPT,
            schema=PAGE_TRANSCRIPTION_SCHEMA,
            images=((("image/png"), base64.b64encode(img).decode("ascii")),),
            prompt_version="transcribe-v1",
            max_tokens=8000,
        )
        res = llm.extract(req)
        texts.append(res.data.get("text", ""))
        for u in res.data.get("uncertain") or []:
            notes.append(f"{doc_id} стр.{i}: {u}")
    return "\n".join(texts), notes


def extract_one(path: Path, llm: LLMClient | None, langs: list[str]) -> DocExtract:
    t0 = time.time()
    doc_id = path.stem
    suffix = path.suffix.lower()
    d = DocExtract(doc_id=doc_id, source=path.name, method="failed")

    try:
        if suffix in TEXT_SUFFIXES:
            text = path.read_text(encoding="utf-8", errors="replace")
            d.method, d.pages = "text", 1

        elif suffix in PDF_SUFFIXES:
            text, n_pages, has_images = _pdf_text(path)
            d.pages = n_pages
            per_page = len(text.strip()) / max(n_pages, 1)
            if per_page >= SCAN_CHARS_PER_PAGE:
                d.method = "pdfplumber"
            else:
                d.warnings.append(
                    f"текстовый слой отсутствует или беден "
                    f"({per_page:.0f} симв./стр., изображения: {has_images}) — распознаю"
                )
                if llm is not None:
                    images, notes = _render_pages(path, VISION_DPI, n_pages)
                    d.warnings.extend(notes)
                    text, vnotes = _vision_pages(images, llm, doc_id)
                    d.method = "vision"
                    d.warnings.extend(vnotes)
                elif langs:
                    images, notes = _render_pages(path, TESSERACT_DPI, n_pages)
                    d.warnings.extend(notes)
                    text = _tesseract_pages(images, langs)
                    d.method = "tesseract"
                    missing = [l for l in PREFERRED_OCR_LANGS if l not in langs]
                    if missing:
                        d.warnings.append(
                            f"tesseract без пакетов {missing} — распознавание ненадёжно"
                        )
                        d.needs_review = True
                else:
                    d.warnings.append("распознать нечем: нет ни LLM-клиента, ни tesseract")
                    return _finish(d, "", t0)

        elif suffix in IMAGE_SUFFIXES:
            data = path.read_bytes()
            if llm is not None:
                text, notes = _vision_pages([data], llm, doc_id)
                d.method, d.pages = "vision", 1
                d.warnings.extend(notes)
            elif langs:
                text = _tesseract_pages([data], langs)
                d.method, d.pages = "tesseract", 1
            else:
                d.warnings.append("изображение, но распознать нечем")
                return _finish(d, "", t0)
        else:
            d.warnings.append(f"неизвестное расширение {suffix}")
            return _finish(d, "", t0)

    except Exception as exc:  # noqa: BLE001 — падение одного файла не роняет прогон
        d.warnings.append(f"{type(exc).__name__}: {exc}")
        log.exception("Не удалось извлечь %s", path.name)
        return _finish(d, "", t0)

    return _finish(d, text, t0)


def _finish(d: DocExtract, text: str, t0: float) -> DocExtract:
    d.chars = len(text)
    d.chars_per_page = d.chars / max(d.pages, 1)
    d.garble_ratio = _garble_ratio(text)
    d.cyrillic_ratio = _cyrillic_ratio(text)
    d.duration_s = round(time.time() - t0, 2)

    if d.method != "failed" and d.chars < 50:
        d.warnings.append(f"подозрительно мало текста: {d.chars} симв.")
        d.needs_review = True
    if d.garble_ratio > GARBLE_RATIO_WARN:
        d.warnings.append(
            f"высокая доля смешанных кириллица/латиница ({d.garble_ratio:.1%}) — "
            f"вероятно неверное распознавание"
        )
        d.needs_review = True
    d._text = text  # type: ignore[attr-defined]
    return d


def _source_fingerprint(path: Path) -> str:
    st = path.stat()
    return hashlib.sha256(f"{path.name}|{st.st_size}|{int(st.st_mtime)}".encode()).hexdigest()[:16]


def run(
    dataset: DatasetPaths,
    paths: RunPaths,
    llm: LLMClient | None = None,
    workers: int = 8,
    force: bool = False,
) -> ExtractReport:
    t0 = time.time()
    out_dir = paths.artifacts / "01_texts"
    out_dir.mkdir(parents=True, exist_ok=True)
    fingerprints_path = paths.artifacts / "01_fingerprints.json"
    fingerprints: dict[str, str] = {}
    if fingerprints_path.exists() and not force:
        try:
            fingerprints = json.loads(fingerprints_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            fingerprints = {}

    langs = tesseract_languages()
    report = ExtractReport(ocr_langs_available=langs)

    for problem in preflight(llm):
        log.warning("PREFLIGHT: %s", problem)

    files: list[Path] = []
    for p in sorted(dataset.documents_dir.iterdir()):
        if not p.is_file():
            continue
        if p.name.lower() in IGNORED_NAMES or p.suffix.lower() in IGNORED_SUFFIXES:
            report.skipped.append(p.name)
            continue
        files.append(p)

    # Инкрементальность: неизменившиеся файлы с готовым артефактом не трогаем.
    todo, reused = [], []
    for p in files:
        fp = _source_fingerprint(p)
        artifact = out_dir / f"{p.stem}.txt"
        if not force and artifact.exists() and fingerprints.get(p.stem) == fp:
            reused.append(p)
        else:
            todo.append((p, fp))

    report.reused = len(reused)
    report.extracted = len(todo)
    if reused:
        log.info("Переиспользую %d ранее извлечённых документов", len(reused))
        meta_path = paths.artifacts / "01_extract_report.json"
        prev = {}
        if meta_path.exists():
            try:
                prev = {
                    d["doc_id"]: d
                    for d in json.loads(meta_path.read_text(encoding="utf-8")).get("documents", [])
                }
            except (json.JSONDecodeError, KeyError):
                prev = {}
        for p in reused:
            if p.stem in prev:
                report.docs.append(DocExtract(**prev[p.stem]))
            else:
                text = (out_dir / f"{p.stem}.txt").read_text(encoding="utf-8")
                report.docs.append(_finish(
                    DocExtract(doc_id=p.stem, source=p.name, method="cached", pages=1), text, time.time()
                ))

    results = LLMClient.map_parallel(
        lambda item: extract_one(item[0], llm, langs), todo, workers=workers
    )
    for (p, fp), res in zip(todo, results):
        if isinstance(res, Exception):
            report.docs.append(DocExtract(
                doc_id=p.stem, source=p.name, method="failed",
                warnings=[f"{type(res).__name__}: {res}"],
            ))
            continue
        (out_dir / f"{res.doc_id}.txt").write_text(
            getattr(res, "_text", ""), encoding="utf-8"
        )
        fingerprints[res.doc_id] = fp
        report.docs.append(res)

    _flag_script_anomalies(report.docs)
    report.total_duration_s = time.time() - t0
    fingerprints_path.write_text(json.dumps(fingerprints, indent=2), encoding="utf-8")
    (paths.artifacts / "01_extract_report.json").write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report
