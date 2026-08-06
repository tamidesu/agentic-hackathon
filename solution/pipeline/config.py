"""Обнаружение датасета и путей запуска.

Принцип: НИ ОДНО имя файла из публичного датасета не зашито в код.
Всё находится по структурным признакам, потому что приватный датасет
почти наверняка назовёт файлы иначе (master_ledger_2026.csv и т.п.).
"""
from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

# Обязательные колонки реестра транзакций. Ищем CSV по ним, а не по имени файла.
LEDGER_REQUIRED_COLUMNS = {"txn_id", "account_id", "amount"}

# Файл с ответами. Пайплайн не имеет права его читать — см. guard ниже.
GROUND_TRUTH_MARKER_KEYS = {"scenarios", "seed"}


class DatasetError(RuntimeError):
    """Датасет не удалось разобрать. Всегда с указанием, что именно не найдено."""


@dataclass(frozen=True)
class DatasetPaths:
    """Разобранный датасет.

    Поля для ground_truth здесь НЕТ намеренно: пайплайн не должен иметь
    к нему доступа даже случайно. Скорер получает его отдельным путём.
    """

    root: Path
    documents_dir: Path
    ledger_csv: Path
    template_json: Path

    def describe(self) -> dict:
        n_docs = sum(1 for _ in self.documents_dir.iterdir() if _.is_file())
        return {
            "root": str(self.root),
            "documents_dir": str(self.documents_dir),
            "documents_files": n_docs,
            "ledger_csv": self.ledger_csv.name,
            "template_json": self.template_json.name,
        }


@dataclass(frozen=True)
class RunPaths:
    """Рабочая область прогона. Всегда вне датасета — датасет только читаем."""

    root: Path
    artifacts: Path
    cache: Path
    logs: Path

    @classmethod
    def create(cls, root: Path) -> "RunPaths":
        root = Path(root).resolve()
        paths = cls(
            root=root,
            artifacts=root / "artifacts",
            cache=root / "cache",
            logs=root / "logs",
        )
        for p in (paths.artifacts, paths.cache, paths.logs):
            p.mkdir(parents=True, exist_ok=True)
        return paths


def _looks_like_ledger(path: Path) -> bool:
    try:
        with path.open(newline="", encoding="utf-8-sig") as fh:
            header = next(csv.reader(fh), [])
    except (OSError, UnicodeDecodeError, StopIteration):
        return False
    return LEDGER_REQUIRED_COLUMNS.issubset({h.strip() for h in header})


def _looks_like_template(path: Path) -> bool:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    if not isinstance(data, dict) or "answers" not in data:
        return False
    answers = data["answers"]
    if not isinstance(answers, dict) or not answers:
        return False
    # Ячейки шаблона пустые: значения None. Так отличаем шаблон от чужого сабмита.
    for cells in answers.values():
        if not isinstance(cells, dict):
            return False
        for cell in cells.values():
            if not isinstance(cell, dict) or "status" not in cell:
                return False
    return True


def _looks_like_ground_truth(path: Path) -> bool:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    return isinstance(data, dict) and GROUND_TRUTH_MARKER_KEYS.issubset(data.keys())


def _find_documents_dir(root: Path, min_files: int = 5) -> Path:
    """Каталог с документами — тот, где больше всего файлов-документов."""
    candidates: list[tuple[int, Path]] = []
    for d in [root, *(p for p in root.rglob("*") if p.is_dir())]:
        n = sum(
            1
            for f in d.iterdir()
            if f.is_file() and f.suffix.lower() in {".pdf", ".txt", ".csv", ".png", ".jpg", ".tif", ".tiff"}
        )
        if n >= min_files:
            candidates.append((n, d))
    if not candidates:
        raise DatasetError(
            f"Не найден каталог с документами внутри {root} "
            f"(искал каталог с >={min_files} файлами документов)"
        )
    candidates.sort(key=lambda x: (-x[0], len(str(x[1]))))
    return candidates[0][1]


def discover_dataset(root: str | Path) -> DatasetPaths:
    """Разобрать произвольный каталог датасета по структурным признакам."""
    root = Path(root).resolve()
    if not root.is_dir():
        raise DatasetError(f"Каталог датасета не существует: {root}")

    ledgers = [p for p in root.rglob("*.csv") if _looks_like_ledger(p)]
    if not ledgers:
        raise DatasetError(
            f"Не найден реестр транзакций в {root}. "
            f"Ожидался CSV с колонками {sorted(LEDGER_REQUIRED_COLUMNS)}"
        )
    if len(ledgers) > 1:
        # Берём самый большой — но громко сообщаем, это аномалия.
        ledgers.sort(key=lambda p: -p.stat().st_size)
        log.warning(
            "Найдено несколько файлов-реестров: %s. Взят самый большой: %s",
            [p.name for p in ledgers],
            ledgers[0].name,
        )

    templates = [p for p in root.rglob("*.json") if _looks_like_template(p)]
    if not templates:
        raise DatasetError(
            f"Не найден submission-шаблон в {root}. "
            f"Ожидался JSON с ключом 'answers' и ячейками status/actual/evidence_txn_id"
        )
    if len(templates) > 1:
        raise DatasetError(
            f"Найдено несколько кандидатов на шаблон: {[p.name for p in templates]}. "
            f"Уточните вручную через --template"
        )

    documents_dir = _find_documents_dir(root)

    leaked = [p for p in root.rglob("*.json") if _looks_like_ground_truth(p)]
    if leaked:
        log.warning(
            "В датасете присутствует файл с ответами: %s. "
            "Пайплайн его НЕ читает; для оценки используйте eval/score.py отдельно.",
            [p.name for p in leaked],
        )

    return DatasetPaths(
        root=root,
        documents_dir=documents_dir,
        ledger_csv=ledgers[0],
        template_json=templates[0],
    )
