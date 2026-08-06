"""Шаг 14: сборка submission.json.

Вход:  <run>/artifacts/09_results.json, <dataset>/submission_template.json
Выход: <run>/submission.json

Структуру задаёт ШАБЛОН, а не результаты расчёта. Ключи не создаются и не
переименовываются: обходится шаблон, и для каждой его ячейки берётся
посчитанное значение. Ячейка, которой не нашлось результата, заполняется
запасным значением — пустая и неверная стоят одинаково, поэтому пустых
быть не должно.

ЗАПАСНОЕ ЗНАЧЕНИЕ — ОСОЗНАННЫЙ ВЫБОР, А НЕ ЗАГЛУШКА. Если расчёт по ячейке
не удался, статус всё равно нужен: пропуск гарантированно даёт ноль, а
угадывание даёт 0.50 с вероятностью около половины. Такие ячейки помечаются
в отчёте, чтобы разбор шага 15 начался именно с них.
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

from .config import DatasetPaths, RunPaths

log = logging.getLogger(__name__)

VALID_STATUSES = ("COMPLIANT", "BREACH")
#: Чем заполняется ячейка, по которой расчёт не дал результата.
FALLBACK_STATUS = "COMPLIANT"
FALLBACK_ACTUAL = 0.0


@dataclass
class AssemblyReport:
    cells_total: int = 0
    cells_computed: int = 0
    cells_fallback: list[str] = field(default_factory=list)
    evidence_filled: int = 0
    problems: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "cells_total": self.cells_total,
            "cells_computed": self.cells_computed,
            "cells_fallback": self.cells_fallback,
            "evidence_filled": self.evidence_filled,
            "problems": self.problems,
        }


def _clean_actual(value) -> float | None:
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(value):
        return None
    # Всегда положительное, ровно два знака: требование условия задачи.
    return round(abs(float(value)), 2)


def build(
    template: dict,
    results: dict,
    team: str,
    contact_email: str,
    model: str,
) -> tuple[dict, AssemblyReport]:
    report = AssemblyReport()
    answers: dict[str, dict] = {}

    for scenario, cells in template.get("answers", {}).items():
        answers[scenario] = {}
        for point in cells:
            report.cells_total += 1
            where = f"{scenario}/{point}"
            cell = (results.get(scenario) or {}).get(point)

            if not isinstance(cell, dict):
                report.cells_fallback.append(where)
                report.problems.append(f"{where}: результата расчёта нет, взято запасное значение")
                answers[scenario][point] = {
                    "status": FALLBACK_STATUS,
                    "actual": FALLBACK_ACTUAL,
                    "evidence_txn_id": None,
                }
                continue

            status = cell.get("status")
            actual = _clean_actual(cell.get("actual"))

            if status not in VALID_STATUSES:
                report.cells_fallback.append(where)
                report.problems.append(f"{where}: статус {status!r} невалиден, взято запасное")
                status = FALLBACK_STATUS
            if actual is None:
                report.cells_fallback.append(where)
                report.problems.append(
                    f"{where}: actual не вычислен, взят 0.0 — ячейка требует разбора"
                )
                actual = FALLBACK_ACTUAL

            evidence = cell.get("evidence_txn_id")
            if evidence is not None and not isinstance(evidence, str):
                report.problems.append(f"{where}: evidence не строка, заменён на null")
                evidence = None
            if evidence:
                report.evidence_filled += 1

            if where not in report.cells_fallback:
                report.cells_computed += 1
            answers[scenario][point] = {
                "status": status,
                "actual": actual,
                "evidence_txn_id": evidence,
            }

    extra = set(results) - set(template.get("answers", {}))
    if extra:
        report.problems.append(
            f"в результатах есть сценарии, которых нет в шаблоне: {sorted(extra)} — "
            f"они не попадут в ответ"
        )

    submission = {
        "team": team,
        "contact_email": contact_email,
        "model": model,
        "answers": answers,
    }
    return submission, report


def run(
    dataset: DatasetPaths,
    paths: RunPaths,
    team: str,
    contact_email: str,
    model: str,
    out_path: Path | None = None,
) -> tuple[dict, AssemblyReport]:
    template = json.loads(dataset.template_json.read_text(encoding="utf-8"))
    results_path = paths.artifacts / "09_results.json"
    results = (
        json.loads(results_path.read_text(encoding="utf-8"))
        if results_path.exists() else {}
    )
    if not results:
        log.warning("СБОРКА: результатов расчёта нет — весь ответ будет запасным")

    submission, report = build(template, results, team, contact_email, model)

    out_path = out_path or (paths.root / "submission.json")
    out_path.write_text(
        json.dumps(submission, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (paths.artifacts / "10_assembly_report.json").write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    for p in report.problems:
        log.warning("СБОРКА: %s", p)
    log.info(
        "Собрано ячеек: %d (посчитано %d, запасных %d), доказательств %d → %s",
        report.cells_total, report.cells_computed, len(report.cells_fallback),
        report.evidence_filled, out_path,
    )
    return submission, report
