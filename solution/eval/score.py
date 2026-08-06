#!/usr/bin/env python3
"""Скорер (шаг 1 плана). Воспроизводит официальную формулу из CASE.ru.md, раздел 4.

Работает ТОЛЬКО на публичном датасете. В боевом окне ключа не будет —
там работает pipeline/validate.py.

Формула на одну ячейку:
    status            0.50  бинарно, точное совпадение строки
    actual            0.30 × max(0, 1 − e/0.05),  e = |ваше − ключ| / |ключ|
    evidence_txn_id   0.20  если в ключе строка — точное совпадение;
                            если в ключе null — убывает вместе с actual
                            по той же шкале

Неверный status обнуляет ВСЮ ячейку. Пропущенная, переименованная или
добавленная ячейка — 0. Битый JSON — затронутые ячейки неоцениваемы, то есть 0.

Использование:
    python eval/score.py --submission out/submission.json
    python eval/score.py --submission out/submission.json --verbose
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

W_STATUS = 0.50
W_ACTUAL = 0.30
W_EVIDENCE = 0.20
TOLERANCE = 0.05  # относительная погрешность, при которой actual даёт ноль

VALID_STATUSES = {"COMPLIANT", "BREACH"}


def actual_scale(submitted, key) -> tuple[float, str]:
    """Множитель шкалы точности [0, 1] и человекочитаемая причина."""
    if submitted is None:
        return 0.0, "actual отсутствует"
    if isinstance(submitted, bool) or not isinstance(submitted, (int, float)):
        return 0.0, f"actual не число: {submitted!r}"
    if not math.isfinite(submitted):
        return 0.0, f"actual не конечное число: {submitted!r}"
    if key is None:
        return 0.0, "в ключе нет actual"
    if key == 0:
        # Деление на ноль. В наборе не встречается, но защищаемся явно.
        return (1.0, "точно") if submitted == 0 else (0.0, "ключ = 0, ответ ≠ 0")
    e = abs(submitted - key) / abs(key)
    if e >= TOLERANCE:
        return 0.0, f"отклонение {e:.1%} ≥ {TOLERANCE:.0%}"
    scale = max(0.0, 1.0 - e / TOLERANCE)
    return scale, ("точно" if e == 0 else f"отклонение {e:.2%}")


@dataclass
class CellScore:
    scenario: str
    point: str
    points: float = 0.0
    status_ok: bool = False
    status_pts: float = 0.0
    actual_pts: float = 0.0
    evidence_pts: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def lost(self) -> float:
        return 1.0 - self.points


def score_cell(scenario: str, point: str, got: dict | None, key: dict) -> CellScore:
    cs = CellScore(scenario=scenario, point=point)

    if got is None:
        cs.notes.append("ячейка отсутствует в submission")
        return cs
    if not isinstance(got, dict):
        cs.notes.append(f"ячейка не объект: {type(got).__name__}")
        return cs

    got_status = got.get("status")
    key_status = key.get("status")

    if got_status not in VALID_STATUSES:
        cs.notes.append(f"status невалиден: {got_status!r}")
        return cs
    if got_status != key_status:
        cs.notes.append(f"status неверен: {got_status} вместо {key_status} → вся ячейка 0")
        return cs

    cs.status_ok = True
    cs.status_pts = W_STATUS

    scale, why = actual_scale(got.get("actual"), key.get("actual"))
    cs.actual_pts = W_ACTUAL * scale
    if scale < 1.0:
        cs.notes.append(f"actual: {why} (ваше {got.get('actual')!r}, ключ {key.get('actual')!r})")

    key_ev = key.get("evidence_txn_id")
    if key_ev is None:
        # Ключ не требует доказательства: эти 0.20 зарабатываются точностью actual.
        cs.evidence_pts = W_EVIDENCE * scale
        if scale < 1.0:
            cs.notes.append("evidence: ключ null → баллы убывают вместе с actual")
    else:
        got_ev = got.get("evidence_txn_id")
        if got_ev == key_ev:
            cs.evidence_pts = W_EVIDENCE
        else:
            cs.notes.append(f"evidence неверен: {got_ev!r} вместо {key_ev!r}")

    cs.points = cs.status_pts + cs.actual_pts + cs.evidence_pts
    return cs


@dataclass
class Report:
    cells: list[CellScore]
    structural: list[str] = field(default_factory=list)

    @property
    def total(self) -> float:
        """Невзвешенное среднее по ячейкам.

        ВНИМАНИЕ: организаторы взвешивают ячейки по сложности, а веса нам
        не выданы. Это число — ориентир, а не предсказание итогового балла.
        """
        return sum(c.points for c in self.cells) / len(self.cells) if self.cells else 0.0

    def by_scenario(self) -> dict[str, float]:
        out: dict[str, list[float]] = {}
        for c in self.cells:
            out.setdefault(c.scenario, []).append(c.points)
        return {k: sum(v) / len(v) for k, v in out.items()}

    def losses(self) -> list[CellScore]:
        return sorted((c for c in self.cells if c.points < 1.0), key=lambda c: c.points)

    def loss_attribution(self) -> dict[str, float]:
        """Куда именно утекли баллы. Задаёт приоритет разбора: чинить надо
        то, что стоит дороже, а не то, что чинится легче."""
        n = len(self.cells) or 1
        return {
            "status": sum(W_STATUS - c.status_pts for c in self.cells) / n,
            "actual": sum(W_ACTUAL - c.actual_pts for c in self.cells) / n,
            "evidence": sum(W_EVIDENCE - c.evidence_pts for c in self.cells) / n,
        }


def score(submission: dict, key: dict) -> Report:
    scenarios = key.get("scenarios", {})
    answers = submission.get("answers", {})
    if not isinstance(answers, dict):
        answers = {}

    cells: list[CellScore] = []
    structural: list[str] = []

    extra_scenarios = set(answers) - set(scenarios)
    if extra_scenarios:
        structural.append(f"лишние сценарии в submission: {sorted(extra_scenarios)}")

    for scenario, sdata in scenarios.items():
        key_cells = sdata.get("covenants", {})
        got_cells = answers.get(scenario)
        if not isinstance(got_cells, dict):
            got_cells = {}
            structural.append(f"{scenario}: сценарий отсутствует в submission")
        extra_points = set(got_cells) - set(key_cells)
        if extra_points:
            structural.append(f"{scenario}: лишние пункты {sorted(extra_points)}")
        for point, kcell in key_cells.items():
            cells.append(score_cell(scenario, point, got_cells.get(point), kcell))

    return Report(cells=cells, structural=structural)


def load_submission(path: Path) -> tuple[dict, list[str]]:
    """Битый JSON не роняет скорер: все ячейки становятся неоцениваемыми."""
    try:
        return json.loads(path.read_text(encoding="utf-8")), []
    except json.JSONDecodeError as exc:
        return {}, [f"submission — битый JSON ({exc}); все ячейки неоцениваемы"]
    except OSError as exc:
        return {}, [f"submission не читается: {exc}"]


def main(argv: list[str] | None = None) -> int:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="Скорер Halyk AI Challenge")
    ap.add_argument("--submission", required=True, type=Path)
    ap.add_argument("--key", type=Path, default=here / "ground_truth.json")
    ap.add_argument("--verbose", "-v", action="store_true", help="показать все потерянные ячейки")
    ap.add_argument("--json", action="store_true", help="машиночитаемый вывод")
    args = ap.parse_args(argv)

    submission, problems = load_submission(args.submission)
    key = json.loads(args.key.read_text(encoding="utf-8"))
    rep = score(submission, key)
    rep.structural = problems + rep.structural

    if args.json:
        print(json.dumps(
            {
                "total": rep.total,
                "by_scenario": rep.by_scenario(),
                "structural": rep.structural,
                "cells": [
                    {"scenario": c.scenario, "point": c.point, "points": round(c.points, 4),
                     "status_ok": c.status_ok, "notes": c.notes}
                    for c in rep.cells
                ],
            },
            ensure_ascii=False, indent=2,
        ))
        return 0

    print(f"ИТОГО (невзвешенно): {rep.total:.4f}  по {len(rep.cells)} ячейкам")
    print("  веса по сложности организаторами не выданы — это ориентир, не прогноз\n")

    if rep.structural:
        print("Структурные проблемы:")
        for p in rep.structural:
            print(f"  ! {p}")
        print()

    attr = rep.loss_attribution()
    if sum(attr.values()) > 1e-9:
        print("Куда утекли баллы (доля от максимума 1.000):")
        for k in ("status", "actual", "evidence"):
            print(f"  {k:9} −{attr[k]:.4f}")
        print()

    per = rep.by_scenario()
    print("По сценариям:")
    for s in sorted(per, key=lambda x: per[x]):
        bar = "█" * int(round(per[s] * 20))
        print(f"  {s:5} {per[s]:.3f}  {bar}")

    losses = rep.losses()
    if losses:
        shown = losses if args.verbose else losses[:10]
        print(f"\nПотери ({len(losses)} ячеек, показано {len(shown)}):")
        for c in shown:
            print(f"  {c.scenario}/{c.point}  {c.points:.3f}  (потеряно {c.lost:.3f})")
            for n in c.notes:
                print(f"      – {n}")
    else:
        print("\nПотерь нет.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
