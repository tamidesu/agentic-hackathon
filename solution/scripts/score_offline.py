#!/usr/bin/env python3
"""Оценка решения без единого вызова модели.

Шаги 11–14 — чистая арифметика над результатами дорогих шагов (5, 7, 8, 10).
Скрипт копирует замороженный снимок fixtures/baseline/artifacts/ в рабочий
каталог, гоняет apply → compute → evidence → assemble и считает балл по
eval/ground_truth.json. Ни одного вызова API.

    cd solution
    python scripts/score_offline.py

ЗАЧЕМ КОПИЯ, А НЕ РАБОТА ПОВЕРХ СНИМКА. Снимок — эталон для сравнения;
шаги 11–14 перезаписывают артефакты, и прогон прямо в fixtures/ уничтожил
бы возможность обнаружить регрессию. fixtures/ только читается.

ПРОВЕРКА ВХОДА ОБЯЗАТЕЛЬНА. Неполный снимок дал бы заниженную оценку по
причине, не связанной с качеством решения, — и правка выглядела бы
регрессией, которой нет.

ВОССТАНОВЛЕННЫЙ ФАЙЛ. Оригинал этого скрипта не был закоммичен (снимок
в коммите 40e016b есть, скрипта нет). Файл восстановлен по описанию
в сообщении коммита; эталонные числа воспроизведены: 70.4% (25.35 из 36),
полностью верных 22, верный status 28.
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

SOLUTION = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOLUTION))

# Консоль Windows по умолчанию cp1251 и роняет вывод на первом же «—».
for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

from pipeline import apply, artifacts as A, assemble, compute, disclosed, evidence  # noqa: E402
from pipeline.config import RunPaths, discover_dataset  # noqa: E402

SNAPSHOT = SOLUTION / "fixtures" / "baseline" / "artifacts"

#: Без этих артефактов шаги 11–14 посчитают мусор молча — поэтому отказ.
REQUIRED = [
    A.COVENANTS,          # спецификации ковенантов (шаг 5)
    A.AUDIT_ADJUSTMENTS,  # корректировки аудитора (шаг 7)
    A.RELATED_PARTIES,    # связанные стороны (шаг 8)
    A.LEDGER_CLEAN,       # очищенный реестр (шаг 9)
    A.TXN_CATEGORIES,     # категории операций (шаг 10)
]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Оценка на снимке, без вызовов модели")
    ap.add_argument("--dataset", type=Path, default=SOLUTION.parent / "agentic-bank-public")
    ap.add_argument("--key", type=Path, default=SOLUTION / "eval" / "ground_truth.json")
    ap.add_argument("--run", type=Path, default=SOLUTION / "runs" / "offline")
    ap.add_argument("--verbose", "-v", action="store_true", help="показать все потерянные ячейки")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    missing = [name for name in REQUIRED if not (SNAPSHOT / name).exists()]
    if missing:
        print(f"ПРОВАЛ: снимок {SNAPSHOT} неполон, отсутствуют: {missing}")
        print("Оценка на неполном снимке была бы занижена не по вине решения — отказ.")
        return 2

    # Свежая копия снимка: прошлый прогон не должен подмешиваться в этот.
    # Чистятся СОДЕРЖИМОЕ каталога, а не сам каталог: OneDrive держит
    # открытый хэндл на каталог, и rmtree падает на rmdir с «отказано
    # в доступе». Устаревший ФАЙЛ исказил бы оценку — его не удалить
    # нельзя; пустой каталог не мешает никому.
    run_root = args.run.resolve()
    stale = run_root / "artifacts"
    if stale.exists():
        for p in sorted(stale.rglob("*"), key=lambda x: len(x.parts), reverse=True):
            try:
                p.unlink() if p.is_file() else p.rmdir()
            except OSError as exc:
                if p.is_file():
                    print(f"ПРОВАЛ: не удалить устаревший артефакт {p}: {exc}")
                    return 2
    paths = RunPaths.create(run_root)
    shutil.copytree(SNAPSHOT, paths.artifacts, dirs_exist_ok=True)

    dataset = discover_dataset(args.dataset)

    # ---------------- шаги 7б, 11–14 ---------------- #
    # Раскрытые величины — производные от снимка (корректировки + ковенанты
    # + реестр), поэтому шаг 7б гоняется здесь наравне с расчётом.
    dreport = disclosed.run(paths)
    n_disclosed = sum(len(v) for v in dreport.values.values())
    print(f"шаг 7б: раскрытых величин {n_disclosed} "
          f"у {len(dreport.values)} заёмщиков")

    apreport = apply.run(paths)
    print(f"шаг 11: строк {len(apreport.rows)}, переклассифицировано "
          f"{len(apreport.reclassified)}, исключено {len(apreport.excluded)}, "
          f"связанных сторон {len(apreport.related_tagged)}")

    results = compute.run(paths)
    print(f"шаг 12: ячеек {sum(len(v) for v in results.values())}")

    evidence.run(paths)

    out_path = paths.root / A.SUBMISSION
    assemble.run(dataset, paths, team="tamidesu", contact_email="mogleg2@gmail.com",
                 model="offline-snapshot", out_path=out_path)
    print(f"шаг 14: {out_path}")

    # ---------------- оценка ---------------- #
    sys.path.insert(0, str(SOLUTION / "eval"))
    from score import load_submission, score  # type: ignore

    submission, problems = load_submission(out_path)
    for p in problems:
        print(f"ПРОБЛЕМА: {p}")
    rep = score(submission, json.loads(args.key.read_text(encoding="utf-8")))

    # total — СВОЙСТВО, а не метод (см. run_all.py: скобки уже роняли вывод).
    exact = sum(1 for c in rep.cells if c.points >= 0.999)
    status_ok = sum(1 for c in rep.cells if c.status_ok)
    print(f"\nИТОГО: {rep.total:.1%}  ({rep.total * len(rep.cells):.2f} из {len(rep.cells)})")
    print(f"полностью верных: {exact}   верный status: {status_ok}")

    for s in rep.structural:
        print(f"структура: {s}")

    losses = rep.losses()
    if losses:
        shown = losses if args.verbose else losses[:10]
        print(f"\nПотери ({len(losses)} ячеек, показано {len(shown)}):")
        for c in shown:
            print(f"  {c.scenario}/{c.point}  {c.points:.3f}")
            for n in c.notes:
                print(f"      – {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
