#!/usr/bin/env python3
"""Полный прогон: от документов до submission.json.

Этот скрипт и есть решение. В боевом окне запускается он, и его вывод —
единственный источник сведений о том, что произошло.

    set GEMINI_API_KEY=ключ1,ключ2,ключ3
    python scripts\\run_all.py --dataset ..\\agentic-bank-private

Полезные ключи:
    --score eval/ground_truth.json   оценить результат по ключу (только
                                     на публичном наборе: приватного ключа
                                     у нас не будет)
    --force                          игнорировать кэш и переспросить модель
    --skip-llm                       прогнать только детерминированные шаги
    --run runs/full                  куда класть артефакты

ЧТО ЗДЕСЬ ВАЖНО ВИДЕТЬ

Каждый шаг печатает не только «сделано», но и ТРЕВОГИ: места, где данные
выглядят подозрительно. Тревога это не ошибка — расчёт продолжится, — но
именно по ней в боевом окне решают, куда смотреть в первую очередь.
Молчаливое «всё хорошо» при пустых агрегатах опаснее громкого отказа.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Консоль Windows по умолчанию cp1251 и роняет вывод на первом же «≤».
for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

from pipeline import (  # noqa: E402
    adjustments,
    apply,
    artifacts as A,
    assemble,
    attribute,
    categorize,
    classify,
    compute,
    covenants,
    disclosed,
    entities,
    evidence,
    extract,
    ledger,
    related,
)
from pipeline.config import RunPaths, discover_dataset  # noqa: E402
from pipeline.gemini import (  # noqa: E402
    DEFAULT_GEMINI_MODEL,
    FREE_TIER_WORKERS_PER_KEY,
    verify_model,
)
from pipeline.llm import LLMClient  # noqa: E402


def banner(step: str, title: str) -> None:
    print(f"\n{'─' * 70}\n{step}  {title}\n{'─' * 70}")


def show(label: str, items, limit: int = 12) -> None:
    """Печатает список замечаний, не заваливая экран."""
    items = list(items)
    if not items:
        return
    for item in items[:limit]:
        print(f"   {label}: {item}")
    if len(items) > limit:
        print(f"   {label}: ...и ещё {len(items) - limit}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Полный прогон решения")
    ap.add_argument("--dataset", type=Path, default=Path("../agentic-bank-public"))
    ap.add_argument("--run", type=Path, default=Path("runs/full"))
    ap.add_argument("--model", default=None)
    ap.add_argument("--provider", choices=["gemini", "anthropic"], default="gemini")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--skip-llm", action="store_true",
                    help="только детерминированные шаги, без вызовов модели")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--score", type=Path, default=None,
                    help="ключ для оценки; на приватном наборе его не будет")
    ap.add_argument("--team", default="tamidesu")
    ap.add_argument("--email", default="mogleg2@gmail.com")
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    started = time.time()

    dataset = discover_dataset(args.dataset)
    paths = RunPaths.create(args.run)
    print(f"датасет:   {dataset.root}\nартефакты: {paths.artifacts}")

    llm = None
    model = args.model or (DEFAULT_GEMINI_MODEL if args.provider == "gemini"
                           else "claude-opus-5")
    workers = args.workers

    if not args.skip_llm:
        os.environ["LLM_MODE"] = args.provider
        if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEYS")
                or os.environ.get("GOOGLE_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")):
            print("ПРОВАЛ: ключ API не задан. Либо задайте его, либо --skip-llm")
            return 2
        llm = LLMClient(cache_dir=paths.cache, force=args.force)
        if args.provider == "gemini":
            model, notes = verify_model(llm.provider, model)
            show("модель", notes)
            llm.model = model
        if not workers:
            keys = getattr(llm.provider, "n_keys", 1)
            workers = max(1, keys * FREE_TIER_WORKERS_PER_KEY)
        print(f"модель:    {model}, параллельных веток {workers}")

    # ---------------- шаг 2: тексты ---------------- #
    banner("шаг 2 ", "извлечение текста")
    t = time.time()
    total = len(list(dataset.documents_dir.iterdir()))
    interactive = sys.stdout.isatty()

    def progress(done: int, all_: int, name: str) -> None:
        if interactive:
            print(f"\r   {done:4}/{all_} {name[:40]:<40}", end="", flush=True)
        elif done == all_ or done % 50 == 0:
            print(f"   {done}/{all_}", flush=True)

    rep = extract.run(dataset, paths, llm=llm, workers=8, force=args.force,
                      progress=progress)
    if interactive:
        print()
    print(f"   документов {total}, извлечено {rep.extracted}, "
          f"переиспользовано {rep.reused}, за {time.time() - t:.1f}с")
    show("ПРОВАЛ", (f"{d.doc_id}: {'; '.join(d.warnings)[:150]}" for d in rep.failed))
    show("на разбор", (d.doc_id for d in rep.review))

    # ---------------- шаг 3: типы ---------------- #
    banner("шаг 3 ", "классификация документов")
    creport = classify.run(paths, llm=llm, model=model if llm else None)
    print(f"   {creport.counts()}")
    show("ТРЕВОГА", creport.alarms())

    # ---------------- шаг 4: привязка ---------------- #
    banner("шаг 4 ", "привязка документов к заёмщикам")
    docs, areport = attribute.run(dataset, paths)
    print(f"   по счёту {areport.by_account}, по названию {areport.by_name}, "
          f"сирот {len(areport.orphans)}, период {areport.reporting_period}")
    show("ПРОБЛЕМА", areport.problems)

    # ---------------- шаг 9: реестр ---------------- #
    banner("шаг 9 ", "очистка реестра")
    txns, lreport = ledger.run(dataset, paths)
    print(f"   строк {lreport.kept_rows} из {lreport.total_rows}, "
          f"шум {lreport.dropped_noise}, курсы {lreport.fx_rates}")
    show("восстановлено", lreport.recovered_amounts)
    show("ПРОБЛЕМА", lreport.problems)

    # ---------------- шаг 9б: граф связей ---------------- #
    banner("шаг 9б", "граф связей сущностей")
    graphs = entities.run(paths)
    print(f"   заёмщиков {len(graphs)}")

    if args.skip_llm:
        print("\n--skip-llm: шаги 5, 7, 8, 10 пропущены, расчёт невозможен")
        return 0

    # ---------------- шаг 5: ковенанты ---------------- #
    banner("шаг 5 ", "ковенанты из договоров")
    t = time.time()
    creport5 = covenants.run(dataset, paths, llm, model=model, workers=workers)
    n_cov = sum(len(s.covenants) for s in creport5.scenarios)
    print(f"   ковенантов {n_cov} у {len(creport5.scenarios)} заёмщиков, "
          f"за {time.time() - t:.1f}с")
    show("ТРЕВОГА", creport5.alarms())
    show("ПРОБЛЕМА", creport5.problems)

    # ---------------- шаг 8: связанные стороны ---------------- #
    banner("шаг 8 ", "связанные стороны из KYC")
    rreport = related.run(paths, llm, model=model, workers=workers)
    thresholds = {s.scenario_id: s.threshold_pct for s in rreport.scenarios}
    print(f"   пороги участия: {thresholds}")
    print(f"   связанных сторон: "
          f"{ {s.scenario_id: len(s.related_names()) for s in rreport.scenarios} }")
    show("ТРЕВОГА", rreport.alarms())
    show("ПРОБЛЕМА", rreport.problems)

    # ---------------- шаг 7: корректировки ---------------- #
    banner("шаг 7 ", "аудиторские корректировки")
    areport7 = adjustments.run(paths, llm, model=model, workers=workers)
    print(f"   применяемых примечаний: "
          f"{ {s.scenario_id: len(s.applied()) for s in areport7.scenarios} }")
    show("не применено", areport7.scenarios and
         [note for s in areport7.scenarios for note in s.remarks])
    show("ТРЕВОГА", areport7.alarms())
    show("ПРОБЛЕМА", areport7.problems)

    # ---------------- шаг 7б: раскрытые величины ---------------- #
    banner("шаг 7б", "раскрытые аудитором величины (DISCLOSED)")
    dreport = disclosed.run(paths)
    print(f"   величин {sum(len(v) for v in dreport.values.values())} "
          f"у {len(dreport.values)} заёмщиков")
    show("учтено", dreport.remarks)
    show("ТРЕВОГА", dreport.alarms())

    # ---------------- шаг 10: категории ---------------- #
    banner("шаг 10", "категоризация транзакций")
    t = time.time()
    rows = [
        {"txn_id": x.txn_id, "description": x.description, "counterparty": x.counterparty,
         "amount": x.amount_usd if x.amount_usd is not None else x.amount,
         "currency": x.currency}
        for x in txns
    ]
    catreport = categorize.run(rows, paths, llm, model=model, workers=workers)
    print(f"   размечено {len(catreport.items)} из {len(rows)} за {time.time() - t:.1f}с")
    print(f"   распределение: {catreport.counts()}")
    show("ТРЕВОГА", catreport.alarms(expected=len(rows)))
    show("низкая уверенность", catreport.low_confidence(), limit=6)

    # ---------------- шаг 11: сборка реестра ---------------- #
    banner("шаг 11", "применение корректировок")
    apreport = apply.run(paths)
    print(f"   строк {len(apreport.rows)}, переклассифицировано "
          f"{len(apreport.reclassified)}, исключено {len(apreport.excluded)}, "
          f"связанных сторон {len(apreport.related_tagged)}")
    show("не применено", apreport.skipped_notes)
    show("ТРЕВОГА", apreport.alarms())

    # ---------------- шаги 12–14: расчёт и сборка ---------------- #
    banner("шаг 12", "расчёт показателей")
    results = compute.run(paths)
    print(f"   ячеек {sum(len(v) for v in results.values())}")

    banner("шаг 13", "поиск доказательств")
    evidence.run(paths)

    banner("шаг 14", "сборка submission")
    out_path = paths.root / A.SUBMISSION
    assemble.run(dataset, paths, team=args.team, contact_email=args.email,
                 model=model, out_path=out_path)
    print(f"   {out_path}")

    # ---------------- оценка ---------------- #
    if args.score and args.score.exists():
        banner("оценка", "сверка с ключом")
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
        from score import load_submission, score  # type: ignore

        submission, problems = load_submission(out_path)
        show("ПРОБЛЕМА", problems)
        report = score(submission, json.loads(args.score.read_text(encoding="utf-8")))

        # total — СВОЙСТВО, а не метод: скобки давали
        # «'float' object is not callable» уже после всех оплаченных вызовов.
        exact = sum(1 for c in report.cells if c.points >= 0.999)
        status_ok = sum(1 for c in report.cells if c.status_ok)
        print(f"   ИТОГО: {report.total:.1%}  "
              f"({report.total * len(report.cells):.2f} из {len(report.cells)})")
        print(f"   полностью верных ячеек: {exact} из {len(report.cells)}; "
              f"верный status: {status_ok}")
        print(f"   по заёмщикам: "
              f"{ {k: round(v, 2) for k, v in report.by_scenario().items()} }")
        print(f"   потери по полям: "
              f"{ {k: round(v, 2) for k, v in report.loss_attribution().items()} }")
        show("структура", report.structural)
        # Худшие ячейки — с них начинается разбор: они показывают, ГДЕ
        # именно теряются очки, а не только сколько.
        # losses() уже отсортирован по возрастанию баллов: сверху самое
        # дорогое. Ни points, ни lost НЕ методы — скобки роняли вывод
        # уже после всех оплаченных вызовов, второй раз за вечер.
        for cell in report.losses()[:10]:
            print(f"   {cell.points:.2f}  {cell.scenario} {cell.point}: "
                  f"{'; '.join(cell.notes)[:110]}")

    usage = llm.usage.as_dict() if llm else {}
    print(f"\n{'═' * 70}")
    print(f"всего: {time.time() - started:.1f}с, расход {json.dumps(usage, ensure_ascii=False)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
