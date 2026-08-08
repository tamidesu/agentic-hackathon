#!/usr/bin/env python3
"""Прогон шагов 1–5 и разбор извлечённых ковенантов.

Шаги 1–4 детерминированные и бесплатные; вызовы модели есть только на шаге 5.
Скрипт печатает по каждому заёмщику извлечённые пункты, пороги, направления
и деревья, а в конце — расход токенов и оценку стоимости.

Запуск (Windows, из каталога solution):
    .venv\\Scripts\\activate
    set GEMINI_API_KEY=...
    python scripts\\run_step5.py

Полезные ключи:
    --model gemini-2.5-flash-lite   какую модель звать (умолчание gemini-2.5-flash)

ВНИМАНИЕ ПРО БЕСПЛАТНЫЙ ТАРИФ. У preview-моделей (например gemini-3-flash-
preview) лимит — 20 запросов в СУТКИ на проект, чего не хватает даже на
двенадцать договоров. Квота считается НА ПРОЕКТ, а не на ключ, поэтому
несколько ключей из одного проекта её не увеличивают. Умолчание — стабильная
модель; при исчерпании берите --model gemini-2.5-flash-lite.
    --run runs\\step5                 куда класть артефакты (кэш переиспользуется)
    --force                          игнорировать кэш и переспросить модель
    --scenario P6                    только один заёмщик, для отладки

Кэш содержательный: повторный запуск без --force не тратит ни одного вызова,
поэтому гонять скрипт можно сколько угодно.
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

# Консоль Windows по умолчанию cp1251, и знак «≤» её роняет с
# UnicodeEncodeError — причём в самом конце, УЖЕ ПОСЛЕ того как все вызовы
# модели оплачены. Потерять результат прогона на выводе абсурдно, поэтому
# кодировка задаётся явно, а не оставляется на усмотрение системы.
for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):  # перенаправление в неперенастраиваемый поток
        pass

from pipeline import attribute, classify, covenants, extract  # noqa: E402
from pipeline.config import RunPaths, discover_dataset  # noqa: E402
from pipeline.gemini import (  # noqa: E402
    DEFAULT_GEMINI_MODEL,
    FREE_TIER_WORKERS_PER_KEY,
    verify_model,
)
from pipeline.llm import LLMClient  # noqa: E402

#: Цены за миллион токенов, только для оценки порядка величины.
PRICES = {
    "gemini-3-flash-preview": (0.075, 0.30),
    "gemini-2.5-flash-lite": (0.037, 0.15),
    "gemini-2.5-flash": (0.075, 0.30),
    "claude-haiku-4-5-20251001": (1.00, 5.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-opus-5": (15.00, 75.00),
}


def shape(node) -> str:
    """Дерево одной строкой — так его видно целиком."""
    if not isinstance(node, dict):
        return "?"
    op = node.get("op")
    if op == "AGG":
        bits = [str(node.get("category"))]
        if node.get("scope") and node["scope"] != "borrower":
            bits.append(f"scope={node['scope']}")
        if node.get("party"):
            bits.append(f"party={node['party']}")
        if node.get("period"):
            bits.append(f"period={node['period'][0]}..{node['period'][1]}")
        return f"AGG({', '.join(bits)})"
    if op == "CONST":
        return f"CONST({node.get('value')})"
    if op == "DISCLOSED":
        return f"DISCLOSED({node.get('key')})"
    return f"{op}({', '.join(shape(a) for a in node.get('args', []))})"


def main() -> int:
    ap = argparse.ArgumentParser(description="Прогон шагов 1–5")
    ap.add_argument("--dataset", type=Path, default=Path("../agentic-bank-public"))
    ap.add_argument("--run", type=Path, default=Path("runs/step5"))
    ap.add_argument("--model", default=None)
    ap.add_argument("--provider", choices=["gemini", "anthropic"], default="gemini")
    ap.add_argument("--force", action="store_true", help="игнорировать кэш")
    ap.add_argument("--scenario", default=None, help="только один заёмщик")
    ap.add_argument("--workers", type=int, default=0,
                    help="0 — подобрать по числу ключей и тарифу")
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    os.environ["LLM_MODE"] = args.provider
    key_var = "GEMINI_API_KEY" if args.provider == "gemini" else "ANTHROPIC_API_KEY"
    if not (os.environ.get(key_var) or os.environ.get("GEMINI_API_KEYS")
            or os.environ.get("GOOGLE_API_KEY")):
        print(f"ПРОВАЛ: переменная {key_var} не задана")
        print("Несколько ключей — через запятую в GEMINI_API_KEYS: лимит частоты "
              "на бесплатном тарифе считается на ключ, ключи перебираются по кругу.")
        return 2

    dataset = discover_dataset(args.dataset)
    paths = RunPaths.create(args.run)
    print(f"датасет: {dataset.root}\nартефакты: {paths.artifacts}\n")

    model = args.model or (DEFAULT_GEMINI_MODEL if args.provider == "gemini"
                           else "claude-opus-5")
    # Клиент создаётся ДО шага 2: сканы распознаёт vision-модель, и без
    # клиента они молча уходят в 'failed'. Первая версия скрипта звала
    # extract.run(llm=None) — в публичном наборе это теряло единственный
    # скан, а он оказался досье KYC, то есть связанными сторонами.
    llm = LLMClient(cache_dir=paths.cache, force=args.force)

    # Имя модели не угадывается, а проверяется по каталогу ключа: имена и
    # доступность меняются без нашего участия (gemini-2.5-flash вернул 404
    # «no longer available to new users»), и захардкоженное имя — это
    # отложенный отказ ровно в боевом окне.
    # Каталог перечислял gemini-2.5-flash, а генерация по ней отвечала 404
    # «no longer available to new users»: список говорит о существовании
    # модели, а не о праве её вызывать. Поэтому модель ПРОВЕРЯЕТСЯ вызовом,
    # один раз и до работы — молчаливая подмена посреди прогона испортила бы
    # кэш, ключ которого содержит имя модели.
    if args.provider == "gemini":
        model, notes = verify_model(llm.provider, model)
        for note in notes:
            print(f"       МОДЕЛЬ: {note}")
        print(f"       МОДЕЛЬ: проверена вызовом — {model}")
        # Клиент подставляет её всем запросам, не указавшим модель явно —
        # в том числе распознаванию сканов на шаге 2. Без этого туда
        # уезжало умолчание, замороженное на момент импорта модуля.
        llm.model = model

    # Параллелизм по числу ключей: на бесплатном тарифе лимит частоты
    # считается НА КЛЮЧ, и шесть веток на один ключ дают сплошные 429 —
    # ретраи с задержками съедают больше времени, чем экономит параллелизм.
    workers = args.workers
    if not workers:
        n_keys = getattr(llm.provider, "n_keys", 1)
        workers = max(1, n_keys * FREE_TIER_WORKERS_PER_KEY)
        print(f"       ключей {n_keys}, параллельных веток {workers}")

    # --- шаги 1–4: модель нужна только для сканов --------------------------- #
    t0 = time.time()
    n_docs = len(list(dataset.documents_dir.iterdir()))
    print(f"шаг 2  документов в каталоге: {n_docs}, извлекаю…")

    # \r рисует одну живую строку в терминале, но при перенаправлении
    # в файл превращает 202 записи в одну простыню на пол-экрана.
    # В файл пишем изредка и с переводом строки.
    interactive = sys.stdout.isatty()

    def progress(done: int, total: int, name: str) -> None:
        if interactive:
            print(f"\r       {done:4}/{total} {name[:40]:<40}", end="", flush=True)
        elif done == total or done % 50 == 0:
            print(f"       {done:4}/{total}", flush=True)

    rep = extract.run(dataset, paths, llm=llm, workers=8, progress=progress)
    print()
    print(f"шаг 2  извлечено {rep.extracted}, переиспользовано {rep.reused}, "
          f"на проверку {len(rep.review)}")
    if rep.failed:
        # Провал извлечения — это НЕ мелочь: непрочитанный документ просто
        # исчезает из расчёта, и заметить это потом уже негде.
        print(f"       ПРОВАЛЫ ИЗВЛЕЧЕНИЯ: {[d.doc_id for d in rep.failed]}")
        for d in rep.failed:
            print(f"         {d.doc_id}: {'; '.join(d.warnings)}")

    creport = classify.run(paths)
    print(f"шаг 3  {creport.counts()}")
    for alarm in creport.alarms():
        print(f"       ТРЕВОГА: {alarm}")

    docs, areport = attribute.run(dataset, paths)
    print(f"шаг 4  по счёту {areport.by_account}, по названию {areport.by_name}, "
          f"сирот {len(areport.orphans)}, период {areport.reporting_period}")
    for problem in areport.problems:
        print(f"       ПРОБЛЕМА: {problem}")
    print(f"       детерминированная часть заняла {time.time() - t0:.1f}с\n")

    # --- шаг 5: вызовы модели ---------------------------------------------- #
    print(f"шаг 5  модель {model}")

    t0 = time.time()
    if args.scenario:
        template = json.loads(dataset.template_json.read_text(encoding="utf-8"))
        wanted = covenants.expected_points(template)
        doc_id = next(i for i, d in sorted(docs.items())
                      if d.type == "LOAN_ACTIVE" and d.scenario_id == args.scenario)
        text = (paths.artifacts / "01_texts" / f"{doc_id}.txt").read_text(encoding="utf-8")
        results = [covenants.extract_one(args.scenario, doc_id, text, llm,
                                         wanted.get(args.scenario), model)]
        report = covenants.CovenantReport(scenarios=results)
    else:
        report = covenants.run(dataset, paths, llm, model=model, workers=workers)
    elapsed = time.time() - t0

    # --- разбор ------------------------------------------------------------- #
    print()
    total = 0
    for scenario in report.scenarios:
        head = f"{scenario.scenario_id:5} {scenario.doc_id or '—':14}"
        if not scenario.covenants:
            print(f"{head} БЕЗ КОВЕНАНТОВ  {scenario.problems}")
            continue
        print(f"{head} пункты {scenario.points()}")
        for c in sorted(scenario.covenants, key=lambda x: x.get("point", "")):
            total += 1
            sign = "≤" if c.get("direction") == "max" else "≥"
            unit = "$" if c.get("unit") == "amount" else ""
            print(f"      {c.get('point'):5} {sign} {unit}{c.get('threshold'):>14,.2f} "
                  f"[{c.get('unit')}]  {c.get('period_start')}..{c.get('period_end')}")
            print(f"            {shape(c.get('metric'))}")
            if c.get("is_conditional"):
                cond_sign = "≤" if c.get("condition_direction") == "max" else "≥"
                print(f"            УСЛОВНЫЙ: {shape(c.get('condition_metric'))} "
                      f"{cond_sign} {c.get('condition_threshold')}")
            if c.get("carve_outs"):
                print(f"            оговорки: {c['carve_outs']}")
        for note in scenario.notes:
            print(f"      прим.: {note}")
        for problem in scenario.problems:
            print(f"      ПРОБЛЕМА: {problem}")

    print(f"\nвсего ковенантов: {total} (ожидалось 36 на публичном наборе)")
    for alarm in report.alarms():
        print(f"ТРЕВОГА: {alarm}")

    usage = llm.usage.as_dict()
    print(f"\nвремя шага 5: {elapsed:.1f}с")
    print(f"расход: {json.dumps(usage, ensure_ascii=False)}")
    price_in, price_out = PRICES.get(model, (0.0, 0.0))
    if price_in:
        cost = (usage.get("input_tokens", 0) / 1e6 * price_in
                + usage.get("output_tokens", 0) / 1e6 * price_out)
        print(f"оценка стоимости прогона: ${cost:.4f} (на бесплатном тарифе $0)")
    print(f"\nартефакт: {paths.artifacts / '04_covenants.json'}")
    return 0 if total == 36 and not report.alarms() else 1


if __name__ == "__main__":
    sys.exit(main())
