#!/usr/bin/env python3
"""Проверка живого провайдера — до того, как на него положиться.

Закрывает три вопроса, которые нельзя проверить без сети:

  1. связь и ключ вообще работают;
  2. наша схема принимается СЕРВЕРОМ, а не только клиентским SDK
     (адаптер проверен локально, но приговор выносит API);
  3. рекурсивное дерево выражений ковенанта возвращается целым —
     на нём держится весь расчётный движок.

Запуск:
    export GEMINI_API_KEY=...      # или ANTHROPIC_API_KEY
    python solution/scripts/check_provider.py --provider gemini
    python solution/scripts/check_provider.py --provider gemini --model gemini-2.5-flash-lite

Ключ берётся ТОЛЬКО из окружения и никуда не записывается.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.gemini import DEFAULT_GEMINI_MODEL  # noqa: E402
from pipeline.llm import LLMClient, LLMRequest  # noqa: E402
from pipeline.schemas import (  # noqa: E402
    COVENANT_SPEC_SCHEMA,
    make_quote_validator,
    validate_covenant_spec,
)

OK, FAIL, WARN = "  OK  ", " ПРОВАЛ", " ВНИМАНИЕ"

#: Рекурсивное дерево — то самое, ради чего затевалась проверка.
#: Рекурсивное поле `args` НЕобязательное: Gemini разворачивает циклы
#: только в необязательных свойствах.
METRIC_TREE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["metric"],
    "$defs": {
        "node": {
            "type": "object",
            "additionalProperties": False,
            "required": ["op"],
            "properties": {
                "op": {
                    "type": "string",
                    "enum": ["AGG", "ADD", "SUB", "MUL", "DIV", "MAX", "MIN"],
                },
                "category": {"type": "string"},
                "args": {"type": "array", "items": {"$ref": "#/$defs/node"}},
            },
        }
    },
    "properties": {"metric": {"$ref": "#/$defs/node"}},
}

TREE_PROMPT = (
    "Переведи формулу ковенанта в дерево выражений.\n\n"
    "Формула: отношение EBITDA (Выручка минус Операционные расходы) "
    "к Процентным расходам.\n\n"
    "Узлы: AGG(category) — агрегат по статье; DIV/SUB/ADD/MUL/MAX/MIN(args) — операции.\n"
    "Статьи: revenue, opex, interest.\n"
    "Верни только дерево, без пояснений."
)

EXPECTED_TREE = {
    "op": "DIV",
    "args": [
        {"op": "SUB", "args": [{"op": "AGG", "category": "revenue"},
                               {"op": "AGG", "category": "opex"}]},
        {"op": "AGG", "category": "interest"},
    ],
}

COVENANT_TEXT = (
    "Статья 6 — Финансовые ковенанты\n"
    "Пункт 6.1 Maximum Capital Intensity Ratio. Заёмщик, Aktau Port Services JSC, "
    "обязуется не допускать, чтобы коэффициент капиталоёмкости за период "
    "с 2025-01-01 по 2025-12-31 превышал 0.42x. Коэффициент капиталоёмкости "
    "означает отношение совокупных капитальных затрат за период к сумме "
    "операционных расходов и арендных платежей за тот же период.\n"
    "Пункт 6.3 Максимальные платежи связанным сторонам. Заёмщик обязуется "
    "не допускать, чтобы совокупный объём платежей в пользу связанных сторон "
    "за период с 2025-01-01 по 2025-12-31 превышал $450,000.00."
)


def _shape(node) -> str:
    if not isinstance(node, dict):
        return "?"
    op = node.get("op")
    if op == "AGG":
        return f"AGG({node.get('category')})"
    args = ", ".join(_shape(a) for a in node.get("args", []))
    return f"{op}({args})"


def main() -> int:
    ap = argparse.ArgumentParser(description="Проверка живого провайдера")
    ap.add_argument("--provider", choices=["gemini", "anthropic"], default="gemini")
    ap.add_argument("--model", default=None)
    ap.add_argument("--cache", default=None, help="каталог кэша (по умолчанию временный)")
    args = ap.parse_args()

    os.environ["LLM_MODE"] = args.provider
    key_var = "GEMINI_API_KEY" if args.provider == "gemini" else "ANTHROPIC_API_KEY"
    if not os.environ.get(key_var) and not os.environ.get("GOOGLE_API_KEY"):
        print(f"{FAIL}  переменная {key_var} не задана")
        return 2

    model = args.model or (
        DEFAULT_GEMINI_MODEL if args.provider == "gemini" else "claude-opus-5"
    )
    print(f"провайдер: {args.provider}   модель: {model}\n")

    import tempfile

    cache = args.cache or tempfile.mkdtemp(prefix="provider-check-")
    client = LLMClient(cache_dir=cache)

    if args.provider == "gemini":
        from pipeline.gemini import warn_about_free_tier

        note = warn_about_free_tier(model)
        if note:
            print(f"{WARN}  {note}\n")

    failures = 0

    # --- 1. связь ---------------------------------------------------------- #
    t0 = time.time()
    try:
        res = client.extract(LLMRequest(
            prompt="Верни ровно {\"ok\": true}.",
            schema={"type": "object", "required": ["ok"],
                    "properties": {"ok": {"type": "boolean"}},
                    "additionalProperties": False},
            model=model, prompt_version="check-ping",
        ))
        print(f"{OK}  связь есть, ответ {res.data}, {time.time() - t0:.1f}с, "
              f"токены {res.input_tokens}/{res.output_tokens}")
    except Exception as exc:
        print(f"{FAIL}  вызов не прошёл: {type(exc).__name__}: {str(exc)[:300]}")
        return 1

    # --- 2. рекурсивная схема ---------------------------------------------- #
    t0 = time.time()
    try:
        res = client.extract(LLMRequest(
            prompt=TREE_PROMPT, schema=METRIC_TREE_SCHEMA,
            model=model, prompt_version="check-tree",
        ))
        tree = res.data.get("metric")
        got, want = _shape(tree), _shape(EXPECTED_TREE)
        if got == want:
            print(f"{OK}  рекурсивная схема принята сервером, дерево верное: {got}")
        else:
            failures += 1
            print(f"{WARN}  дерево вернулось, но не то, что ожидалось")
            print(f"          получено: {got}")
            print(f"          ожидалось: {want}")
        print(f"          {time.time() - t0:.1f}с, токены {res.input_tokens}/{res.output_tokens}")
    except Exception as exc:
        failures += 1
        print(f"{FAIL}  рекурсивная схема отвергнута: {type(exc).__name__}: {str(exc)[:300]}")
        print("          без неё дерево выражений придётся передавать строкой")

    # --- 3. извлечение ковенантов с проверкой цитат ------------------------- #
    t0 = time.time()
    validator = lambda p: (  # noqa: E731
        validate_covenant_spec(p) + make_quote_validator(COVENANT_TEXT)(p)
    )
    try:
        res = client.extract(
            LLMRequest(
                prompt=(
                    "Извлеки все финансовые ковенанты из текста ниже.\n"
                    "Цитата обязана быть ДОСЛОВНОЙ из документа.\n\n" + COVENANT_TEXT
                ),
                schema=COVENANT_SPEC_SCHEMA, model=model, prompt_version="check-covenants",
            ),
            validator=validator,
        )
        covenants = {c["point"]: c for c in res.data.get("covenants", [])}
        print(f"{OK}  ковенанты извлечены: {sorted(covenants)}, "
              f"{time.time() - t0:.1f}с, токены {res.input_tokens}/{res.output_tokens}")
        for point, expect in (("6.1", (0.42, "max", "ratio")), ("6.3", (450000.0, "max", "amount"))):
            c = covenants.get(point)
            if not c:
                failures += 1
                print(f"{FAIL}  пункт {point} не извлечён")
                continue
            th, direction, unit = expect
            ok = (abs(float(c["threshold"]) - th) < 1e-6
                  and c["direction"] == direction and c["unit"] == unit)
            mark = OK if ok else FAIL
            failures += 0 if ok else 1
            print(f"{mark}  {point}: порог {c['threshold']} {c['direction']} {c['unit']} "
                  f"(ожидалось {th} {direction} {unit})")
        if client.usage.repairs:
            print(f"{WARN}  потребовалось исправлений: {client.usage.repairs} "
                  f"(цитаты или значения приходили неверными)")
    except Exception as exc:
        failures += 1
        print(f"{FAIL}  извлечение не прошло: {type(exc).__name__}: {str(exc)[:400]}")

    print(f"\nрасход: {json.dumps(client.usage.as_dict(), ensure_ascii=False)}")
    print(f"кэш: {cache}")
    print("\nИТОГ:", "провайдер пригоден" if failures == 0
          else f"проблем: {failures} — см. выше")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
