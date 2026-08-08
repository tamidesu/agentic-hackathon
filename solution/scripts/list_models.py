#!/usr/bin/env python3
"""Какие модели реально доступны нашему ключу.

Имена моделей и их доступность меняются без нашего участия:
`gemini-2.5-flash` вернул 404 «no longer available to new users» — при том,
что модель существует и документирована. Захардкоженное имя — отложенный
отказ, который сработает в самый неудобный момент.

Запуск:
    set GEMINI_API_KEYS=ключ1,ключ2,ключ3
    python scripts\\list_models.py

Печатает список, отмечает выбор пайплайна и поясняет, почему именно он.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.gemini import (  # noqa: E402
    DEFAULT_GEMINI_MODEL,
    GeminiProvider,
    rank_model,
    resolve_model,
    verify_model,
    warn_about_free_tier,
)


def main() -> int:
    ap = argparse.ArgumentParser(description="Доступные модели Gemini")
    ap.add_argument("--prefer", default=DEFAULT_GEMINI_MODEL,
                    help="желаемая модель; если недоступна, будет выбрана лучшая")
    ap.add_argument("--all", action="store_true", help="показать и специализированные")
    ap.add_argument("--verify", action="store_true",
                    help="проверить выбор настоящим вызовом: каталог перечисляет "
                         "модели, вызывать которые нельзя")
    args = ap.parse_args()

    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEYS")
            or os.environ.get("GOOGLE_API_KEY")):
        print("ПРОВАЛ: задайте GEMINI_API_KEY или GEMINI_API_KEYS")
        return 2

    provider = GeminiProvider()
    print(f"ключей: {provider.n_keys}\n")

    try:
        models = provider.available_models()
    except Exception as exc:  # noqa: BLE001
        print(f"ПРОВАЛ: не удалось получить список: {type(exc).__name__}: {str(exc)[:300]}")
        return 1

    if args.verify:
        chosen, notes = verify_model(provider, args.prefer, models)
    else:
        chosen, notes = resolve_model(args.prefer, models)
        notes.append("выбор НЕ проверен вызовом — добавьте --verify")

    shown = models if args.all else [m for m in models if rank_model(m)[0]]
    print(f"доступно моделей: {len(models)} (пригодных для текста: {len(shown)})\n")
    for name in sorted(shown, key=rank_model, reverse=True):
        mark = "→" if name == chosen else " "
        tags = []
        if "preview" in name:
            tags.append("preview: 20 запросов/сутки на бесплатном")
        if "lite" in name:
            tags.append("lite")
        print(f" {mark} {name:<42} {'; '.join(tags)}")

    print(f"\nвыбор пайплайна: {chosen}")
    for note in notes:
        print(f"  {note}")
    warning = warn_about_free_tier(chosen)
    if warning:
        print(f"  ВНИМАНИЕ: {warning}")
    print(f"\nЗапуск с ней:\n  python scripts\\run_step5.py --model {chosen}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
