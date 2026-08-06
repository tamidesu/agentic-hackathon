"""Валидатор submission.json БЕЗ ключа (шаг 1 плана).

Это тот инструмент, который поедет в боевое окно: на приватном датасете
ground truth недоступен, и единственный способ убедиться, что сабмит не
обнулится по формальным причинам, — проверить его на инварианты.

Уровни:
    ERROR — ячейка (или весь файл) гарантированно получит 0. Чинить обязательно.
    WARN  — подозрительно, но не смертельно. Смотреть глазами.

Использование:
    python -m pipeline.validate --submission out/submission.json \
        --template <dataset>/submission_template.json --ledger <dataset>/ledger.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path

VALID_STATUSES = {"COMPLIANT", "BREACH"}
TOP_LEVEL_REQUIRED = ("team", "contact_email", "model", "answers")
CELL_FIELDS = ("status", "actual", "evidence_txn_id")
TXN_ID_RE = re.compile(r"^TXN-([A-Za-z0-9]+)-\d+$")

# Дублируется намеренно: валидатор обязан работать в одиночку, без импорта
# llm.py и без установленного SDK — он поедет в боевое окно.
KNOWN_MODELS = ("claude-opus-5", "claude-sonnet-5", "claude-haiku-4-5-20251001", "claude-fable-5")


@dataclass
class Issue:
    level: str  # ERROR | WARN
    where: str
    message: str

    def __str__(self) -> str:
        return f"[{self.level}] {self.where}: {self.message}"


@dataclass
class ValidationReport:
    issues: list[Issue] = field(default_factory=list)
    cells_checked: int = 0

    def error(self, where: str, msg: str) -> None:
        self.issues.append(Issue("ERROR", where, msg))

    def warn(self, where: str, msg: str) -> None:
        self.issues.append(Issue("WARN", where, msg))

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.level == "ERROR"]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.level == "WARN"]

    @property
    def ok(self) -> bool:
        return not self.errors


def _decimals(value: float) -> int:
    try:
        d = Decimal(str(value)).normalize()
    except InvalidOperation:
        return 0
    exp = d.as_tuple().exponent
    return -exp if isinstance(exp, int) and exp < 0 else 0


def _load_ledger_index(path: Path) -> dict[str, str]:
    """txn_id -> scenario_id, выведенный из самого txn_id."""
    index: dict[str, str] = {}
    with path.open(newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            tid = (row.get("txn_id") or "").strip()
            m = TXN_ID_RE.match(tid)
            if m:
                index[tid] = m.group(1)
    return index


def validate(
    submission: dict,
    template: dict,
    ledger_index: dict[str, str] | None = None,
) -> ValidationReport:
    rep = ValidationReport()

    for fld in TOP_LEVEL_REQUIRED:
        if fld not in submission:
            rep.error("<root>", f"отсутствует поле {fld!r}")
        elif fld != "answers" and not str(submission.get(fld) or "").strip():
            rep.error("<root>", f"поле {fld!r} пустое")

    contact = str(submission.get("contact_email") or "")
    if contact and "@" not in contact:
        rep.warn("<root>", f"contact_email не похож на адрес: {contact!r}")

    model = str(submission.get("model") or "")
    if model and model not in KNOWN_MODELS:
        rep.warn(
            "<root>",
            f"model={model!r} не входит в список известных строк {KNOWN_MODELS} — "
            f"проверьте, что имя модели указано точно",
        )

    answers = submission.get("answers")
    tpl_answers = template.get("answers", {})
    if not isinstance(answers, dict):
        rep.error("<root>", "answers не объект — все ячейки неоцениваемы")
        return rep

    missing_scen = set(tpl_answers) - set(answers)
    extra_scen = set(answers) - set(tpl_answers)
    for s in sorted(missing_scen):
        rep.error(s, "сценарий отсутствует — все его ячейки получат 0")
    for s in sorted(extra_scen):
        rep.error(s, "лишний сценарий, которого нет в шаблоне")

    for scenario, tpl_cells in tpl_answers.items():
        got_cells = answers.get(scenario)
        if not isinstance(got_cells, dict):
            continue

        missing_pts = set(tpl_cells) - set(got_cells)
        extra_pts = set(got_cells) - set(tpl_cells)
        for p in sorted(missing_pts):
            rep.error(f"{scenario}/{p}", "пункт отсутствует")
        for p in sorted(extra_pts):
            rep.error(f"{scenario}/{p}", "лишний пункт, которого нет в шаблоне")

        for point in tpl_cells:
            if point not in got_cells:
                continue
            rep.cells_checked += 1
            where = f"{scenario}/{point}"
            cell = got_cells[point]

            if not isinstance(cell, dict):
                rep.error(where, f"ячейка не объект: {type(cell).__name__}")
                continue
            for f_ in CELL_FIELDS:
                if f_ not in cell:
                    rep.error(where, f"в ячейке нет поля {f_!r}")
            extra_fields = set(cell) - set(CELL_FIELDS)
            if extra_fields:
                # Условие прямо запрещает добавлять ключи. Скорее всего это
                # забытая отладочная информация — вычистить перед отправкой.
                rep.warn(where, f"лишние поля в ячейке: {sorted(extra_fields)}")

            # --- status: 0.50 и обнуление всей ячейки при ошибке ---
            st = cell.get("status")
            if st is None:
                rep.error(where, "status не заполнен")
            elif not isinstance(st, str):
                rep.error(where, f"status не строка: {st!r}")
            elif st not in VALID_STATUSES:
                if st.upper() in VALID_STATUSES:
                    rep.error(where, f"status в неверном регистре: {st!r} — нужен {st.upper()!r}")
                else:
                    rep.error(where, f"status не COMPLIANT/BREACH: {st!r}")

            # --- actual: 0.30, плюс 0.20 в ячейках без evidence ---
            act = cell.get("actual")
            if act is None:
                rep.error(where, "actual не заполнен — потеряно 0.30, а возможно и 0.50")
            elif isinstance(act, bool) or not isinstance(act, (int, float)):
                rep.error(where, f"actual не число (возможно строка): {act!r}")
            elif not math.isfinite(act):
                rep.error(where, f"actual не конечное число: {act!r}")
            elif act < 0:
                rep.error(where, f"actual отрицателен: {act} — должен быть модулем")
            elif act == 0:
                rep.warn(where, "actual = 0 — возможен пропуск данных, проверьте")
            elif _decimals(act) > 2:
                rep.warn(where, f"actual с {_decimals(act)} знаками после запятой: {act}")

            # --- evidence: 0.20 ---
            ev = cell.get("evidence_txn_id")
            if ev is not None:
                if not isinstance(ev, str):
                    rep.error(where, f"evidence_txn_id не строка и не null: {ev!r}")
                elif not TXN_ID_RE.match(ev):
                    rep.error(where, f"evidence_txn_id не похож на идентификатор: {ev!r}")
                elif ledger_index is not None:
                    if ev not in ledger_index:
                        rep.error(where, f"evidence_txn_id отсутствует в реестре: {ev}")
                    elif ledger_index[ev] != scenario:
                        rep.error(
                            where,
                            f"evidence_txn_id {ev} принадлежит сценарию "
                            f"{ledger_index[ev]}, а не {scenario}",
                        )
    return rep


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Валидатор submission.json (работает без ключа)")
    ap.add_argument("--submission", required=True, type=Path)
    ap.add_argument("--template", required=True, type=Path)
    ap.add_argument("--ledger", type=Path, help="реестр — включает проверку evidence по существованию и владельцу")
    ap.add_argument("--strict", action="store_true", help="ненулевой код возврата и при WARN")
    args = ap.parse_args(argv)

    try:
        submission = json.loads(args.submission.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"[ERROR] submission — битый JSON: {exc}")
        print("Все ячейки неоцениваемы. Итог: 0.")
        return 2

    template = json.loads(args.template.read_text(encoding="utf-8"))
    ledger_index = _load_ledger_index(args.ledger) if args.ledger else None

    rep = validate(submission, template, ledger_index)

    for issue in rep.issues:
        print(issue)

    print(
        f"\nПроверено ячеек: {rep.cells_checked}  "
        f"ошибок: {len(rep.errors)}  предупреждений: {len(rep.warnings)}"
    )
    if rep.ok:
        print("ГОТОВ К ОТПРАВКЕ" if not rep.warnings else "готов, но есть предупреждения — посмотрите глазами")
    else:
        print("НЕ ОТПРАВЛЯТЬ: перечисленные ячейки получат 0")

    if rep.errors:
        return 1
    return 1 if (args.strict and rep.warnings) else 0


if __name__ == "__main__":
    sys.exit(main())
