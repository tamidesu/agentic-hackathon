"""Шаг 15: флаги уверенности — где ответ похож на правильный, но может им не быть.

Вход:  <run>/artifacts/09_results.json (после шага 13),
       <run>/artifacts/08_apply_report.json, 10_assembly_report.json,
       06_ledger_report.json — если есть
Выход: <run>/artifacts/11_confidence.json и список CellRisk по убыванию риска

ЗАЧЕМ. На приватном наборе ключа не будет: отличить верный ответ от
правдоподобного можно только по внутренним признакам сомнения. Этот модуль
их собирает и упорядочивает ячейки по риску — в боевом окне разбор идёт
сверху этого списка. Модуль ТОЛЬКО наблюдает: ни одного значения он
не меняет.

ПРИЗНАКИ И ИХ ЛОГИКА

  расчёт не удался      пустой агрегат, NaN, запасное значение — ячейка
                        честно кричит о себе сама
  далеко от порога      банк ставит порог рядом с ожидаемым значением;
                        отличие в 5+ раз — почти наверняка ошибка расчёта,
                        а не грандиозное нарушение
  вплотную к порогу     в пределах 5% — допуска, с которым скорер вообще
                        засчитывает actual: вердикт переворачивается
                        ошибкой в третьем знаке (5–10% — жёлтая зона,
                        учитывается в ранге, но не флагует)
  хрупкая разность      метрика-отношение, где за доминирующим агрегатом
                        стоит сравнимая масса других: ошибка входа
                        усиливается кратно (у B1 5% ошибки выручки дают
                        13% ошибки числителя)
  вердикт на одной      контрфактуально исход держится на единственной
  KYC-операции          операции, признанной связанной стороной по
                        сопоставлению имён с досье — самый шаткий шаг
                        пайплайна
  провал уровня группы  у заёмщика есть невычислимая ячейка уровня Группы:
                        периметр его отчётности под сомнением целиком
  промах адресации      применяемое примечание аудитора не нашло свою
                        операцию — корректировка потеряна
  неопознанные данные   валюта, форма или статья вне словарей — шаги
                        обязаны сообщать об этом в отчётах (см. их problems)

Порог «вплотную» взят 5%, а не 10% из постановки: 5% — собственный допуск
скорера по actual, внутри него вердикт неотличим от шума измерения.
Ячейки 5–10% остаются видимыми в ранжированном списке.
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field

from . import artifacts as A
from .config import RunPaths

log = logging.getLogger(__name__)

#: Далеко от порога: расхождение в 5 раз в любую сторону.
FAR_HIGH = 5.0
FAR_LOW = 0.2
#: Вплотную: внутри допуска скорера по actual.
NEAR_FLAG = 0.05
#: Жёлтая зона из постановки — в ранг, но не во флаг.
NEAR_RANK = 0.10
#: Хрупкая разность: прочие агрегаты ≥ половины доминирующего.
AMPLIFICATION = 1.5
#: Не больше стольких ячеек имеет смысл флаговать: на 36 ячеек длиннее
#: список — не приоритизация, а перепись.
FLAG_BUDGET = 20


@dataclass
class CellRisk:
    scenario_id: str
    point: str
    risk: float = 0.0
    flagged: bool = False
    signals: list[str] = field(default_factory=list)

    @property
    def where(self) -> str:
        return f"{self.scenario_id}/{self.point}"

    def add(self, weight: float, label: str, flags: bool = True) -> None:
        self.risk += weight
        self.signals.append(label)
        if flags:
            self.flagged = True


# --------------------------------------------------------------------------- #
# Отдельные признаки
# --------------------------------------------------------------------------- #


def _margin(actual: float, threshold: float) -> float:
    return abs(abs(actual) - abs(threshold)) / abs(threshold)


def _amplification(trace: list[dict]) -> float:
    """Сколько агрегатной массы стоит за доминирующим агрегатом.

    1.0 — метрика держится на одном агрегате; 2.0 — прочие в сумме равны
    доминирующему, и их ошибки взаимно усиливаются при вычитании.
    Повторные обращения к одному агрегату (условие springing-теста)
    не считаются дважды.
    """
    seen: dict[tuple, float] = {}
    for t in trace:
        key = (t.get("category"), t.get("scope"), t.get("party"),
               tuple(t.get("period") or ()))
        seen[key] = abs(t.get("value") or 0.0)
    values = [v for v in seen.values() if v > 0]
    if not values:
        return 1.0
    return sum(values) / max(values)


def _group_gap_scenarios(results: dict) -> set[str]:
    """Заёмщики, у которых не вычислилась ячейка уровня Группы."""
    out = set()
    for scenario, cells in results.items():
        for cell in cells.values():
            for problem in cell.get("problems", []):
                if "/group пуст" in problem:
                    out.add(scenario)
    return out


def _addressing_misses(apply_report: dict | None) -> dict[str, list[str]]:
    """Сценарий → применяемые примечания, не нашедшие свою операцию."""
    out: dict[str, list[str]] = {}
    for line in (apply_report or {}).get("skipped_notes", []):
        if "не найдена операция" in line:
            scenario = line.split(" ", 1)[0]
            out.setdefault(scenario, []).append(line)
    return out


def _scenario_report_problems(paths: RunPaths) -> dict[str, list[str]]:
    """Строки «неопознанное» из отчётов шагов, привязанные к заёмщику.

    Шаги обязаны не терять незнакомое молча (валюта вне словаря, форма
    организации, статья) — они пишут это в problems своих отчётов
    с пометкой «вне словаря» или «неопознан». Здесь эти строки
    подтягиваются к ячейкам заёмщика.
    """
    markers = ("вне словаря", "неопознан", "неизвестн")
    sources = [A.LEDGER_REPORT, A.APPLY_REPORT]
    out: dict[str, list[str]] = {}
    for name in sources:
        path = paths.artifacts / name
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        for line in data.get("problems", []):
            if not isinstance(line, str) or not any(m in line for m in markers):
                continue
            scenario = line.split(":", 1)[0].strip()
            out.setdefault(scenario, []).append(line)
    return out


# --------------------------------------------------------------------------- #
# Сборка
# --------------------------------------------------------------------------- #


def assess(
    results: dict,
    assembly_report: dict | None = None,
    apply_report: dict | None = None,
    scenario_problems: dict[str, list[str]] | None = None,
) -> list[CellRisk]:
    """Признаки по каждой ячейке, отсортированные по убыванию риска."""
    fallback_cells = set((assembly_report or {}).get("cells_fallback", []))
    group_gaps = _group_gap_scenarios(results)
    misses = _addressing_misses(apply_report)
    scenario_problems = scenario_problems or {}

    out: list[CellRisk] = []
    for scenario, cells in sorted(results.items()):
        for point, cell in sorted(cells.items()):
            r = CellRisk(scenario_id=scenario, point=point)
            actual = cell.get("actual")
            threshold = cell.get("threshold")
            problems = cell.get("problems", [])

            if r.where in fallback_cells:
                r.add(1.0, "результата расчёта нет — запасное значение")
            if actual is None or (isinstance(actual, float) and math.isnan(actual)):
                r.add(1.0, "actual не вычислен")
            for p in problems:
                if "пуст" in p:
                    r.add(1.0, p[:120])
                    break

            if isinstance(actual, (int, float)) and threshold:
                ratio = abs(actual) / abs(threshold)
                margin = _margin(actual, threshold)
                if ratio > FAR_HIGH or ratio < FAR_LOW:
                    r.add(0.9, f"далеко от порога: actual/threshold = {ratio:.2f}")
                elif margin <= NEAR_FLAG:
                    r.add(0.7, f"вплотную к порогу: запас {margin:.1%} — в пределах "
                               f"допуска скорера, вердикт неотличим от шума")
                elif margin <= NEAR_RANK:
                    r.add(0.25, f"близко к порогу: запас {margin:.1%}", flags=False)

            if cell.get("unit") == "ratio":
                amp = _amplification(cell.get("trace", []))
                if amp >= AMPLIFICATION:
                    r.add(0.5, f"хрупкая разность: за доминирующим агрегатом "
                               f"масса ×{amp:.2f} — ошибка входа усиливается")

            if cell.get("evidence_basis") == "related_party":
                r.add(0.5, "вердикт держится на одной операции, признанной "
                           "связанной стороной по сопоставлению имён с KYC")

            if scenario in group_gaps and "/group пуст" not in " ".join(problems):
                r.add(0.4, "у заёмщика есть невычислимая ячейка уровня Группы — "
                           "периметр отчётности под сомнением")

            for line in misses.get(scenario, ()):
                r.add(0.4, f"промах адресации: {line[:120]}")
            for line in scenario_problems.get(scenario, ()):
                r.add(0.3, f"неопознанные данные: {line[:120]}")

            if cell.get("status") == "BREACH" and not cell.get("evidence_txn_id"):
                r.add(0.25, "BREACH без доказательства", flags=False)

            out.append(r)

    out.sort(key=lambda r: (-r.risk, r.where))
    return out


def run(paths: RunPaths) -> list[CellRisk]:
    results_path = paths.artifacts / A.RESULTS
    if not results_path.exists():
        log.warning("ФЛАГИ: результатов расчёта нет — оценивать нечего")
        return []
    results = json.loads(results_path.read_text(encoding="utf-8"))

    def _optional(name: str) -> dict | None:
        p = paths.artifacts / name
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None

    risks = assess(
        results,
        assembly_report=_optional(A.ASSEMBLY_REPORT),
        apply_report=_optional(A.APPLY_REPORT),
        scenario_problems=_scenario_report_problems(paths),
    )

    flagged = [r for r in risks if r.flagged]
    if len(flagged) > FLAG_BUDGET:
        log.warning(
            "ФЛАГИ: помечено %d ячеек из %d — список перестаёт быть "
            "приоритизацией, разбирайте сверху по рангу",
            len(flagged), len(risks),
        )

    (paths.artifacts / A.CONFIDENCE).write_text(
        json.dumps(
            [
                {"scenario": r.scenario_id, "point": r.point,
                 "risk": round(r.risk, 3), "flagged": r.flagged,
                 "signals": r.signals}
                for r in risks
            ],
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    return risks


def render(risks: list[CellRisk], limit: int = 36) -> list[str]:
    """Строки для консоли: флаги сверху, дальше жёлтая зона ранга."""
    lines: list[str] = []
    flagged = [r for r in risks if r.flagged]
    watch = [r for r in risks if not r.flagged and r.risk > 0]
    lines.append(
        f"помечено {len(flagged)} из {len(risks)} ячеек; "
        f"ещё {len(watch)} в жёлтой зоне"
    )
    for r in flagged[:limit]:
        lines.append(f"  {r.risk:4.2f}  {r.where}")
        for s in r.signals:
            lines.append(f"          – {s}")
    if watch:
        lines.append("  жёлтая зона (не флаг, но рядом):")
        for r in watch[:limit]:
            lines.append(f"  {r.risk:4.2f}  {r.where}: {'; '.join(r.signals)[:100]}")
    return lines
