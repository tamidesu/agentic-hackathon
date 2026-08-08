"""Шаг 13: определение evidence_txn_id.

Вход:  <run>/artifacts/09_results.json, подготовленный реестр,
       <run>/artifacts/04_adjustments.json (корректировки аудитора)
Выход: <run>/artifacts/09_results.json (заполнено поле evidence_txn_id)

ОПРЕДЕЛЕНИЕ ИЗ УСЛОВИЯ ЗАДАЧИ

«evidence_txn_id — единственная транзакция, определяющая результат: та, чья
переклассификация, включение, исключение или исправление и приводит
к нарушению ковенанта. Уберите её — и вердикт изменится.

Транзакция, которая лишь вносит вклад в сумму, доказательством НЕ является:
ни самая крупная строка в проверяемой категории, ни последняя перед
закрытием периода, ни та, что случайно вывела накопленную сумму за порог.»

ОТСЮДА ДВА УСЛОВИЯ, А НЕ ОДНО

Контрфактуальная проверка «уберите — вердикт изменится» необходима, но её
НЕДОСТАТОЧНО. В агрегате из десяти платежей, превысившем порог на немного,
удаление почти любого крупного платежа перевернёт вердикт — и все они
окажутся «доказательствами». Именно это условие и запрещает.

Отличает настоящее доказательство ПРОИСХОЖДЕНИЕ его трактовки: операция
попала в расчёт (или выпала из него) не по своим собственным реквизитам,
а по решению, записанному в документе:

  * аудитор переклассифицировал её в проверяемую статью;
  * её сумма отсутствовала в реестре и восстановлена из документа;
  * правило отсечения перенесло её в другой период;
  * контрагент признан связанной стороной по досье KYC, хотя из назначения
    платежа это не следует;
  * аудитор исправил ошибочно отражённую сумму.

Поэтому: кандидаты берутся только из документально обусловленных операций,
и уже среди них ищется та, что решает исход.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .compute import LedgerAggregateSource, Row
from . import artifacts as A
from .config import RunPaths
from .covenant_types import CovenantTest, run_test

log = logging.getLogger(__name__)


#: Основания попадания в кандидаты, от сильного к слабому. Порядок решает
#: спор, если контрфактуально подходят несколько операций.
BASIS_WEIGHT: dict[str, int] = {
    "reclassified": 5,      # аудитор перенёс операцию в проверяемую статью
    "recovered_amount": 4,  # сумма отсутствовала в реестре, взята из документа
    "corrected": 4,         # аудитор исправил ошибочно отражённую сумму
    "cutoff": 3,            # операция отнесена к другому периоду
    "related_party": 2,     # контрагент признан связанным по досье, а не по описанию
}


@dataclass
class Candidate:
    txn_id: str
    basis: str
    detail: str = ""
    #: Чем заменить поля строки при контрфактуальном прогоне. Для
    #: переклассификации это возврат в исходную статью, а не удаление:
    #: операция никуда не исчезала.
    revert: dict = field(default_factory=dict)

    @property
    def weight(self) -> int:
        return BASIS_WEIGHT.get(self.basis, 1)


def candidates_from_artifacts(
    rows: list[Row], adjustments: list[dict]
) -> dict[str, list[Candidate]]:
    """Документально обусловленные операции, сгруппированные по заёмщику."""
    by_scenario: dict[str, list[Candidate]] = {}
    scenario_of = {r.txn_id: r.scenario_id for r in rows}
    original_category = {r.txn_id: r.category for r in rows}

    for adj in adjustments:
        txn_id = adj.get("target_txn_id")
        if not txn_id or txn_id not in scenario_of:
            continue
        kind = adj.get("kind", "")
        basis = {
            "reclassification": "reclassified",
            "missing_amount": "recovered_amount",
            "cutoff": "cutoff",
            "correction": "corrected",
        }.get(kind)
        if not basis:
            continue
        revert: dict = {}
        if basis == "reclassified" and adj.get("from_category"):
            revert = {"category": adj["from_category"]}
        by_scenario.setdefault(scenario_of[txn_id], []).append(
            Candidate(txn_id=txn_id, basis=basis,
                      detail=adj.get("description", "")[:160], revert=revert)
        )

    # Включение по признаку связанной стороны: из реквизитов самой операции
    # это не следует, признак приходит из досье KYC.
    for r in rows:
        if r.party == "related":
            by_scenario.setdefault(r.scenario_id, []).append(
                Candidate(txn_id=r.txn_id, basis="related_party",
                          detail=f"контрагент {r.counterparty} признан связанной стороной",
                          revert={"party": None})
            )

    for scenario in by_scenario:
        # Дедупликация: одна операция может иметь несколько оснований,
        # оставляем сильнейшее.
        best: dict[str, Candidate] = {}
        for c in by_scenario[scenario]:
            if c.txn_id not in best or c.weight > best[c.txn_id].weight:
                best[c.txn_id] = c
        by_scenario[scenario] = sorted(
            best.values(), key=lambda c: (-c.weight, c.txn_id)
        )
    _ = original_category
    return by_scenario


@dataclass
class EvidenceResult:
    txn_id: str | None
    decisive: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    basis: str | None = None


def find_evidence(
    test: CovenantTest,
    rows: list[Row],
    candidates: list[Candidate],
    disclosed: dict[str, float] | None = None,
    baseline_status: str | None = None,
) -> EvidenceResult:
    """Ищет единственную операцию, определяющую исход.

    Проверка контрфактуальная: убрать операцию (или вернуть ей исходную
    трактовку) и пересчитать. Если вердикт изменился — операция решает.
    """
    base_src = LedgerAggregateSource(rows, disclosed=disclosed)
    base = baseline_status or run_test(test, base_src).status

    decisive: list[Candidate] = []
    for c in candidates:
        if c.revert:
            src = LedgerAggregateSource(rows, disclosed=disclosed,
                                        overrides={c.txn_id: c.revert})
        else:
            src = LedgerAggregateSource(rows, disclosed=disclosed, exclude={c.txn_id})
        if run_test(test, src).status != base:
            decisive.append(c)

    if not decisive:
        return EvidenceResult(txn_id=None, decisive=[])

    decisive.sort(key=lambda c: (-c.weight, c.txn_id))
    winner = decisive[0]
    problems: list[str] = []
    if len(decisive) > 1:
        same_weight = [c for c in decisive if c.weight == winner.weight]
        if len(same_weight) > 1:
            problems.append(
                f"решающих операций несколько с равным основанием "
                f"({winner.basis}): {[c.txn_id for c in same_weight]} — "
                f"выбрана первая, требуется ручная проверка"
            )
        else:
            problems.append(
                f"решающих операций несколько: "
                f"{[(c.txn_id, c.basis) for c in decisive]}; выбрана "
                f"с сильнейшим основанием {winner.basis}"
            )
    return EvidenceResult(
        txn_id=winner.txn_id,
        decisive=[c.txn_id for c in decisive],
        problems=problems,
        basis=winner.basis,
    )


def load_adjustment_notes(path: Path) -> list[dict]:
    """Собирает примечания аудитора из артефакта шага 7.

    ФОРМА АРТЕФАКТА — КОНТРАКТ, И ИМЕНИ ФАЙЛА МАЛО. Прежняя версия
    перебирала `data.values()` в надежде, что это словарь «заёмщик →
    примечания». Шаг 7 пишет иначе: сверху лежат `alarms`, `problems`
    и `scenarios`, поэтому перебор натыкался на список тревог и падал
    с `'list' object has no attribute 'get'`.

    Разбор ведётся от известной структуры, а не от догадки о ней.
    """
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        log.warning("ДОКАЗАТЕЛЬСТВА: артефакт корректировок не читается: %s", exc)
        return []

    scenarios = data.get("scenarios") if isinstance(data, dict) else None
    if not isinstance(scenarios, dict):
        return []

    notes: list[dict] = []
    for scenario, payload in scenarios.items():
        if not isinstance(payload, dict):
            continue
        for note in payload.get("notes", []):
            if isinstance(note, dict):
                # Заёмщик нужен дальше: доказательство ищется среди строк
                # ЕГО реестра, а примечания разных заёмщиков смешивать нельзя.
                notes.append({**note, "scenario_id": scenario})
    return notes


def run(paths: RunPaths) -> dict:
    from .compute import load_rows, load_tests

    results_path = paths.artifacts / A.RESULTS
    results = json.loads(results_path.read_text(encoding="utf-8"))

    ledger_path = paths.artifacts / A.LEDGER_FINAL
    if not ledger_path.exists():
        ledger_path = paths.artifacts / A.LEDGER_CLEAN
    rows = load_rows(ledger_path)
    tests = load_tests(paths.artifacts / A.COVENANTS)

    adjustments = load_adjustment_notes(paths.artifacts / A.AUDIT_ADJUSTMENTS)

    disclosed_path = paths.artifacts / A.DISCLOSED
    disclosed_all = (
        json.loads(disclosed_path.read_text(encoding="utf-8"))
        if disclosed_path.exists() else {}
    )

    by_scenario_rows: dict[str, list[Row]] = {}
    for r in rows:
        by_scenario_rows.setdefault(r.scenario_id, []).append(r)
    cands = candidates_from_artifacts(rows, adjustments)

    n_found = 0
    for scenario, scenario_tests in tests.items():
        for test in scenario_tests:
            cell = results.get(scenario, {}).get(test.point)
            if cell is None:
                continue
            res = find_evidence(
                test,
                by_scenario_rows.get(scenario, []),
                cands.get(scenario, []),
                disclosed=disclosed_all.get(scenario, {}),
                baseline_status=cell.get("status"),
            )
            cell["evidence_txn_id"] = res.txn_id
            cell["evidence_basis"] = res.basis
            cell["evidence_decisive"] = res.decisive
            cell["problems"] = list(cell.get("problems", [])) + res.problems
            if res.txn_id:
                n_found += 1

    results_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log.info("Доказательство найдено для %d ячеек", n_found)
    return results
