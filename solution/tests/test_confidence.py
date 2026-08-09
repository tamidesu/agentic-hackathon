"""Тесты шага 15: флаги уверенности.

Главная проверка — приёмочная: на публичном наборе флаги обязаны накрыть
ВСЕ ячейки, потерявшие баллы по ключу, и пометить не больше 20 из 36.
Список длиннее — не приоритизация; список с дырами — ложное спокойствие,
которое на приватном наборе никто не поймает.

Ключ здесь используется КАК ИЗМЕРЕНИЕ качества эвристик, а не как вход
пайплайна: сами флаги считаются без него.
"""
from __future__ import annotations

import json
import math
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline import artifacts as A  # noqa: E402
from pipeline import confidence  # noqa: E402
from pipeline.confidence import CellRisk, assess  # noqa: E402
from pipeline.config import RunPaths  # noqa: E402

SNAPSHOT = ROOT / "fixtures" / "baseline" / "artifacts"
KEY = ROOT / "eval" / "ground_truth.json"


def _cell(status="COMPLIANT", actual=1.0, threshold=1.0, unit="amount",
          trace=(), problems=(), evidence=None, basis=None):
    return {
        "status": status, "actual": actual, "threshold": threshold,
        "unit": unit, "trace": list(trace), "problems": list(problems),
        "evidence_txn_id": evidence, "evidence_basis": basis,
    }


# --------------------------------------------------------------------------- #
# Отдельные признаки
# --------------------------------------------------------------------------- #


def test_far_from_threshold_flags():
    """Банк ставит порог рядом с ожидаемым значением: расхождение
    в 17 раз — ошибка расчёта, а не грандиозное нарушение."""
    risks = assess({"X": {"6.1": _cell(actual=17.44, threshold=1.0)}})
    assert risks[0].flagged
    assert any("далеко" in s for s in risks[0].signals)


def test_near_threshold_within_scorer_tolerance_flags():
    risks = assess({"X": {"6.1": _cell(actual=1.02, threshold=1.0)}})
    assert risks[0].flagged
    assert any("вплотную" in s for s in risks[0].signals)


def test_the_5_to_10_percent_band_ranks_but_does_not_flag():
    """Жёлтая зона: видима в ранге, не тратит бюджет флагов."""
    risks = assess({"X": {"6.1": _cell(actual=1.08, threshold=1.0)}})
    assert not risks[0].flagged
    assert risks[0].risk > 0


def test_empty_aggregate_flags():
    risks = assess({"X": {"6.1": _cell(
        problems=["агрегат capex/group пуст — вероятно расхождение словаря"])}})
    assert risks[0].flagged


def test_nan_actual_flags():
    risks = assess({"X": {"6.1": _cell(actual=float("nan"))}})
    assert risks[0].flagged


def test_fallback_cell_flags():
    risks = assess({"X": {"6.1": _cell()}},
                   assembly_report={"cells_fallback": ["X/6.1"]})
    assert risks[0].flagged


def test_ill_conditioned_ratio_flags():
    """Форма B1/6.1: (выручка − расходы) / проценты. За доминирующим
    агрегатом стоит сравнимая масса — ошибка входа усиливается кратно."""
    trace = [
        {"category": "revenue", "scope": "borrower", "party": None, "value": 11.4e6},
        {"category": "opex", "scope": "borrower", "party": None, "value": 7.1e6},
        {"category": "interest", "scope": "borrower", "party": None, "value": 1.5e6},
    ]
    risks = assess({"X": {"6.1": _cell(actual=2.81, threshold=2.0,
                                       unit="ratio", trace=trace)}})
    assert risks[0].flagged
    assert any("хрупкая разность" in s for s in risks[0].signals)


def test_a_single_aggregate_ratio_is_not_ill_conditioned():
    trace = [{"category": "opex", "scope": "borrower", "party": None, "value": 5e6}]
    risks = assess({"X": {"6.1": _cell(actual=1.5, threshold=1.0,
                                       unit="ratio", trace=trace)}})
    assert not any("хрупкая" in s for s in risks[0].signals)


def test_duplicate_traces_are_not_double_counted():
    """Условие springing-теста обращается к тому же агрегату второй раз —
    это не вторая масса."""
    t = {"category": "financing_inflow", "scope": "borrower", "party": None,
         "value": 5e6}
    risks = assess({"X": {"6.1": _cell(actual=1.5, threshold=1.0,
                                       unit="ratio", trace=[t, dict(t)])}})
    assert not any("хрупкая" in s for s in risks[0].signals)


def test_aggregate_that_is_a_corpus_outlier_flags():
    """Форма P5: «выручка» из двух строк при медиане корпуса в одну —
    вторая оказалась платежом консультанту по налогам. Заёмщики устроены
    одинаково, поэтому вдвое больший состав статьи требует объяснения."""
    from collections import Counter

    counts = {f"S{i}": Counter({"revenue": 1, "opex": 2}) for i in range(11)}
    counts["P5"] = Counter({"revenue": 2, "opex": 2})
    trace = [{"category": "revenue", "scope": "borrower", "party": None,
              "value": 10.3e6, "txn_ids": ["T1", "T2"]}]
    risks = {r.where: r for r in assess(
        {"P5": {"6.2": _cell(actual=10.3e6, threshold=7.5e6, trace=trace)}},
        counts=counts)}
    assert risks["P5/6.2"].flagged
    assert any("выброс против корпуса" in s for s in risks["P5/6.2"].signals)


def test_a_typical_aggregate_is_not_an_outlier():
    from collections import Counter

    counts = {f"S{i}": Counter({"revenue": 1}) for i in range(12)}
    trace = [{"category": "revenue", "scope": "borrower", "party": None,
              "value": 7e6, "txn_ids": ["T1"]}]
    risks = assess({"S1": {"6.2": _cell(actual=7e6, threshold=5e6, trace=trace)}},
                   counts=counts)
    assert not any("выброс" in s for s in risks[0].signals)


def test_outlier_signal_is_silent_on_a_tiny_corpus():
    """На двух заёмщиках медиана ничего не значит — сигнал молчит,
    а не сыплет ложными тревогами."""
    from collections import Counter

    counts = {"A": Counter({"revenue": 1}), "B": Counter({"revenue": 9})}
    trace = [{"category": "revenue", "scope": "borrower", "party": None,
              "value": 1.0, "txn_ids": [f"T{i}" for i in range(9)]}]
    risks = assess({"B": {"6.1": _cell(trace=trace)}}, counts=counts)
    assert not any("выброс" in s for s in risks[0].signals)


def test_verdict_hanging_on_one_kyc_txn_flags():
    """Сопоставление имён с досье KYC — самый шаткий шаг пайплайна;
    ячейка, чей исход контрфактуально держится на одной такой операции,
    обязана попасть в разбор."""
    risks = assess({"X": {"6.1": _cell(status="BREACH", actual=0.09,
                                       threshold=0.08, unit="ratio",
                                       evidence="TXN-1", basis="related_party")}})
    assert risks[0].flagged
    assert any("KYC" in s for s in risks[0].signals)


def test_group_gap_spills_over_the_scenario():
    """Невычислимая ячейка уровня Группы ставит под сомнение периметр
    отчётности заёмщика целиком — соседние ячейки тоже в разбор."""
    results = {"X": {
        "6.1": _cell(actual=0.0, threshold=1.0,
                     problems=["агрегат capex/group пуст — расхождение"]),
        "6.2": _cell(actual=10.3e6, threshold=7.5e6),
    }}
    risks = {r.where: r for r in assess(results)}
    assert risks["X/6.2"].flagged
    assert any("Группы" in s for s in risks["X/6.2"].signals)


def test_addressing_miss_flags_the_scenario():
    apply_report = {"skipped_notes": [
        "X п.7.1: не найдена операция (txn=None, контрагент=None, сумма=1.0)"]}
    risks = assess({"X": {"6.1": _cell()}}, apply_report=apply_report)
    assert risks[0].flagged
    assert any("промах адресации" in s for s in risks[0].signals)


def test_unknown_data_from_step_reports_flags():
    risks = assess({"X": {"6.1": _cell()}},
                   scenario_problems={"X": ["X: валюта 'KZT' вне словаря"]})
    assert risks[0].flagged


def test_breach_without_evidence_ranks_but_does_not_flag():
    risks = assess({"X": {"6.1": _cell(status="BREACH", actual=0.5,
                                       threshold=1.0)}})
    assert not risks[0].flagged
    assert risks[0].risk > 0


def test_the_list_is_sorted_by_risk():
    results = {"X": {
        "6.1": _cell(),                                  # чисто
        "6.2": _cell(actual=17.0, threshold=1.0),        # далеко
        "6.3": _cell(actual=1.01, threshold=1.0),        # вплотную
    }}
    risks = assess(results)
    assert [r.risk for r in risks] == sorted((r.risk for r in risks), reverse=True)
    assert risks[0].point == "6.2"


# --------------------------------------------------------------------------- #
# Приёмка на публичном наборе
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def offline_run(tmp_path_factory):
    """Шаги 7б и 11–13 поверх копии снимка — как в score_offline.py."""
    from pipeline import apply, compute, disclosed, evidence
    from pipeline.config import RunPaths

    if not SNAPSHOT.exists() or not KEY.exists():
        pytest.skip("нет снимка или ключа")
    rp = RunPaths.create(tmp_path_factory.mktemp("confidence"))
    shutil.copytree(SNAPSHOT, rp.artifacts, dirs_exist_ok=True)
    disclosed.run(rp)
    apply.run(rp)
    compute.run(rp)
    evidence.run(rp)
    return rp


def _lost_cells(rp) -> set[str]:
    """Ячейки, теряющие баллы по ключу, — считаются скорером eval/score.py."""
    sys.path.insert(0, str(ROOT / "eval"))
    from score import score  # type: ignore

    results = json.loads((rp.artifacts / A.RESULTS).read_text(encoding="utf-8"))
    answers = {
        s: {p: {"status": c.get("status"),
                "actual": None if c.get("actual") is None or (
                    isinstance(c.get("actual"), float) and math.isnan(c["actual"]))
                else round(abs(c["actual"]), 2),
                "evidence_txn_id": c.get("evidence_txn_id")}
            for p, c in cells.items()}
        for s, cells in results.items()
    }
    rep = score({"answers": answers}, json.loads(KEY.read_text(encoding="utf-8")))
    return {f"{c.scenario}/{c.point}" for c in rep.cells if c.points < 0.999}


def test_flags_cover_every_lost_cell_within_budget(offline_run):
    """Приёмочный критерий задания: все неполные ячейки накрыты,
    помечено не больше 20 из 36. Ключ — измерение, не вход."""
    risks = confidence.run(offline_run)
    flagged = {r.where for r in risks if r.flagged}
    lost = _lost_cells(offline_run)

    missed = lost - flagged
    assert missed == set(), (
        f"неполные ячейки без флага: {sorted(missed)} — "
        f"на приватном наборе их никто не откроет"
    )
    assert len(flagged) <= confidence.FLAG_BUDGET, (
        f"помечено {len(flagged)} из {len(risks)} — список перестал "
        f"быть приоритизацией"
    )


@pytest.mark.slow
def test_flags_cover_every_lost_cell_on_the_live_run(corpus_report, public_dataset,
                                                     tmp_path):
    """Приёмка в той конфигурации, которая будет завтра: снимок дорогих
    шагов ПЛЮС живые тексты корпуса.

    Разница не косметическая. На снимке нет 01_texts, поэтому шаг 9б
    не выводит показатели Группы, и ячейки P5 держатся флагом «пустой
    агрегат». На живом прогоне агрегат наполняется — и P5/6.2 теряла
    флаг вместе с ним: покрытие 12 из 13 вместо полного. Проверять
    покрытие только на снимке значит не проверять его вовсе.
    """
    from pipeline import apply, attribute, compute, disclosed, entities, evidence

    _, rp = corpus_report
    run = RunPaths.create(tmp_path / "live")
    shutil.copytree(rp.artifacts, run.artifacts, dirs_exist_ok=True)
    attribute.run(public_dataset, run)
    shutil.copytree(SNAPSHOT, run.artifacts, dirs_exist_ok=True)

    entities.run(run)
    disclosed.run(run)
    apply.run(run)
    compute.run(run)
    evidence.run(run)

    risks = confidence.run(run)
    flagged = {r.where for r in risks if r.flagged}
    lost = _lost_cells(run)

    assert lost, "на публичном наборе потери есть — иначе проверять нечего"
    assert lost - flagged == set(), (
        f"на живом прогоне без флага остались: {sorted(lost - flagged)}"
    )
    assert len(flagged) <= confidence.FLAG_BUDGET


def test_confidence_only_observes(offline_run):
    """Шаг 15 не имеет права менять расчёт: артефакт результатов
    до и после — байт в байт."""
    results_path = offline_run.artifacts / A.RESULTS
    before = results_path.read_bytes()
    confidence.run(offline_run)
    assert results_path.read_bytes() == before
    assert (offline_run.artifacts / A.CONFIDENCE).exists()
