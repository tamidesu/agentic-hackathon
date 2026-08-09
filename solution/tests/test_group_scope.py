"""Тесты задачи «пустые агрегаты scope=group и party=unrestricted_subsidiary».

Две ячейки публичного набора (P5/6.1, P9/6.1) давали ровно 0.00: дерево
ковенанта просит разрез, которого нет в реестре. Здесь проверяется вся
цепочка починки:

  * шаг 9б читает артефакт шага 8 в его НАСТОЯЩЕЙ форме (scenarios внутри
    отчёта) — прежний код искал сценарии на верхнем уровне и молча
    получал пустой граф;
  * таблица обеспечительного покрытия дочерних организаций разбирается
    из досье KYC детерминированно;
  * показатель Группы доезжает до расчёта с пометкой о происхождении;
  * метка unrestricted_subsidiary доезжает до строк реестра и до
    доказательства.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline import artifacts as A  # noqa: E402
from pipeline import entities as entities_module  # noqa: E402
from pipeline.apply import FinalRow, apply_unrestricted  # noqa: E402
from pipeline.compute import LedgerAggregateSource, Row  # noqa: E402
from pipeline.config import RunPaths  # noqa: E402
from pipeline.entities import (  # noqa: E402
    Entity,
    EntityGraph,
    build_graph,
    parse_subsidiary_pledges,
)

SNAPSHOT = ROOT / "fixtures" / "baseline" / "artifacts"

#: Фрагмент настоящего досье P9 — вместе с мусором OCR («BHe» латиницей).
PLEDGE_TEXT = """
Обеспечительное покрытие дочерних организаций

Ниже приведена доля активов каждой дочерней организации, переданных в залог по Договору
обеспечения, по состоянию на дату проверки.

Дочерняя организация Доля активов в залоге
Zhezkazgan Conveyor Assets LLP 87.6%
Zhezkazgan Processing Holdings LLP 11.4%

Дочерние организации, у которых доля активов в залоге ниже 50.0%, находятся BHe периметра
обеспечения и для целей Договора рассматриваются как неограниченные.
"""


# --------------------------------------------------------------------------- #
# Разбор таблицы обеспечительного покрытия
# --------------------------------------------------------------------------- #


def test_pledge_table_is_parsed_with_rule():
    rows, below, problems = parse_subsidiary_pledges(PLEDGE_TEXT)
    assert rows == [("Zhezkazgan Conveyor Assets LLP", 87.6),
                    ("Zhezkazgan Processing Holdings LLP", 11.4)]
    assert below == 50.0
    assert problems == []


def test_ownership_rows_are_not_mistaken_for_pledges():
    """Строки долей участия выглядят так же — отличает только заголовок."""
    text = "Ulytau Capital LLP. 39.7%\nUral Haul Systems LLP 31.4%\n"
    rows, below, problems = parse_subsidiary_pledges(text)
    assert rows == [] and below is None and problems == []


def test_missing_rule_reports_and_derives_nothing():
    """Порог не найден — статус не выводится: назначить организацию
    неограниченной без основания хуже, чем спросить человека."""
    text = PLEDGE_TEXT.split("Дочерние организации, у которых")[0]
    rows, below, problems = parse_subsidiary_pledges(text)
    assert len(rows) == 2 and below is None
    assert any("правило" in p for p in problems)


def test_build_graph_marks_unrestricted_by_the_rule(tmp_path):
    g = build_graph("P9", "Zhezkazgan Mining Services JSC",
                    {"threshold_pct": 34.0, "parties": []}, {},
                    kyc_text=PLEDGE_TEXT)
    subs = {e.name: e.unrestricted for e in g.entities if e.role == "subsidiary"}
    assert subs == {"Zhezkazgan Conveyor Assets LLP": False,
                    "Zhezkazgan Processing Holdings LLP": True}


# --------------------------------------------------------------------------- #
# Форма артефакта шага 8 — регресс формы (третий случай в проекте)
# --------------------------------------------------------------------------- #


def test_step9b_reads_the_report_shape_of_step8(tmp_path):
    """Шаг 8 пишет {alarms, problems, scenarios: {...}}. Прежний код искал
    сценарии на верхнем уровне, находил пустоту, и граф — стороны,
    родитель, показатели Группы — молча выходил пустым."""
    paths = RunPaths.create(tmp_path / "run")
    (paths.artifacts / "01_texts").mkdir(parents=True)
    (paths.artifacts / A.DOC_INDEX).write_text(json.dumps({
        "documents": {"d1": {"doc_id": "d1", "type": "KYC", "rule": None,
                             "confidence": 1.0, "scenario_id": "P9", "notes": []}}
    }), encoding="utf-8")
    (paths.artifacts / "01_texts" / "d1.txt").write_text(PLEDGE_TEXT, encoding="utf-8")
    (paths.artifacts / A.RELATED_PARTIES).write_text(json.dumps({
        "alarms": [], "problems": [],
        "scenarios": {"P9": {"scenario_id": "P9", "doc_id": "d1",
                             "threshold_pct": 34.0,
                             "parties": [{"name": "Ulytau Capital LLP.",
                                          "ownership_pct": 39.7}]}},
    }, ensure_ascii=False), encoding="utf-8")

    graphs = entities_module.run(paths)

    assert graphs["P9"].related_names() == ["Ulytau Capital LLP."]
    assert any(e.unrestricted for e in graphs["P9"].entities)


def test_load_round_trips_and_tolerates_the_old_artifact(tmp_path):
    """Артефакт старого прогона — без поля unrestricted — не роняет load."""
    paths = RunPaths.create(tmp_path / "run")
    (paths.artifacts / A.ENTITY_GRAPH).write_text(json.dumps({
        "P1": {"scenario_id": "P1", "borrower": "X", "threshold_pct": 40.0,
               "entities": [{"name": "Alpha LLP", "role": "counterparty",
                             "ownership_pct": 46.8, "is_related": True,
                             "basis": "", "source_doc": ""}],
               "group": {"parent": None, "source_doc": None,
                         "values": {}, "derivations": {}},
               "problems": [], "related_names": ["Alpha LLP"]},
    }, ensure_ascii=False), encoding="utf-8")

    graphs = entities_module.load(paths)
    assert graphs["P1"].related_names() == ["Alpha LLP"]
    assert graphs["P1"].entities[0].unrestricted is False


# --------------------------------------------------------------------------- #
# Метка на строках и доказательство
# --------------------------------------------------------------------------- #


def _graph_with_subsidiary() -> EntityGraph:
    g = EntityGraph(scenario_id="P9")
    g.entities.append(Entity(
        name="Zhezkazgan Processing Holdings LLP", role="subsidiary",
        unrestricted=True, basis="доля активов в залоге 11.4% < 50.0%"))
    return g


def test_rows_to_unrestricted_subsidiaries_are_tagged():
    rows = [FinalRow(txn_id="T1", scenario_id="P9", date="2025-09-08",
                     counterparty="Zhezkazgan Processing Holdings LLP",
                     amount_usd=-418204.37)]
    tagged, problems = apply_unrestricted(rows, _graph_with_subsidiary())
    assert tagged == ["T1"] and problems == []
    assert rows[0].party == "unrestricted_subsidiary"


def test_related_label_wins_over_subsidiary_and_is_reported():
    """Метка связанной стороны нужна ковенанту 6.3, который есть у всех;
    конфликт ролей — повод для ручной проверки, а не молчаливого выбора."""
    rows = [FinalRow(txn_id="T1", scenario_id="P9", date="",
                     counterparty="Zhezkazgan Processing Holdings LLP",
                     amount_usd=-1.0, party="related")]
    tagged, problems = apply_unrestricted(rows, _graph_with_subsidiary())
    assert tagged == [] and rows[0].party == "related"
    assert any("одновременно" in p for p in problems)


def test_group_value_feeds_the_aggregate_with_a_note():
    src = LedgerAggregateSource([], group_values={"capex": 21_847_362.55})
    assert src.aggregate("capex", scope="group") == pytest.approx(21_847_362.55)
    assert src.missing_categories == set()
    assert any("консолидированной" in n for n in src.group_notes)


def test_group_rows_have_priority_over_the_derived_value():
    """Строки реестра — данные, выведенный показатель — вывод.
    Появятся строки со scope=group — они и считаются."""
    rows = [Row(txn_id="T1", scenario_id="P5", date="2025-01-01",
                counterparty="X", amount_usd=-100.0, category="capex",
                scope="group")]
    src = LedgerAggregateSource(rows, group_values={"capex": 999.0})
    assert src.aggregate("capex", scope="group") == 100.0
    assert src.group_notes == []


def test_without_group_values_the_aggregate_stays_loudly_empty():
    src = LedgerAggregateSource([])
    assert src.aggregate("capex", scope="group") == 0.0
    assert "capex/group" in src.missing_categories


def test_evidence_finds_the_unrestricted_transfer():
    """Контрфакт: убери признак «неограниченная» у единственного перевода —
    числитель обнулится и вердикт перевернётся. Это и есть доказательство."""
    from pipeline.covenant_types import CovenantTest
    from pipeline.evidence import Candidate, find_evidence

    test = CovenantTest(
        point="6.1", direction="max", threshold=0.15, metric={
            "op": "DIV", "args": [
                {"op": "AGG", "category": "capex", "party": "unrestricted_subsidiary"},
                {"op": "AGG", "category": "capex"},
            ]}, unit="ratio")
    rows = [
        Row("T1", "P9", "2025-09-08", "Processing", -418204.37, "capex",
            party="unrestricted_subsidiary"),
        Row("T2", "P9", "2025-11-12", "Conveyor", -302118.64, "capex"),
        Row("T3", "P9", "2025-05-01", "Ural", -1204663.28, "capex"),
    ]
    cands = [Candidate(txn_id="T1", basis="unrestricted_subsidiary",
                       revert={"party": None})]
    res = find_evidence(test, rows, cands)
    assert res.txn_id == "T1"
    assert res.basis == "unrestricted_subsidiary"


# --------------------------------------------------------------------------- #
# Сквозной прогон на настоящем корпусе
# --------------------------------------------------------------------------- #


@pytest.mark.slow
def test_both_cells_come_alive_on_the_real_corpus(corpus_report, public_dataset, tmp_path):
    """P5/6.1 получает капзатраты Группы из отчётности Sarybel, P9/6.1 —
    операцию, переданную неограниченной дочерней. Оба агрегата перестают
    быть пусто-нулевыми, происхождение видно в problems и evidence.

    Привязка (шаг 4) гоняется на СВОЕЙ копии артефактов: она дополняет
    индекс документов на месте, а общий сессионный прогон читают и другие
    тесты, в том числе проверяющие индекс ДО привязки."""
    from pipeline import apply, attribute, compute, disclosed, evidence

    _, rp = corpus_report
    run = RunPaths.create(tmp_path / "run")
    shutil.copytree(rp.artifacts, run.artifacts, dirs_exist_ok=True)
    attribute.run(public_dataset, run)
    shutil.copytree(SNAPSHOT, run.artifacts, dirs_exist_ok=True)

    graphs = entities_module.run(run)
    assert graphs["P5"].group.values.get("capex") == pytest.approx(21_847_362.55, abs=0.01)
    assert any(e.unrestricted for e in graphs["P9"].entities)

    disclosed.run(run)
    report = apply.run(run)
    assert "TXN-P9-0025" in report.subsidiary_tagged

    results = compute.run(run)
    p5 = next(c for c in results["P5"] if c.point == "6.1")
    assert p5.actual > 0
    assert any("консолидированной" in p for p in p5.problems)

    p9 = next(c for c in results["P9"] if c.point == "6.1")
    assert p9.status == "BREACH"
    assert p9.actual == pytest.approx(0.2173, abs=0.001)

    out = evidence.run(run)
    assert out["P9"]["6.1"]["evidence_txn_id"] == "TXN-P9-0025"
