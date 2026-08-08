"""Тесты слоя сущностей и связей.

39% ячеек зависят от вопроса «кем приходится этот контрагент заёмщику».
Здесь проверяется, что связь устанавливается устойчиво к тому, как имя
записано в разных документах, и что многошаговые выводы прослеживаются.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline import artifacts as A  # noqa: E402
from pipeline.compute import Row  # noqa: E402
from pipeline.config import RunPaths  # noqa: E402
from pipeline.entities import (  # noqa: E402
    Entity,
    EntityGraph,
    EntityIndex,
    build_graph,
    core_name,
    derive_group_capex,
    explain,
    find_group_parent,
    normalize_entity_name,
    tag_rows,
)
from pipeline.entities import run as entities_run  # noqa: E402


def row(txn_id, cp, amount=-100.0, cat="opex"):
    return Row(txn_id=txn_id, scenario_id="P1", date="2025-06-01", counterparty=cp,
               amount_usd=amount, category=cat)


# --------------------------------------------------------------------------- #
# Нормализация названий
# --------------------------------------------------------------------------- #


def test_quotes_are_stripped():
    """Реальный случай P6: в отсканированном досье единственная связанная
    сторона записана как «"Taraz Holding Group" LLP», а в реестре — без
    кавычек. Без нормализации ковенант теряет её целиком."""
    assert (normalize_entity_name('"Taraz Holding Group" LLP')
            == normalize_entity_name("Taraz Holding Group LLP"))


def test_all_quote_styles_are_handled():
    for quoted in ['«Alpha» LLP', '„Alpha" LLP', "'Alpha' LLP", '“Alpha” LLP']:
        assert normalize_entity_name(quoted) == normalize_entity_name("Alpha LLP")


def test_parenthetical_suffix_is_stripped():
    """Реестр добавляет к названию уточнение площадки."""
    assert (normalize_entity_name("Foxridge Telecom LP Holding (Ekibastuz block B)")
            == normalize_entity_name("Foxridge Telecom LP Holding"))


def test_case_and_whitespace_are_normalised():
    assert normalize_entity_name("  TARAZ   KILN\nSERVICES  LLP ") == "taraz kiln services llp"


def test_core_name_drops_legal_form():
    assert core_name("Taraz Holding Group LLP") == "taraz holding group"
    assert core_name("Alpha JSC") == "alpha"


# --------------------------------------------------------------------------- #
# Сопоставление контрагентов
# --------------------------------------------------------------------------- #


def index(*names) -> EntityIndex:
    return EntityIndex([Entity(name=n, role="counterparty") for n in names])


def test_exact_match_wins():
    e, how = index("Taraz Holding Group LLP").match("Taraz Holding Group LLP")
    assert e is not None and how == "exact"


def test_quoted_kyc_name_matches_plain_ledger_name():
    e, how = index('"Taraz Holding Group" LLP').match("Taraz Holding Group LLP")
    assert e is not None and how == "exact"


def test_ledger_suffix_does_not_break_match():
    e, how = index("Ural Grinding Works LLP").match("Ural Grinding Works LLP (Taraz plant)")
    assert e is not None and how == "exact"


def test_different_legal_form_is_a_loose_match_not_silent():
    """«Alpha JSC» и «Alpha LLP» могут быть разными лицами — совпадение
    возможно, но обязано быть помечено."""
    e, how = index("Alpha JSC").match("Alpha LLP")
    assert e is not None and how == "core"


def test_ambiguous_core_name_is_refused():
    e, how = index("Alpha JSC", "Alpha LLP").match("Alpha Ltd")
    assert e is None and how == "ambiguous"


def test_unknown_counterparty_is_not_matched():
    e, how = index("Alpha LLP").match("Совершенно другая компания LLP")
    assert e is None and how == "none"


# --------------------------------------------------------------------------- #
# Разметка реестра
# --------------------------------------------------------------------------- #


def graph_with(*parties) -> EntityGraph:
    g = EntityGraph(scenario_id="P6", borrower="Taraz Cement Works JSC", threshold_pct=40.0)
    for name, pct in parties:
        g.entities.append(Entity(name=name, role="counterparty", ownership_pct=pct,
                                 is_related=pct >= 40.0, basis=f"доля {pct}%"))
    return g


def test_only_related_parties_are_tagged():
    """Порог решает, а похожесть названия — нет. «Taraz Cement Personnel LLP»
    выглядит внутригрупповой, но в досье её нет."""
    g = graph_with(('"Taraz Holding Group" LLP', 46.8), ("Taraz Kiln Services LLP", 38.1))
    rows = [row("TXN-P6-1", "Taraz Holding Group LLP"),
            row("TXN-P6-2", "Taraz Kiln Services LLP"),
            row("TXN-P6-3", "Taraz Cement Personnel LLP")]

    tag_rows(rows, g)
    assert rows[0].party == "related"
    assert rows[1].party is None, "38.1% ниже порога 40%"
    assert rows[2].party is None, "в досье отсутствует"


def test_loose_match_is_reported():
    g = graph_with(("Alpha JSC", 51.0))
    notes = tag_rows([row("TXN-P1-1", "Alpha LLP")], g)
    assert any("без учёта организационно-правовой формы" in n for n in notes)


def test_zero_matches_is_reported_as_naming_mismatch():
    g = graph_with(("Alpha LLP", 51.0))
    notes = tag_rows([row("TXN-P1-1", "Beta LLP")], g)
    assert any("расхождение в написании" in n for n in notes)


def test_ambiguity_is_reported_and_row_left_untagged():
    g = graph_with(("Alpha JSC", 51.0), ("Alpha LLP", 60.0))
    rows = [row("TXN-P1-1", "Alpha Ltd")]
    notes = tag_rows(rows, g)
    assert rows[0].party is None
    assert any("совпал с несколькими" in n for n in notes)


# --------------------------------------------------------------------------- #
# Показатели уровня Группы
# --------------------------------------------------------------------------- #

PPE_REPORT = (
    "Note 7 — Property, Plant and Equipment There were no disposals of property, "
    "plant and equipment during the year. Year ended 2025-12-31 "
    "Net book value at the beginning of the year $148,028,989.69 "
    "Depreciation charge for the year $15,826,229.43 "
    "Net book value at the end of the year $154,050,122.81"
)


def test_group_capex_is_derived_from_the_roll_forward():
    """Прямой строки «капитальные затраты Группы» в отчётности нет —
    значение выводится из движения основных средств."""
    value, why = derive_group_capex(PPE_REPORT)
    assert value == pytest.approx(21_847_362.55, abs=0.01)
    assert "выбытий не было" in why


def test_group_capex_is_refused_without_the_no_disposals_clause():
    """Формула «конец − начало + амортизация» верна только при отсутствии
    выбытий. Молча посчитать по неприменимой формуле хуже, чем не считать."""
    without = PPE_REPORT.replace("There were no disposals of property, "
                                 "plant and equipment during the year.", "")
    value, why = derive_group_capex(without)
    assert value is None and "выбытий" in why


def test_group_capex_is_refused_without_a_roll_forward():
    value, why = derive_group_capex("Обычный отчёт без движения основных средств")
    assert value is None and "не найдена свёртка" in why


def test_group_parent_is_found_by_subsidiary_mention():
    texts = {
        "report": (
            "SARYBEL ENERGY HOLDING JSC Consolidated Financial Statements. "
            "The Group's segment is conducted through Ekibastuz Power Services JSC, "
            "which operates generating plant."
        ),
        "noise": "Политика информационной безопасности",
    }
    parent, doc = find_group_parent("Ekibastuz Power Services JSC", texts)
    assert doc == "report" and parent and "SARYBEL" in parent.upper()


def test_own_documents_are_excluded_from_parent_search():
    texts = {"own": "Ekibastuz Power Services JSC Consolidated ALPHA HOLDING JSC"}
    parent, doc = find_group_parent("Ekibastuz Power Services JSC", texts,
                                    exclude={"own"})
    assert parent is None and doc is None


# --------------------------------------------------------------------------- #
# Сборка графа
# --------------------------------------------------------------------------- #


def test_related_flag_is_derived_from_threshold():
    kyc = {"threshold_pct": 40.0, "parties": [
        {"name": "Alpha LLP", "ownership_pct": 46.8},
        {"name": "Beta LLP", "ownership_pct": 38.1},
    ]}
    g = build_graph("P6", "Taraz Cement Works JSC", kyc, {})
    assert g.related_names() == ["Alpha LLP"]
    assert "доля 46.8% при пороге 40.0%" in g.entities[0].basis


def test_explicit_flag_from_the_dossier_overrides_the_threshold():
    kyc = {"threshold_pct": 40.0, "parties": [
        {"name": "Alpha LLP", "ownership_pct": 10.0, "is_related": True},
    ]}
    assert build_graph("P1", "X", kyc, {}).related_names() == ["Alpha LLP"]


def test_entity_without_share_or_flag_is_flagged():
    kyc = {"threshold_pct": 40.0, "parties": [{"name": "Alpha LLP"}]}
    g = build_graph("P1", "X", kyc, {})
    assert g.related_names() == []
    assert any("нет ни доли, ни признака" in p for p in g.problems)


def test_empty_dossier_is_reported():
    g = build_graph("P1", "X", {"threshold_pct": 40.0, "parties": []}, {})
    assert any("6.3 неразрешим" in p for p in g.problems)


def test_group_facts_reach_the_graph():
    texts = {"parent": "SARYBEL ENERGY HOLDING JSC Consolidated Financial Statements "
                       "through Ekibastuz Power Services JSC. " + PPE_REPORT}
    g = build_graph("P5", "Ekibastuz Power Services JSC",
                    {"threshold_pct": 40.0, "parties": []}, texts)
    assert g.group.parent and "SARYBEL" in g.group.parent.upper()
    assert g.group.values["capex"] == pytest.approx(21_847_362.55, abs=0.01)
    assert "амортизация" in g.group.derivations["capex"]


# --------------------------------------------------------------------------- #
# Прослеживаемость
# --------------------------------------------------------------------------- #


def test_explanation_is_a_readable_chain():
    g = graph_with(('"Taraz Holding Group" LLP', 46.8))
    g.entities[0].source_doc = "f3fa6d20c8a1"
    chain = explain("TXN-P6-0040", "Taraz Holding Group LLP", g)
    joined = " | ".join(chain)
    assert "TXN-P6-0040" in joined
    assert "f3fa6d20c8a1" in joined
    assert "связанная сторона" in joined


def test_explanation_says_when_entity_is_absent():
    g = graph_with(("Alpha LLP", 51.0))
    chain = explain("TXN-P1-1", "Taraz Cement Personnel LLP", g)
    assert any("не найден" in c for c in chain)


# --------------------------------------------------------------------------- #
# Артефакт
# --------------------------------------------------------------------------- #


def test_run_writes_artifact_even_without_step_eight(tmp_path):
    paths = RunPaths.create(tmp_path / "run")
    texts = paths.artifacts / "01_texts"
    texts.mkdir(parents=True)
    (texts / "d1.txt").write_text("текст", encoding="utf-8")
    (paths.artifacts / A.DOC_INDEX).write_text(json.dumps({
        "documents": {"d1": {"doc_id": "d1", "type": "KYC", "rule": None,
                             "confidence": 1.0, "scenario_id": "P1", "notes": []}}
    }), encoding="utf-8")

    graphs = entities_run(paths)
    assert set(graphs) == {"P1"}
    saved = json.loads((paths.artifacts / A.ENTITY_GRAPH).read_text(encoding="utf-8"))
    assert "P1" in saved and "related_names" in saved["P1"]


# --------------------------------------------------------------------------- #
# Точность поиска материнской компании
# --------------------------------------------------------------------------- #

from pipeline.entities import trim_leading_noise  # noqa: E402


def test_trim_leading_noise_strips_header_tail():
    """В шапке отчётности название соседствует с датой."""
    assert trim_leading_noise("DECEMBER 2025 Sarybel Energy Holding JSC") == \
        "Sarybel Energy Holding JSC"
    assert trim_leading_noise("2025 Alpha Beta LLP") == "Alpha Beta LLP"


def test_trim_does_not_eat_an_all_caps_name():
    assert trim_leading_noise("SARYBEL ENERGY HOLDING JSC") == "SARYBEL ENERGY HOLDING JSC"


def test_parent_requires_an_explicit_consolidation_marker():
    """Свободный поиск подцеплял название банка из шапки кредитного
    договора. Неверный родитель молча отравляет ковенант по Группе."""
    loan = ("КОНФИДЕНЦИАЛЬНО ДОГОВОР БАНКОВСКОГО ЗАЙМА Halyk Bank of Kazakhstan JSC "
            "Заёмщик Ekibastuz Power Services JSC входит в Группу")
    parent, doc = find_group_parent("Ekibastuz Power Services JSC", {"loan": loan})
    assert parent is None and doc is None


def test_parent_is_found_by_each_supported_phrasing():
    borrower = "Ekibastuz Power Services JSC"
    variants = {
        "a": "Sarybel Energy Holding JSC Consolidated Financial Statements ... "
             "Ekibastuz Power Services JSC",
        "b": "We have audited the consolidated financial statements of "
             "Sarybel Energy Holding JSC ... Ekibastuz Power Services JSC",
        "c": "Sarybel Energy Holding JSC and its subsidiaries ... "
             "Ekibastuz Power Services JSC",
    }
    for key, text in variants.items():
        parent, doc = find_group_parent(borrower, {key: text})
        assert parent == "Sarybel Energy Holding JSC", key


def test_borrower_is_never_its_own_parent():
    text = "Ekibastuz Power Services JSC Consolidated Financial Statements"
    assert find_group_parent("Ekibastuz Power Services JSC", {"x": text}) == (None, None)


def test_parent_report_attributed_to_the_borrower_is_still_searched():
    """Отчётность материнской компании называет дочернюю по имени, поэтому
    привязка относит её к тому же заёмщику. Исключать «свои» документы
    нельзя — иначе выбрасывается ровно нужный."""
    kyc = {"threshold_pct": 40.0, "parties": []}
    texts = {"parent_report": (
        "Sarybel Energy Holding JSC Consolidated Financial Statements. "
        "The Group's segment is conducted through Ekibastuz Power Services JSC. "
        + PPE_REPORT)}
    g = build_graph("P5", "Ekibastuz Power Services JSC", kyc, texts,
                    own_docs={"parent_report"})
    assert g.group.parent == "Sarybel Energy Holding JSC"
    assert g.group.values["capex"] == pytest.approx(21_847_362.55, abs=0.01)
