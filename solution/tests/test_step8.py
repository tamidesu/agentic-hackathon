"""Тесты шага 8: связанные стороны из досье KYC.

Главный риск шага — не падение, а ЗАНИЖЕННЫЙ агрегат. Связанная сторона,
которую не опознали, уносит с собой свои платежи, `actual` выходит меньше
настоящего, и ковенант «не более 0.04x выручки» показывает COMPLIANT там,
где на деле BREACH. Ни одна из таких ошибок не выглядит как ошибка,
поэтому проверяется каждое место, где число может потеряться.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline import related  # noqa: E402
from pipeline.llm import LLMClient, MockProvider  # noqa: E402
from pipeline.related import Party, ScenarioParties, decide_related  # noqa: E402
from pipeline.schemas import validate_related_parties  # noqa: E402

PUBLIC = ROOT.parent / "agentic-bank-public"
needs_public = pytest.mark.skipif(not PUBLIC.exists(), reason="нет публичного датасета")


# --------------------------------------------------------------------------- #
# Правило признания связанной стороной
# --------------------------------------------------------------------------- #


def test_threshold_is_inclusive():
    """В досье написано «20.0% И БОЛЕЕ». Строгое сравнение потеряло бы
    участника с долей ровно 20.0% — и его платежи вместе с ним."""
    assert decide_related(20.0, 20.0, None) is True


def test_below_threshold_is_not_related():
    assert decide_related(19.99, 20.0, None) is False


def test_other_basis_wins_without_a_share():
    """Связанность бывает не только через долю: общий контроль, прямое
    указание в тексте."""
    assert decide_related(None, None, "общий контроль") is True
    assert decide_related(5.0, 40.0, "прямое указание в досье") is True


def test_missing_threshold_means_no_decision():
    """Без порога сравнивать не с чем. Признать связанной «на всякий
    случай» значило бы завысить агрегат — ошибка в другую сторону,
    но такая же."""
    assert decide_related(46.8, None, None) is False


@pytest.mark.parametrize("threshold", [20.0, 25.0, 30.0, 32.0, 34.0, 35.0, 36.0, 38.0, 40.0])
def test_every_observed_threshold_works(threshold):
    """Все девять значений, встреченные в публичном наборе."""
    assert decide_related(threshold, threshold, None)
    assert not decide_related(threshold - 0.1, threshold, None)


# --------------------------------------------------------------------------- #
# Валидатор: сверка мнения модели с расчётом
# --------------------------------------------------------------------------- #


def _payload(**over):
    base = {
        "has_ownership_section": True,
        "threshold_pct": 20.0,
        "parties": [
            {"name": "Aktau Holdings LLP", "ownership_pct": 34.5, "is_related": True,
             "quote": "Aktau Holdings LLP 34.5%"},
        ],
    }
    base.update(over)
    return base


def test_agreeing_answer_passes():
    assert validate_related_parties(_payload()) == []


def test_disagreement_between_model_and_arithmetic_is_reported():
    """Расхождение означает, что доля или порог прочитаны неверно."""
    payload = _payload(parties=[
        {"name": "X LLP", "ownership_pct": 34.5, "is_related": False, "quote": "q"}])
    problems = validate_related_parties(payload)
    assert problems and "34.5" in problems[0]


def test_disagreement_is_allowed_when_the_basis_explains_it():
    payload = _payload(parties=[
        {"name": "X LLP", "ownership_pct": 5.0, "is_related": True,
         "basis": "общий контроль", "quote": "q"}])
    assert validate_related_parties(payload) == []


def test_threshold_out_of_range_is_rejected():
    assert validate_related_parties(_payload(threshold_pct=140.0))
    assert validate_related_parties(_payload(threshold_pct=0))
    assert validate_related_parties(_payload(threshold_pct=None))


def test_share_out_of_range_is_rejected():
    payload = _payload(parties=[
        {"name": "X", "ownership_pct": 460.0, "is_related": True, "quote": "q"}])
    assert validate_related_parties(payload)


def test_document_without_an_ownership_section_is_legitimate():
    """Реальный случай P2: раздела о владении нет вовсе. Это значит
    «связанных сторон не заявлено», а не «извлечение провалилось»."""
    payload = {"has_ownership_section": False, "threshold_pct": None, "parties": []}
    assert validate_related_parties(payload) == []


def test_no_section_but_parties_listed_is_contradictory():
    payload = {"has_ownership_section": False, "threshold_pct": None,
               "parties": [{"name": "X", "ownership_pct": 50.0, "is_related": True,
                            "quote": "q"}]}
    assert validate_related_parties(payload)


# --------------------------------------------------------------------------- #
# Извлечение
# --------------------------------------------------------------------------- #

KYC_TEXT = (
    "Досье «Знай своего клиента» (KYC)\n"
    "Проверка связанных сторон · Aktau Port Services JSC\n"
    "Бенефициарное владение и контроль\n"
    "Организация Доля голосующих прав\n"
    "Aktau Holdings LLP 34.5%\n"
    "Kaspi Marine Engineering LLP 18.7%\n"
    "Ural Crane Works LLP 6.2%\n"
    "Организации, в которых Группа владеет 20.0% и более голосующих прав, "
    "признаются связанными сторонами для целей Договора.\n"
)


def _client(responses, tmp_path):
    mock = MockProvider()
    state = {"i": 0}

    def reply(req):
        i = min(state["i"], len(responses) - 1)
        state["i"] += 1
        return responses[i]

    mock.register_rule(lambda r: True, reply)
    return LLMClient(cache_dir=tmp_path / "c", provider=mock), mock


FULL_ANSWER = {
    "has_ownership_section": True,
    "threshold_pct": 20.0,
    "threshold_quote": "Группа владеет 20.0% и более голосующих прав",
    "parties": [
        {"name": "Aktau Holdings LLP", "ownership_pct": 34.5, "is_related": True,
         "quote": "Aktau Holdings LLP 34.5%"},
        {"name": "Kaspi Marine Engineering LLP", "ownership_pct": 18.7,
         "is_related": False, "quote": "Kaspi Marine Engineering LLP 18.7%"},
        {"name": "Ural Crane Works LLP", "ownership_pct": 6.2, "is_related": False,
         "quote": "Ural Crane Works LLP 6.2%"},
    ],
}


def test_extraction_decides_by_code_not_by_the_model(tmp_path):
    client, _ = _client([FULL_ANSWER], tmp_path)
    result = related.extract_one("P1", "d1", KYC_TEXT, client)
    assert result.threshold_pct == 20.0
    assert result.related_names() == ["Aktau Holdings LLP"]
    assert result.problems == []


def test_code_overrides_a_wrong_model_verdict(tmp_path):
    """Модель ошиблась вердиктом при верных числах. Решает арифметика,
    а расхождение попадает в примечания."""
    answer = json.loads(json.dumps(FULL_ANSWER))
    answer["parties"][1]["is_related"] = True  # 18.7% при пороге 20% — нет
    client, _ = _client([answer, FULL_ANSWER], tmp_path)
    result = related.extract_one("P1", "d1", KYC_TEXT, client)
    assert "Kaspi Marine Engineering LLP" not in result.related_names()


def test_missing_threshold_with_a_section_is_a_problem(tmp_path):
    answer = {"has_ownership_section": True, "threshold_pct": None, "parties": []}
    client, _ = _client([answer], tmp_path)
    result = related.extract_one("P1", "d1", KYC_TEXT, client)
    assert result.problems
    assert any("порог" in p for p in result.problems)


def test_absent_section_is_not_a_problem(tmp_path):
    """P2: агрегат честно равен нулю, и это не повод для тревоги."""
    answer = {"has_ownership_section": False, "threshold_pct": None, "parties": []}
    client, _ = _client([answer], tmp_path)
    result = related.extract_one("P2", "d2", "Досье без раздела о владении.", client)
    assert result.problems == []
    assert result.related_names() == []


def test_invented_party_is_rejected_by_the_quote_check(tmp_path):
    answer = json.loads(json.dumps(FULL_ANSWER))
    answer["parties"].append({"name": "Выдуманная LLP", "ownership_pct": 99.0,
                              "is_related": True, "quote": "Выдуманная LLP 99.0%"})
    client, mock = _client([answer], tmp_path)
    related.extract_one("P1", "d1", KYC_TEXT, client)
    assert len(mock.calls) > 1, "должна была быть попытка исправления"


def test_prompt_warns_against_a_default_threshold():
    """Самая дорогая ошибка шага — подставить привычные 40%."""
    assert "РАЗНЫЙ" in related._PROMPT
    assert "не подставляй привычное значение" in related._PROMPT


# --------------------------------------------------------------------------- #
# Связывание с реестром
# --------------------------------------------------------------------------- #


def test_ledger_name_with_a_suffix_is_matched():
    """В реестре к названию прирастают уточнения вида «(Taraz yard)»."""
    parties = ScenarioParties("P6", parties=[
        Party("Taraz Holding Group LLP", 46.8, is_related=True)])
    matched, notes = match = related.match_against_ledger(
        parties, ["Taraz Holding Group LLP (Taraz yard)", "Прочие Поставки"])
    assert matched == {"Taraz Holding Group LLP (Taraz yard)": "Taraz Holding Group LLP"}
    assert notes == []


def test_unrelated_counterparty_is_not_matched():
    parties = ScenarioParties("P6", parties=[
        Party("Ural Grinding Works LLP", 11.5, is_related=False)])
    matched, _ = related.match_against_ledger(parties, ["Ural Grinding Works LLP"])
    assert matched == {}


def test_a_related_party_absent_from_the_ledger_is_reported():
    """Либо платежей не было, либо название не совпало. Первое безобидно,
    второе занижает агрегат — поэтому сообщается всегда."""
    parties = ScenarioParties("P6", parties=[
        Party("Совсем Другая LLP", 80.0, is_related=True)])
    _, notes = related.match_against_ledger(parties, ["Кто-то Ещё LLP"])
    assert notes and "не найдена" in notes[0]


# --------------------------------------------------------------------------- #
# Тревоги
# --------------------------------------------------------------------------- #


def test_alarm_when_no_dossier_has_an_ownership_section():
    report = related.RelatedReport(scenarios=[
        ScenarioParties("P1"), ScenarioParties("P2"),
    ])
    assert any("НИ В ОДНОМ" in a for a in report.alarms())


def test_alarm_when_every_threshold_is_identical():
    """В публичном наборе пороги различались у каждого заёмщика. Одинаковые
    пороги у всех — повод заподозрить подстановку значения по умолчанию."""
    report = related.RelatedReport(scenarios=[
        ScenarioParties(f"P{i}", has_ownership_section=True, threshold_pct=40.0)
        for i in range(1, 6)
    ])
    assert any("по умолчанию" in a for a in report.alarms())


def test_no_alarm_when_thresholds_differ():
    report = related.RelatedReport(scenarios=[
        ScenarioParties("P1", has_ownership_section=True, threshold_pct=20.0),
        ScenarioParties("P2", has_ownership_section=True, threshold_pct=35.0),
        ScenarioParties("P3", has_ownership_section=True, threshold_pct=40.0),
        ScenarioParties("P4", has_ownership_section=True, threshold_pct=25.0),
    ])
    assert report.alarms() == []


def test_report_survives_a_failed_scenario():
    report = related.RelatedReport(scenarios=[
        ScenarioParties("P1", has_ownership_section=True, threshold_pct=20.0,
                        parties=[Party("A LLP", 50.0, is_related=True)]),
        ScenarioParties("P2", problems=["извлечение упало: LLMError"]),
    ])
    data = json.loads(json.dumps(report.to_dict(), ensure_ascii=False))
    assert data["scenarios"]["P1"]["related_names"] == ["A LLP"]
    assert data["scenarios"]["P2"]["related_names"] == []


# --------------------------------------------------------------------------- #
# На реальных досье — без единого вызова модели
# --------------------------------------------------------------------------- #


@pytest.mark.slow
@needs_public
def test_every_dossier_states_its_own_threshold(attributed, corpus_report):
    """Проверка допущения, на котором держится весь шаг: порог написан
    в тексте и у разных заёмщиков РАЗНЫЙ. Если это перестанет быть так,
    тест обязан заметить это раньше, чем расчёт."""
    import re

    from pipeline.classify import DocType

    docs, _ = attributed
    _, rp = corpus_report

    thresholds: dict[str, float] = {}
    for doc_id, d in sorted(docs.items()):
        if d.type != DocType.KYC or not d.scenario_id:
            continue
        text = (rp.artifacts / "01_texts" / f"{doc_id}.txt").read_text(encoding="utf-8")
        found = re.search(r"владеет\s+([\d.]+)%\s+и более", re.sub(r"\s+", " ", text))
        if found:
            thresholds[d.scenario_id] = float(found.group(1))

    assert len(thresholds) >= 10, f"пороги найдены только у {len(thresholds)} заёмщиков"
    assert len(set(thresholds.values())) > 1, (
        f"все пороги одинаковы ({set(thresholds.values())}) — допущение шага неверно"
    )


def test_a_missing_section_flag_does_not_silently_disable_the_checks():
    """Отсутствие поля и значение false — не одно и то же. Раньше ответ
    без этого поля молча пропускал ВСЕ последующие проверки: и порог,
    и доли, и сверку с мнением модели."""
    payload = {"threshold_pct": 999.0,
               "parties": [{"name": "X", "ownership_pct": 500.0, "is_related": False,
                            "quote": "q"}]}
    problems = validate_related_parties(payload)
    assert problems and "has_ownership_section" in problems[0]
