"""Тесты шага 7: аудиторские корректировки.

Опасность этого шага — обратная остальным. Везде страшен пропуск, здесь —
ЛИШНЕЕ ПРИМЕНЕНИЕ. Приложение упоминает суммы, которые применять нельзя,
и выглядят они как обычные корректировки: с суммой, контрагентом и словом
«переклассификация». Применить такую — значит переложить сотни тысяч
долларов между статьями без основания, и расчёт этого не заметит.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline import adjustments  # noqa: E402
from pipeline.adjustments import Note, apply_materiality, find_section  # noqa: E402
from pipeline.llm import LLMClient, MockProvider  # noqa: E402
from pipeline.schemas import validate_audit_adjustments  # noqa: E402


# --------------------------------------------------------------------------- #
# Статус: применять или нет
# --------------------------------------------------------------------------- #


def _note(**over):
    base = dict(note_id="9.1", kind="reclassification", status="applied",
                from_category="opex", to_category="insurance", value_usd=142118.64)
    base.update(over)
    return Note(**base)


def test_an_applied_note_is_applied():
    assert _note().applies


@pytest.mark.parametrize("status", ["considered_but_rejected", "referred_elsewhere",
                                    "informational"])
def test_a_non_applied_note_is_not_applied(status):
    """Обе ловушки публичного набора: B1 отправляет вывод в другой отчёт,
    P10 сохраняет первоначальную классификацию."""
    assert not _note(status=status).applies


# --------------------------------------------------------------------------- #
# Порог существенности
# --------------------------------------------------------------------------- #

P4_ITEMS = [
    Note("8.1", "ebitda_adjustment", "applied", value_usd=251338.94),
    Note("8.2", "ebitda_adjustment", "applied", value_usd=342905.28),
    Note("8.3", "ebitda_adjustment", "applied", value_usd=481247.63),
]


def test_items_below_the_threshold_are_dropped():
    """Настоящие числа P4. Сумма всех трёх дала бы 1,075,491.85 вместо
    824,152.91 — расхождение в 30%."""
    notes = [Note(**vars(n)) for n in P4_ITEMS]
    apply_materiality(notes, 300000.0)
    total = sum(n.value_usd for n in notes if n.applies)
    assert total == pytest.approx(824152.91)
    assert [n.note_id for n in notes if n.applies] == ["8.2", "8.3"]


def test_the_dropped_item_says_why():
    notes = [Note(**vars(n)) for n in P4_ITEMS]
    apply_materiality(notes, 300000.0)
    dropped = notes[0]
    assert not dropped.applies
    assert "ниже порога" in dropped.skipped_because


def test_an_item_exactly_at_the_threshold_counts():
    """Написано «не менее $300,000.00» — граница нестрогая."""
    notes = [Note("8.1", "ebitda_adjustment", "applied", value_usd=300000.0)]
    apply_materiality(notes, 300000.0)
    assert notes[0].applies


def test_no_threshold_means_no_filtering():
    notes = [Note(**vars(n)) for n in P4_ITEMS]
    apply_materiality(notes, None)
    assert all(n.applies for n in notes)


def test_the_threshold_does_not_touch_reclassifications():
    """Аудитор либо сделал переклассификацию, либо нет. Её размер тут ни
    при чём: порог существенности назван для РАЗОВЫХ СТАТЕЙ EBITDA."""
    notes = [Note("9.1", "reclassification", "applied", value_usd=1000.0,
                  from_category="opex", to_category="insurance")]
    apply_materiality(notes, 300000.0)
    assert notes[0].applies


def test_an_item_without_an_amount_is_dropped_loudly():
    notes = [Note("8.1", "ebitda_adjustment", "applied", value_usd=None)]
    remarks = apply_materiality(notes, 300000.0)
    assert not notes[0].applies
    assert remarks and "сумма не указана" in remarks[0]


# --------------------------------------------------------------------------- #
# Обрезка раздела
# --------------------------------------------------------------------------- #


def test_section_starts_at_the_supplement():
    text = ("Примечание 1 — Основа подготовки\nдлинная учётная политика\n"
            "ДОПОЛНЕНИЕ О СОБЛЮДЕНИИ КОВЕНАНТОВ\nздесь важное")
    section, anchor = find_section(text)
    assert anchor == "ДОПОЛНЕНИЕ О СОБЛЮДЕНИИ КОВЕНАНТОВ"
    assert "учётная политика" not in section
    assert "здесь важное" in section


def test_english_supplement_is_found():
    section, anchor = find_section("Note 1\nCOVENANT COMPLIANCE SUPPLEMENT\nbody")
    assert anchor == "COVENANT COMPLIANCE SUPPLEMENT" and "body" in section


def test_missing_anchor_keeps_the_whole_document():
    """Документ невелик, и потерять раздел дороже, чем прочитать лишнее."""
    text = "Совсем другой заголовок\nтело документа"
    section, anchor = find_section(text)
    assert anchor is None and section == text


# --------------------------------------------------------------------------- #
# Валидатор
# --------------------------------------------------------------------------- #


def _payload(**over):
    base = {
        "notes": [{"note_id": "9.1", "kind": "reclassification", "status": "applied",
                   "from_category": "opex", "to_category": "insurance",
                   "value_usd": 142118.64, "description": "d", "quote": "q"}],
        "no_adjustments_stated": False,
    }
    base.update(over)
    return base


def test_a_valid_payload_passes():
    assert validate_audit_adjustments(_payload()) == []


def test_saying_nothing_changed_while_applying_something_is_caught():
    problems = validate_audit_adjustments(_payload(no_adjustments_stated=True))
    assert problems and "определись" in problems[0]


def test_a_rejected_note_needs_no_categories():
    """У неприменяемых примечаний суммы и статьи названы вскользь —
    требовать от них полноты значит провоцировать выдумывание."""
    payload = _payload(notes=[{"note_id": "7.2", "kind": "reclassification",
                               "status": "considered_but_rejected",
                               "description": "d", "quote": "q"}])
    assert validate_audit_adjustments(payload) == []


def test_an_unknown_status_is_caught():
    payload = _payload(notes=[{"note_id": "9.1", "kind": "reclassification",
                               "status": "выдумка", "description": "d", "quote": "q"}])
    assert validate_audit_adjustments(payload)


def test_a_negative_threshold_is_caught():
    assert validate_audit_adjustments(_payload(materiality_threshold_usd=-5.0))


# --------------------------------------------------------------------------- #
# Извлечение
# --------------------------------------------------------------------------- #

B1_TRAP = (
    "ДОПОЛНЕНИЕ О СОБЛЮДЕНИИ КОВЕНАНТОВ\n"
    "Примечание 9 — Переклассификации для целей соблюдения ковенантов\n"
    "(9.1) Сумма в размере $592,296.10, выплаченная контрагенту Irtysh Advisory "
    "Bureau, была отобрана для проверки классификации. Вывод по данной сумме "
    "изложен в отчёте о выполнении согласованных процедур № AR-2025-0634 и "
    "в настоящих примечаниях не повторяется.\n"
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


def test_a_referred_elsewhere_note_is_extracted_but_not_applied(tmp_path):
    """Настоящая ловушка B1: сумма, контрагент и слово «переклассификация»
    на месте, а вывода нет."""
    answer = {"notes": [{
        "note_id": "9.1", "kind": "reclassification", "status": "referred_elsewhere",
        "target_counterparty": "Irtysh Advisory Bureau", "value_usd": 592296.10,
        "description": "вывод в отчёте AR-2025-0634",
        "quote": "была отобрана для проверки классификации",
    }], "no_adjustments_stated": False}
    client, _ = _client([answer], tmp_path)
    result = adjustments.extract_one("B1", "d1", B1_TRAP, client)
    assert result.notes and result.applied() == []
    assert result.notes[0].skipped_because == "статус referred_elsewhere"


def test_an_out_of_vocabulary_category_is_reported(tmp_path):
    """Статья вне словаря не упадёт — она молча промахнётся мимо агрегата."""
    answer = {"notes": [{
        "note_id": "9.1", "kind": "reclassification", "status": "applied",
        "from_category": "Операционные расходы", "to_category": "insurance",
        "value_usd": 1.0, "description": "d", "quote": "ДОПОЛНЕНИЕ",
    }], "no_adjustments_stated": False}
    client, _ = _client([answer], tmp_path)
    result = adjustments.extract_one("P10", "d1", B1_TRAP, client)
    assert any("вне словаря" in p for p in result.problems)


def test_materiality_is_applied_during_extraction(tmp_path):
    answer = {
        "notes": [
            {"note_id": "8.1", "kind": "ebitda_adjustment", "status": "applied",
             "value_usd": 251338.94, "description": "d", "quote": "ДОПОЛНЕНИЕ"},
            {"note_id": "8.2", "kind": "ebitda_adjustment", "status": "applied",
             "value_usd": 342905.28, "description": "d", "quote": "ДОПОЛНЕНИЕ"},
        ],
        "materiality_threshold_usd": 300000.0,
        "no_adjustments_stated": False,
    }
    client, _ = _client([answer], tmp_path)
    result = adjustments.extract_one("P4", "d1", B1_TRAP, client)
    assert [n.note_id for n in result.applied()] == ["8.2"]
    assert result.remarks


def test_no_adjustments_is_a_clean_result(tmp_path):
    answer = {"notes": [], "no_adjustments_stated": True}
    client, _ = _client([answer], tmp_path)
    result = adjustments.extract_one("P5", "d1", "ДОПОЛНЕНИЕ О СОБЛЮДЕНИИ КОВЕНАНТОВ\n"
                                                 "Переклассификаций не требовалось.", client)
    assert result.problems == [] and result.applied() == []
    assert result.no_adjustments_stated


def test_prompt_names_both_traps():
    prompt = adjustments.build_prompt("x")
    assert "не повторяется" in prompt
    assert "первоначальная классификация сохраняется" in prompt
    assert "Не додумывай вывод за аудитора" in prompt


def test_prompt_carries_the_category_vocabulary():
    from pipeline.covenant_types import CATEGORIES

    prompt = adjustments.build_prompt("x")
    for category in CATEGORIES:
        assert category in prompt


# --------------------------------------------------------------------------- #
# Тревоги
# --------------------------------------------------------------------------- #


def test_alarm_when_a_scenario_is_silent():
    report = adjustments.AdjustmentsReport(scenarios=[
        adjustments.ScenarioAdjustments("P1", section_anchor="ДОПОЛНЕНИЕ",
                                        no_adjustments_stated=True),
        adjustments.ScenarioAdjustments("P2", section_anchor="ДОПОЛНЕНИЕ"),
    ])
    alarms = report.alarms()
    assert any("P2" in a for a in alarms)


def test_no_alarm_when_everyone_answered():
    report = adjustments.AdjustmentsReport(scenarios=[
        adjustments.ScenarioAdjustments("P1", section_anchor="ДОПОЛНЕНИЕ",
                                        no_adjustments_stated=True),
        adjustments.ScenarioAdjustments("P2", section_anchor="ДОПОЛНЕНИЕ",
                                        notes=[_note()]),
    ])
    assert report.alarms() == []
