"""Тесты шага 10: категоризация транзакций.

Два риска, и оба не выглядят как ошибка.

ПЕРВЫЙ — потеря строки. Пропущенная операция не роняет расчёт, она молча
уменьшает сумму агрегата. Поэтому проверяется, что ни на одном пути —
ни при сбое пакета, ни при неполном ответе, ни при выдуманном id — строка
не может исчезнуть.

ВТОРОЙ — разметка по названию контрагента вместо описания. В публичном
наборе 96 строк из 673 устроены так, что название указывает на одну статью,
а описание на другую. Четырнадцать процентов, разложенных не туда, сдвинут
агрегаты достаточно, чтобы часть вердиктов перевернулась.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline import artifacts as A  # noqa: E402
from pipeline import categorize  # noqa: E402
from pipeline.categorize import TxnCategory, chunk, format_rows  # noqa: E402
from pipeline.config import RunPaths  # noqa: E402
from pipeline.covenant_types import CATEGORIES  # noqa: E402
from pipeline.llm import LLMClient, MockProvider  # noqa: E402


def _row(txn_id, description, counterparty="Некто", amount=-1000.0):
    return {"txn_id": txn_id, "description": description, "counterparty": counterparty,
            "amount": amount, "currency": "USD"}


ROWS = [
    _row("TXN-P4-0056", "Revolver interest — November", "Cedarville Payroll LP"),
    _row("TXN-P6-0005", "Franchise tax filing", "Glenwood Property Trust"),
    _row("TXN-P1-0012", "Payroll for laboratory staff — April 2025", "Ironwood Power Trust"),
]


# --------------------------------------------------------------------------- #
# Промпт: описание важнее названия
# --------------------------------------------------------------------------- #


def test_prompt_states_the_trap_explicitly():
    """Четырнадцать процентов строк — намеренная ловушка. Молчать о ней
    в промпте значит рассчитывать на везение."""
    prompt = categorize.build_prompt(ROWS)
    assert "ОПИСАНИЕ, А НЕ НАЗВАНИЕ КОНТРАГЕНТА" in prompt
    assert "Cedarville Payroll LP" in prompt and "interest" in prompt


def test_prompt_lists_every_category():
    prompt = categorize.build_prompt(ROWS)
    for category in CATEGORIES:
        assert category in prompt


def test_description_comes_before_the_counterparty():
    """Порядок полей — тоже подсказка: важное впереди."""
    line = format_rows([ROWS[0]])
    assert line.index("Revolver interest") < line.index("Cedarville Payroll LP")


def test_a_missing_amount_is_stated_not_blanked():
    """Пустая сумма — известный случай (её восстанавливают из документов).
    Пустое место в строке модель может принять за ноль."""
    line = format_rows([_row("TXN-P7-0033", "Оплата", amount="")])
    assert "СУММА ОТСУТСТВУЕТ" in line


# --------------------------------------------------------------------------- #
# Пакеты
# --------------------------------------------------------------------------- #


def test_chunking_covers_everything_exactly_once():
    rows = [_row(f"T{i}", "d") for i in range(97)]
    batches = chunk(rows, 40)
    assert [len(b) for b in batches] == [40, 40, 17]
    seen = [r["txn_id"] for b in batches for r in b]
    assert seen == [r["txn_id"] for r in rows]


def test_zero_batch_size_is_rejected():
    with pytest.raises(ValueError):
        chunk(ROWS, 0)


# --------------------------------------------------------------------------- #
# Разметка пакета
# --------------------------------------------------------------------------- #


def _client(responses, tmp_path):
    mock = MockProvider()
    state = {"i": 0}

    def reply(req):
        i = min(state["i"], len(responses) - 1)
        state["i"] += 1
        return responses[i]

    mock.register_rule(lambda r: True, reply)
    return LLMClient(cache_dir=tmp_path / "c", provider=mock), mock


def _answer(*pairs):
    return {"items": [
        {"txn_id": t, "category": c, "flow": "outflow", "confidence": 0.9}
        for t, c in pairs
    ]}


GOOD = _answer(("TXN-P4-0056", "interest"), ("TXN-P6-0005", "taxes"),
               ("TXN-P1-0012", "payroll"))


def test_a_complete_batch_is_accepted(tmp_path):
    client, _ = _client([GOOD], tmp_path)
    marked, notes = categorize.categorize_batch(ROWS, client)
    assert set(marked) == {r["txn_id"] for r in ROWS}
    assert marked["TXN-P4-0056"].category == "interest"
    assert notes == []


def test_an_invented_transaction_is_dropped(tmp_path):
    """Придуманная строка искажает счёт: её нет в реестре, а сумма
    у неё возьмётся ниоткуда."""
    answer = json.loads(json.dumps(GOOD))
    answer["items"].append({"txn_id": "TXN-ВЫДУМКА", "category": "opex",
                            "flow": "outflow", "confidence": 1.0})
    client, _ = _client([answer, GOOD], tmp_path)
    marked, _ = categorize.categorize_batch(ROWS, client)
    assert "TXN-ВЫДУМКА" not in marked


def test_a_duplicate_row_is_counted_once(tmp_path):
    answer = json.loads(json.dumps(GOOD))
    answer["items"].append({"txn_id": "TXN-P4-0056", "category": "opex",
                            "flow": "outflow", "confidence": 0.1})
    client, _ = _client([answer, GOOD], tmp_path)
    marked, _ = categorize.categorize_batch(ROWS, client)
    assert marked["TXN-P4-0056"].category == "interest", "победила первая запись"


def test_an_out_of_vocabulary_category_is_refused(tmp_path):
    """Выдуманная статья не упадёт — она молча выпадет из агрегата."""
    answer = _answer(("TXN-P4-0056", "процентные расходы"))
    client, _ = _client([answer], tmp_path)
    marked, notes = categorize.categorize_batch([ROWS[0]], client)
    assert marked == {}
    assert notes


def test_a_partial_answer_keeps_what_arrived(tmp_path):
    """Разложенные строки уже верны. Бросить всё значило бы потерять
    и то, что есть."""
    partial = _answer(("TXN-P4-0056", "interest"))
    client, _ = _client([partial], tmp_path)
    marked, notes = categorize.categorize_batch(ROWS, client)
    assert set(marked) == {"TXN-P4-0056"}
    assert notes


# --------------------------------------------------------------------------- #
# Полный прогон: ни одна строка не исчезает
# --------------------------------------------------------------------------- #


def test_every_row_is_marked(tmp_path):
    paths = RunPaths.create(tmp_path / "run")
    client, _ = _client([GOOD], tmp_path)
    report = categorize.run(ROWS, paths, client, batch_size=40)
    assert set(report.items) == {r["txn_id"] for r in ROWS}
    assert report.alarms(expected=len(ROWS)) == []


def test_an_incomplete_answer_is_repaired_before_any_second_pass(tmp_path):
    """Первая линия защиты — repair-петля клиента: она сообщает модели,
    каких строк не хватает, и та дополняет ответ. Второй проход нужен
    только если и это не помогло."""
    paths = RunPaths.create(tmp_path / "run")
    partial = _answer(("TXN-P4-0056", "interest"))
    client, _ = _client([partial, GOOD], tmp_path)
    report = categorize.run(ROWS, paths, client, batch_size=40)
    assert report.retried == 0, "до второго прохода дойти не должно было"
    assert set(report.items) == {r["txn_id"] for r in ROWS}


def test_rows_missing_after_repairs_are_asked_again(tmp_path):
    """Когда repair-петля исчерпана, недостающие строки собираются заново
    и запрашиваются маленькими пакетами, где ответ обозрим."""
    paths = RunPaths.create(tmp_path / "run")
    partial = _answer(("TXN-P4-0056", "interest"))
    # Три неполных ответа исчерпывают исправления, четвёртый — полный.
    client, _ = _client([partial, partial, partial, GOOD], tmp_path)
    report = categorize.run(ROWS, paths, client, batch_size=40)
    assert report.retried == 2, "второй проход обязан был случиться"
    assert set(report.items) == {r["txn_id"] for r in ROWS}
    assert not report.fallbacks()


def test_a_row_that_never_arrives_gets_a_loud_fallback(tmp_path):
    """Молчаливого исчезновения нет ни на одном пути."""
    paths = RunPaths.create(tmp_path / "run")
    partial = _answer(("TXN-P4-0056", "interest"))
    client, _ = _client([partial], tmp_path)
    report = categorize.run(ROWS, paths, client, batch_size=40)

    assert set(report.items) == {r["txn_id"] for r in ROWS}, "строка исчезла"
    assert sorted(report.fallbacks()) == ["TXN-P1-0012", "TXN-P6-0005"]
    assert all(report.items[t].category == "other" for t in report.fallbacks())
    assert report.problems


def test_a_crashed_batch_does_not_lose_its_rows(tmp_path):
    """Падение пакета — не повод потерять сорок операций."""
    paths = RunPaths.create(tmp_path / "run")

    class Boom(MockProvider):
        def call(self, req, extra=()):
            raise RuntimeError("нет сети")

    client = LLMClient(cache_dir=tmp_path / "c", provider=Boom())
    report = categorize.run(ROWS, paths, client, batch_size=2)
    assert set(report.items) == {r["txn_id"] for r in ROWS}
    assert len(report.fallbacks()) == 3
    assert report.problems


def test_the_artifact_records_what_needs_review(tmp_path):
    paths = RunPaths.create(tmp_path / "run")
    answer = {"items": [
        {"txn_id": "TXN-P4-0056", "category": "interest", "flow": "outflow",
         "confidence": 0.95},
        {"txn_id": "TXN-P6-0005", "category": "taxes", "flow": "outflow",
         "confidence": 0.30},
        {"txn_id": "TXN-P1-0012", "category": "payroll", "flow": "outflow",
         "confidence": 0.99},
    ]}
    client, _ = _client([answer], tmp_path)
    categorize.run(ROWS, paths, client, batch_size=40)

    data = json.loads((paths.artifacts / A.TXN_CATEGORIES).read_text(encoding="utf-8"))
    assert data["low_confidence"] == ["TXN-P6-0005"]
    assert data["counts"] == {"interest": 1, "payroll": 1, "taxes": 1}


# --------------------------------------------------------------------------- #
# Тревоги
# --------------------------------------------------------------------------- #


def test_alarm_when_rows_went_missing():
    report = categorize.CategoryReport(items={
        "a": TxnCategory("a", "opex"), "b": TxnCategory("b", "opex")})
    assert any("уменьшат агрегаты" in a for a in report.alarms(expected=5))


def test_alarm_on_degenerate_labelling():
    """Если почти всё уехало в одну статью, разметка, скорее всего,
    не состоялась — а агрегаты при этом заполнены и выглядят правдоподобно."""
    items = {f"t{i}": TxnCategory(f"t{i}", "opex") for i in range(9)}
    items["t9"] = TxnCategory("t9", "revenue")
    report = categorize.CategoryReport(items=items)
    assert any("вырожденную" in a for a in report.alarms(expected=10))


def test_no_alarm_on_a_healthy_distribution():
    items = {}
    for i, category in enumerate(["opex", "revenue", "capex", "payroll", "taxes"] * 2):
        items[f"t{i}"] = TxnCategory(f"t{i}", category)
    report = categorize.CategoryReport(items=items)
    assert report.alarms(expected=10) == []


# --------------------------------------------------------------------------- #
# Группировка по смыслу
#
# На публичном наборе 673 строки дают лишь 244 различных смысловых
# описания: «Management advisory retainer» встречается одиннадцать раз.
# Спрашивать модель об одном и том же одиннадцать раз значит платить
# одиннадцать раз за один ответ — и именно на этом шаге кончилась
# суточная квота, оставив 323 строки из 673 неразмеченными.
# --------------------------------------------------------------------------- #

from pipeline.categorize import build_groups, group_key  # noqa: E402


def test_location_and_period_do_not_split_a_group():
    """«Corporate income tax instalment — Kostanay centre, H1 2025»
    и «… — Almaty office» это одна статья расхода."""
    a = _row("T1", "Corporate income tax instalment — Kostanay centre, H1 2025")
    b = _row("T2", "Corporate income tax instalment — Almaty office")
    assert group_key(a) == group_key(b)


def test_different_meanings_stay_apart():
    assert group_key(_row("T1", "Revolver interest — November")) != \
           group_key(_row("T2", "Land tax instalment — November"))


def test_case_and_spacing_do_not_split_a_group():
    assert group_key(_row("T1", "Revolver  Interest")) == \
           group_key(_row("T2", "revolver interest"))


def test_every_row_belongs_to_exactly_one_group():
    """Строка, не попавшая ни в одну группу, исчезнет из агрегата."""
    rows = [_row(f"T{i}", d) for i, d in enumerate(
        ["Payroll — March", "Payroll — April", "Rent — Q1", "Tax filing"])]
    representatives, members = build_groups(rows)
    covered = [t for group in members.values() for t in group]
    assert sorted(covered) == sorted(r["txn_id"] for r in rows)
    assert len(covered) == len(set(covered)), "строка попала в две группы"
    assert len(representatives) == 3


def test_the_representative_is_the_first_row_of_its_group():
    rows = [_row("T1", "Payroll — March"), _row("T2", "Payroll — April")]
    representatives, members = build_groups(rows)
    assert [r["txn_id"] for r in representatives] == ["T1"]
    assert members["T1"] == ["T1", "T2"]


def test_the_group_answer_spreads_to_every_member(tmp_path):
    """Ради этого группировка и делается: один вызов — вся группа."""
    paths = RunPaths.create(tmp_path / "run")
    rows = [_row("T1", "Revolver interest — November"),
            _row("T2", "Revolver interest — December"),
            _row("T3", "Land tax instalment — H1")]
    client, mock = _client([_answer(("T1", "interest"), ("T3", "taxes"))], tmp_path)

    report = categorize.run(rows, paths, client, batch_size=60)

    assert report.groups == 2, "две группы вместо трёх строк"
    assert len(mock.calls) == 1
    assert report.items["T2"].category == "interest"
    assert "то же описание" in report.items["T2"].reason
    assert set(report.items) == {"T1", "T2", "T3"}


def test_grouping_does_not_hide_a_missing_row(tmp_path):
    """Отчёт считает СТРОКИ, а не группы: из строк складываются агрегаты."""
    paths = RunPaths.create(tmp_path / "run")
    rows = [_row("T1", "Payroll — March"), _row("T2", "Payroll — April"),
            _row("T3", "Rent — Q1")]
    client, _ = _client([_answer(("T1", "payroll"))], tmp_path)

    report = categorize.run(rows, paths, client, batch_size=60)

    assert set(report.items) == {"T1", "T2", "T3"}
    assert report.items["T3"].fallback, "непришедшая группа обязана быть заметна"
    assert report.items["T2"].category == "payroll"


def test_a_fallback_group_marks_all_its_rows_as_fallback(tmp_path):
    paths = RunPaths.create(tmp_path / "run")
    rows = [_row("T1", "Payroll — March"), _row("T2", "Payroll — April")]

    class Boom(MockProvider):
        def call(self, req, extra=()):
            raise RuntimeError("нет сети")

    client = LLMClient(cache_dir=tmp_path / "c", provider=Boom())
    report = categorize.run(rows, paths, client, batch_size=60)
    assert sorted(report.fallbacks()) == ["T1", "T2"]


def test_marketing_is_its_own_category():
    """Выяснено измерением, а не рассуждением. Без отдельной статьи
    129 строк из 157 в `opex` оказывались маркетинговыми — 291 млн
    из 339 — и операционные расходы раздувались в разы.

    У P1 `opex` выходил 19.27 млн вместо 4.00 млн, ковенант
    «капиталоёмкость» показывал 0.09 при настоящих 0.46, и вердикт
    переворачивался. Выделение статьи подняло долю верных вердиктов
    с 25 до 29 из 36."""
    assert "marketing" in CATEGORIES

    prompt = categorize.build_prompt([])
    assert "marketing" in prompt
    assert "НЕ смешивай с opex" in prompt, (
        "модель обязана знать, что это отдельная статья, а не разновидность opex"
    )


def test_the_vocabulary_is_shared_by_every_step_that_uses_it():
    """Статья, известная одному шагу и неизвестная другому, даёт ПУСТОЙ
    агрегат: шаг 5 попросит AGG(marketing), а шаг 10 такую не назначит."""
    from pipeline import adjustments, covenants

    for build in (categorize.build_prompt, covenants.build_prompt,
                  adjustments.build_prompt):
        prompt = build([] if build is categorize.build_prompt else "x")
        for category in CATEGORIES:
            assert category in prompt, f"{build.__module__}: нет статьи {category}"
