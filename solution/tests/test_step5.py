"""Тесты шага 5: извлечение ковенантов.

Проверяется не «модуль вызывает модель», а три места, где ошибка тихая:
обрезка раздела (можно отдать модели заголовок вместо договора), сверка
с шаблоном (можно недосчитаться пункта и потерять треть ячеек заёмщика)
и проверка цитаты (можно принять выдумку за извлечение).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline import covenants  # noqa: E402
from pipeline.classify import DocType  # noqa: E402
from pipeline.covenant_types import ANY_CATEGORY, CATEGORIES, OBSERVED_FORMS  # noqa: E402
from pipeline.llm import LLMClient, MockProvider  # noqa: E402
from pipeline.schemas import _validate_metric  # noqa: E402

PUBLIC = ROOT.parent / "agentic-bank-public"
needs_public = pytest.mark.skipif(not PUBLIC.exists(), reason="нет публичного датасета")


# --------------------------------------------------------------------------- #
# Обрезка раздела
# --------------------------------------------------------------------------- #

TOC_THEN_BODY = (
    "Статья 5 Случаи неисполнения обязательств\n"
    "Статья 6 Финансовые ковенанты\n"
    "Статья 7 Ограничительные обязательства\n"
    "Статья 8 Погашение\n"
    + "прочий текст договора. " * 200
    + "\nСтатья 6 — Финансовые ковенанты\n"
    + "Пункт 6.1 Заёмщик обязуется не допускать превышения $1,000,000.00. " * 12
    + "\nСтатья 7 Ограничительные обязательства\nдалее не нужно"
)


def test_table_of_contents_does_not_win_over_the_real_section():
    """Реальный провал: первое вхождение якоря стоит в ОГЛАВЛЕНИИ, сразу
    за ним «Статья 7», и раздел получался длиной в 21 знак — молча."""
    section = covenants.find_section(TOC_THEN_BODY)
    assert section.anchor is not None
    assert len(section.text) > covenants.MIN_SECTION_CHARS
    assert "Пункт 6.1" in section.text
    assert "далее не нужно" not in section.text, "раздел не обрезан по следующей статье"
    assert any("оглавление" in n for n in section.notes)


def test_section_stops_at_the_next_article():
    text = ("Финансовые ковенанты\n" + "Пункт 6.1 порог $500,000.00. " * 30
            + "\nСтатья 7 Ограничительные обязательства\nНЕ ДОЛЖНО ПОПАСТЬ")
    assert "НЕ ДОЛЖНО ПОПАСТЬ" not in covenants.find_section(text).text


def test_english_section_is_found():
    text = ("FINANCIAL COVENANTS\n"
            + "Clause 6.1 The Borrower shall not exceed $750,000.00. " * 20
            + "\nArticle 7 Negative undertakings\ntail")
    section = covenants.find_section(text)
    assert section.anchor == "FINANCIAL COVENANTS"
    assert "Clause 6.1" in section.text and "tail" not in section.text


def test_missing_anchor_sends_the_whole_contract_loudly():
    """Обрезка вслепую отсекла бы половину пунктов. Целый текст дороже,
    но полон — и о переходе на этот путь обязано быть сказано."""
    text = "Раздел о показателях\n" + "Пункт 6.1 порог $100,000.00. " * 30
    section = covenants.find_section(text)
    assert section.anchor is None
    assert section.text == text
    assert section.notes and "целиком" in section.notes[0]


def test_toc_only_document_is_reported_not_silently_trimmed():
    text = "Статья 6 Финансовые ковенанты\nСтатья 7 Прочее\n" + "хвост. " * 200
    section = covenants.find_section(text)
    assert section.anchor is None, "вырожденный фрагмент не должен считаться разделом"
    assert any("оглавление" in n for n in section.notes)


def test_overlong_section_is_truncated_loudly(monkeypatch):
    monkeypatch.setattr(covenants, "SECTION_MAX_CHARS", 600)
    text = "Финансовые ковенанты\n" + "Пункт 6.1 текст. " * 400
    section = covenants.find_section(text)
    assert len(section.text) == 600
    assert section.truncated and any("обрезан" in n for n in section.notes)


# --------------------------------------------------------------------------- #
# Промпт
# --------------------------------------------------------------------------- #


def test_prompt_carries_the_exact_category_vocabulary():
    """Главный стык проекта: категория, названная в промпте, обязана
    совпадать со словарём движка. Расхождение даёт ПУСТОЙ агрегат —
    ковенант «соблюдён» со значением 0, и это не падает."""
    prompt = covenants.build_prompt("текст")
    for category in CATEGORIES:
        assert category in prompt, f"статья {category} не названа модели"
    assert ANY_CATEGORY in prompt


def test_prompt_examples_are_executable_by_the_engine():
    """Few-shot материал берётся из OBSERVED_FORMS. Если пример содержит
    статью вне словаря, промпт УЧИТ модель ошибке — так и было с AGG('ebitda')."""
    for name, form in OBSERVED_FORMS.items():
        assert _validate_metric(form["дерево"]) == [], f"пример {name} неисполним"


def test_prompt_states_both_languages():
    prompt = covenants.build_prompt("x")
    assert "английском" in prompt


# --------------------------------------------------------------------------- #
# Извлечение: сверка с шаблоном и цитаты
# --------------------------------------------------------------------------- #

SECTION_TEXT = (
    "Финансовые ковенанты\n"
    "Пункт 6.1 Максимальные капитальные затраты. Заёмщик обязуется не допускать, "
    "чтобы совокупный объём расходов по статье «Капитальные затраты» за период "
    "с 2025-01-01 по 2025-12-31 превышал $2,000,000.00.\n"
    "Пункт 6.2 Минимальное покрытие. Заёмщик обеспечивает, чтобы Выручка за период "
    "с 2025-01-01 по 2025-12-31 составляла не менее 3.00x Операционных расходов.\n"
    + "дополнительный текст раздела. " * 20
)


def _covenant(point, quote, **over):
    base = {
        "point": point, "title": "т", "direction": "max", "threshold": 2000000.0,
        "unit": "amount", "period_start": "2025-01-01", "period_end": "2025-12-31",
        "metric_definition": "капитальные затраты",
        "metric": {"op": "AGG", "category": "capex"},
        "quote": quote,
    }
    base.update(over)
    return base


GOOD_QUOTE_61 = "совокупный объём расходов по статье «Капитальные затраты»"
GOOD_QUOTE_62 = "Выручка за период с 2025-01-01 по 2025-12-31 составляла не менее 3.00x"


def _client(responses, tmp_path):
    """responses — список ответов подряд; последний повторяется."""
    mock = MockProvider()
    state = {"i": 0}

    def reply(req):
        i = min(state["i"], len(responses) - 1)
        state["i"] += 1
        return responses[i]

    mock.register_rule(lambda r: True, reply)
    return LLMClient(cache_dir=tmp_path / "c", provider=mock), mock


def test_extraction_succeeds_and_keeps_the_tree(tmp_path):
    payload = {"covenants": [_covenant("6.1", GOOD_QUOTE_61),
                             _covenant("6.2", GOOD_QUOTE_62, unit="ratio", threshold=3.0,
                                       direction="min")]}
    client, _ = _client([payload], tmp_path)
    res = covenants.extract_one("P1", "d1", SECTION_TEXT, client, ["6.1", "6.2"])
    assert res.points() == ["6.1", "6.2"]
    assert res.problems == []
    assert res.covenants[0]["metric"] == {"op": "AGG", "category": "capex"}


def test_missing_template_point_is_asked_for_and_then_reported(tmp_path):
    """Шаблон — данность: пункт из него обязан быть. Сначала просим
    исправить, и только неисправленное становится проблемой.

    ЧАСТИЧНОЕ ИЗВЛЕЧЕНИЕ СОХРАНЯЕТСЯ. Замечание относится к ответу целиком,
    но ячейки независимы: уронив ответ, мы потеряли бы и исправный 6.1."""
    incomplete = {"covenants": [_covenant("6.1", GOOD_QUOTE_61)]}
    client, mock = _client([incomplete], tmp_path)
    res = covenants.extract_one("P1", "d1", SECTION_TEXT, client, ["6.1", "6.2"])
    assert len(mock.calls) > 1, "клиент обязан был запросить исправление"
    assert res.points() == ["6.1"], "исправный пункт обязан был уцелеть"
    assert res.problems and any("6.2" in p for p in res.problems)


def test_repair_recovers_the_missing_point(tmp_path):
    incomplete = {"covenants": [_covenant("6.1", GOOD_QUOTE_61)]}
    complete = {"covenants": [_covenant("6.1", GOOD_QUOTE_61),
                              _covenant("6.2", GOOD_QUOTE_62)]}
    client, _ = _client([incomplete, complete], tmp_path)
    res = covenants.extract_one("P1", "d1", SECTION_TEXT, client, ["6.1", "6.2"])
    assert res.points() == ["6.1", "6.2"]
    assert res.problems == []


def test_invented_quote_is_dropped_not_kept(tmp_path):
    """Цитата — единственная нить между деревом и договором. Пункт без
    подтверждённой цитаты в расчёт не идёт: лучше запасное значение,
    чем уверенно неверное."""
    bad = {"covenants": [_covenant("6.1", "такой строки в договоре нет вовсе")]}
    client, mock = _client([bad], tmp_path)
    res = covenants.extract_one("P1", "d1", SECTION_TEXT, client, ["6.1"])
    assert len(mock.calls) > 1, "должна была быть попытка исправления"
    assert res.covenants == []
    assert res.problems


def test_a_good_point_survives_a_neighbour_with_a_bad_quote(tmp_path):
    """Именно ради этого случая ответ не роняется целиком."""
    mixed = {"covenants": [_covenant("6.1", GOOD_QUOTE_61),
                           _covenant("6.2", "выдуманная строка")]}
    client, _ = _client([mixed], tmp_path)
    res = covenants.extract_one("P1", "d1", SECTION_TEXT, client, ["6.1", "6.2"])
    assert res.points() == ["6.1"]
    assert any("отброшены" in p for p in res.problems)


def test_out_of_vocabulary_category_is_rejected(tmp_path):
    """AGG('capital_expenditure') не упадёт — он вернёт ПУСТУЮ сумму
    и ложный COMPLIANT со значением 0. Это ловится здесь, а не в отчёте."""
    bad = {"covenants": [_covenant("6.1", GOOD_QUOTE_61,
                                   metric={"op": "AGG", "category": "capital_expenditure"})]}
    client, mock = _client([bad], tmp_path)
    res = covenants.extract_one("P1", "d1", SECTION_TEXT, client, ["6.1"])
    assert res.covenants == [], "пункт с выдуманной статьёй дал бы actual=0 и ложный COMPLIANT"
    assert len(mock.calls) > 1


def test_extra_point_is_noted_but_not_fatal(tmp_path):
    payload = {"covenants": [_covenant("6.1", GOOD_QUOTE_61),
                             _covenant("6.4", GOOD_QUOTE_62)]}
    client, _ = _client([payload], tmp_path)
    res = covenants.extract_one("P1", "d1", SECTION_TEXT, client, ["6.1"])
    assert res.problems == []
    assert any("вне шаблона" in n for n in res.notes)


def test_quote_is_checked_against_the_trimmed_text_only(tmp_path):
    """Модель видела раздел, а не договор. Проверять цитату по целому
    тексту значило бы принимать цитаты из пунктов, которых она не видела."""
    contract = "шапка договора с фразой ПОСТОРОННЯЯ СТРОКА ИЗ ШАПКИ\n" + SECTION_TEXT
    bad = {"covenants": [_covenant("6.1", "ПОСТОРОННЯЯ СТРОКА ИЗ ШАПКИ")]}
    client, _ = _client([bad], tmp_path)
    res = covenants.extract_one("P1", "d1", contract, client, ["6.1"])
    assert res.covenants == []


# --------------------------------------------------------------------------- #
# Прогон
# --------------------------------------------------------------------------- #


def test_expected_points_reads_the_template():
    template = {"answers": {"P1": {"6.3": {}, "6.1": {}}, "B1": {"6.2": {}}}}
    assert covenants.expected_points(template) == {"P1": ["6.1", "6.3"], "B1": ["6.2"]}


def test_alarm_when_no_section_was_recognised_anywhere():
    rep = covenants.CovenantReport(scenarios=[
        covenants.ScenarioCovenants("P1", "d1", covenants=[{"point": "6.1"}]),
        covenants.ScenarioCovenants("P2", "d2", covenants=[{"point": "6.1"}]),
    ])
    assert any("НИ В ОДНОМ" in a for a in rep.alarms())


def test_alarm_when_a_borrower_has_no_covenants():
    rep = covenants.CovenantReport(scenarios=[
        covenants.ScenarioCovenants("P1", "d1", covenants=[], section_anchor="x"),
    ])
    assert any("без ковенантов" in a for a in rep.alarms())


def test_report_survives_a_missing_active_loan(tmp_path):
    """Заёмщик без действующего договора не должен ронять остальных."""
    rep = covenants.CovenantReport(scenarios=[
        covenants.ScenarioCovenants("P1", None, problems=["нет действующего договора"]),
        covenants.ScenarioCovenants("P2", "d2", covenants=[{"point": "6.1"}],
                                    section_anchor="Финансовые ковенанты"),
    ])
    data = json.loads(json.dumps(rep.to_dict(), ensure_ascii=False))
    assert data["scenarios"]["P1"]["covenants"] == []
    assert data["scenarios"]["P2"]["points"] == ["6.1"]


# --------------------------------------------------------------------------- #
# На реальном корпусе
# --------------------------------------------------------------------------- #


@pytest.mark.slow
@needs_public
def test_every_active_contract_yields_a_section_with_all_template_points(attributed, corpus_report):
    """Проверка без единого вызова модели: раздел, вырезанный из каждого
    действующего договора, обязан содержать ВСЕ пункты, которые называет
    шаблон. Если это не так, обрезка теряет пункты, и никакая модель
    их уже не вернёт."""
    import re

    docs, _ = attributed
    _, rp = corpus_report
    template = json.loads((PUBLIC / "submission_template.json").read_text(encoding="utf-8"))
    wanted = covenants.expected_points(template)

    active = {d.scenario_id: doc_id for doc_id, d in sorted(docs.items())
              if d.type == DocType.LOAN_ACTIVE and d.scenario_id}
    assert set(active) == set(wanted), "не у каждого заёмщика есть действующий договор"

    for scenario, doc_id in sorted(active.items()):
        text = (rp.artifacts / "01_texts" / f"{doc_id}.txt").read_text(encoding="utf-8")
        section = covenants.find_section(text)
        assert section.anchor is not None, f"{scenario}: раздел не опознан"
        found = sorted(set(re.findall(r"Пункт\s*(\d+\.\d+)", section.text)))
        assert found == wanted[scenario], f"{scenario}: в разделе {found}, шаблон ждёт {wanted[scenario]}"


@pytest.mark.slow
@needs_public
def test_trimming_removes_the_bulk_of_the_contract(attributed, corpus_report):
    """Смысл обрезки — деньги и точность. Если экономии нет, обрезка
    только добавляет риск, и от неё надо отказаться."""
    docs, _ = attributed
    _, rp = corpus_report
    total_full = total_section = 0
    for doc_id, d in sorted(docs.items()):
        if d.type != DocType.LOAN_ACTIVE or not d.scenario_id:
            continue
        text = (rp.artifacts / "01_texts" / f"{doc_id}.txt").read_text(encoding="utf-8")
        total_full += len(text)
        total_section += len(covenants.find_section(text).text)
    assert total_section < total_full * 0.10, (
        f"обрезка оставила {total_section / total_full:.1%} текста — слишком много"
    )


# --------------------------------------------------------------------------- #
# Порог обязан встречаться в тексте
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("value,text", [
    (2000000.0, "превышал $2,000,000.00 за период"),
    (2000000.0, "не более 2 000 000 тенге"),
    (2000000.0, "shall not exceed 2000000"),
    (0.08, "превышал 0.08x Операционных расходов"),
    (3.0, "составляла не менее 3.00x"),
    (450000.0, "$450,000.00"),
])
def test_threshold_is_recognised_in_its_written_forms(value, text):
    assert covenants.threshold_appears(text, value)


def test_invented_threshold_is_caught():
    text = "превышал $2,000,000.00 за период"
    assert not covenants.threshold_appears(text, 12345.0)
    problems = covenants.check_thresholds(
        {"covenants": [{"point": "6.1", "threshold": 12345.0}]}, text)
    assert problems and "не встречается" in problems[0]


def test_threshold_check_is_silent_on_a_correct_answer():
    text = "превышал $2,000,000.00 и не ниже 3.00x"
    payload = {"covenants": [{"point": "6.1", "threshold": 2000000.0},
                             {"point": "6.2", "threshold": 3.0}]}
    assert covenants.check_thresholds(payload, text) == []


@pytest.mark.slow
@needs_public
def test_no_false_rejections_on_the_real_corpus(attributed, corpus_report):
    """Проверка полезна ровно настолько, насколько она не мешает. Каждый
    порог, реально написанный в разделе, обязан опознаваться — иначе
    начнутся круги исправления на верных ответах, а это деньги и время."""
    import re

    docs, _ = attributed
    _, rp = corpus_report
    checked = 0
    for doc_id, d in sorted(docs.items()):
        if d.type != DocType.LOAN_ACTIVE:
            continue
        text = (rp.artifacts / "01_texts" / f"{doc_id}.txt").read_text(encoding="utf-8")
        section = covenants.find_section(text).text
        values = [float(x.replace("$", "").replace(",", ""))
                  for x in re.findall(r"\$[\d,]+\.\d\d", section)]
        values += [float(x[:-1]) for x in re.findall(r"\b\d+\.\d\dx\b", section)]
        for value in values:
            checked += 1
            assert covenants.threshold_appears(section, value), (
                f"{d.scenario_id}: порог {value} написан в тексте, но не опознан"
            )
    assert checked >= 36, f"проверено только {checked} порогов"


def test_force_bypasses_the_cache_but_still_writes_it(tmp_path):
    """Нужно после правки промпта и для честного замера времени прогона:
    с кэшем шаг 5 идёт за секунду и ничего не говорит о боевом окне."""
    payload = {"covenants": [_covenant("6.1", GOOD_QUOTE_61)]}
    cache = tmp_path / "c"

    client, mock = _client([payload], tmp_path)
    covenants.extract_one("P1", "d1", SECTION_TEXT, client, ["6.1"])
    covenants.extract_one("P1", "d1", SECTION_TEXT, client, ["6.1"])
    assert len(mock.calls) == 1, "второй вызов обязан был прийти из кэша"

    mock2 = MockProvider()
    mock2.register_rule(lambda r: True, lambda r: payload)
    forced = LLMClient(cache_dir=cache, provider=mock2, force=True)
    covenants.extract_one("P1", "d1", SECTION_TEXT, forced, ["6.1"])
    assert len(mock2.calls) == 1, "--force обязан игнорировать накопленный кэш"


# --------------------------------------------------------------------------- #
# Плоское представление дерева
#
# РЕАЛЬНЫЙ ПРОВАЛ, РАДИ КОТОРОГО ОНО ПОЯВИЛОСЬ. Схема описывала дерево
# рекурсивно, через $ref. Она валидна, SDK её принимал, и на маленьком
# проверочном примере всё работало. На живом прогоне gemini-3.6-flash
# вернул ВСЕ узлы-операции пустыми: {"op": "DIV"} без args. Уцелели только
# листья — голые AGG. Из 36 ковенантов 17 являются отношениями, то есть
# это стоило бы половины ответов.
#
# Отказ был не громким, а коварным: JSON валиден по схеме, поля на месте,
# и только семантический валидатор поймал «DIV без аргументов».
# --------------------------------------------------------------------------- #

from pipeline.schemas import (  # noqa: E402
    MetricGraphError,
    flatten_metric,
    metric_from_payload,
    nest_metric,
)

RATIO_FLAT = [
    {"id": "root", "op": "DIV", "args": ["ebitda", "int"]},
    {"id": "ebitda", "op": "SUB", "args": ["rev", "opx"]},
    {"id": "int", "op": "AGG", "category": "interest"},
    {"id": "rev", "op": "AGG", "category": "revenue"},
    {"id": "opx", "op": "AGG", "category": "opex"},
]
RATIO_TREE = {
    "op": "DIV",
    "args": [
        {"op": "SUB", "args": [{"op": "AGG", "category": "revenue"},
                               {"op": "AGG", "category": "opex"}]},
        {"op": "AGG", "category": "interest"},
    ],
}


def test_flat_list_becomes_the_expected_tree():
    assert nest_metric(RATIO_FLAT, "root") == RATIO_TREE


def test_argument_order_is_preserved():
    """DIV — не коммутативная операция: перепутать числитель со знаменателем
    значит получить обратное число и, скорее всего, обратный статус."""
    tree = nest_metric(RATIO_FLAT, "root")
    assert tree["args"][1] == {"op": "AGG", "category": "interest"}
    assert tree["args"][0]["args"][0]["category"] == "revenue"


def test_leaf_has_no_args_key():
    tree = nest_metric([{"id": "a", "op": "AGG", "category": "capex"}], "a")
    assert tree == {"op": "AGG", "category": "capex"}


def test_round_trip_through_the_flat_form_is_lossless():
    nodes, root = flatten_metric(RATIO_TREE)
    assert nest_metric(nodes, root) == RATIO_TREE


def test_dangling_reference_is_caught():
    """Ссылки по имени открывают три способа сломать дерево, которых при
    вложенности физически не было. Каждый даёт неверное число, а не сбой."""
    with pytest.raises(MetricGraphError, match="не объявлен"):
        nest_metric([{"id": "root", "op": "DIV", "args": ["a", "нет-такого"]},
                     {"id": "a", "op": "AGG", "category": "revenue"}], "root")


def test_cycle_is_caught_and_not_infinite():
    with pytest.raises(MetricGraphError, match="цикл"):
        nest_metric([{"id": "a", "op": "DIV", "args": ["b"]},
                     {"id": "b", "op": "DIV", "args": ["a"]}], "a")


def test_missing_root_is_caught():
    with pytest.raises(MetricGraphError, match="корневой узел"):
        nest_metric([{"id": "a", "op": "AGG", "category": "capex"}], "root")


def test_duplicate_id_is_caught():
    with pytest.raises(MetricGraphError, match="дважды"):
        nest_metric([{"id": "a", "op": "AGG", "category": "capex"},
                     {"id": "a", "op": "AGG", "category": "opex"}], "a")


def test_nested_form_is_still_accepted():
    """Провайдеры различаются: если какой-то рекурсию всё же отдаёт,
    незачем ломать работающий ответ."""
    assert metric_from_payload({"metric": RATIO_TREE}) == RATIO_TREE


def test_flat_form_is_read_from_the_payload():
    payload = {"metric_nodes": RATIO_FLAT, "metric_root": "root"}
    assert metric_from_payload(payload) == RATIO_TREE


def test_schema_contains_no_recursion_at_all():
    """Именно рекурсию провайдер и не разворачивает."""
    from pipeline.schemas import COVENANT_SPEC_SCHEMA

    assert "$ref" not in json.dumps(COVENANT_SPEC_SCHEMA)


def test_args_are_strings_not_objects_in_the_schema():
    from pipeline.schemas import COVENANT_SPEC_SCHEMA

    node = COVENANT_SPEC_SCHEMA["properties"]["covenants"]["items"] \
        ["properties"]["metric_nodes"]["items"]
    assert node["properties"]["args"]["items"] == {"type": "string"}


def test_prompt_examples_use_the_flat_form():
    """Пример во вложенной форме учил бы модель отвечать так, как схема
    не разрешает — ровно тот разрыв, который стоил половины ковенантов."""
    prompt = covenants.build_prompt("x")
    assert "metric_nodes" in prompt and "metric_root" in prompt
    assert '"args": [{' not in prompt, "в примерах остались вложенные объекты"


def test_extraction_accepts_the_flat_answer(tmp_path):
    payload = {"covenants": [{
        "point": "6.1", "title": "т", "direction": "max", "threshold": 2000000.0,
        "unit": "amount", "period_start": "2025-01-01", "period_end": "2025-12-31",
        "metric_definition": "отношение",
        "metric_nodes": RATIO_FLAT, "metric_root": "root",
        "quote": GOOD_QUOTE_61,
    }]}
    client, _ = _client([payload], tmp_path)
    res = covenants.extract_one("P1", "d1", SECTION_TEXT, client, ["6.1"])
    assert res.problems == []
    assert res.covenants[0]["metric"] == RATIO_TREE, "дерево не собралось из плоского списка"


def test_operation_without_args_is_still_rejected(tmp_path):
    """Ровно то, что вернула модель на живом прогоне."""
    payload = {"covenants": [{
        "point": "6.1", "title": "т", "direction": "max", "threshold": 2000000.0,
        "unit": "amount", "period_start": "2025-01-01", "period_end": "2025-12-31",
        "metric_definition": "отношение",
        "metric_nodes": [{"id": "root", "op": "DIV"}], "metric_root": "root",
        "quote": GOOD_QUOTE_61,
    }]}
    client, _ = _client([payload], tmp_path)
    res = covenants.extract_one("P1", "d1", SECTION_TEXT, client, ["6.1"])
    assert res.covenants == []
    assert res.problems


# --------------------------------------------------------------------------- #
# Артефакты вёрстки против сверки цитат
# --------------------------------------------------------------------------- #

P5_REAL_TEXT = (
    "Пункт 6.3 Максимальные платежи связанным сторонам. За период с 2025-01-01 "
    "по 2025-12-31\nсовокупные платежи Заёмщика (Ekibastuz Power Services JSC) "
    "в адрес аффилированных и связанных\n5\nсторон не должны превышать "
    "$260,000.00. Принадлежность контрагента к связанным сторонам\n"
    "устанавливается с учётом раскрытий в комплаенс-досье Заёмщика.\n"
)


def test_quote_survives_a_page_number_inside_the_sentence():
    """Реальный случай P5: колонтитул «5» попал в середину предложения.
    Модель процитировала осмысленно и цифру опустила — и правильно
    сделала. Первая версия сверки объявила цитату выдуманной и выбросила
    верный ковенант."""
    from pipeline.schemas import make_quote_validator

    validator = make_quote_validator(P5_REAL_TEXT)
    quote = ("совокупные платежи Заёмщика (Ekibastuz Power Services JSC) в адрес "
             "аффилированных и связанных сторон не должны превышать $260,000.00")
    assert validator({"quote": quote}) == []


def test_invented_quote_is_still_caught_after_the_relaxation():
    """Послабление не должно превращать проверку в формальность."""
    from pipeline.schemas import make_quote_validator

    validator = make_quote_validator(P5_REAL_TEXT)
    assert validator({"quote": "совокупные платежи не должны превышать $999,999.00"})


def test_amounts_are_not_damaged_by_page_number_stripping():
    """Гасится только ОТДЕЛЬНО СТОЯЩЕЕ число между переводами строк.
    Цифры внутри сумм трогать нельзя — на них держится сверка порогов."""
    from pipeline.schemas import _normalize

    assert "$260,000.00" in _normalize(P5_REAL_TEXT)
    assert "1,600,000.00" in _normalize("порог\n7\n$1,600,000.00 за период")


# --------------------------------------------------------------------------- #
# Springing-тест в плоской форме
# --------------------------------------------------------------------------- #

def _springing(**over):
    base = {
        # Порог обязан встречаться в SECTION_TEXT: проверка порогов —
        # не декорация, и подобрать «любое число» она не даст.
        "point": "6.1", "title": "t", "direction": "max", "threshold": 3.0,
        "unit": "ratio", "period_start": "2025-01-01", "period_end": "2025-12-31",
        "metric_definition": "d",
        "metric_nodes": [{"id": "a", "op": "AGG", "category": "financing_inflow"}],
        "metric_root": "a", "quote": "q", "is_conditional": True,
        "condition_nodes": [{"id": "c", "op": "AGG", "category": "financing_inflow"}],
        "condition_root": "c",
    }
    base.update(over)
    return base


def test_conditional_covenant_with_a_flat_condition_is_accepted():
    """Мой собственный провал: поля условия переехали в плоскую форму,
    а проверка осталась на старом имени condition_metric — и объявила
    верный springing-тест неполным (реальный случай: 6.1 у P3)."""
    from pipeline.schemas import validate_covenant_spec

    assert validate_covenant_spec({"covenants": [_springing()]}) == []


def test_conditional_covenant_without_any_condition_is_rejected():
    from pipeline.schemas import validate_covenant_spec

    problems = validate_covenant_spec(
        {"covenants": [_springing(condition_nodes=None, condition_root=None)]})
    assert problems and "условия" in problems[0]


def test_condition_tree_is_nested_during_extraction(tmp_path):
    payload = {"covenants": [_springing(
        quote=GOOD_QUOTE_61,
        condition_nodes=[{"id": "c", "op": "DIV", "args": ["x", "y"]},
                         {"id": "x", "op": "AGG", "category": "financing_inflow"},
                         {"id": "y", "op": "AGG", "category": "revenue"}],
        condition_root="c",
    )]}
    client, _ = _client([payload], tmp_path)
    res = covenants.extract_one("P3", "d1", SECTION_TEXT, client, ["6.1"])
    assert res.covenants, res.problems
    condition = res.covenants[0]["condition_metric"]
    assert condition["op"] == "DIV" and len(condition["args"]) == 2


# --------------------------------------------------------------------------- #
# Смена модели при исчерпании суточной квоты
#
# РЕАЛЬНЫЙ СЛУЧАЙ. На прогоне квота gemini-3.6-flash кончилась после
# четырёх заёмщиков, и восемь из двенадцати остались без ковенантов —
# две трети возможных очков. Квота считается на проект И НА МОДЕЛЬ,
# то есть соседняя модель это не «попробовать то же ещё раз», а
# полностью свежий счётчик. Ретраями не лечится принципиально: суточная
# квота не рассасывается за секунды.
# --------------------------------------------------------------------------- #


def test_client_fills_in_the_model_for_requests_that_omit_it(tmp_path):
    """Реальный провал: распознавание сканов не указывало модель, и туда
    подставлялось умолчание, ЗАМОРОЖЕННОЕ на момент импорта модуля —
    claude-opus-5 уехал на Gemini-ключ и получил 404."""
    from pipeline.llm import LLMRequest

    mock = MockProvider()
    mock.register_rule(lambda r: True, lambda r: {"ok": True})
    client = LLMClient(cache_dir=tmp_path / "c", provider=mock, model="gemini-3.6-flash")

    client.extract(LLMRequest(prompt="p", schema={"type": "object"}))
    assert mock.calls[0].model == "gemini-3.6-flash"


def test_an_explicit_model_is_never_overridden(tmp_path):
    from pipeline.llm import LLMRequest

    mock = MockProvider()
    mock.register_rule(lambda r: True, lambda r: {"ok": True})
    client = LLMClient(cache_dir=tmp_path / "c", provider=mock, model="gemini-3.6-flash")

    client.extract(LLMRequest(prompt="p", schema={"type": "object"}, model="другая"))
    assert mock.calls[0].model == "другая"


def test_model_substitution_happens_before_the_cache_key(tmp_path):
    """Иначе ответы легли бы под ключом с пустым именем модели, и кэш
    перестал бы различать модели."""
    from pipeline.llm import LLMRequest

    mock = MockProvider()
    mock.register_rule(lambda r: True, lambda r: {"ok": True})
    request = LLMRequest(prompt="p", schema={"type": "object"})

    a = LLMClient(cache_dir=tmp_path / "c", provider=mock, model="модель-А")
    b = LLMClient(cache_dir=tmp_path / "c", provider=mock, model="модель-Б")
    a.extract(request)
    b.extract(request)
    assert len(mock.calls) == 2, "разные модели обязаны иметь разные ключи кэша"

    again = LLMClient(cache_dir=tmp_path / "c", provider=mock, model="модель-А")
    result = again.extract(request)
    assert result.cached, "та же модель обязана попадать в кэш"


def test_switch_model_is_skipped_for_providers_without_a_catalogue(tmp_path):
    mock = MockProvider()
    client = LLMClient(cache_dir=tmp_path / "c", provider=mock)
    spare, notes = covenants._switch_model(client, "какая-то")
    assert spare is None and notes


def test_switch_model_excludes_the_exhausted_one(tmp_path, monkeypatch):
    """Предлагать ту же модель, чья квота кончилась, бессмысленно."""
    mock = MockProvider()
    client = LLMClient(cache_dir=tmp_path / "c", provider=mock)
    asked = {}

    def catalogue():
        return ["gemini-3.6-flash", "gemini-3.5-flash"]

    monkeypatch.setattr(mock, "available_models", catalogue, raising=False)

    def fake_verify(provider, preferred, cat):
        asked["catalogue"] = cat
        return "gemini-3.5-flash", []

    import pipeline.gemini as gemini_module

    monkeypatch.setattr(gemini_module, "verify_model", fake_verify)
    spare, _ = covenants._switch_model(client, "gemini-3.6-flash")
    assert spare == "gemini-3.5-flash"
    assert "gemini-3.6-flash" not in asked["catalogue"]


def test_switch_model_failure_does_not_break_the_run(tmp_path, monkeypatch):
    mock = MockProvider()
    client = LLMClient(cache_dir=tmp_path / "c", provider=mock)
    monkeypatch.setattr(
        mock, "available_models",
        lambda: (_ for _ in ()).throw(RuntimeError("нет сети")), raising=False)
    spare, notes = covenants._switch_model(client, "m")
    assert spare is None and any("не удалось" in n for n in notes)
