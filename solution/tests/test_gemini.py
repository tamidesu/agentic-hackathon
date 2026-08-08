"""Тесты провайдера Gemini и адаптера схем.

Главный риск, ради которого всё проверяется: наши схемы написаны под
forced tool use Anthropic, а Gemini поддерживает лишь подмножество
JSON Schema. Расхождение между тем, что видит модель, и тем, чем мы
проверяем ответ, — источник тихих ошибок.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.gemini import (  # noqa: E402
    adapt_schema,
    recursive_refs_are_optional,
    warn_about_free_tier,
)
from pipeline.schemas import (  # noqa: E402
    AUDIT_ADJUSTMENTS_SCHEMA,
    COVENANT_SPEC_SCHEMA,
    PAGE_TRANSCRIPTION_SCHEMA,
    RELATED_PARTIES_SCHEMA,
    TXN_CATEGORY_SCHEMA,
)

genai_types = pytest.importorskip("google.genai").types


# --------------------------------------------------------------------------- #
# Адаптер: что именно переписывается
# --------------------------------------------------------------------------- #


def test_nullable_union_becomes_anyof():
    """`type: ["string","null"]` Gemini не принимает."""
    out = adapt_schema({"type": ["string", "null"]})
    assert out == {"anyOf": [{"type": "string"}, {"type": "null"}]}


def test_nullable_number_union():
    out = adapt_schema({"type": ["number", "null"]})
    assert {"type": "number"} in out["anyOf"] and {"type": "null"} in out["anyOf"]


def test_enum_with_null_is_split():
    """`enum` допустим только для строк и чисел — null выносится."""
    out = adapt_schema({"type": ["string", "null"], "enum": ["expense", "income", None]})
    flat = str(out)
    assert "None" not in flat and "null" in flat
    assert "expense" in flat


def test_plain_types_pass_through():
    src = {"type": "object", "additionalProperties": False,
           "properties": {"a": {"type": "number", "minimum": 0, "maximum": 1}},
           "required": ["a"]}
    assert adapt_schema(src) == src


def test_refs_and_defs_are_preserved():
    """Рекурсия — единственный способ описать дерево выражений ковенанта."""
    src = {"type": "object",
           "$defs": {"node": {"type": "object",
                              "properties": {"op": {"type": "string"},
                                             "args": {"type": "array",
                                                      "items": {"$ref": "#/$defs/node"}}},
                              "required": ["op"]}},
           "properties": {"metric": {"$ref": "#/$defs/node"}},
           "required": ["metric"]}
    out = adapt_schema(src)
    assert out["$defs"]["node"]["properties"]["args"]["items"] == {"$ref": "#/$defs/node"}


def test_adapter_is_idempotent():
    once = adapt_schema(COVENANT_SPEC_SCHEMA)
    assert adapt_schema(once) == once


def test_adapter_does_not_mutate_the_original():
    import copy

    before = copy.deepcopy(COVENANT_SPEC_SCHEMA)
    adapt_schema(COVENANT_SPEC_SCHEMA)
    assert COVENANT_SPEC_SCHEMA == before


# --------------------------------------------------------------------------- #
# Все наши схемы принимаются SDK
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name,schema", [
    ("covenant_spec", COVENANT_SPEC_SCHEMA),
    ("audit_adjustments", AUDIT_ADJUSTMENTS_SCHEMA),
    ("related_parties", RELATED_PARTIES_SCHEMA),
    ("txn_category", TXN_CATEGORY_SCHEMA),
    ("page_transcription", PAGE_TRANSCRIPTION_SCHEMA),
])
def test_every_schema_is_accepted_after_adaptation(name, schema):
    genai_types.GenerateContentConfig(
        response_mime_type="application/json",
        response_json_schema=adapt_schema(schema),
    )


def test_raw_schema_with_union_is_rejected_by_the_strict_path():
    """Подтверждает, что адаптер решает настоящую проблему, а не мнимую:
    через типизированный `response_schema` наша схема не проходит."""
    with pytest.raises(Exception):
        genai_types.Schema.model_validate(COVENANT_SPEC_SCHEMA)


# --------------------------------------------------------------------------- #
# Оговорка про циклические ссылки
# --------------------------------------------------------------------------- #


def test_cycle_through_required_properties_is_reported():
    """Документация: «Cyclic references … may only be used within
    non-required properties». Здесь node → args → node, и args обязателен:
    развернуть такую рекурсию нельзя, она бесконечна."""
    bad = {"type": "object",
           "$defs": {"node": {"type": "object",
                              "properties": {"args": {"$ref": "#/$defs/node"}},
                              "required": ["args"]}},
           "properties": {"metric": {"$ref": "#/$defs/node"}}}
    problems = recursive_refs_are_optional(bad)
    assert problems and "цикл по обязательным" in problems[0]


def test_recursive_ref_in_optional_property_is_fine():
    good = {"type": "object",
            "$defs": {"node": {"type": "object",
                               "properties": {"op": {"type": "string"},
                                              "args": {"type": "array",
                                                       "items": {"$ref": "#/$defs/node"}}},
                               "required": ["op"]}},
            "properties": {"metric": {"$ref": "#/$defs/node"}}}
    assert recursive_refs_are_optional(good) == []


def test_required_reference_into_a_recursive_definition_is_not_a_violation():
    """Поле `metric` обязательно и указывает на рекурсивное определение,
    но сам цикл замыкается через необязательное `args` — глубина конечна.
    Первая версия проверки помечала и этот случай, что было неверно."""
    schema = {"type": "object",
              "required": ["metric"],
              "$defs": {"node": {"type": "object", "required": ["op"],
                                 "properties": {"op": {"type": "string"},
                                                "args": {"type": "array",
                                                         "items": {"$ref": "#/$defs/node"}}}}},
              "properties": {"metric": {"$ref": "#/$defs/node"}}}
    assert recursive_refs_are_optional(schema) == []


def test_indirect_cycle_through_two_definitions_is_reported():
    schema = {"type": "object",
              "$defs": {
                  "a": {"type": "object", "required": ["b"],
                        "properties": {"b": {"$ref": "#/$defs/b"}}},
                  "b": {"type": "object", "required": ["a"],
                        "properties": {"a": {"$ref": "#/$defs/a"}}}},
              "properties": {"root": {"$ref": "#/$defs/a"}}}
    problems = recursive_refs_are_optional(schema)
    assert len(problems) == 2


def test_current_schemas_have_no_cyclic_problem():
    for schema in (COVENANT_SPEC_SCHEMA, AUDIT_ADJUSTMENTS_SCHEMA,
                   RELATED_PARTIES_SCHEMA, TXN_CATEGORY_SCHEMA):
        assert recursive_refs_are_optional(schema) == []


# --------------------------------------------------------------------------- #
# Провайдер
# --------------------------------------------------------------------------- #


def test_missing_key_fails_with_a_useful_message(monkeypatch):
    from pipeline.gemini import GeminiProvider
    from pipeline.llm import LLMError

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(LLMError, match="GEMINI_API_KEY"):
        GeminiProvider()


def test_provider_implements_the_common_interface():
    from pipeline.gemini import GeminiProvider
    from pipeline.llm import Provider

    assert issubclass(GeminiProvider, Provider)
    assert GeminiProvider.name == "gemini"


def test_free_tier_warning_is_explicit():
    """На бесплатном тарифе данные идут в обучение — это осознанный выбор,
    а не умолчание."""
    warning = warn_about_free_tier("gemini-3-flash-preview")
    assert warning and "бесплатном тарифе" in warning
    assert warn_about_free_tier("gemini-3.1-pro-preview") is None


# --------------------------------------------------------------------------- #
# Обрыв ответа по лимиту вывода
#
# Реальный провал первого боевого прогона: Gemini 3 думает по умолчанию,
# мысли считаются в max_output_tokens вместе с ответом, и JSON обрывался
# на середине строки. Снаружи это выглядело как ошибка ФОРМАТА
# («Unterminated string»), хотя причина была в БЮДЖЕТЕ.
# --------------------------------------------------------------------------- #


class _FakeResponse:
    def __init__(self, text, finish="STOP", thoughts=0):
        self.text = text
        self.usage_metadata = type("U", (), {
            "prompt_token_count": 100, "candidates_token_count": 50,
            "thoughts_token_count": thoughts})()
        self.candidates = [type("C", (), {"finish_reason": type("F", (), {"name": finish})()})()]


@pytest.fixture
def provider(monkeypatch):
    from pipeline.gemini import GeminiProvider

    monkeypatch.setenv("GEMINI_API_KEY", "k1,k2,k3")
    return GeminiProvider()


def test_several_keys_are_rotated(provider):
    """Лимит частоты на бесплатном тарифе считается НА КЛЮЧ."""
    assert provider.n_keys == 3
    seen = {id(provider._next_client()) for _ in range(6)}
    assert len(seen) == 3


def test_single_key_still_works(monkeypatch):
    from pipeline.gemini import GeminiProvider

    monkeypatch.setenv("GEMINI_API_KEY", "one")
    p = GeminiProvider()
    assert p.n_keys == 1 and p._next_client() is p._clients[0]


def test_thinking_level_is_set_explicitly(provider):
    """Без явного уровня модель тратит бюджет вывода на размышления."""
    cfg = provider._thinking_config()
    assert cfg is not None
    assert str(getattr(cfg, "thinking_level", "")).lower().endswith("low")


def test_thinking_can_be_left_to_the_model(monkeypatch):
    from pipeline.gemini import GeminiProvider

    monkeypatch.setenv("GEMINI_API_KEY", "k")
    assert GeminiProvider(thinking_level=None)._thinking_config() is None


def test_truncation_is_diagnosed_not_reported_as_bad_json(provider):
    from pipeline.gemini import TruncatedResponse

    with pytest.raises(TruncatedResponse) as exc:
        provider._parse(_FakeResponse('{"covenants":[{"point":"6.1"',
                                      finish="MAX_TOKENS", thoughts=7800), budget=8000)
    message = str(exc.value)
    assert "оборван по лимиту вывода" in message
    assert "7800" in message, "без числа токенов размышлений причина неочевидна"
    assert "8000" in message


def test_broken_json_without_max_tokens_is_also_retryable(provider):
    """Обрыв не всегда сопровождается finish_reason=MAX_TOKENS."""
    from pipeline.gemini import TruncatedResponse

    with pytest.raises(TruncatedResponse):
        provider._parse(_FakeResponse('{"covenants":[{"point"'), budget=8000)


def test_truncation_is_retryable(provider):
    from pipeline.gemini import TruncatedResponse

    assert provider.retryable(TruncatedResponse("оборван"))


def test_valid_response_passes_through(provider):
    payload, in_tok, out_tok = provider._parse(_FakeResponse('{"ok": true}'), budget=8000)
    assert payload == {"ok": True} and in_tok == 100 and out_tok == 50


def test_empty_response_names_the_finish_reason(provider):
    from pipeline.llm import LLMError

    with pytest.raises(LLMError, match="SAFETY"):
        provider._parse(_FakeResponse("", finish="SAFETY"), budget=8000)


def test_budget_escalates_after_truncation(provider, monkeypatch):
    """Повтор того же запроса с тем же лимитом бессмыслен: обрыв
    лечится увеличением бюджета, а не задержкой."""
    from pipeline.llm import LLMRequest

    budgets = []

    class FakeModels:
        def generate_content(self, model, contents, config):
            budgets.append(config.max_output_tokens)
            if len(budgets) < 2:
                return _FakeResponse('{"a"', finish="MAX_TOKENS", thoughts=999)
            return _FakeResponse('{"a": 1}')

    # genai.Client.models — свойство только для чтения, поэтому
    # подменяется выбор клиента, а не его содержимое.
    fake = type("Client", (), {"models": FakeModels()})()
    monkeypatch.setattr(provider, "_next_client_with_index", lambda: (fake, 0))

    payload, _, _ = provider.call(LLMRequest(prompt="p", schema={"type": "object"},
                                             max_tokens=4000))
    assert payload == {"a": 1}
    assert budgets[1] > budgets[0], f"бюджет не поднялся: {budgets}"


def test_escalation_gives_up_at_the_ceiling(provider, monkeypatch):
    """Бесконечно поднимать бюджет нельзя — это деньги и время."""
    from pipeline.gemini import MAX_OUTPUT_TOKENS, TruncatedResponse
    from pipeline.llm import LLMRequest

    calls = []

    class AlwaysTruncated:
        def generate_content(self, model, contents, config):
            calls.append(config.max_output_tokens)
            return _FakeResponse('{"a"', finish="MAX_TOKENS", thoughts=1)

    fake = type("Client", (), {"models": AlwaysTruncated()})()
    monkeypatch.setattr(provider, "_next_client_with_index", lambda: (fake, 0))

    with pytest.raises(TruncatedResponse):
        provider.call(LLMRequest(prompt="p", schema={"type": "object"}, max_tokens=8000))
    assert len(calls) <= 3, f"слишком много попыток: {calls}"
    assert max(calls) <= MAX_OUTPUT_TOKENS


# --------------------------------------------------------------------------- #
# Самоограничение частоты
# --------------------------------------------------------------------------- #


def test_rate_limit_is_respected_per_key(monkeypatch):
    """Ретрай после 429 — обнаружение лимита постфактум. Дешевле его
    не превышать: на 673 транзакциях шага 10 очередь ретраев растёт
    быстрее, чем разгребается."""
    import time

    from pipeline.gemini import GeminiProvider

    monkeypatch.setenv("GEMINI_API_KEY", "a,b,c")
    monkeypatch.setenv("GEMINI_MIN_INTERVAL_S", "0.2")
    p = GeminiProvider()

    start = time.monotonic()
    for _ in range(9):
        p._next_client()
    elapsed = time.monotonic() - start
    # 9 запросов / 3 ключа = по 3 на ключ = 2 интервала ожидания.
    assert 0.3 < elapsed < 0.9, f"ожидание не соответствует лимиту: {elapsed:.2f}с"


def test_more_keys_mean_proportionally_less_waiting(monkeypatch):
    """Ради этого и заведено несколько ключей.

    Замеряется ТОЛЬКО ожидание между запросами: конструктор провайдера —
    вне секундомера. Прежде он был внутри, и на машине, где genai.Client()
    строится по ~2.5с (Windows), четыре клиента стоили дороже трёх пауз
    по 0.2с — тест мерил скорость конструктора, а не пользу ключей.
    """
    import time

    from pipeline.gemini import GeminiProvider

    monkeypatch.setenv("GEMINI_MIN_INTERVAL_S", "0.2")

    monkeypatch.setenv("GEMINI_API_KEY", "solo")
    one = GeminiProvider()
    monkeypatch.setenv("GEMINI_API_KEY", "a,b,c,d")
    many = GeminiProvider()

    t0 = time.monotonic()
    for _ in range(4):
        one._next_client()
    single = time.monotonic() - t0

    t0 = time.monotonic()
    for _ in range(4):
        many._next_client()
    quad = time.monotonic() - t0

    assert quad < single / 2, f"четыре ключа не ускорили: {quad:.2f} против {single:.2f}"


def test_rate_limiting_can_be_switched_off(monkeypatch):
    """На платном тарифе ограничение только мешает."""
    import time

    from pipeline.gemini import GeminiProvider

    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("GEMINI_MIN_INTERVAL_S", "0")
    p = GeminiProvider()
    t0 = time.monotonic()
    for _ in range(20):
        p._next_client()
    assert time.monotonic() - t0 < 0.1


def test_waiting_happens_outside_the_lock(monkeypatch):
    """Держать блокировку во время сна значило бы выстроить все ветки
    в одну очередь и свести параллелизм к нулю."""
    import threading
    import time

    from pipeline.gemini import GeminiProvider

    monkeypatch.setenv("GEMINI_API_KEY", "a,b,c,d,e,f")
    monkeypatch.setenv("GEMINI_MIN_INTERVAL_S", "0.4")
    p = GeminiProvider()

    start = time.monotonic()
    threads = [threading.Thread(target=p._next_client) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # Шесть веток на шесть ключей: ждать не должен никто.
    assert time.monotonic() - start < 0.2


# --------------------------------------------------------------------------- #
# Суточная квота против лимита частоты
#
# Оба приходят как 429, но лечатся противоположно: лимит частоты
# рассасывается за секунды, суточная квота — только к полуночи.
# Пять кругов экспоненциальной задержки на суточной квоте — это полторы
# минуты впустую и всё равно провал.
# --------------------------------------------------------------------------- #

DAILY_429 = ("429 RESOURCE_EXHAUSTED. Quota exceeded for metric: "
             "generate_content_free_tier_requests, limit: 20 "
             "quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier")
MINUTE_429 = ("429 RESOURCE_EXHAUSTED. quotaId: "
              "GenerateRequestsPerMinutePerProjectPerModel-FreeTier")


class _Err(Exception):
    def __init__(self, text, code=429):
        super().__init__(text)
        self.code = code


def test_daily_quota_is_told_apart_from_a_rate_limit(provider):
    assert provider._is_daily_quota(_Err(DAILY_429))
    assert not provider._is_daily_quota(_Err(MINUTE_429))


def test_rate_limit_is_retried(provider, monkeypatch):
    from google.genai import errors

    monkeypatch.setattr(provider, "_errors", errors)
    exc = errors.ClientError.__new__(errors.ClientError)
    Exception.__init__(exc, MINUTE_429)
    exc.code = 429
    assert provider.retryable(exc)


def test_daily_quota_is_not_retried(provider, monkeypatch):
    """Ждать до полуночи в трёхчасовом окне бессмысленно."""
    from google.genai import errors

    monkeypatch.setattr(provider, "_errors", errors)
    exc = errors.ClientError.__new__(errors.ClientError)
    Exception.__init__(exc, DAILY_429)
    exc.code = 429
    assert not provider.retryable(exc)


def test_daily_quota_message_says_what_to_do(monkeypatch):
    """С ОДНИМ ключом переходить некуда, и тогда сообщение — единственное,
    что остаётся пользователю. С несколькими ключами провайдер сначала
    перебирает их, и это проверяется отдельным тестом."""
    from pipeline.gemini import DailyQuotaExhausted, GeminiProvider
    from pipeline.llm import LLMRequest

    monkeypatch.setenv("GEMINI_API_KEY", "solo")
    monkeypatch.setenv("GEMINI_MIN_INTERVAL_S", "0")
    provider = GeminiProvider()

    class Exhausted:
        def generate_content(self, model, contents, config):
            raise _Err(DAILY_429)

    fake = type("Client", (), {"models": Exhausted()})()
    monkeypatch.setattr(provider, "_next_client_with_index", lambda: (fake, 0))

    with pytest.raises(DailyQuotaExhausted) as exc:
        provider.call(LLMRequest(prompt="p", schema={"type": "object"}))
    message = str(exc.value)
    assert "СУТОЧНАЯ" in message
    assert "НА ПРОЕКТ" in message, "иначе непонятно, что три ключа могут не помочь"
    assert "preview" in message


def test_other_client_errors_are_not_retried(provider, monkeypatch):
    """400 повторять бессмысленно — запрос не станет валиднее."""
    from google.genai import errors

    monkeypatch.setattr(provider, "_errors", errors)
    exc = errors.ClientError.__new__(errors.ClientError)
    Exception.__init__(exc, "400 INVALID_ARGUMENT")
    exc.code = 400
    assert not provider.retryable(exc)


def test_default_model_is_not_a_preview(monkeypatch):
    """У preview-моделей бесплатный тариф — 20 запросов в СУТКИ на проект.
    Умолчание, упирающееся в квоту на двенадцати договорах, непригодно."""
    from pipeline.gemini import DEFAULT_GEMINI_MODEL

    assert "preview" not in DEFAULT_GEMINI_MODEL, "preview даёт 20 запросов в сутки"

    monkeypatch.setenv("LLM_MODE", "gemini")
    monkeypatch.delenv("LLM_MODEL", raising=False)
    from pipeline.llm import _default_model

    assert _default_model() == DEFAULT_GEMINI_MODEL


def test_thinking_is_switched_off_entirely_on_2_5(provider):
    """У 2.5 размышления отключаются полностью — нулевым бюджетом."""
    cfg = provider._thinking_config("gemini-2.5-flash")
    assert cfg.thinking_budget == 0
    assert cfg.thinking_level is None


def test_thinking_level_is_used_on_generation_3(provider):
    """У 3 отключить нельзя, есть только уровень. Отправить не то поле
    не той модели — 400 на каждом вызове."""
    cfg = provider._thinking_config("gemini-3-flash-preview")
    assert cfg.thinking_budget is None
    assert cfg.thinking_level is not None


# --------------------------------------------------------------------------- #
# Каталог моделей
#
# Реальный провал: gemini-2.5-flash вернул 404 «no longer available to new
# users». Имя модели, зашитое в код, — отложенный отказ.
# --------------------------------------------------------------------------- #

CATALOGUE = [
    "gemini-3.6-flash", "gemini-3.1-pro-preview", "gemini-3.5-flash-lite",
    "gemini-2.5-flash-lite", "gemini-2.0-flash", "gemini-2.5-flash-tts",
    "gemini-embedding-001", "gemini-3-flash-preview", "veo-3.0-generate",
]


def test_specialised_models_are_never_chosen():
    """TTS, эмбеддинги и видео структурный JSON не отдают."""
    from pipeline.gemini import resolve_model

    chosen, _ = resolve_model(None, CATALOGUE)
    assert not any(x in chosen for x in ("tts", "embedding", "veo"))


def test_preview_loses_to_a_stable_model():
    """Не из-за качества, а из-за 20 запросов в сутки."""
    from pipeline.gemini import rank_model

    assert rank_model("gemini-3.6-flash") > rank_model("gemini-3-flash-preview")


def test_newer_generation_beats_full_older_one():
    """Свежая lite разбирает юридический текст лучше, чем полная модель
    на два поколения старше."""
    from pipeline.gemini import rank_model

    assert rank_model("gemini-3.5-flash-lite") > rank_model("gemini-2.0-flash")


def test_full_model_beats_lite_of_the_same_generation():
    from pipeline.gemini import rank_model

    assert rank_model("gemini-3.5-flash") > rank_model("gemini-3.5-flash-lite")


def test_requested_model_wins_when_available():
    from pipeline.gemini import resolve_model

    chosen, notes = resolve_model("gemini-2.0-flash", CATALOGUE)
    assert chosen == "gemini-2.0-flash" and notes == []


def test_unavailable_model_is_replaced_and_the_swap_is_announced():
    """Подмена модели меняет качество и стоимость — она обязана быть видимой."""
    from pipeline.gemini import resolve_model

    chosen, notes = resolve_model("gemini-2.5-flash", CATALOGUE)
    assert chosen == "gemini-3.6-flash"
    assert any("недоступна" in n for n in notes)


def test_empty_catalogue_falls_back_to_the_requested_model():
    """Сбой листинга не повод отказываться от прогона."""
    from pipeline.gemini import resolve_model

    chosen, notes = resolve_model("gemini-2.5-flash", [])
    assert chosen == "gemini-2.5-flash" and notes


def test_lite_only_catalogue_is_flagged():
    from pipeline.gemini import resolve_model

    chosen, notes = resolve_model("gemini-2.5-flash", ["gemini-3.5-flash-lite"])
    assert chosen == "gemini-3.5-flash-lite"
    assert any("lite" in n for n in notes)


def test_available_models_filters_and_strips_the_prefix(provider, monkeypatch):
    class Model:
        def __init__(self, name, actions):
            self.name, self.supported_actions = name, actions

    class Models:
        def list(self):
            return [Model("models/gemini-3.6-flash", ["generateContent"]),
                    Model("models/gemini-embedding-001", ["embedContent"]),
                    Model("models/gemini-2.0-flash", ["generateContent"])]

    monkeypatch.setattr(provider, "_clients",
                        [type("C", (), {"models": Models()})()])
    assert provider.available_models() == ["gemini-2.0-flash", "gemini-3.6-flash"]


# --------------------------------------------------------------------------- #
# Каталог перечисляет модели, вызывать которые нельзя
#
# Реальный провал: gemini-2.5-flash есть в списке моделей ключа, но
# генерация по ней отвечает 404 «no longer available to new users».
# Листинг говорит о существовании модели, а не о праве её вызывать.
# --------------------------------------------------------------------------- #

GONE_404 = ("404 NOT_FOUND. This model models/gemini-2.5-flash is no longer "
            "available to new users.")


def _provider_where(fails: dict, monkeypatch, provider):
    """fails: имя модели -> текст ошибки; остальные отвечают успешно."""
    calls = []

    def fake_call(req, extra=()):
        calls.append(req.model)
        if req.model in fails:
            raise _Err(fails[req.model], code=404)
        return {"ok": True}, 1, 1

    monkeypatch.setattr(provider, "call", fake_call)
    return calls


def test_a_model_listed_but_not_callable_is_skipped(provider, monkeypatch):
    from pipeline.gemini import verify_model

    calls = _provider_where({"gemini-2.5-flash": GONE_404}, monkeypatch, provider)
    chosen, notes = verify_model(
        provider, "gemini-2.5-flash",
        ["gemini-2.5-flash", "gemini-3.6-flash", "gemini-2.0-flash"])

    assert chosen == "gemini-3.6-flash"
    assert calls[0] == "gemini-2.5-flash", "желаемая обязана быть проверена первой"
    assert any("вызвать нельзя" in n for n in notes)
    assert any("вместо" in n for n in notes)


def test_a_working_model_is_kept_and_costs_one_call(provider, monkeypatch):
    from pipeline.gemini import verify_model

    calls = _provider_where({}, monkeypatch, provider)
    chosen, notes = verify_model(provider, "gemini-3.6-flash",
                                 ["gemini-3.6-flash", "gemini-2.0-flash"])
    assert chosen == "gemini-3.6-flash"
    assert len(calls) == 1, "проверка не должна перебирать лишнее"
    assert not any("вместо" in n for n in notes)


def test_daily_quota_moves_on_to_the_next_model(provider, monkeypatch):
    """Квота считается на модель — соседняя может быть свежей."""
    from pipeline.gemini import verify_model

    _provider_where({"gemini-3.6-flash": DAILY_429}, monkeypatch, provider)
    chosen, notes = verify_model(provider, "gemini-3.6-flash",
                                 ["gemini-3.6-flash", "gemini-3.5-flash"])
    assert chosen == "gemini-3.5-flash"
    assert any("квота" in n for n in notes)


def test_an_unrelated_error_does_not_trigger_a_model_hunt(provider, monkeypatch):
    """Сетевой сбой повторится на любой модели — перебор только потратит
    вызовы и время."""
    from pipeline.gemini import verify_model

    calls = _provider_where({"gemini-3.6-flash": "соединение разорвано"},
                            monkeypatch, provider)
    chosen, notes = verify_model(provider, "gemini-3.6-flash",
                                 ["gemini-3.6-flash", "gemini-3.5-flash"])
    assert chosen == "gemini-3.6-flash"
    assert len(calls) == 1
    assert any("беру как есть" in n for n in notes)


def test_probing_stops_after_a_few_attempts(provider, monkeypatch):
    """Перебирать все 42 модели каталога — это 42 вызова из суточной квоты."""
    from pipeline.gemini import verify_model

    catalogue = [f"gemini-3.{i}-flash" for i in range(10)]
    calls = _provider_where({m: GONE_404 for m in catalogue}, monkeypatch, provider)
    chosen, notes = verify_model(provider, catalogue[0], catalogue, max_tries=3)
    assert len(calls) == 3
    assert any("ни одна модель не ответила" in n for n in notes)


def test_catalogue_failure_does_not_stop_the_run(provider, monkeypatch):
    from pipeline.gemini import verify_model

    monkeypatch.setattr(provider, "available_models",
                        lambda: (_ for _ in ()).throw(RuntimeError("нет сети")))
    chosen, notes = verify_model(provider, "gemini-3.6-flash")
    assert chosen == "gemini-3.6-flash"
    assert any("каталог недоступен" in n for n in notes)


# --------------------------------------------------------------------------- #
# Вывод исчерпанного ключа из круга
#
# Квота считается на ПРОЕКТ. Ключи перебирались вслепую, и один
# исчерпанный проект отравлял треть всех вызовов: каждый третий запрос
# был гарантированным отказом.
# --------------------------------------------------------------------------- #


def test_a_retired_key_leaves_the_rotation(provider):
    assert provider.live_keys == 3
    assert provider.retire_key(1, "квота")
    assert provider.live_keys == 2

    used = {provider._clients.index(provider._next_client()) for _ in range(12)}
    assert used == {0, 2}, "выбывший ключ продолжает участвовать"


def test_the_last_key_is_never_retired(provider):
    """Иначе работать станет нечем, а осмысленное сообщение об исчерпании
    квоты полезнее ошибки «нет доступных ключей»."""
    assert provider.retire_key(0, "квота")
    assert provider.retire_key(1, "квота")
    assert not provider.retire_key(2, "квота")
    assert provider.live_keys == 1


def test_rotation_does_not_hang_when_everything_is_retired(provider):
    """Обход ограничен числом ключей: бесконечного цикла не выйдет."""
    provider._retired = {0, 1, 2}
    assert provider._next_client() is not None


def test_daily_quota_moves_to_the_next_key(provider, monkeypatch):
    """Соседний ключ может быть из другого проекта и потому свежим."""
    from pipeline.llm import LLMRequest

    seen: list[int] = []

    class PerKey:
        def __init__(self, index):
            self.index = index

        def generate_content(self, model, contents, config):
            seen.append(self.index)
            if self.index == 0:
                raise _Err(DAILY_429)
            return _FakeResponse('{"ok": true}')

    fakes = [type("C", (), {"models": PerKey(i)})() for i in range(3)]
    order = iter([(fakes[0], 0), (fakes[1], 1)])
    monkeypatch.setattr(provider, "_next_client_with_index", lambda: next(order))

    payload, _, _ = provider.call(LLMRequest(prompt="p", schema={"type": "object"}))
    assert payload == {"ok": True}
    assert seen == [0, 1], "запрос обязан был уйти на следующий ключ"
    assert 0 in provider._retired
