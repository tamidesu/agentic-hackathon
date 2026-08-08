"""Тесты шага 0: инфраструктура LLM-вызовов и обнаружение датасета.

Каждый тест проверяет заявленную в плане гарантию, а не факт «код запускается».
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline import schemas  # noqa: E402
from pipeline.config import DatasetError, RunPaths, discover_dataset  # noqa: E402
from pipeline.llm import (  # noqa: E402
    LLMClient,
    LLMError,
    LLMRequest,
    MockProvider,
    ValidationFailed,
)

SCHEMA = {
    "type": "object",
    "required": ["value"],
    "properties": {"value": {"type": "number"}},
    "additionalProperties": False,
}


def req(prompt="привет", **kw) -> LLMRequest:
    return LLMRequest(prompt=prompt, schema=SCHEMA, **kw)


# --------------------------------------------------------------------------- #
# Кэш
# --------------------------------------------------------------------------- #


def test_cache_makes_second_call_free_and_identical(tmp_path):
    """Гарантия из плана: повторный прогон не делает сетевых вызовов
    и даёт побитово тот же результат."""
    mock = MockProvider()
    mock.register_rule(lambda r: True, lambda r: {"value": 42})
    client = LLMClient(cache_dir=tmp_path / "cache", provider=mock)

    first = client.extract(req())
    assert first.data == {"value": 42} and not first.cached
    assert len(mock.calls) == 1

    # Новый клиент, тот же кэш — имитация перезапуска процесса.
    client2 = LLMClient(cache_dir=tmp_path / "cache", provider=mock)
    second = client2.extract(req())
    assert second.cached is True
    assert second.data == first.data
    assert len(mock.calls) == 1, "повторный вызов ушёл в сеть — кэш не работает"


def test_cache_key_separates_prompt_versions(tmp_path):
    """Правка промпта обязана инвалидировать кэш, иначе получим старые ответы."""
    mock = MockProvider()
    mock.register_rule(lambda r: True, lambda r: {"value": 1 if r.prompt_version == "v1" else 2})
    client = LLMClient(cache_dir=tmp_path / "cache", provider=mock)

    assert client.extract(req(prompt_version="v1")).data == {"value": 1}
    assert client.extract(req(prompt_version="v2")).data == {"value": 2}
    assert len(mock.calls) == 2


def test_corrupt_cache_entry_is_ignored(tmp_path):
    mock = MockProvider()
    mock.register_rule(lambda r: True, lambda r: {"value": 7})
    client = LLMClient(cache_dir=tmp_path / "cache", provider=mock)
    client.extract(req())

    victim = next((tmp_path / "cache").glob("*.json"))
    victim.write_text("{битый json", encoding="utf-8")

    assert client.extract(req()).data == {"value": 7}
    assert len(mock.calls) == 2, "битый кэш должен приводить к перезапросу, а не к падению"


# --------------------------------------------------------------------------- #
# Ретраи и repair-петля
# --------------------------------------------------------------------------- #


def test_transient_error_is_retried(tmp_path):
    class Flaky(MockProvider):
        def __init__(self):
            super().__init__()
            self.n = 0

        def call(self, r, extra_user_turns=()):
            self.n += 1
            if self.n < 3:
                raise TimeoutError("временная сетевая ошибка")
            return {"value": 5}, 10, 3

        def retryable(self, exc):
            return isinstance(exc, TimeoutError)

    client = LLMClient(cache_dir=tmp_path / "c", provider=Flaky(), base_backoff_s=0.001)
    assert client.extract(req()).data == {"value": 5}
    assert client.usage.retries == 2


def test_non_retryable_error_propagates(tmp_path):
    mock = MockProvider()  # ничего не зарегистрировано -> LLMError, retryable=False
    client = LLMClient(cache_dir=tmp_path / "c", provider=mock)
    with pytest.raises(LLMError):
        client.extract(req())


def test_repair_loop_fixes_semantic_error(tmp_path):
    """Схема пропускает, валидатор ловит, модель получает обратную связь."""
    state = {"n": 0}

    def produce(r):
        state["n"] += 1
        return {"value": -1} if state["n"] == 1 else {"value": 10}

    mock = MockProvider()
    mock.register_rule(lambda r: True, produce)
    client = LLMClient(cache_dir=tmp_path / "c", provider=mock)

    def validator(p):
        return [] if p["value"] > 0 else ["value должно быть положительным"]

    res = client.extract(req(), validator=validator)
    assert res.data == {"value": 10}
    assert client.usage.repairs == 1


def test_repair_gives_up_and_raises(tmp_path):
    mock = MockProvider()
    mock.register_rule(lambda r: True, lambda r: {"value": -1})
    client = LLMClient(cache_dir=tmp_path / "c", provider=mock, max_repairs=2)

    with pytest.raises(ValidationFailed) as exc:
        client.extract(req(), validator=lambda p: ["всегда плохо"])
    assert exc.value.last_payload == {"value": -1}


def test_cache_hit_is_revalidated(tmp_path):
    """Если валидатор ужесточили после записи кэша, старый ответ не должен пролезть."""
    mock = MockProvider()
    mock.register_rule(lambda r: True, lambda r: {"value": 3})
    client = LLMClient(cache_dir=tmp_path / "c", provider=mock)
    client.extract(req())
    assert len(mock.calls) == 1

    with pytest.raises(ValidationFailed):
        client.extract(req(), validator=lambda p: ["теперь это невалидно"])
    assert len(mock.calls) > 1, "кэш отдал ответ в обход нового валидатора"


# --------------------------------------------------------------------------- #
# Параллельность
# --------------------------------------------------------------------------- #


def test_map_parallel_preserves_order_and_isolates_failures():
    def fn(x):
        if x == 3:
            raise ValueError("ветка 3 упала")
        return x * 10

    out = LLMClient.map_parallel(fn, [1, 2, 3, 4], workers=4)
    assert out[0] == 10 and out[1] == 20 and out[3] == 40
    assert isinstance(out[2], ValueError), "падение ветки обнулило соседние"


def test_self_consistency_reports_agreement(tmp_path):
    seq = [{"value": 1}, {"value": 1}, {"value": 2}]
    state = {"i": 0}

    def produce(r):
        v = seq[state["i"] % len(seq)]
        state["i"] += 1
        return v

    mock = MockProvider()
    mock.register_rule(lambda r: True, produce)
    client = LLMClient(cache_dir=tmp_path / "c", provider=mock)

    winner, agreement = client.extract_consistent(req(), n=3)
    assert winner.data == {"value": 1}
    assert agreement == pytest.approx(2 / 3)


# --------------------------------------------------------------------------- #
# Содержательные валидаторы
# --------------------------------------------------------------------------- #


def test_quote_validator_catches_hallucinated_quote():
    source = "Пункт 6.3. Заёмщик обязуется не допускать, чтобы платежи связанным сторонам превышали $450,000.00."
    v = schemas.make_quote_validator(source)
    assert v({"quote": "платежи связанным сторонам превышали $450,000.00"}) == []
    problems = v({"quote": "платежи связанным сторонам превышали $999,999.00"})
    assert problems and "отсутствует в документе" in problems[0]


def test_quote_validator_tolerates_pdf_whitespace_and_dashes():
    source = "коэффициент\nкапиталоёмкости   за период с 2025-01-01 по 2025-12-31 превышал 0.42x"
    v = schemas.make_quote_validator(source)
    assert v({"quote": "коэффициент капиталоёмкости за период с 2025—01—01 по 2025—12—31 превышал 0.42x"}) == []


def test_quote_validator_walks_nested_structures():
    v = schemas.make_quote_validator("альфа бета гамма дельта эпсилон")
    problems = v({"covenants": [{"point": "6.1", "quote": "выдуманная цитата целиком"}]})
    assert problems and "covenants[0]" in problems[0]


def test_covenant_validator_catches_bad_period_and_threshold():
    bad = {
        "covenants": [
            {"point": "6.1", "threshold": -5, "unit": "amount",
             "period_start": "2025-01-01", "period_end": "2024-12-31"},
            {"point": "6.1", "threshold": 450000, "unit": "ratio",
             "period_start": "01.01.2025", "period_end": "2025-12-31"},
        ]
    }
    problems = schemas.validate_covenant_spec(bad)
    joined = " | ".join(problems)
    assert "положительным" in joined
    assert "перевёрнут" in joined
    assert "YYYY-MM-DD" in joined
    assert "выглядит как сумма" in joined
    assert "дважды" in joined


def test_related_parties_validator_catches_threshold_inconsistency():
    """Прямая защита от ошибки, которая переворачивает вердикт по 6.3."""
    payload = {
        "has_ownership_section": True,
        "threshold_pct": 40.0,
        "parties": [
            {"name": "Taraz Holding Group LLP", "ownership_pct": 46.8, "is_related": True},
            {"name": "Taraz Kiln Services LLP", "ownership_pct": 38.1, "is_related": True},
        ],
        "quote": "x",
    }
    problems = schemas.validate_related_parties(payload)
    assert len(problems) == 1 and "Kiln" in problems[0]

    payload["parties"][1]["is_related"] = False
    assert schemas.validate_related_parties(payload) == []


def test_txn_category_validator_catches_missing_and_invented():
    v = schemas.make_txn_category_validator(["T-1", "T-2"], ["revenue", "capex"])
    problems = v({"items": [
        {"txn_id": "T-1", "category": "revenue"},
        {"txn_id": "T-9", "category": "wat"},
    ]})
    joined = " | ".join(problems)
    assert "пропущены" in joined and "придуманы" in joined and "вне таксономии" in joined


def test_audit_adjustment_validator_requires_amount_for_missing_amount():
    problems = schemas.validate_audit_adjustments(
        # status обязателен: без него валидатор не знает, применяется ли
        # примечание, и требовать от него полноты полей нельзя.
        {"notes": [{"note_id": "8.1", "kind": "missing_amount",
                    "status": "applied", "value_usd": None}]}
    )
    joined = " | ".join(problems)
    assert "требует value_usd" in joined and "требует target_txn_id" in joined


# --------------------------------------------------------------------------- #
# Обнаружение датасета
# --------------------------------------------------------------------------- #

PUBLIC = Path(__file__).resolve().parents[2] / "agentic-bank-public"


@pytest.mark.skipif(not PUBLIC.exists(), reason="публичный датасет недоступен")
def test_discovers_public_dataset():
    ds = discover_dataset(PUBLIC)
    assert ds.documents_dir.name == "documents"
    assert ds.ledger_csv.suffix == ".csv"
    assert ds.template_json.name.endswith(".json")
    assert not hasattr(ds, "ground_truth"), "пайплайн не должен видеть ответы"


@pytest.mark.skipif(not PUBLIC.exists(), reason="публичный датасет недоступен")
def test_discovery_survives_renaming(tmp_path):
    """Переносимость: имена файлов приватного датасета будут другими."""
    clone = tmp_path / "private_like"
    clone.mkdir()
    shutil.copytree(PUBLIC / "documents", clone / "docs_2026")
    shutil.copy(PUBLIC / "master_ledger_2025.csv", clone / "transactions_export_Q4.csv")
    shutil.copy(PUBLIC / "submission_template.json", clone / "answers_skeleton.json")

    ds = discover_dataset(clone)
    assert ds.documents_dir.name == "docs_2026"
    assert ds.ledger_csv.name == "transactions_export_Q4.csv"
    assert ds.template_json.name == "answers_skeleton.json"


def test_discovery_reports_what_is_missing(tmp_path):
    (tmp_path / "documents").mkdir()
    for i in range(6):
        (tmp_path / "documents" / f"{i}.pdf").write_bytes(b"%PDF-1.4")
    with pytest.raises(DatasetError, match="реестр транзакций"):
        discover_dataset(tmp_path)


def test_run_paths_are_created(tmp_path):
    rp = RunPaths.create(tmp_path / "run")
    assert rp.artifacts.is_dir() and rp.cache.is_dir() and rp.logs.is_dir()


def test_chunked_splits_evenly_and_remainder():
    assert LLMClient.chunked([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]
    assert LLMClient.chunked([], 3) == []
    with pytest.raises(ValueError):
        LLMClient.chunked([1], 0)


def test_default_model_is_a_known_string():
    """Защита от выдуманного имени модели: боевой вызов упал бы с 404."""
    from pipeline.llm import DEFAULT_MODEL, KNOWN_MODELS
    assert DEFAULT_MODEL in KNOWN_MODELS
