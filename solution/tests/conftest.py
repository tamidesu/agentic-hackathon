"""Общие фикстуры.

Полное извлечение корпуса стоит ~80 секунд. Оно обязано выполняться
РОВНО ОДИН РАЗ за сессию и переиспользоваться всеми медленными тестами,
иначе набор перестают запускать — а незапускаемый набор бесполезен.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PUBLIC = ROOT.parent / "agentic-bank-public"


@pytest.fixture(scope="session")
def public_dataset():
    from pipeline.config import discover_dataset

    if not PUBLIC.exists():
        pytest.skip("нет публичного датасета")
    return discover_dataset(PUBLIC)


@pytest.fixture(scope="session")
def full_run(public_dataset, tmp_path_factory):
    """Единственный полный прогон извлечения на всю сессию."""
    from pipeline import extract
    from pipeline.config import RunPaths

    rp = RunPaths.create(tmp_path_factory.mktemp("corpus"))
    return extract.run(public_dataset, rp, llm=None, workers=8), rp


@pytest.fixture(scope="session")
def corpus_report(full_run):
    """Классификация поверх того же прогона — без повторного извлечения."""
    from pipeline import classify

    _, rp = full_run
    return classify.run(rp), rp


@pytest.fixture(scope="session")
def attributed(corpus_report, public_dataset):
    """Привязка поверх той же классификации — общая для шагов 4 и 9."""
    from pipeline import attribute

    _, rp = corpus_report
    return attribute.run(public_dataset, rp)
