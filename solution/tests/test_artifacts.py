"""Тесты реестра артефактов.

Шаги общаются через файлы: один пишет, другой читает. Пока имена были
строковыми литералами, разбросанными по модулям, случилось три вещи,
и ни одна не выглядела как ошибка:

  * entities.py и related.py писали ОДИН И ТОТ ЖЕ файл с разным
    содержимым — кто последний, тот и прав;
  * compute.py читал 03_covenants.json, а covenants.py писал
    04_covenants.json — расчёт находил пустоту, а не падал;
  * evidence.py читал 04_adjustments.json, а adjustments.py писал
    06_adjustments.json — то же самое.

Эти тесты следят, чтобы связь между шагами оставалась именованной.
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline import artifacts as A  # noqa: E402

PIPELINE = ROOT / "pipeline"
MODULES = sorted(p for p in PIPELINE.glob("*.py") if p.name != "artifacts.py")


def test_no_module_builds_an_artifact_path_from_a_literal():
    """Опечатка в имени должна быть ошибкой импорта, а не тихой пустотой."""
    offenders = []
    for path in MODULES:
        for match in re.finditer(r'artifacts\s*/\s*"([^"]+)"', path.read_text(encoding="utf-8")):
            offenders.append(f"{path.name}: {match.group(1)}")
    assert offenders == [], f"имена артефактов заданы литералами: {offenders}"


#: Признаки ЗАПИСИ. Чтение и запись отличаются режимом открытия, и не
#: различать их — значит получать ложные тревоги: apply.py читает
#: LEDGER_CLEAN и был обвинён в том, что пишет его наравне с ledger.py.
#: Тест, который кричит на исправный код, перестают читать.
_WRITE_MARKERS = ('.write_text(', '.open("w"', ".open('w'", "to_csv(")


def _written_by(source: str) -> set[str]:
    """Константы артефактов, в которые модуль ПИШЕТ."""
    written = set()
    for match in re.finditer(r"artifacts / A\.([A-Z_]+)", source):
        name = match.group(1)
        window = source[match.end(): match.end() + 200]
        if any(marker in window for marker in _WRITE_MARKERS):
            written.add(name)
    return written


def test_the_write_detector_tells_reading_from_writing():
    """Проверка самой проверки: без этого различия тест обвинял
    читающий модуль в записи."""
    reading = 'path = paths.artifacts / A.LEDGER_CLEAN\n    with path.open(newline="") as fh:'
    writing = 'out = paths.artifacts / A.LEDGER_FINAL\n    with out.open("w", newline="") as fh:'
    assert _written_by(reading) == set()
    assert _written_by(writing) == {"LEDGER_FINAL"}


def test_no_two_modules_write_the_same_artifact():
    """Ровно та ошибка, из-за которой список связанных сторон затирался
    графом связей."""
    owners: dict[str, list[str]] = {}
    for path in MODULES:
        for name in _written_by(path.read_text(encoding="utf-8")):
            owners.setdefault(name, []).append(path.name)
    clashes = {
        k: sorted(set(v)) for k, v in owners.items()
        if len(set(v)) > 1 and sorted(set(v)) != sorted(A.SHARED_BY_DESIGN.get(k, ()))
    }
    assert clashes == {}, f"один артефакт пишут несколько шагов: {clashes}"


def test_the_shared_artifact_is_shared_by_exactly_the_declared_steps():
    """Исключение не должно превратиться в лазейку: если индекс начнёт
    писать кто-то третий, это снова гонка."""
    owners: dict[str, set[str]] = {}
    for path in MODULES:
        for name in _written_by(path.read_text(encoding="utf-8")):
            owners.setdefault(name, set()).add(path.name)
    for name, declared in A.SHARED_BY_DESIGN.items():
        assert owners.get(name, set()) == set(declared), (
            f"{name}: заявлено {declared}, а пишут {owners.get(name)}"
        )


def test_every_constant_is_used_somewhere():
    """Объявленное, но никем не используемое имя — след переименования,
    доведённого не до конца."""
    sources = "\n".join(p.read_text(encoding="utf-8") for p in MODULES)
    sources += "\n".join(
        p.read_text(encoding="utf-8") for p in (ROOT / "scripts").glob("*.py")
    )
    unused = [name for name in A.all_names()
              if f"A.{name}" not in sources and name not in A.SHARED_BY_DESIGN]
    assert unused == [], f"объявлены, но не используются: {unused}"


def test_names_are_unique():
    values = list(A.all_names().values())
    assert len(values) == len(set(values)), "две константы указывают на один файл"


def test_the_graph_and_the_dossier_list_are_different_files():
    """Граф связей и список связанных сторон из досье — разные сведения.
    Раньше они делили одно имя."""
    assert A.ENTITY_GRAPH != A.RELATED_PARTIES


@pytest.mark.parametrize("module,constant", [
    ("covenants", "COVENANTS"),
    ("related", "RELATED_PARTIES"),
    ("adjustments", "AUDIT_ADJUSTMENTS"),
    ("categorize", "TXN_CATEGORIES"),
])
def test_a_step_reads_back_what_it_writes(module, constant, tmp_path):
    """Круговая проверка контракта: модуль обязан уметь прочитать
    собственный артефакт. Расхождение имён ломается здесь."""
    import importlib

    mod = importlib.import_module(f"pipeline.{module}")
    assert hasattr(mod, "load"), f"{module}.load отсутствует"
    source = Path(mod.__file__).read_text(encoding="utf-8")
    assert f"A.{constant}" in source, (
        f"{module} не использует A.{constant} — имя артефакта разъехалось"
    )


# --------------------------------------------------------------------------- #
# Форма артефакта — тоже контракт
#
# Реестр имён свёл шаги к одному файлу, но ФОРМА расходилась: шаг 5 пишет
# scenarios[X] = {covenants: [...]}, а шаг 12 ожидал словарь «пункт →
# спецификация». Расчёт падал с «'str' object has no attribute 'get'»,
# потому что перебор шёл по служебным полям, а не по ковенантам.
#
# Общее имя необходимо, но недостаточно.
# --------------------------------------------------------------------------- #


def test_step12_reads_what_step5_writes(tmp_path):
    """Круговая проверка: артефакт шага 5 обязан читаться шагом 12."""
    import json as _json

    from pipeline.compute import load_tests
    from pipeline.config import RunPaths
    from pipeline.covenants import CovenantReport, ScenarioCovenants

    paths = RunPaths.create(tmp_path / "run")
    report = CovenantReport(scenarios=[ScenarioCovenants(
        scenario_id="P1", doc_id="d1", section_anchor="Финансовые ковенанты",
        covenants=[{
            "point": "6.1", "title": "t", "direction": "max", "threshold": 0.42,
            "unit": "ratio", "period_start": "2025-01-01", "period_end": "2025-12-31",
            "metric_definition": "d", "quote": "q",
            "metric": {"op": "DIV", "args": [{"op": "AGG", "category": "capex"},
                                             {"op": "AGG", "category": "opex"}]},
        }],
    )])
    (paths.artifacts / A.COVENANTS).write_text(
        _json.dumps(report.to_dict(), ensure_ascii=False), encoding="utf-8")

    tests = load_tests(paths.artifacts / A.COVENANTS)
    assert list(tests) == ["P1"]
    assert tests["P1"][0].point == "6.1"
    assert tests["P1"][0].threshold == 0.42
    assert tests["P1"][0].period == ("2025-01-01", "2025-12-31")
    assert tests["P1"][0].metric["op"] == "DIV"


def test_step12_carries_over_a_springing_condition(tmp_path):
    """Условие springing-теста хранится отдельными полями и обязано
    собраться обратно — иначе ковенант проверится всегда, а не при условии."""
    import json as _json

    from pipeline.compute import load_tests
    from pipeline.config import RunPaths

    paths = RunPaths.create(tmp_path / "run")
    (paths.artifacts / A.COVENANTS).write_text(_json.dumps({"scenarios": {"P3": {
        "covenants": [{
            "point": "6.1", "direction": "max", "threshold": 1.7, "unit": "ratio",
            "period_start": "2025-01-01", "period_end": "2025-12-31",
            "metric": {"op": "AGG", "category": "financing_inflow"},
            "is_conditional": True,
            "condition_metric": {"op": "AGG", "category": "financing_inflow"},
            "condition_direction": "max", "condition_threshold": 4000000.0,
        }]}}}, ensure_ascii=False), encoding="utf-8")

    test = load_tests(paths.artifacts / A.COVENANTS)["P3"][0]
    assert test.condition is not None
    assert test.condition["threshold"] == 4000000.0


def test_a_broken_covenant_does_not_kill_the_rest(tmp_path):
    """Один неполный ковенант не должен обнулять остальных заёмщиков."""
    import json as _json

    from pipeline.compute import load_tests
    from pipeline.config import RunPaths

    paths = RunPaths.create(tmp_path / "run")
    (paths.artifacts / A.COVENANTS).write_text(_json.dumps({"scenarios": {
        "P1": {"covenants": [{"point": "6.1"}]},
        "P2": {"covenants": [{
            "point": "6.1", "direction": "max", "threshold": 1.0,
            "metric": {"op": "AGG", "category": "opex"}}]},
    }}, ensure_ascii=False), encoding="utf-8")

    tests = load_tests(paths.artifacts / A.COVENANTS)
    assert tests["P1"] == []
    assert len(tests["P2"]) == 1
