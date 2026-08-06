"""Тесты шага 4: привязка документов к заёмщикам.

Главное здесь — что ни одно имя счёта и ни одно название компании
не зашито в код: всё выводится из данных в рантайме.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline import attribute  # noqa: E402
from pipeline.attribute import (  # noqa: E402
    attribute_by_name,
    build_account_map,
    check_invariants,
    learn_company_names,
    normalize_name,
)
from pipeline.classify import DocClass, DocType  # noqa: E402


def write_ledger(path: Path, rows: list[tuple[str, str]]) -> Path:
    lines = ["txn_id,date,account_id,counterparty,description,amount,currency"]
    lines += [f"{t},2025-01-01,{a},CP,desc,-100.00,USD" for t, a in rows]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# Карта счетов
# --------------------------------------------------------------------------- #


def test_account_map_is_derived_from_ledger(tmp_path):
    """Идентификаторы приватного набора будут другими — карта обязана
    строиться в рантайме, а не браться из константы."""
    led = write_ledger(tmp_path / "l.csv", [
        ("TXN-ALPHA-0001", "BANK-001"),
        ("TXN-ALPHA-0002", "BANK-001"),
        ("TXN-BETA-0001", "BANK-002"),
    ])
    mapping, problems = build_account_map(led, {"ALPHA", "BETA"})
    assert mapping == {"BANK-001": "ALPHA", "BANK-002": "BETA"}
    assert problems == []


def test_noise_accounts_are_excluded_by_template_not_by_prefix(tmp_path):
    """В публичном наборе шум имеет префикс 9xxx, но отсекать надо
    по отсутствию сценария в шаблоне — префикс может быть любым."""
    led = write_ledger(tmp_path / "l.csv", [
        ("TXN-P1-0001", "ACC-7801"),
        ("TXN-9001-0001", "ACC-9001"),
        ("TXN-ZZZ-0001", "ACC-1234"),
    ])
    mapping, _ = build_account_map(led, {"P1"})
    assert mapping == {"ACC-7801": "P1"}


def test_account_shared_between_scenarios_is_reported(tmp_path):
    led = write_ledger(tmp_path / "l.csv", [
        ("TXN-P1-0001", "ACC-1"), ("TXN-P1-0002", "ACC-1"), ("TXN-P2-0001", "ACC-1"),
    ])
    mapping, problems = build_account_map(led, {"P1", "P2"})
    assert mapping == {"ACC-1": "P1"}, "берётся преобладающий"
    assert problems and "нескольких сценариев" in problems[0]


def test_account_pattern_prefers_longer_id():
    pat = attribute._account_pattern(["ACC-780", "ACC-7801"])
    assert pat.findall("счёт ACC-7801 указан") == ["ACC-7801"]
    assert pat.findall("счёт ACC-780 указан") == ["ACC-780"]


# --------------------------------------------------------------------------- #
# Обучение названий
# --------------------------------------------------------------------------- #


def test_normalize_name_flattens_and_strips_layout_noise():
    assert normalize_name("Almaty Cold\nChain JSC") == "Almaty Cold Chain JSC"
    assert normalize_name("Организация Aktau Port Services JSC") == "Aktau Port Services JSC"
    assert normalize_name("За Кредитора\nEkibastuz Energy JSC") == "Ekibastuz Energy JSC"


def test_auditor_names_are_not_learned_as_borrowers():
    """Аудитор обслуживает нескольких заёмщиков — различительность
    обязана его отсеять без всякого списка аудиторов."""
    texts = {
        "d1": "Aktau Port Services JSC. Аудитор: Turan Verity Audit LLP",
        "d2": "Aktau Port Services JSC. Отчёт Turan Verity Audit LLP",
        "d3": "Almaty Cold Chain JSC. Аудитор: Turan Verity Audit LLP",
        "d4": "Almaty Cold Chain JSC. Отчёт Turan Verity Audit LLP",
    }
    learned = learn_company_names(texts, {"d1": "P1", "d2": "P1", "d3": "P2", "d4": "P2"})
    flat = [n for names in learned.values() for n in names]
    assert "Aktau Port Services JSC" in flat
    assert "Almaty Cold Chain JSC" in flat
    assert "Turan Verity Audit LLP" not in flat


def test_rare_name_is_not_learned():
    """Название, мелькнувшее в одном документе из многих, не является
    признаком заёмщика."""
    texts = {f"d{i}": "Aktau Port Services JSC" for i in range(10)}
    texts["d0"] += " однократно Random Counterparty LLP"
    learned = learn_company_names(texts, {f"d{i}": "P1" for i in range(10)})
    assert "Random Counterparty LLP" not in learned.get("P1", [])


# --------------------------------------------------------------------------- #
# Привязка по названию
# --------------------------------------------------------------------------- #


def test_longest_match_resolves_prefix_collision():
    """Критическая коллизия корпуса: «Shymkent Refinery JSC» — один
    заёмщик, «Shymkent Refinery Services JSC» — другой."""
    learned = {
        "B4": ["Shymkent Refinery JSC"],
        "P3": ["Shymkent Refinery Services JSC"],
    }
    assert attribute_by_name("Отчёт Shymkent Refinery Services JSC", learned)[0] == "P3"
    assert attribute_by_name("Отчёт Shymkent Refinery JSC", learned)[0] == "B4"


def test_name_match_survives_line_breaks():
    learned = {"P2": ["Almaty Cold Chain JSC"]}
    assert attribute_by_name("... Almaty Cold\nChain JSC ...", learned)[0] == "P2"


def test_equal_length_conflict_is_ambiguous_not_guessed():
    learned = {"P1": ["Alpha Works JSC"], "P2": ["Omega Works JSC"]}
    scenario, ambiguous = attribute_by_name("Alpha Works JSC и Omega Works JSC", learned)
    assert scenario is None and ambiguous is True


def test_no_match_returns_none():
    assert attribute_by_name("посторонний текст", {"P1": ["Alpha JSC"]}) == (None, False)


# --------------------------------------------------------------------------- #
# Инварианты
# --------------------------------------------------------------------------- #


def _docs(**per_scenario) -> dict[str, DocClass]:
    out, i = {}, 0
    for scenario, types in per_scenario.items():
        for t in types:
            out[f"d{i}"] = DocClass(doc_id=f"d{i}", type=t, scenario_id=scenario)
            i += 1
    return out


def test_invariants_pass_on_complete_scenario():
    docs = _docs(P1=[DocType.LOAN_ACTIVE, DocType.AUDIT_FINAL, DocType.KYC])
    assert check_invariants(docs, {"P1"}) == []


def test_invariant_catches_two_active_loans():
    """Два действующих договора означают, что отменённая редакция
    не распознана — это главная ловушка набора."""
    docs = _docs(P1=[DocType.LOAN_ACTIVE, DocType.LOAN_ACTIVE, DocType.AUDIT_FINAL, DocType.KYC])
    problems = check_invariants(docs, {"P1"})
    assert any("должен быть один" in p for p in problems)


def test_invariant_catches_missing_documents():
    docs = _docs(P1=[DocType.LOAN_ACTIVE])
    problems = " | ".join(check_invariants(docs, {"P1", "P2"}))
    assert "нет финального аудита" in problems
    assert "нет KYC" in problems
    assert "P2: нет действующего договора" in problems


# --------------------------------------------------------------------------- #
# Реальный корпус
# --------------------------------------------------------------------------- #


# attributed — сессионная фикстура из conftest.py


@pytest.mark.slow
def test_corpus_attribution_is_clean(attributed):
    docs, rep = attributed
    assert len(set(rep.account_to_scenario.values())) == 12
    assert rep.ambiguous == []
    assert rep.problems == [], f"проблемы: {rep.problems}"
    assert rep.by_account + rep.by_name == 192
    assert len(rep.orphans) == 10


@pytest.mark.slow
def test_every_orphan_is_genuine_noise(attributed):
    """Сирота авторитетного типа означала бы выпадение документа
    из расчёта. Все сироты обязаны быть фоном."""
    docs, rep = attributed
    assert all(docs[o].type == DocType.BACKGROUND for o in rep.orphans)


@pytest.mark.slow
def test_scan_is_attributed_despite_broken_ocr(attributed):
    docs, _ = attributed
    assert docs["f3fa6d20c8a1"].scenario_id == "P6"


@pytest.mark.slow
def test_group_parent_report_is_found_by_name(attributed):
    """Консолидированная отчётность Sarybel Energy Holding не содержит
    ни номера счёта, ни имени заёмщика в шапке — но упоминает дочернюю
    структуру. Я однажды уже отнёс её к шуму по ошибке."""
    docs, _ = attributed
    assert docs["a5cc1400b640"].scenario_id == "P5"


@pytest.mark.slow
def test_each_scenario_has_the_full_document_set(attributed):
    docs, rep = attributed
    for scenario in set(rep.account_to_scenario.values()):
        mine = [d for d in docs.values() if d.scenario_id == scenario]
        types = [d.type for d in mine]
        assert types.count(DocType.LOAN_ACTIVE) == 1, scenario
        assert types.count(DocType.AUDIT_FINAL) == 1, scenario
        assert types.count(DocType.KYC) >= 1, scenario


@pytest.mark.slow
def test_attribution_survives_renamed_identifiers(tmp_path, public_dataset, full_run):
    """Предвестник шага 16: в приватном наборе будут другие счета
    и другие scenario_id. Привязка обязана работать без правок кода."""
    import shutil

    from pipeline import classify
    from pipeline.config import RunPaths, discover_dataset

    _, orig_run = full_run          # переиспользуем единственное извлечение сессии

    root = tmp_path / "renamed"
    (root / "documents").mkdir(parents=True)

    rename = {f"ACC-780{i}": f"BANK-X{i}" for i in range(1, 10)}
    rename.update({"ACC-7810": "BANK-X10", "ACC-7201": "BANK-Y1", "ACC-7204": "BANK-Y4"})
    scen = {f"P{i}": f"CASE{i}" for i in range(1, 11)}
    scen.update({"B1": "CASE11", "B4": "CASE12"})

    def swap(text: str) -> str:
        for old, new in sorted(rename.items(), key=lambda kv: -len(kv[0])):
            text = text.replace(old, new)
        return text

    # PDF нужны только для обнаружения датасета; тексты подменяем напрямую.
    for p in list(public_dataset.documents_dir.glob("*.pdf"))[:6]:
        shutil.copy(p, root / "documents" / p.name)

    rp = RunPaths.create(root / "run")
    (rp.artifacts / "01_texts").mkdir(parents=True, exist_ok=True)
    for t in (orig_run.artifacts / "01_texts").glob("*.txt"):
        (rp.artifacts / "01_texts" / t.name).write_text(
            swap(t.read_text(encoding="utf-8")), encoding="utf-8"
        )

    ledger_lines = public_dataset.ledger_csv.read_text(encoding="utf-8")
    for old, new in sorted(rename.items(), key=lambda kv: -len(kv[0])):
        ledger_lines = ledger_lines.replace(old, new)
    for old, new in sorted(scen.items(), key=lambda kv: -len(kv[0])):
        ledger_lines = ledger_lines.replace(f"TXN-{old}-", f"TXN-{new}-")
    (root / "ledger.csv").write_text(ledger_lines, encoding="utf-8")

    tpl = json.loads(public_dataset.template_json.read_text(encoding="utf-8"))
    tpl["answers"] = {scen[k]: v for k, v in tpl["answers"].items()}
    (root / "tpl.json").write_text(json.dumps(tpl, ensure_ascii=False), encoding="utf-8")

    ds = discover_dataset(root)
    classify.run(rp)
    docs, rep = attribute.run(ds, rp)

    assert set(rep.account_to_scenario.values()) == set(scen.values())
    assert rep.problems == [], f"на переименованном наборе: {rep.problems}"
    assert rep.by_account + rep.by_name >= 190


# --------------------------------------------------------------------------- #
# Актуальность редакции: вторая ось, независимая от текстовой пометки
# --------------------------------------------------------------------------- #

from pipeline.attribute import (  # noqa: E402
    all_dates,
    document_date,
    infer_reporting_period,
    resolve_revisions,
)


def test_document_date_prefers_header_over_latest_mention():
    """В договоре самая поздняя дата — конец ковенантного периода,
    а не дата подписания. Максимум по датам здесь даёт неверный ответ."""
    loan = (
        "ДОГОВОР БАНКОВСКОГО ЗАЙМА г. Алматы · от 1 января 2025 года ... "
        "за период с 2025-01-01 по 2025-12-31 ..."
    )
    assert document_date(loan) == "2025-01-01"


def test_document_date_uses_labelled_date_for_dossier():
    kyc = "Досье «Знай своего клиента» Счёт ACC-1 Дата проверки 31 декабря 2025 года"
    assert document_date(kyc) == "2025-12-31"


def test_document_date_falls_back_to_signature_block():
    audit = "ПРИМЕЧАНИЯ К ОТЧЁТНОСТИ ... За аудитора и от его имени 31 декабря 2025 года"
    assert document_date(audit) == "2025-12-31"


def test_document_date_none_when_absent():
    assert document_date("текст без единой даты") is None


def test_all_dates_collects_both_formats():
    assert all_dates("от 1 января 2025 года и 2024-06-15") == ["2025-01-01", "2024-06-15"]


def test_stale_revision_without_marker_is_demoted():
    """Случая нет в публичном наборе: там устаревший договор помечен
    текстом. В приватном он может отличаться только датой."""
    docs = {
        "new": DocClass("new", DocType.LOAN_ACTIVE, scenario_id="P1"),
        "old": DocClass("old", DocType.LOAN_ACTIVE, scenario_id="P1"),
    }
    texts = {
        "new": "ДОГОВОР от 1 января 2025 года",
        "old": "ДОГОВОР от 1 января 2024 года",
    }
    problems = resolve_revisions(docs, texts, ("2025-01-01", "2025-12-31"))

    assert docs["new"].type == DocType.LOAN_ACTIVE
    assert docs["old"].type == DocType.LOAN_ACTIVE + "__STALE"
    assert any("вытеснен более актуальным new" in n for n in docs["old"].notes)
    assert problems and "актуальным принят new" in problems[0]


def test_in_period_beats_later_date():
    """Документ следующего года новее, но отчётный период важнее давности."""
    docs = {
        "cur": DocClass("cur", DocType.KYC, scenario_id="P1"),
        "future": DocClass("future", DocType.KYC, scenario_id="P1"),
    }
    texts = {
        "cur": "Дата проверки 31 декабря 2025 года",
        "future": "Дата проверки 15 марта 2026 года",
    }
    resolve_revisions(docs, texts, ("2025-01-01", "2025-12-31"))
    assert docs["cur"].type == DocType.KYC
    assert docs["future"].type == DocType.KYC + "__STALE"


def test_undated_duplicates_are_flagged_not_guessed():
    docs = {
        "a": DocClass("a", DocType.AUDIT_FINAL, scenario_id="P1"),
        "b": DocClass("b", DocType.AUDIT_FINAL, scenario_id="P1"),
    }
    problems = resolve_revisions(docs, {"a": "без даты", "b": "тоже без даты"},
                                 ("2025-01-01", "2025-12-31"))
    assert any("ни у одного нет даты" in p for p in problems)
    assert docs["a"].type == DocType.AUDIT_FINAL and docs["b"].type == DocType.AUDIT_FINAL


def test_single_document_is_left_alone():
    docs = {"a": DocClass("a", DocType.KYC, scenario_id="P1")}
    assert resolve_revisions(docs, {"a": "Дата проверки 31 декабря 2025 года"},
                             ("2025-01-01", "2025-12-31")) == []
    assert docs["a"].type == DocType.KYC


@pytest.mark.slow
def test_corpus_has_no_revision_conflicts(attributed):
    """У каждого заёмщика два договора, и разрешение по периоду обязано
    выбрать один. Записи о выборе информационные; настоящих конфликтов
    (пропавший период, расхождение с пометкой) быть не должно."""
    docs, rep = attributed
    resolutions = [r for r in rep.revisions if "действующим принят" in r]
    assert len(resolutions) == 12

    assert not [r for r in rep.revisions if "не найден период" in r]
    assert not [r for r in rep.revisions if "Решает период" in r]
    assert not [r for r in rep.revisions if "перекрывает" in r]

    assert rep.reporting_period == ("2025-01-02", "2025-12-31")
    assert not [d for d in docs.values() if d.type.endswith("__STALE")]


@pytest.mark.slow
def test_corpus_document_dates_are_extracted(attributed):
    docs, _ = attributed
    dated = {}
    for d in docs.values():
        for n in d.notes:
            if n.startswith("дата документа"):
                dated.setdefault(d.type, set()).add(n.split(": ")[1])
    assert dated["LOAN_ACTIVE"] == {"2025-01-01"}
    assert dated["AUDIT_FINAL"] == {"2025-12-31"}


# --------------------------------------------------------------------------- #
# Действующий договор выбирается ПО ПЕРИОДУ, а не по пометке
# --------------------------------------------------------------------------- #

from pipeline.attribute import (  # noqa: E402
    covenant_period,
    padded_period,
    resolve_active_loans,
)

ART6 = "Статья 6 — Финансовые ковенанты Пункт 6.1 ... за период с {a} по {b} превышал 0.42x"


def test_covenant_period_is_read_from_article_six():
    assert covenant_period(ART6.format(a="2025-01-01", b="2025-12-31")) == (
        "2025-01-01", "2025-12-31")


def test_covenant_period_reads_russian_dates():
    text = "Финансовые ковенанты. Обязуется за период с 1 апреля 2025 года по 31 марта 2026 года ..."
    assert covenant_period(text) == ("2025-04-01", "2026-03-31")


def test_covenant_period_none_when_absent():
    assert covenant_period("Договор без указания периода") is None


def test_period_beats_the_superseded_mark():
    """Правило организаторов: «отметка "недействующая редакция" не влияет».
    Если период помеченного договора соответствует отчётному, а у другого
    нет — действующим считается помеченный."""
    docs = {
        "marked": DocClass("marked", DocType.LOAN, scenario_id="P1",
                           notes=["присутствует пометка об отмене редакции"]),
        "clean": DocClass("clean", DocType.LOAN, scenario_id="P1"),
    }
    texts = {
        "marked": ART6.format(a="2025-01-01", b="2025-12-31"),
        "clean": ART6.format(a="2023-01-01", b="2023-12-31"),
    }
    problems = resolve_active_loans(docs, texts, ("2025-01-02", "2025-12-31"))

    assert docs["marked"].type == DocType.LOAN_ACTIVE
    assert docs["clean"].type == DocType.LOAN_SUPERSEDED
    assert any("стоит пометка об" in p and "Решает период" in p for p in problems)


def test_period_selection_works_without_any_mark():
    """Устаревший договор в приватном наборе может не иметь пометки вовсе."""
    docs = {
        "new": DocClass("new", DocType.LOAN, scenario_id="P1"),
        "old": DocClass("old", DocType.LOAN, scenario_id="P1"),
    }
    texts = {
        "new": ART6.format(a="2025-01-01", b="2025-12-31"),
        "old": ART6.format(a="2024-01-01", b="2024-12-31"),
    }
    resolve_active_loans(docs, texts, ("2025-01-02", "2025-12-31"))
    assert docs["new"].type == DocType.LOAN_ACTIVE
    assert docs["old"].type == DocType.LOAN_SUPERSEDED
    assert any("отклонён по периоду" in n for n in docs["old"].notes)


def test_non_calendar_period_is_handled():
    """Период не обязан быть календарным годом — это прямо следует
    из ответа организаторов «в каждом договоре написан период его действия»."""
    docs = {
        "fy": DocClass("fy", DocType.LOAN, scenario_id="P1"),
        "prev": DocClass("prev", DocType.LOAN, scenario_id="P1"),
    }
    texts = {
        "fy": ART6.format(a="2025-04-01", b="2026-03-31"),
        "prev": ART6.format(a="2024-04-01", b="2025-03-31"),
    }
    resolve_active_loans(docs, texts, ("2025-04-05", "2026-03-28"))
    assert docs["fy"].type == DocType.LOAN_ACTIVE


def test_single_loan_is_active_without_ceremony():
    docs = {"only": DocClass("only", DocType.LOAN, scenario_id="P1")}
    problems = resolve_active_loans(
        docs, {"only": ART6.format(a="2025-01-01", b="2025-12-31")},
        ("2025-01-02", "2025-12-31"))
    assert docs["only"].type == DocType.LOAN_ACTIVE
    assert problems == []


def test_missing_period_is_flagged_not_guessed():
    docs = {"a": DocClass("a", DocType.LOAN, scenario_id="P1")}
    problems = resolve_active_loans(docs, {"a": "договор без периода"},
                                    ("2025-01-02", "2025-12-31"))
    assert docs["a"].type == DocType.LOAN_ACTIVE
    assert any("не найден период" in p for p in problems)


def test_reporting_period_is_not_rounded_to_calendar_year(tmp_path):
    """Округление до календарного года сломало бы период «апрель — март»."""
    led = tmp_path / "l.csv"
    led.write_text(
        "txn_id,date,account_id,counterparty,description,amount,currency\n"
        "TXN-P1-0001,2025-04-05,A,CP,d,-1,USD\n"
        "TXN-P1-0002,2026-03-28,A,CP,d,-1,USD\n",
        encoding="utf-8",
    )
    from pipeline.attribute import infer_reporting_period

    assert infer_reporting_period(led, {"P1"}) == ("2025-04-05", "2026-03-28")


def test_padding_only_applies_to_document_date_checks():
    assert padded_period(("2025-01-02", "2025-12-31"), days=45)[0] < "2025-01-01"
    assert padded_period(None) is None


@pytest.mark.slow
def test_corpus_resolution_gives_exactly_one_active_loan(attributed):
    docs, rep = attributed
    active = [d for d in docs.values() if d.type == DocType.LOAN_ACTIVE]
    stale = [d for d in docs.values() if d.type == DocType.LOAN_SUPERSEDED]
    assert len(active) == 12 and len(stale) == 12
    assert len({d.scenario_id for d in active}) == 12
    assert all(any("период договора: 2025-01-01..2025-12-31" in n for n in d.notes)
               for d in active)


@pytest.mark.slow
def test_period_and_mark_agree_on_public_corpus(attributed):
    """На публичном наборе оба признака согласны — расхождений быть не должно."""
    _, rep = attributed
    assert not [p for p in rep.revisions if "Решает период" in p]
