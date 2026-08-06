"""Шаг 3: классификация типа документа по структурным маркерам.

Вход:  <run>/artifacts/01_texts/
Выход: <run>/artifacts/02_doc_index.json  (поле `type`; scenario_id добавит шаг 4)

Два принципа, оба выстраданы на публичном наборе.

ПОРЯДОК ПРАВИЛ ВАЖНЕЕ САМИХ ПРАВИЛ. Тематические слова текут между
категориями: фраза «Знай своего клиента» встречается в 18 документах, тогда
как KYC-досье всего 12 — остальные это кредитные договоры, ссылающиеся на
KYC в пункте 6.3. Правило договора обязано сработать раньше правила KYC.

МАРКЕР ПРЕДПОЧТИТЕЛЬНО ЯЗЫКОНЕЗАВИСИМЫЙ. Единственный скан корпуса потерял
всю кириллицу при распознавании, но «KYC-ACC-7806-2025» уцелел символ
в символ. Латиница и цифры переживают то, что убивает кириллицу, поэтому
первичным маркером берётся код, а текстовая формулировка — вторичным.

ЛОВУШКА, НА КОТОРОЙ ЛЕГКО ОШИБИТЬСЯ. Пометка «ИСПОЛНИТЕЛЬНЫЙ ЭКЗЕМПЛЯР»
стоит на ОБЕИХ редакциях договора — и на действующей, и на отменённой.
Признаком актуальности она не является ни при каких условиях.
"""
from __future__ import annotations

import json
import logging
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .config import RunPaths

log = logging.getLogger(__name__)


class DocType:
    #: Договор до разрешения актуальности. Шаг 3 не решает, действующий он
    #: или нет: по разъяснению организаторов критерий — период действия,
    #: написанный в самом договоре, а не текстовая пометка. Разрешение
    #: происходит на шаге 4, где известен отчётный период.
    LOAN = "LOAN"
    #: Результаты разрешения (проставляет шаг 4).
    LOAN_SUPERSEDED = "LOAN_SUPERSEDED"
    LOAN_ACTIVE = "LOAN_ACTIVE"
    AUDIT_DRAFT = "AUDIT_DRAFT"
    AUDIT_FINAL = "AUDIT_FINAL"
    KYC = "KYC"
    TREASURY_MEMO = "TREASURY_MEMO"
    BACKGROUND = "BACKGROUND"


#: Типы, участвующие в расчёте ПОСЛЕ разрешения актуальности (шаг 4).
AUTHORITATIVE = {DocType.LOAN_ACTIVE, DocType.AUDIT_FINAL, DocType.KYC, DocType.TREASURY_MEMO}
#: Типы, участвующие в расчёте ДО разрешения (выход шага 3).
CLASSIFIED_AUTHORITATIVE = {DocType.LOAN, DocType.AUDIT_FINAL, DocType.KYC, DocType.TREASURY_MEMO}
#: Типы, которые выглядят авторитетно, но использовать их нельзя.
POISONED = {DocType.LOAN_SUPERSEDED, DocType.AUDIT_DRAFT}


@dataclass(frozen=True)
class Rule:
    name: str
    doc_type: str
    pattern: str
    confidence: float
    why: str

    def compiled(self) -> re.Pattern:
        return re.compile(self.pattern, re.IGNORECASE | re.UNICODE)


#: ПОРЯДОК ЗНАЧИМ. Первое совпадение выигрывает.
RULES: tuple[Rule, ...] = (
    Rule(
        "loan", DocType.LOAN,
        r"Стать[яи]\s*6\s*[—\-–]\s*Финансовые\s+ковенанты|FINANCIAL\s+COVENANTS"
        r"|НЕДЕЙСТВУЮЩАЯ\s+РЕДАКЦИЯ|SUPERSEDED\s+AND\s+RESTATED",
        1.0,
        "Кредитный договор — любой редакции. Разделение на действующий и "
        "отменённый здесь НЕ делается: по разъяснению организаторов критерий — "
        "период действия, указанный в самом договоре, а пометка «недействующая "
        "редакция» решающей не является. Разрешение — на шаге 4.",
    ),
    Rule(
        "audit_draft", DocType.AUDIT_DRAFT,
        r"ПРОЕКТ\s*[—\-–]\s*ПРОМЕЖУТОЧНАЯ\s+ВЕДОМОСТЬ"
        r"|НЕ\s+ЯВЛЯЕТСЯ\s+ОКОНЧАТЕЛЬНОЙ\s+ПОЗИЦИЕЙ",
        1.0,
        "Черновая ведомость аудитора. Раньше финального правила: если документ "
        "помечен как черновик, он черновик, что бы ещё в нём ни нашлось.",
    ),
    Rule(
        "audit_final", DocType.AUDIT_FINAL,
        r"ДОПОЛНЕНИЕ\s+О\s+СОБЛЮДЕНИИ\s+КОВЕНАНТОВ",
        1.0,
        "Приложение к аудиторскому отчёту — единственный источник "
        "переклассификаций, курсов и внебалансовых сумм.",
    ),
    Rule(
        "kyc_code", DocType.KYC,
        # Требование только одно: после «KYC-» идёт идентификатор, где
        # где-то встречается цифра.формат самого идентификатора не
        # предполагается — «KYC-ACC-7806» и «KYC-BANK-X6» одинаково валидны.
        # Первая версия правила требовала «буквы, затем цифры» и разваливалась
        # на переименованных счетах; поймал тест переносимости.
        r"\bKYC[-\s][A-Za-z0-9][A-Za-z0-9-]*\d",
        1.0,
        "Регистрационный номер досье. Латиница с цифрами — переживает "
        "распознавание там, где кириллица теряется.",
    ),
    Rule(
        "kyc_phrase", DocType.KYC,
        r"Знай\s+своего\s+клиента|Know\s+Your\s+Customer",
        0.7,
        "Текстовая формулировка. Ниже уверенность: та же фраза встречается "
        "в договорах, поэтому правило работает только после них.",
    ),
    Rule(
        "treasury_memo", DocType.TREASURY_MEMO,
        r"Служебная\s+записка\s+казначейства|ЗАПИСКА\s+КАЗНАЧЕЙСТВА",
        1.0,
        "Казначейская записка. В публичном наборе одна на весь корпус, но "
        "содержит сумму транзакции, которой нет в реестре.",
    ),
)


def normalize(text: str) -> str:
    """Гасит артефакты PDF, ломающие поиск маркера.

    На публичном наборе фраза «Знай своего клиента» разорвана переносом
    строки в трёх документах: без нормализации они теряются.
    """
    text = text.replace("­", "")
    for dash in "‐‑‒–—―−":
        text = text.replace(dash, "-")
    return re.sub(r"\s+", " ", text)


@dataclass
class DocClass:
    doc_id: str
    type: str
    rule: str | None = None
    confidence: float = 0.0
    scenario_id: str | None = None  # заполняет шаг 4
    notes: list[str] = field(default_factory=list)


SUPERSEDED_MARK_RE = re.compile(
    r"НЕДЕЙСТВУЮЩАЯ\s+РЕДАКЦИЯ|SUPERSEDED\s+AND\s+RESTATED", re.IGNORECASE
)


def classify_text(text: str) -> DocClass:
    norm = normalize(text)
    for rule in RULES:
        if rule.compiled().search(norm):
            d = DocClass(doc_id="", type=rule.doc_type, rule=rule.name,
                         confidence=rule.confidence)
            if d.type == DocType.LOAN and SUPERSEDED_MARK_RE.search(norm):
                # Признак справочный: решает период, а не пометка. Но
                # расхождение между ними обязано быть заметно (шаг 4).
                d.notes.append("присутствует пометка об отмене редакции")
            return d
    return DocClass(doc_id="", type=DocType.BACKGROUND, rule=None, confidence=1.0)


def _all_matches(text: str) -> list[str]:
    """Все сработавшие правила — для диагностики коллизий."""
    norm = normalize(text)
    return [r.name for r in RULES if r.compiled().search(norm)]


@dataclass
class ClassifyReport:
    docs: list[DocClass] = field(default_factory=list)
    collisions: list[str] = field(default_factory=list)

    def counts(self) -> dict[str, int]:
        return dict(Counter(d.type for d in self.docs))

    def of_type(self, doc_type: str) -> list[DocClass]:
        return [d for d in self.docs if d.type == doc_type]

    def alarms(self) -> list[str]:
        """Признаки того, что маркеры не подошли к этому датасету.

        Нужны в первые минуты боевого окна: если формулировки в приватном
        наборе окажутся другими, документы молча уедут в BACKGROUND, и без
        этой проверки выяснится это слишком поздно.
        """
        out: list[str] = []
        counts = self.counts()
        total = len(self.docs) or 1

        for t in sorted(CLASSIFIED_AUTHORITATIVE):
            if counts.get(t, 0) == 0:
                out.append(
                    f"{t}: не найдено ни одного документа — маркер, вероятно, "
                    f"не подходит к этому набору, проверьте формулировки"
                )
        n_loans, n_audits = counts.get(DocType.LOAN, 0), counts.get(DocType.AUDIT_FINAL, 0)
        if n_loans and n_audits and n_loans < n_audits:
            out.append(
                f"договоров {n_loans}, финальных аудитов {n_audits} — договоров "
                f"не может быть меньше: у каждого заёмщика есть хотя бы один"
            )
        background_share = counts.get(DocType.BACKGROUND, 0) / total
        if background_share > 0.9:
            out.append(
                f"в фон ушло {background_share:.0%} документов — похоже, "
                f"маркеры не сработали почти нигде"
            )
        return out

    def to_dict(self) -> dict:
        return {
            "counts": self.counts(),
            "alarms": self.alarms(),
            "collisions": self.collisions,
            "documents": {d.doc_id: asdict(d) for d in sorted(self.docs, key=lambda x: x.doc_id)},
        }


def run(paths: RunPaths, texts_dir: Path | None = None) -> ClassifyReport:
    texts_dir = texts_dir or (paths.artifacts / "01_texts")
    if not texts_dir.is_dir():
        raise FileNotFoundError(f"Нет каталога с текстами: {texts_dir}. Сначала выполните шаг 2.")

    report = ClassifyReport()
    for path in sorted(texts_dir.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        d = classify_text(text)
        d.doc_id = path.stem

        matched = _all_matches(text)
        if len(matched) > 1:
            # Не ошибка: маркеры пересекаются по устройству корпуса. Но полезно
            # видеть, какое правило кого перебило — при разборе спорных случаев.
            report.collisions.append(f"{d.doc_id}: {matched} -> {d.rule}")
        if not text.strip():
            d.notes.append("пустой текст — классификация недостоверна")
            d.confidence = 0.0
        report.docs.append(d)

    out = paths.artifacts / "02_doc_index.json"
    out.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    for alarm in report.alarms():
        log.warning("КЛАССИФИКАЦИЯ: %s", alarm)
    log.info("Классифицировано %d документов: %s", len(report.docs), report.counts())
    return report


def load(paths: RunPaths) -> dict[str, DocClass]:
    """Чтение артефакта следующими шагами."""
    data = json.loads((paths.artifacts / "02_doc_index.json").read_text(encoding="utf-8"))
    return {k: DocClass(**v) for k, v in data["documents"].items()}
