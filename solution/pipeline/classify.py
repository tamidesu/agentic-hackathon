"""Шаг 3: классификация типа документа по структурным маркерам.

Вход:  <run>/artifacts/01_texts/
Выход: <run>/artifacts/02_doc_index.json  (поле `type`; scenario_id добавит шаг 4)

Два принципа, оба выстраданы на публичном наборе.

ПОРЯДОК ПРАВИЛ ВАЖНЕЕ САМИХ ПРАВИЛ. Тематические слова текут между
категориями: фраза «Знай своего клиента» встречается в 18 документах, тогда
как KYC-досье всего 12 — остальные это кредитные договоры, ссылающиеся на
KYC в пункте 6.3. Правило договора обязано сработать раньше правила KYC.

ЯЗЫК НЕ ГАРАНТИРОВАН. Организаторы предупредили: документы приватного
набора «могут быть на английском, но в основном на русском». Поэтому
каждое правило несёт обе формулировки, а английские варианты помечены как
ГИПОТЕЗЫ — они выведены из парных строк публичного набора (там, где
русская пометка уже соседствует с английской: «НЕДЕЙСТВУЮЩАЯ РЕДАКЦИЯ» /
«SUPERSEDED AND RESTATED»), но подтвердить их не на чем. Настоящая
страховка — не угадать формулировку, а ЗАМЕТИТЬ промах: `alarms()` ловит
нулевые категории, а `llm_fallback()` доклассифицирует то, что уехало
в фон. Правила бесплатны, модель включается только там, где они видимо
не сработали.

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

from . import artifacts as A
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
        r"|НЕ\s+ЯВЛЯЕТСЯ\s+ОКОНЧАТЕЛЬНОЙ\s+ПОЗИЦИЕЙ"
        # Гипотезы английских форм. «DRAFT» в одиночку намеренно НЕ берётся:
        # слово стоит в колонтитулах документов, черновиками не являющихся,
        # а ложный AUDIT_DRAFT выбрасывает настоящие корректировки.
        r"|DRAFT\s*[—\-–]\s*INTERIM\s+SCHEDULE"
        r"|NOT\s+(?:THE\s+)?FINAL\s+POSITION"
        r"|SUBJECT\s+TO\s+(?:FURTHER\s+)?REVIEW",
        1.0,
        "Черновая ведомость аудитора. Раньше финального правила: если документ "
        "помечен как черновик, он черновик, что бы ещё в нём ни нашлось.",
    ),
    Rule(
        "audit_final", DocType.AUDIT_FINAL,
        r"ДОПОЛНЕНИЕ\s+О\s+СОБЛЮДЕНИИ\s+КОВЕНАНТОВ"
        r"|COVENANT\s+COMPLIANCE\s+(?:SUPPLEMENT|SCHEDULE|ANNEX)",
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
        r"Служебная\s+записка\s+казначейства|ЗАПИСКА\s+КАЗНАЧЕЙСТВА"
        r"|TREASURY\s+(?:MEMORANDUM|MEMO)\b",
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
    r"НЕДЕЙСТВУЮЩАЯ\s+РЕДАКЦИЯ|SUPERSEDED\s+AND\s+RESTATED"
    r"|NO\s+LONGER\s+IN\s+(?:FORCE|EFFECT)", re.IGNORECASE
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
    fallback_notes: list[str] = field(default_factory=list)

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
            "fallback_notes": self.fallback_notes,
            "documents": {d.doc_id: asdict(d) for d in sorted(self.docs, key=lambda x: x.doc_id)},
        }


# --------------------------------------------------------------------------- #
# Запасной путь: доклассификация моделью
# --------------------------------------------------------------------------- #

#: Сколько символов документа показывать модели. Тип определяется шапкой:
#: заголовок, реквизиты, первая пара разделов. Слать документ целиком —
#: платить за 40 000 знаков ради ответа, который виден в первых двух тысячах.
FALLBACK_HEAD_CHARS = 3000
#: Предохранитель от разорительного прогона, если промахнулись все правила
#: разом: в фон уедет весь корпус, и без потолка это сотни вызовов.
FALLBACK_MAX_DOCS = 220

_FALLBACK_PROMPT = """Определи тип финансового документа. Документ может быть \
на русском или на английском языке.

Типы:
  LOAN          — кредитный договор (содержит статью о финансовых ковенантах)
  AUDIT_DRAFT   — ЧЕРНОВАЯ ведомость аудитора: помечена как проект, промежуточная,
                  не окончательная позиция, подлежит пересмотру
  AUDIT_FINAL   — ОКОНЧАТЕЛЬНОЕ приложение аудитора о соблюдении ковенантов:
                  переклассификации, курсы валют, внебалансовые суммы
  KYC           — досье «Знай своего клиента» на контрагента
  TREASURY_MEMO — служебная записка казначейства о конкретном платеже
  BACKGROUND    — всё остальное: новости, справки, переписка, маркетинг

Различие AUDIT_DRAFT и AUDIT_FINAL критично: черновик содержит цифры, \
которые в итоге ИЗМЕНИЛИСЬ. Если документ помечен как проект — это AUDIT_DRAFT, \
что бы ещё в нём ни было.

Приведи в evidence_quote ДОСЛОВНУЮ строку из документа, по которой ты определил \
тип. Если такой строки нет — тип BACKGROUND.

ДОКУМЕНТ:
"""


def llm_fallback(
    report: ClassifyReport,
    texts: dict[str, str],
    llm,
    model: str | None = None,
    force: bool = False,
) -> list[str]:
    """Доклассифицирует документы, уехавшие в фон, — если правила промахнулись.

    ЗАЧЕМ. Правила опознают формулировку, а формулировка в приватном наборе
    может оказаться иной — организаторы предупредили про английский язык.
    Промах правила выглядит не как ошибка, а как тишина: документ спокойно
    лежит в BACKGROUND, аудиторские корректировки не применяются, и ковенант
    считается по неверным цифрам. Молчаливая деградация — худший из режимов
    отказа, поэтому здесь она превращается в вызов модели.

    КОГДА. Только при сработавших `alarms()`: если каждая нужная категория
    представлена, правила работают, и платить не за что. `force=True` —
    для проверки самого механизма.

    ЧЕМ РИСКУЕМ. Модель может назвать фоном настоящий документ или наоборот.
    Поэтому: (1) результат применяется только к документам, которые правила
    НЕ опознали — вердикт правила модель переписать не может; (2) требуется
    дословная цитата, и она проверяется на вхождение в текст; (3) низкая
    уверенность не применяется. Правила остаются главным источником истины.
    """
    alarms = report.alarms()
    if not alarms and not force:
        return []
    if llm is None:
        return [f"правила промахнулись ({len(alarms)} тревог), но модель недоступна"]

    from .schemas import DOC_TYPE_SCHEMA
    from .llm import LLMRequest

    unknown = [d for d in report.docs if d.type == DocType.BACKGROUND]
    notes = [f"запасная классификация моделью: тревог {len(alarms)}, "
             f"кандидатов {len(unknown)}"]
    if len(unknown) > FALLBACK_MAX_DOCS:
        notes.append(f"кандидатов больше потолка {FALLBACK_MAX_DOCS} — взяты первые")
        unknown = unknown[:FALLBACK_MAX_DOCS]

    def request(d: DocClass) -> LLMRequest:
        head = (texts.get(d.doc_id, "") or "")[:FALLBACK_HEAD_CHARS]
        kwargs = {"model": model} if model else {}
        return LLMRequest(
            prompt=_FALLBACK_PROMPT + head,
            schema=DOC_TYPE_SCHEMA,
            prompt_version="classify-fallback-v1",
            **kwargs,
        )

    changed = 0
    for d in unknown:
        try:
            res = llm.extract(request(d))
        except Exception as exc:  # noqa: BLE001 — один документ не должен ронять шаг
            d.notes.append(f"запасная классификация не удалась: {type(exc).__name__}")
            continue
        payload = res.data
        new_type = payload.get("doc_type", DocType.BACKGROUND)
        quote = (payload.get("evidence_quote") or "").strip()
        confidence = float(payload.get("confidence") or 0.0)
        if new_type == DocType.BACKGROUND:
            continue
        if confidence < 0.6:
            d.notes.append(f"модель предположила {new_type}, но уверенность {confidence:.2f}")
            continue
        if quote and normalize(quote) not in normalize(texts.get(d.doc_id, "")):
            # Цитата, которой нет в документе, — признак выдумки. Такой
            # вердикт не применяется, но и не замалчивается.
            d.notes.append(f"модель предположила {new_type}, но цитата не найдена в тексте")
            continue
        d.type, d.rule, d.confidence = new_type, "llm_fallback", confidence
        d.notes.append(f"тип определён моделью по строке: {quote[:120]}")
        changed += 1

    notes.append(f"переклассифицировано {changed}")
    return notes


def run(paths: RunPaths, texts_dir: Path | None = None,
        llm=None, model: str | None = None) -> ClassifyReport:
    texts_dir = texts_dir or (paths.artifacts / A.TEXTS_DIR)
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

    if report.alarms() and llm is not None:
        texts = {p.stem: p.read_text(encoding="utf-8") for p in texts_dir.glob("*.txt")}
        for note in llm_fallback(report, texts, llm, model):
            log.warning("КЛАССИФИКАЦИЯ: %s", note)
            report.fallback_notes.append(note)

    out = paths.artifacts / A.DOC_INDEX
    out.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    for alarm in report.alarms():
        log.warning("КЛАССИФИКАЦИЯ: %s", alarm)
    log.info("Классифицировано %d документов: %s", len(report.docs), report.counts())
    return report


def load(paths: RunPaths) -> dict[str, DocClass]:
    """Чтение артефакта следующими шагами."""
    data = json.loads((paths.artifacts / A.DOC_INDEX).read_text(encoding="utf-8"))
    return {k: DocClass(**v) for k, v in data["documents"].items()}
