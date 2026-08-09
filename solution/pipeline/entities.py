"""Слой сущностей и связей.

Вход:  <run>/artifacts/05_related_parties.json (шаг 8, извлечение из KYC),
       <run>/artifacts/02_doc_index.json, <run>/artifacts/01_texts/,
       подготовленный реестр
Выход: <run>/artifacts/05_entities.json

ЗАЧЕМ ОТДЕЛЬНЫЙ СЛОЙ

39% ячеек (14 из 36 в публичном наборе) отвечают на вопрос «кем приходится
этот контрагент заёмщику»:

  * 12 ковенантов о платежах связанным сторонам;
  * P5/6.1 — капитальные затраты ГРУППЫ к EBITDA заёмщика;
  * P9/6.1 — активы, переданные НЕОГРАНИЧЕННЫМ дочерним организациям.

Раньше это жило плоскими признаками `party` и `scope` в строках реестра,
причём `scope="group"` не заполнял никто — ковенант P5/6.1 посчитать было
нечем. Здесь связи становятся явными, а цепочка вывода — прослеживаемой.

ПОЧЕМУ НЕ ГРАФОВАЯ БАЗА

Структура мелкая: почти везде один переход «заёмщик → контрагент», изредка
два («заёмщик → материнская компания → её отчётность»). Графовый движок дал
бы зависимость и риск регрессии, не добавив ни балла. Связи хранятся
таблицей, но обходятся как граф.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import artifacts as A
from .config import RunPaths

log = logging.getLogger(__name__)

#: Организационно-правовые формы. Не отбрасываются при точном сравнении:
#: «Alpha JSC» и «Alpha LLP» — разные лица.
LEGAL_FORMS = ("JSC", "LLP", "LLC", "LTD", "PLC", "GMBH", "INC", "АО", "ТОО", "ЖШС", "АҚ")

#: Кавычки задаются кодами, а не литералами: в исходнике «умные» кавычки
#: визуально неотличимы от обычных, и одна из них уже потерялась при наборе.
_QUOTES = frozenset(
    '"' "'"
    "\u00ab\u00bb"                      # « »
    "\u201c\u201d\u201e\u201f"            # " " „ ‟
    "\u2018\u2019\u201a\u201b"            # ' ' ‚ ‛
    "\u2039\u203a"                      # ‹ ›
)
#: Уточнение в скобках, которое реестр добавляет к названию:
#: «Foxridge Telecom LP Holding (Ekibastuz block B)».
_PARENTHETICAL_RE = re.compile(r"\s*\([^)]*\)\s*")

#: Аббревиатура, записанная через точки: «L.L.P.», «L.L.C.», «А.О.».
#:
#: РЕАЛЬНАЯ ПОТЕРЯ ДАННЫХ. Набор пишет организационно-правовую форму
#: по-разному в досье и в реестре, причём в обе стороны:
#:
#:     P1  досье «Aktau Holdings LLP»        реестр «Aktau Holdings L.L.P.»
#:     P7  досье «Atyrau Holding Group L.L.P.» реестр «Atyrau Holding Group LLP»
#:
#: Точки гасились наравне с прочей пунктуацией, и «L.L.P.» распадалось
#: на три отдельных слова «l l p». Отбрасывание формы их не узнавало,
#: сравнение не срабатывало, и ВЕСЬ агрегат по связанным сторонам этих
#: заёмщиков выходил нулевым. Ковенант «платежи связанным сторонам»
#: показывал COMPLIANT со значением 0 при настоящих 283 664.18.
#:
#: Поэтому точечные аббревиатуры схлопываются ДО общей чистки пунктуации.
_DOTTED_ACRONYM_RE = re.compile(r"\b(?:[^\W\d_]\.){2,}")


def normalize_entity_name(name: str) -> str:
    """Приводит название к сопоставимому виду.

    Кавычки гасятся обязательно: в отсканированном досье P6 единственная
    связанная сторона записана как «"Taraz Holding Group" LLP», а в реестре
    — «Taraz Holding Group LLP». Без нормализации ковенант P6/6.1 теряет
    свою единственную связанную сторону целиком.
    """
    s = _PARENTHETICAL_RE.sub(" ", name)
    s = "".join(" " if ch in _QUOTES else ch for ch in s)
    # Сначала склеиваем «L.L.P.» в «LLP», иначе точки распадутся на слова.
    s = _DOTTED_ACRONYM_RE.sub(lambda m: m.group(0).replace(".", ""), s)
    s = re.sub(r"[.,;:]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s.casefold()


def core_name(name: str) -> str:
    """Название без организационно-правовой формы — для нестрогого сравнения."""
    parts = normalize_entity_name(name).split()
    while parts and parts[-1].upper() in LEGAL_FORMS:
        parts.pop()
    return " ".join(parts)


def suspect_unknown_legal_form(name: str) -> str | None:
    """Хвост названия, похожий на форму вне словаря LEGAL_FORMS.

    Форма вне словаря не отбрасывается при нестрогом сравнении, и «Alpha
    OOO» из досье молча не совпадёт с «Alpha ООО́» из реестра — агрегат по
    связанным сторонам обнулится без единой ошибки. Заметить это можно
    только по подозрительному хвосту: короткая аббревиатура заглавными
    (или через точки), которой нет в словаре.
    """
    tokens = name.replace('"', " ").split()
    if len(tokens) < 2:
        return None
    tail = _DOTTED_ACRONYM_RE.sub(lambda m: m.group(0).replace(".", ""), tokens[-1])
    tail = tail.strip(".,;:")
    if not (2 <= len(tail) <= 4):
        return None
    if not (tail.isalpha() and tail.isupper()):
        return None
    return tail if tail.upper() not in LEGAL_FORMS else None


# --------------------------------------------------------------------------- #
# Сущности и связи
# --------------------------------------------------------------------------- #


@dataclass
class Entity:
    name: str
    role: str                       # counterparty | parent | subsidiary
    ownership_pct: float | None = None
    is_related: bool = False
    #: Неограниченная дочерняя организация: вне периметра обеспечения
    #: по правилу из досье KYC. Осмысленно только при role="subsidiary".
    unrestricted: bool = False
    basis: str = ""                 # почему признан связанным
    source_doc: str = ""


@dataclass
class GroupFacts:
    """Показатели уровня Группы, взятые из отчётности материнской компании."""

    parent: str | None = None
    source_doc: str | None = None
    values: dict[str, float] = field(default_factory=dict)
    derivations: dict[str, str] = field(default_factory=dict)


@dataclass
class EntityGraph:
    scenario_id: str
    borrower: str = ""
    threshold_pct: float | None = None
    entities: list[Entity] = field(default_factory=list)
    group: GroupFacts = field(default_factory=GroupFacts)
    problems: list[str] = field(default_factory=list)

    def related_names(self) -> list[str]:
        return [e.name for e in self.entities if e.is_related]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["related_names"] = self.related_names()
        return d


class EntityIndex:
    """Сопоставление названия контрагента из реестра с сущностью из досье.

    Три уровня, от строгого к нестрогому. Нестрогое совпадение не молчит:
    оно возвращается вместе с пометкой, чтобы шаг 15 мог его проверить.
    """

    def __init__(self, entities: list[Entity]):
        self.entities = entities
        self._exact = {normalize_entity_name(e.name): e for e in entities}
        self._core: dict[str, list[Entity]] = {}
        for e in entities:
            self._core.setdefault(core_name(e.name), []).append(e)

    def match(self, counterparty: str) -> tuple[Entity | None, str]:
        """Возвращает (сущность, как найдено)."""
        norm = normalize_entity_name(counterparty)
        if norm in self._exact:
            return self._exact[norm], "exact"

        core = core_name(counterparty)
        candidates = self._core.get(core, [])
        if len(candidates) == 1:
            return candidates[0], "core"
        if len(candidates) > 1:
            return None, "ambiguous"
        return None, "none"


# --------------------------------------------------------------------------- #
# Показатели уровня Группы
# --------------------------------------------------------------------------- #

_MONEY = r"\$?\s*([\d]{1,3}(?:[,\s][\d]{3})*(?:\.\d{1,2})?)"
_PPE_BEGIN = re.compile(
    r"(?:Net book value at the beginning|Балансовая стоимость на начало)[^$\d]{0,40}" + _MONEY,
    re.IGNORECASE)
_PPE_END = re.compile(
    r"(?:Net book value at the end|Балансовая стоимость на конец)[^$\d]{0,40}" + _MONEY,
    re.IGNORECASE)
_PPE_DEPR = re.compile(
    r"(?:Depreciation charge|Амортизац\w+[^$\d]{0,30})[^$\d]{0,40}" + _MONEY,
    re.IGNORECASE)
#: Название материнской компании берётся ТОЛЬКО из явных признаков
#: консолидации. Свободный поиск «любое название рядом со словом Group»
#: подцеплял название банка из шапки кредитного договора: неверный родитель
#: молча отравил бы весь ковенант по Группе, поэтому здесь точность важнее
#: полноты — не нашли, значит сообщаем, а не угадываем.
_NAME = r"((?:[A-Z][\w\-&'\u2019.]*\s+){1,5}(?:JSC|LLP|LLC|Ltd|LTD|PLC))"
_PARENT_PATTERNS = (
    re.compile(_NAME + r"\s*Consolidated\s+Financial\s+Statements", re.IGNORECASE),
    re.compile(r"consolidated financial statements of\s+" + _NAME, re.IGNORECASE),
    re.compile(_NAME + r"\s+and its subsidiaries", re.IGNORECASE),
    re.compile(r"консолидированн\w+ отчётност\w+\s+" + _NAME, re.IGNORECASE),
)

_NO_DISPOSALS = re.compile(
    r"(?:no disposals|выбыти\w+ не (?:было|производилось))", re.IGNORECASE)


def _to_float(s: str) -> float:
    return float(re.sub(r"[,\s]", "", s))


def trim_leading_noise(name: str) -> str:
    """Отрезает хвост заголовка, прилипший к названию.

    В шапке отчётности название соседствует с датой и служебными словами:
    «... DECEMBER 2025 Sarybel Energy Holding JSC». Отбрасываются ведущие
    числа и слова целиком заглавными — но только если дальше есть слово
    в обычном регистре: иначе название вида «SARYBEL ENERGY HOLDING JSC»
    было бы съедено целиком.
    """
    tokens = name.split()
    has_titlecase = any(t[:1].isupper() and not t.isupper() for t in tokens)
    if not has_titlecase:
        return " ".join(tokens)
    while tokens and (tokens[0].isdigit() or (tokens[0].isupper() and len(tokens) > 1)):
        if tokens[0].upper() in LEGAL_FORMS:
            break
        tokens.pop(0)
    return " ".join(tokens)


def derive_group_capex(text: str) -> tuple[float | None, str]:
    """Выводит капитальные затраты Группы из свёртки основных средств.

    Прямой строки «капитальные затраты» в консолидированной отчётности нет.
    Она выводится из движения основных средств:

        поступления = стоимость на конец − стоимость на начало + амортизация

    Формула верна ТОЛЬКО при отсутствии выбытий, и отчётность это прямо
    оговаривает. Если оговорки нет, значение не выводится: молча посчитать
    по неприменимой формуле хуже, чем не посчитать.
    """
    flat = re.sub(r"\s+", " ", text)
    b, e, d = _PPE_BEGIN.search(flat), _PPE_END.search(flat), _PPE_DEPR.search(flat)
    if not (b and e and d):
        return None, "в отчётности не найдена свёртка основных средств"
    if not _NO_DISPOSALS.search(flat):
        return None, (
            "в отчётности нет оговорки об отсутствии выбытий — формула "
            "«конец − начало + амортизация» неприменима"
        )
    begin, end, depr = _to_float(b.group(1)), _to_float(e.group(1)), _to_float(d.group(1))
    value = end - begin + depr
    return value, (
        f"конец {end:,.2f} − начало {begin:,.2f} + амортизация {depr:,.2f} "
        f"= {value:,.2f}; выбытий не было"
    )


def find_group_parent(
    borrower: str, texts: dict[str, str], exclude: set[str] | None = None
) -> tuple[str | None, str | None]:
    """Ищет документ, в котором заёмщик назван частью группы.

    Признак — упоминание заёмщика рядом со словами о консолидации или
    дочерней структуре. Возвращает (название материнской компании, doc_id).

    Документы заёмщика из поиска НЕ исключаются. Отчётность материнской
    компании упоминает дочернюю структуру по названию, поэтому привязка
    относит её к тому же заёмщику — и исключение «своих» документов
    выбросило бы ровно тот документ, ради которого поиск и затевался.
    Защита от самосовпадения другая: найденное название обязано отличаться
    от названия заёмщика.
    """
    exclude = exclude or set()
    core = core_name(borrower)
    if not core:
        return None, None

    def acceptable(candidate: str) -> bool:
        return bool(candidate) and core_name(candidate) != core

    for doc_id, text in texts.items():
        if doc_id in exclude:
            continue
        flat = re.sub(r"\s+", " ", text)
        if core not in flat.casefold():
            continue
        for pattern in _PARENT_PATTERNS:
            m = pattern.search(flat)
            if not m:
                continue
            candidate = trim_leading_noise(m.group(1).strip())
            if acceptable(candidate):
                return candidate, doc_id
    return None, None


# --------------------------------------------------------------------------- #
# Неограниченные дочерние организации — из досье KYC
# --------------------------------------------------------------------------- #

#: Заголовок таблицы обеспечительного покрытия. Именно он отличает её от
#: таблицы долей участия выше по досье: строки обеих выглядят одинаково
#: («Название LLP 31.4%»), и без привязки к заголовку дочерние организации
#: перепутались бы с контрагентами.
_PLEDGE_HEADER_RE = re.compile(
    r"(?:дол[яи]\s+активов[^\n]{0,60}залог|залог[^\n]{0,60}дол[яи]\s+активов"
    r"|assets?\s+pledged|pledged\s+assets?|pledge\s+share|share\s+of\s+assets\s+pledged)",
    re.IGNORECASE)
_SUBSIDIARY_ROW_RE = re.compile(
    r"^\s*([^\n%]{2,70}?\b(?:" + "|".join(LEGAL_FORMS) + r")\b\.?)"
    r"\s+(\d{1,3}(?:[.,]\d+)?)\s*%\s*$",
    re.IGNORECASE | re.MULTILINE)
#: Правило «ниже X%% — неограниченная». Порог берётся ТОЛЬКО из предложения,
#: где упомянута неограниченность: числа в досье встречаются всюду.
_UNRESTRICTED_WORD_RE = re.compile(r"неограниченн|unrestricted", re.IGNORECASE)
_BELOW_PCT_RE = re.compile(
    r"(?:ниже|менее|below|under|less\s+than)\s*(\d{1,3}(?:[.,]\d+)?)\s*%",
    re.IGNORECASE)
#: Дальше этого от заголовка таблица не живёт: окно ограничено, чтобы
#: не подцепить строки следующих разделов.
_PLEDGE_WINDOW = 900


def parse_subsidiary_pledges(text: str) -> tuple[list[tuple[str, float]], float | None, list[str]]:
    """Таблица «дочерняя организация → доля активов в залоге» и порог правила.

    Возвращает (строки, порог_неограниченности, проблемы). Дочерняя
    организация с долей НИЖЕ порога — неограниченная (вне периметра
    обеспечения). Порог не найден — статус не выводится: угадать его
    значит назначить организацию неограниченной без основания.
    """
    problems: list[str] = []
    header = _PLEDGE_HEADER_RE.search(text)
    if not header:
        return [], None, problems

    window = text[header.end(): header.end() + _PLEDGE_WINDOW]
    rows = [
        (m.group(1).strip(), float(m.group(2).replace(",", ".")))
        for m in _SUBSIDIARY_ROW_RE.finditer(window)
    ]
    if not rows:
        problems.append(
            "найден заголовок обеспечительного покрытия, но ни одной строки "
            "«организация → доля» — формат таблицы не разобран"
        )
        return [], None, problems

    threshold: float | None = None
    for m in _UNRESTRICTED_WORD_RE.finditer(text):
        vicinity = text[max(0, m.start() - 300): m.start() + 300]
        pct = _BELOW_PCT_RE.search(vicinity)
        if pct:
            threshold = float(pct.group(1).replace(",", "."))
            break
    if threshold is None:
        problems.append(
            "таблица обеспечительного покрытия есть, а правило «ниже X% — "
            "неограниченная» не найдено — статусы не выводятся"
        )
    return rows, threshold, problems


# --------------------------------------------------------------------------- #
# Сборка графа
# --------------------------------------------------------------------------- #


def build_graph(
    scenario_id: str,
    borrower: str,
    kyc: dict,
    texts: dict[str, str],
    own_docs: set[str] | None = None,
    kyc_text: str = "",
) -> EntityGraph:
    """Собирает связи заёмщика из результата извлечения KYC и документов.

    `kyc` — сценарная запись шага 8: {'threshold_pct': 40.0, 'parties': [...]}.
    `kyc_text` — текст самого досье: из него детерминированно разбирается
    таблица обеспечительного покрытия дочерних организаций, которую
    извлечение шага 8 не покрывает.
    """
    graph = EntityGraph(scenario_id=scenario_id, borrower=borrower)
    threshold = kyc.get("threshold_pct")
    graph.threshold_pct = threshold

    for p in kyc.get("parties", []):
        pct = p.get("ownership_pct")
        if p.get("is_related") is not None:
            related = bool(p["is_related"])
        elif pct is not None and threshold is not None:
            related = pct >= threshold
        else:
            related = False
            graph.problems.append(
                f"{p.get('name')}: нет ни доли, ни признака связанности — "
                f"считается несвязанной, проверьте вручную"
            )
        basis = (
            f"доля {pct}% при пороге {threshold}%"
            if pct is not None and threshold is not None
            else p.get("basis") or "указано в досье"
        )
        name = (p.get("name") or "").strip()
        suspect = suspect_unknown_legal_form(name)
        if suspect:
            graph.problems.append(
                f"{name}: хвост {suspect!r} похож на организационно-правовую "
                f"форму вне словаря LEGAL_FORMS — нестрогое сопоставление "
                f"имён с реестром может молча не сработать"
            )
        graph.entities.append(Entity(
            name=name, role="counterparty",
            ownership_pct=pct, is_related=related, basis=basis,
            source_doc=kyc.get("source_doc", ""),
        ))

    if not graph.entities:
        graph.problems.append("в досье не найдено ни одной сущности — 6.3 неразрешим")

    # --- дочерние организации и периметр обеспечения ---
    if kyc_text:
        pledges, below_pct, pledge_problems = parse_subsidiary_pledges(kyc_text)
        graph.problems.extend(pledge_problems)
        for name, pct in pledges:
            unrestricted = below_pct is not None and pct < below_pct
            graph.entities.append(Entity(
                name=name, role="subsidiary", unrestricted=unrestricted,
                basis=(
                    f"доля активов в залоге {pct}% "
                    + (f"{'<' if unrestricted else '≥'} {below_pct}% — "
                       f"{'неограниченная' if unrestricted else 'ограниченная'}"
                       if below_pct is not None else "— правило не найдено")
                ),
                source_doc=kyc.get("doc_id", "") or kyc.get("source_doc", ""),
            ))

    # --- уровень Группы ---
    # own_docs не исключаются: отчётность материнской компании привязана
    # к тому же заёмщику, потому что называет его дочерней структурой.
    parent, doc_id = find_group_parent(borrower, texts)
    if parent:
        graph.group.parent = parent
        graph.group.source_doc = doc_id
        graph.entities.append(Entity(
            name=parent, role="parent", basis="заёмщик назван дочерней структурой",
            source_doc=doc_id or "",
        ))
        capex, why = derive_group_capex(texts.get(doc_id or "", ""))
        if capex is not None:
            graph.group.values["capex"] = round(capex, 2)
            graph.group.derivations["capex"] = why
        else:
            graph.problems.append(f"капзатраты Группы не выведены: {why}")
    return graph


def tag_rows(rows: list, graph: EntityGraph) -> list[str]:
    """Проставляет строкам реестра признак связанной стороны.

    Строки меняются на месте. Возвращает список замечаний: нестрогие
    совпадения и неоднозначности обязаны быть видимы.
    """
    index = EntityIndex([e for e in graph.entities if e.role == "counterparty"])
    notes: list[str] = []
    matched = 0
    for r in rows:
        entity, how = index.match(r.counterparty)
        if how == "ambiguous":
            notes.append(f"{r.txn_id}: контрагент {r.counterparty!r} совпал с несколькими сущностями")
            continue
        if entity is None:
            continue
        matched += 1
        if how == "core":
            notes.append(
                f"{r.txn_id}: {r.counterparty!r} сопоставлен с {entity.name!r} "
                f"без учёта организационно-правовой формы"
            )
        if entity.is_related:
            r.party = "related"
    if graph.entities and matched == 0:
        notes.append(
            f"{graph.scenario_id}: ни один контрагент реестра не сопоставлен "
            f"с досье — вероятно расхождение в написании названий"
        )
    return notes


def explain(txn_id: str, counterparty: str, graph: EntityGraph) -> list[str]:
    """Цепочка вывода для одной операции — прослеживаемое обоснование."""
    index = EntityIndex([e for e in graph.entities if e.role == "counterparty"])
    entity, how = index.match(counterparty)
    chain = [f"операция {txn_id} → контрагент {counterparty!r}"]
    if entity is None:
        chain.append("в досье не найден → связанной стороной не является")
        return chain
    chain.append(f"сопоставлен ({how}) с {entity.name!r} из досье {entity.source_doc}")
    chain.append(f"{entity.basis} → {'связанная сторона' if entity.is_related else 'не связанная'}")
    return chain


# --------------------------------------------------------------------------- #
# Прогон
# --------------------------------------------------------------------------- #


def run(paths: RunPaths) -> dict[str, EntityGraph]:
    from . import classify

    docs = classify.load(paths)
    texts_dir = paths.artifacts / A.TEXTS_DIR
    texts = {p.stem: p.read_text(encoding="utf-8") for p in texts_dir.glob("*.txt")}

    # ФОРМА АРТЕФАКТА — КОНТРАКТ (третий раз в этом проекте, см. artifacts.py).
    # Шаг 8 пишет {alarms, problems, scenarios: {...}}, а здесь сценарии
    # искались на ВЕРХНЕМ уровне: kyc_all.get("P5") находил пустоту, и весь
    # граф — стороны, родитель, показатели Группы — молча выходил пустым.
    kyc_path = paths.artifacts / A.RELATED_PARTIES
    kyc_raw: dict = (
        json.loads(kyc_path.read_text(encoding="utf-8")) if kyc_path.exists() else {}
    )
    kyc_all: dict[str, dict] = kyc_raw.get("scenarios", kyc_raw)
    if not kyc_all:
        log.warning("СУЩНОСТИ: нет артефакта шага 8 — связанные стороны неизвестны")

    # Название заёмщика в артефакте шага 8 не хранится — оно выучено шагом 4
    # по различительности и лежит в отчёте привязки. Без названия не найти
    # отчётность материнской компании: та упоминает заёмщика по имени.
    borrowers: dict[str, str] = {}
    attribution_path = paths.artifacts / A.ATTRIBUTION_REPORT
    if attribution_path.exists():
        learned = json.loads(
            attribution_path.read_text(encoding="utf-8")
        ).get("learned_names", {})
        for scenario, variants in learned.items():
            # Вариант в обычном регистре читабельнее ЗАГЛАВНОГО из шапки.
            titled = [v for v in variants if any(c.islower() for c in v)]
            borrowers[scenario] = (titled or variants or [""])[0]

    own_docs: dict[str, set[str]] = {}
    kyc_docs: dict[str, str] = {}
    for doc_id, d in docs.items():
        if d.scenario_id:
            own_docs.setdefault(d.scenario_id, set()).add(doc_id)
            if d.type == classify.DocType.KYC and d.scenario_id not in kyc_docs:
                kyc_docs[d.scenario_id] = doc_id

    graphs: dict[str, EntityGraph] = {}
    for scenario in sorted(own_docs):
        kyc = kyc_all.get(scenario, {})
        kyc_doc = kyc.get("doc_id") or kyc_docs.get(scenario, "")
        graphs[scenario] = build_graph(
            scenario, borrowers.get(scenario, ""), kyc,
            texts, own_docs=own_docs[scenario],
            kyc_text=texts.get(kyc_doc, ""),
        )

    (paths.artifacts / A.ENTITY_GRAPH).write_text(
        json.dumps({s: g.to_dict() for s, g in graphs.items()},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    for s, g in graphs.items():
        for p in g.problems:
            log.warning("СУЩНОСТИ %s: %s", s, p)
    log.info(
        "Собрано графов: %d, связанных сторон всего: %d, показатели Группы у %d",
        len(graphs), sum(len(g.related_names()) for g in graphs.values()),
        sum(1 for g in graphs.values() if g.group.values),
    )
    return graphs


def load(paths: RunPaths) -> dict[str, EntityGraph]:
    """Читает обратно 05_entities.json — для шагов 11 и 12.

    Лишние ключи (related_names — производное поле) отбрасываются,
    отсутствующие получают значения по умолчанию: артефакт из старого
    прогона не должен ронять новый код.
    """
    path = paths.artifacts / A.ENTITY_GRAPH
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        log.warning("СУЩНОСТИ: артефакт не читается: %s", exc)
        return {}

    entity_fields = {f for f in Entity.__dataclass_fields__}
    out: dict[str, EntityGraph] = {}
    for scenario, payload in data.items():
        if not isinstance(payload, dict):
            continue
        graph = EntityGraph(
            scenario_id=scenario,
            borrower=payload.get("borrower", ""),
            threshold_pct=payload.get("threshold_pct"),
            problems=payload.get("problems", []),
        )
        for e in payload.get("entities", []):
            if isinstance(e, dict):
                graph.entities.append(
                    Entity(**{k: v for k, v in e.items() if k in entity_fields})
                )
        group = payload.get("group") or {}
        graph.group = GroupFacts(
            parent=group.get("parent"),
            source_doc=group.get("source_doc"),
            values=group.get("values", {}) or {},
            derivations=group.get("derivations", {}) or {},
        )
        out[scenario] = graph
    return out
