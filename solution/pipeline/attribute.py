"""Шаг 4: привязка документов к заёмщикам.

Вход:  <run>/artifacts/02_doc_index.json, <run>/artifacts/01_texts/,
       реестр транзакций, submission-шаблон
Выход: <run>/artifacts/02_doc_index.json (заполнено поле scenario_id)
       <run>/artifacts/03_attribution_report.json

Ни одно имя счёта и ни одно название компании в коде не зашито. Всё
выводится в рантайме, потому что приватный датасет назовёт и то и другое
иначе.

Три прохода:

1. ПО НОМЕРУ СЧЁТА. Карта account_id → scenario_id строится из реестра:
   в каждой строке txn_id начинается со scenario_id, а account_id стоит
   рядом. Искомые строки берутся буквально из колонки account_id — никаких
   догадок о формате идентификатора.

2. ОБУЧЕНИЕ НАЗВАНИЙ. По документам, привязанным на первом проходе,
   выучиваются названия компаний. Отбор по различительности: название
   аудитора встречается у многих заёмщиков, название заёмщика — только
   у своего. Это отделяет одно от другого без списка аудиторов.

3. ПО НАЗВАНИЮ. Оставшиеся документы привязываются самым длинным
   совпадением среди выученных названий. Длинное побеждает короткое:
   «Shymkent Refinery Services JSC» не должен проиграть подстроке
   «Shymkent Refinery JSC», принадлежащей другому заёмщику.
"""
from __future__ import annotations

import csv
import json
import logging
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .classify import AUTHORITATIVE, DocClass, DocType
from .config import DatasetPaths, RunPaths

log = logging.getLogger(__name__)

#: Организационно-правовые формы, по которым опознаётся название компании.
ORG_SUFFIXES = r"(?:JSC|LLP|LLC|Ltd|PLC|GmbH|Inc|АО|ТОО|ЖШС|АҚ)"
ORG_NAME_RE = re.compile(
    rf"\b([A-ZА-ЯЁ][\w\-&'’.]*(?:\s+[\w\-&'’.]+){{0,4}}\s+{ORG_SUFFIXES})\b"
)

#: Название считается принадлежащим заёмщику, если встречается не реже
#: чем в такой доле его документов...
MIN_COVERAGE = 0.30
#: ...и при этом такая доля всех его вхождений приходится на этого заёмщика.
MIN_DISCRIMINATION = 0.80


@dataclass
class AttributionReport:
    account_to_scenario: dict[str, str] = field(default_factory=dict)
    learned_names: dict[str, list[str]] = field(default_factory=dict)
    by_account: int = 0
    by_name: int = 0
    orphans: list[str] = field(default_factory=list)
    ambiguous: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    reporting_period: tuple[str, str] | None = None
    revisions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "scenarios": sorted(set(self.account_to_scenario.values())),
            "account_to_scenario": self.account_to_scenario,
            "learned_names": self.learned_names,
            "attributed_by_account": self.by_account,
            "attributed_by_name": self.by_name,
            "orphans": self.orphans,
            "ambiguous": self.ambiguous,
            "problems": self.problems,
            "reporting_period": list(self.reporting_period) if self.reporting_period else None,
            "revisions": self.revisions,
        }


# --------------------------------------------------------------------------- #
# Карта счетов
# --------------------------------------------------------------------------- #


def build_account_map(ledger_csv: Path, known_scenarios: set[str]) -> tuple[dict[str, str], list[str]]:
    """account_id → scenario_id по реестру.

    Счета, чей scenario_id отсутствует в шаблоне, — шумовые. Отсекаются
    именно по этому признаку, а не по виду идентификатора: в публичном
    наборе шум имеет префикс 9xxx, но полагаться на это нельзя.
    """
    pairs: dict[str, Counter] = defaultdict(Counter)
    with ledger_csv.open(newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            txn = (row.get("txn_id") or "").strip()
            acc = (row.get("account_id") or "").strip()
            if not txn or not acc:
                continue
            parts = txn.split("-")
            if len(parts) < 3:
                continue
            pairs[acc][parts[1]] += 1

    mapping: dict[str, str] = {}
    problems: list[str] = []
    for acc, counter in pairs.items():
        scenario, n = counter.most_common(1)[0]
        if len(counter) > 1:
            problems.append(
                f"счёт {acc} встречается у нескольких сценариев {dict(counter)}; "
                f"взят преобладающий {scenario}"
            )
        if scenario in known_scenarios:
            mapping[acc] = scenario
    return mapping, problems


def _account_pattern(accounts: list[str]) -> re.Pattern:
    """Литеральный поиск идентификаторов, длинные раньше коротких.

    Длина важна: если в реестре есть и ACC-780, и ACC-7801, короткий
    не должен перехватывать длинный.
    """
    ordered = sorted(accounts, key=len, reverse=True)
    return re.compile(r"(?<![\w-])(" + "|".join(re.escape(a) for a in ordered) + r")(?![\w-])")


# --------------------------------------------------------------------------- #
# Обучение названий компаний
# --------------------------------------------------------------------------- #


#: Служебные слова, прилипающие к названию из-за вёрстки PDF
#: («Организация Aktau Port Services JSC», «За Кредитора\nEkibastuz...»).
LEADING_NOISE_RE = re.compile(
    r"^(?:[А-ЯЁа-яё][\wё]*\.?\s+)*?(?=[A-Z][A-Za-z]*\s|[A-Z]{2,})", re.UNICODE
)


def normalize_name(name: str) -> str:
    """Схлопывает пробелы и отрезает служебный префикс из вёрстки.

    Без этого «Almaty Cold\\nChain JSC» и «Almaty Cold Chain JSC» живут
    как два разных названия, а «Организация Aktau Port Services JSC»
    засоряет словарь.
    """
    name = re.sub(r"\s+", " ", name).strip(" .,;:")
    return LEADING_NOISE_RE.sub("", name).strip(" .,;:")


def learn_company_names(
    texts: dict[str, str], attributed: dict[str, str]
) -> dict[str, list[str]]:
    """Выучивает названия заёмщиков по уже привязанным документам.

    Различительность отделяет заёмщика от аудитора: аудитор обслуживает
    нескольких заёмщиков и потому встречается в документах разных
    сценариев, заёмщик — только в своих. Списка аудиторов не требуется.
    """
    docs_per_scenario: Counter = Counter(attributed.values())
    name_scenario_docs: dict[str, Counter] = defaultdict(Counter)

    for doc_id, scenario in attributed.items():
        flat = re.sub(r"\s+", " ", texts.get(doc_id, ""))
        names = {normalize_name(n) for n in ORG_NAME_RE.findall(flat)}
        for name in names:
            if len(name) >= 6:
                name_scenario_docs[name][scenario] += 1

    learned: dict[str, list[str]] = defaultdict(list)
    for name, per_scenario in name_scenario_docs.items():
        total = sum(per_scenario.values())
        scenario, hits = per_scenario.most_common(1)[0]
        coverage = hits / max(docs_per_scenario[scenario], 1)
        discrimination = hits / total
        if coverage >= MIN_COVERAGE and discrimination >= MIN_DISCRIMINATION:
            learned[scenario].append(name)

    for scenario in learned:
        learned[scenario] = sorted(set(learned[scenario]), key=len, reverse=True)
    return dict(learned)


def attribute_by_name(text: str, learned: dict[str, list[str]]) -> tuple[str | None, bool]:
    """Самое длинное совпадение выигрывает.

    Длина решает исход в критическом случае: «Shymkent Refinery JSC»
    и «Shymkent Refinery Services JSC» — разные заёмщики с общим началом.
    """
    flat = re.sub(r"\s+", " ", text)
    best_len = 0
    winners: set[str] = set()
    for scenario, names in learned.items():
        for name in names:
            if len(name) >= best_len and name in flat:
                if len(name) > best_len:
                    best_len, winners = len(name), {scenario}
                else:
                    winners.add(scenario)
    if not winners:
        return None, False
    if len(winners) > 1:
        return None, True
    return winners.pop(), False


# --------------------------------------------------------------------------- #
# Инварианты
# --------------------------------------------------------------------------- #


MONTHS_RU = {
    "января": 1, "февраля": 2, "марта": 3, "апреля": 4, "мая": 5, "июня": 6,
    "июля": 7, "августа": 8, "сентября": 9, "октября": 10, "ноября": 11, "декабря": 12,
}
_RU_DATE_RE = re.compile(r"(\d{1,2})\s+(" + "|".join(MONTHS_RU) + r")\s+(20\d\d)")
_ISO_DATE_RE = re.compile(r"\b(20\d\d)-(\d{2})-(\d{2})\b")


def all_dates(text: str) -> list[str]:
    """Все даты документа в ISO, в порядке появления."""
    flat = re.sub(r"\s+", " ", text)
    found: list[tuple[int, str]] = []
    for m in _RU_DATE_RE.finditer(flat):
        d, mo, y = m.groups()
        found.append((m.start(), f"{y}-{MONTHS_RU[mo]:02d}-{int(d):02d}"))
    for m in _ISO_DATE_RE.finditer(flat):
        y, mo, d = m.groups()
        found.append((m.start(), f"{y}-{mo}-{d}"))
    return [d for _, d in sorted(found)]


def document_date(text: str) -> str | None:
    """Собственная дата документа в ISO.

    Нужна как ВТОРАЯ, независимая ось определения актуальности. README
    архива говорит: «действующей считается только текущая редакция за
    отчётный период». Текстовая пометка об отмене — не единственный
    признак: устаревший документ может её не иметь и отличаться только
    датой.

    Приоритеты выведены из устройства документов, а не угаданы:
      1. «... от 1 января 2025 года» — шапка договора;
      2. «Дата проверки 31 декабря 2025 года» — шапка досье;
      3. последняя дата в тексте — блок подписи аудиторского отчёта.

    Максимум по всем датам не годится: в договоре самая поздняя дата —
    это конец ковенантного периода, а не дата подписания.
    """
    flat = re.sub(r"\s+", " ", text)
    months = "|".join(MONTHS_RU)

    header = re.search(rf"\bот\s+(\d{{1,2}})\s+({months})\s+(20\d\d)", flat[:1500])
    if header:
        d, mo, y = header.groups()
        return f"{y}-{MONTHS_RU[mo]:02d}-{int(d):02d}"

    labelled = re.search(
        rf"Дат[аеу][^.]{{0,40}}?(\d{{1,2}})\s+({months})\s+(20\d\d)", flat[:2500]
    )
    if labelled:
        d, mo, y = labelled.groups()
        return f"{y}-{MONTHS_RU[mo]:02d}-{int(d):02d}"

    ru = [
        f"{y}-{MONTHS_RU[mo]:02d}-{int(d):02d}"
        for d, mo, y in _RU_DATE_RE.findall(flat)
    ]
    if ru:
        return ru[-1]
    iso = all_dates(text)
    return iso[-1] if iso else None


def infer_reporting_period(ledger_csv: Path, scenarios: set[str]) -> tuple[str, str] | None:
    """Отчётный период выводится из дат реестра целевых операций.

    Реестр — это выгрузка ровно за проверяемый период, поэтому его границы
    и есть период. Ни год, ни границы в коде не зашиты.
    """
    dates: list[str] = []
    with ledger_csv.open(newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            parts = (row.get("txn_id") or "").split("-")
            if len(parts) >= 3 and parts[1] in scenarios:
                d = (row.get("date") or "").strip()
                if _ISO_DATE_RE.fullmatch(d):
                    dates.append(d)
    # Возвращается НАБЛЮДАЕМЫЙ диапазон, без округления до календарного года.
    # Организаторы уточнили, что период действия написан в самом договоре и
    # не обязан быть календарным годом: округление сломало бы сопоставление
    # для периодов вида «апрель — март».
    return (min(dates), max(dates)) if dates else None


#: Допуск при проверке «дата документа внутри периода». Реестр может не
#: содержать операций на самой границе: в публичном наборе первая операция —
#: 2 января, а договор подписан 1 января.
PERIOD_MARGIN_DAYS = 45


def padded_period(period: tuple[str, str] | None, days: int = PERIOD_MARGIN_DAYS):
    """Период с допуском по краям — только для проверки дат документов.

    Для сопоставления периодов договоров допуск не нужен: там решает
    величина пересечения, а она устойчива к смещению границ на день.
    """
    if not period:
        return None
    from datetime import date, timedelta

    def d(s: str) -> date:
        y, m, dd = s.split("-")
        return date(int(y), int(m), int(dd))

    return ((d(period[0]) - timedelta(days=days)).isoformat(),
            (d(period[1]) + timedelta(days=days)).isoformat())


_PERIOD_ISO_RE = re.compile(r"с\s*(20\d\d-\d\d-\d\d)\s*по\s*(20\d\d-\d\d-\d\d)")


def covenant_period(text: str) -> tuple[str, str] | None:
    """Период действия, указанный в самом договоре.

    По разъяснению организаторов это ЕДИНСТВЕННЫЙ критерий актуальности:
    «в каждом договоре написан период его действия… отметка "недействующая
    редакция" не влияет на настоящий договор».

    Ищется сначала в статье о финансовых ковенантах, затем по всему тексту:
    период ковенанта и есть период, за который проверяется соблюдение.
    """
    flat = re.sub(r"\s+", " ", text)
    months = "|".join(MONTHS_RU)

    def scan(fragment: str) -> tuple[str, str] | None:
        m = _PERIOD_ISO_RE.search(fragment)
        if m:
            return m.group(1), m.group(2)
        ru = re.search(
            rf"с\s*(\d{{1,2}})\s*({months})\s*(20\d\d)[^.]{{0,20}}?по\s*(\d{{1,2}})\s*({months})\s*(20\d\d)",
            fragment,
        )
        if ru:
            d1, m1, y1, d2, m2, y2 = ru.groups()
            return (f"{y1}-{MONTHS_RU[m1]:02d}-{int(d1):02d}",
                    f"{y2}-{MONTHS_RU[m2]:02d}-{int(d2):02d}")
        return None

    i = flat.find("Финансовые ковенанты")
    if i >= 0:
        found = scan(flat[i : i + 6000])
        if found:
            return found
    return scan(flat)


def _overlap_days(a: tuple[str, str], b: tuple[str, str]) -> int:
    from datetime import date

    def d(s: str) -> date:
        y, m, dd = s.split("-")
        return date(int(y), int(m), int(dd))

    start, end = max(d(a[0]), d(b[0])), min(d(a[1]), d(b[1]))
    return max(0, (end - start).days + 1)


def resolve_active_loans(
    docs: dict[str, DocClass],
    texts: dict[str, str],
    period: tuple[str, str] | None,
) -> list[str]:
    """Выбирает действующий договор ПО ПЕРИОДУ, а не по текстовой пометке.

    Правило организаторов: действующим считается договор, чей указанный
    период действия соответствует отчётному. Пометка «недействующая
    редакция» — справочный признак; расхождение между ней и периодом
    сообщается, но решает период.

    Действующий договор у заёмщика всегда ровно один — это тоже
    подтверждено организаторами, поэтому конкуренция разрешается всегда.
    """
    problems: list[str] = []
    by_scenario: dict[str, list[str]] = defaultdict(list)
    for doc_id, d in docs.items():
        if d.type == DocType.LOAN and d.scenario_id:
            by_scenario[d.scenario_id].append(doc_id)

    for scenario, ids in sorted(by_scenario.items()):
        periods = {i: covenant_period(texts.get(i, "")) for i in ids}
        marked = {
            i: any("пометка об отмене" in n for n in docs[i].notes) for i in ids
        }
        for doc_id, per in periods.items():
            if per:
                docs[doc_id].notes.append(f"период договора: {per[0]}..{per[1]}")
            else:
                docs[doc_id].notes.append("период действия в договоре не найден")

        def rank(doc_id: str):
            per = periods[doc_id]
            overlap = _overlap_days(per, period) if (per and period) else -1
            exact = 1 if (per and period and per == period) else 0
            # Пометка — последний, самый слабый разряд ключа.
            return (exact, overlap, 0 if marked[doc_id] else 1, doc_id)

        winner = max(ids, key=rank)
        for doc_id in ids:
            docs[doc_id].type = (
                DocType.LOAN_ACTIVE if doc_id == winner else DocType.LOAN_SUPERSEDED
            )

        if periods[winner] is None:
            problems.append(
                f"{scenario}: у выбранного договора {winner} не найден период "
                f"действия — выбор сделан по косвенным признакам, проверьте вручную"
            )
        elif period:
            # Точное совпадение границ не требуется: реестр может не содержать
            # операций в первый или последний день периода. Сообщаем только
            # о существенном расхождении.
            span = _overlap_days(periods[winner], periods[winner])
            covered = _overlap_days(periods[winner], period)
            if span and covered / span < 0.5:
                problems.append(
                    f"{scenario}: период договора {periods[winner]} перекрывает "
                    f"отчётный {period} лишь на {covered / span:.0%} — проверьте вручную"
                )
        if marked[winner]:
            problems.append(
                f"{scenario}: у выбранного договора {winner} стоит пометка об "
                f"отмене, но его период подходит лучше прочих. Решает период — "
                f"проверьте вручную, это расхождение"
            )
        for doc_id in ids:
            if doc_id != winner and not marked[doc_id] and periods[doc_id] != periods[winner]:
                docs[doc_id].notes.append(
                    "отклонён по периоду, пометки об отмене нет"
                )
        if len(ids) > 1:
            problems.append(
                f"{scenario}: договоров {len(ids)}, действующим принят {winner} "
                f"(период {periods[winner]})"
            )
    return problems


def resolve_revisions(
    docs: dict[str, DocClass],
    texts: dict[str, str],
    period: tuple[str, str] | None,
) -> list[str]:
    """Оставляет по одному авторитетному документу каждого типа на заёмщика.

    Правило README: действующей считается текущая редакция за отчётный
    период. Реализуется в два приоритета — сначала документы, попадающие
    в период, затем самая поздняя дата. Проигравшие не удаляются, а
    помечаются: молча выбрасывать документ опаснее, чем показать выбор.
    """
    problems: list[str] = []
    period = padded_period(period)
    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for doc_id, d in docs.items():
        if d.scenario_id and d.type in AUTHORITATIVE:
            groups[(d.scenario_id, d.type)].append(doc_id)

    for (scenario, doc_type), ids in sorted(groups.items()):
        dated = {i: document_date(texts.get(i, "")) for i in ids}
        for doc_id, dt in dated.items():
            if dt:
                docs[doc_id].notes.append(f"дата документа: {dt}")
            # Проверяем по ВСЕМ датам документа, а не только по собственной:
            # одиночное упоминание будущего периода (например, правило
            # отсечения со ссылкой на 2026 год) не делает документ устаревшим.
            if period and dt:
                touches = [d for d in all_dates(texts.get(doc_id, ""))
                           if period[0] <= d <= period[1]]
                if not touches:
                    docs[doc_id].notes.append(
                        f"ни одна дата документа не попадает в отчётный период "
                        f"{period[0]}..{period[1]}"
                    )
        if len(ids) < 2:
            continue

        def rank(doc_id: str) -> tuple[int, str]:
            dt = dated[doc_id]
            in_period = bool(dt and period and period[0] <= dt <= period[1])
            return (1 if in_period else 0, dt or "")


        ordered = sorted(ids, key=rank, reverse=True)
        winner, losers = ordered[0], ordered[1:]
        if all(dated[i] is None for i in ids):
            problems.append(
                f"{scenario}: несколько документов типа {doc_type} "
                f"({len(ids)}) и ни у одного нет даты — выбрать актуальный нечем"
            )
            continue
        for doc_id in losers:
            docs[doc_id].type = f"{doc_type}__STALE"
            docs[doc_id].notes.append(
                f"вытеснен более актуальным {winner} (дата {dated[winner]})"
            )
        problems.append(
            f"{scenario}: документов типа {doc_type} было {len(ids)}; "
            f"актуальным принят {winner} (дата {dated[winner]}), "
            f"остальные помечены устаревшими: {losers}"
        )
    return problems


def check_invariants(
    docs: dict[str, DocClass], scenarios: set[str]
) -> list[str]:
    """Структурные требования, которые обязаны выполняться на любом наборе.

    Это главная страховка на случай, если маркеры шага 3 не подойдут
    к приватному датасету: пропавший договор виден здесь, а не в конце.
    """
    problems: list[str] = []
    per_scenario: dict[str, Counter] = defaultdict(Counter)
    for d in docs.values():
        if d.scenario_id:
            per_scenario[d.scenario_id][d.type] += 1

    for scenario in sorted(scenarios):
        counts = per_scenario.get(scenario, Counter())
        n_loans = counts.get(DocType.LOAN_ACTIVE, 0)
        if n_loans == 0:
            problems.append(f"{scenario}: нет действующего договора — ковенанты неоткуда взять")
        elif n_loans > 1:
            problems.append(
                f"{scenario}: действующих договоров {n_loans}, должен быть один — "
                f"вероятно, отменённая редакция не распознана"
            )
        if counts.get(DocType.AUDIT_FINAL, 0) == 0:
            problems.append(f"{scenario}: нет финального аудита — корректировки недоступны")
        if counts.get(DocType.KYC, 0) == 0:
            problems.append(f"{scenario}: нет KYC — связанные стороны неопределимы")
    return problems


# --------------------------------------------------------------------------- #
# Прогон
# --------------------------------------------------------------------------- #


def run(dataset: DatasetPaths, paths: RunPaths) -> tuple[dict[str, DocClass], AttributionReport]:
    from . import classify

    docs = classify.load(paths)
    texts_dir = paths.artifacts / "01_texts"
    texts = {
        p.stem: p.read_text(encoding="utf-8") for p in texts_dir.glob("*.txt")
    }

    template = json.loads(dataset.template_json.read_text(encoding="utf-8"))
    scenarios = set(template.get("answers", {}))

    account_map, problems = build_account_map(dataset.ledger_csv, scenarios)
    report = AttributionReport(account_to_scenario=account_map, problems=problems)
    if not account_map:
        report.problems.append(
            "Карта счетов пуста: ни один scenario_id из реестра не совпал с шаблоном. "
            "Проверьте формат txn_id."
        )
        return docs, report

    pattern = _account_pattern(list(account_map))

    # --- проход 1: по номеру счёта ---
    attributed: dict[str, str] = {}
    for doc_id, text in texts.items():
        found = {account_map[a] for a in set(pattern.findall(text)) if a in account_map}
        if len(found) == 1:
            scenario = found.pop()
            docs[doc_id].scenario_id = scenario
            attributed[doc_id] = scenario
            report.by_account += 1
        elif len(found) > 1:
            report.ambiguous.append(f"{doc_id}: счета нескольких сценариев {sorted(found)}")

    # --- проход 2: обучение названий ---
    report.learned_names = learn_company_names(texts, attributed)

    # --- проход 3: по названию ---
    for doc_id, text in texts.items():
        if docs[doc_id].scenario_id:
            continue
        scenario, ambiguous = attribute_by_name(text, report.learned_names)
        if ambiguous:
            report.ambiguous.append(f"{doc_id}: название совпало с несколькими сценариями")
        elif scenario:
            docs[doc_id].scenario_id = scenario
            docs[doc_id].notes.append("привязан по названию компании, счёт в тексте отсутствует")
            report.by_name += 1
        else:
            report.orphans.append(doc_id)

    # --- выбор актуальной редакции (вторая ось после текстовой пометки) ---
    report.reporting_period = infer_reporting_period(dataset.ledger_csv, scenarios)
    # Договоры разрешаются по периоду действия (правило организаторов),
    # остальные типы — по дате документа.
    report.revisions = resolve_active_loans(docs, texts, report.reporting_period)
    report.revisions += resolve_revisions(docs, texts, report.reporting_period)

    # --- инварианты ---
    report.problems.extend(check_invariants(docs, scenarios))

    # Сироты авторитетных типов — это всегда проблема, а не фон.
    for doc_id in report.orphans:
        if docs[doc_id].type in AUTHORITATIVE:
            report.problems.append(
                f"{doc_id}: тип {docs[doc_id].type}, но заёмщик не определён — "
                f"документ выпадет из расчёта, разберите вручную"
            )

    index_path = paths.artifacts / "02_doc_index.json"
    data = json.loads(index_path.read_text(encoding="utf-8"))
    data["documents"] = {k: asdict(v) for k, v in sorted(docs.items())}
    index_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    (paths.artifacts / "03_attribution_report.json").write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    for p in report.problems:
        log.warning("ПРИВЯЗКА: %s", p)
    log.info(
        "Привязано: %d по счёту, %d по названию, сирот %d",
        report.by_account, report.by_name, len(report.orphans),
    )
    return docs, report
