"""Шаг 9: подготовка реестра транзакций.

Вход:  реестр, submission-шаблон, <run>/artifacts/02_doc_index.json,
       <run>/artifacts/01_texts/
Выход: <run>/artifacts/06_ledger_clean.csv
       <run>/artifacts/06_ledger_report.json

Три задачи, и все три решаются кодом, без обращения к модели.

ОТСЕЧЕНИЕ ШУМА. Строка относится к делу, если её scenario_id есть
в шаблоне. Не «префикс счёта 9xxx» — это свойство публичного набора,
а не правило.

ВОССТАНОВЛЕНИЕ ПРОПУЩЕННЫХ СУММ. У части операций поле amount пустое,
и настоящая сумма раскрыта в документах: у одной — в приложении
аудитора, у другой — в служебной записке казначейства, единственной
на весь корпус. Механизм общий: если сумма отсутствует, идентификатор
операции ищется во всех авторитетных документах её заёмщика, а рядом
с ним — денежная величина. Конкретные идентификаторы в коде не зашиты.

ВАЛЮТА. Курс не берётся извне: он выводится из раскрытой аудитором пары
«счёт в валюте — платёж в долларах». Внешний рыночный курс дал бы
расхождение легко больше 5%, а это обнуляет и actual, и evidence.
"""
from __future__ import annotations

import csv
import json
import logging
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .classify import AUTHORITATIVE, DocClass
from . import artifacts as A
from .config import DatasetPaths, RunPaths

log = logging.getLogger(__name__)

#: «фактическая сумма операции составляет $884,204.16 (расход)»
DISCLOSED_AMOUNT_RE = re.compile(
    r"\$\s*([\d]{1,3}(?:[,  ][\d]{3})*(?:\.\d{1,2})?)"
)
EXPENSE_WORDS = ("расход", "списан", "уплач", "платёж", "платеж", "outflow", "expense")
INCOME_WORDS = ("поступлен", "доход", "получен", "inflow", "income")

#: «счёт на сумму 72,146.75 EUR урегулирован платежом в долларах США
#:  в размере $83,690.23»
FX_PAIR_RE = re.compile(
    r"([\d]{1,3}(?:[,  ][\d]{3})*(?:\.\d{1,2})?)\s*([A-Z]{3})"
    r"[^.$]{0,160}?\$\s*([\d]{1,3}(?:[,  ][\d]{3})*(?:\.\d{1,2})?)",
    re.IGNORECASE,
)

BASE_CURRENCY = "USD"


def _to_float(s: str) -> float:
    return float(re.sub(r"[,  ]", "", s))


@dataclass
class Txn:
    txn_id: str
    scenario_id: str
    date: str
    account_id: str
    counterparty: str
    description: str
    amount: float | None
    currency: str
    amount_usd: float | None = None
    recovered: bool = False
    fx_rate: float | None = None
    evidence: str | None = None
    problems: list[str] = field(default_factory=list)


@dataclass
class LedgerReport:
    total_rows: int = 0
    kept_rows: int = 0
    dropped_noise: int = 0
    recovered_amounts: list[str] = field(default_factory=list)
    fx_rates: dict[str, float] = field(default_factory=dict)
    fx_evidence: dict[str, str] = field(default_factory=dict)
    unresolved: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    per_scenario: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Восстановление пропущенных сумм
# --------------------------------------------------------------------------- #


def find_disclosed_amount(
    txn_id: str, text: str, window: int = 320
) -> tuple[float, str, bool] | None:
    """Ищет раскрытую сумму рядом с упоминанием операции.

    Возвращает (сумма со знаком, доказательство, направление определено?)
    либо None. Знак берётся из слов «расход» / «поступление» поблизости:
    в реестре списания отрицательны, и восстановленная сумма обязана
    следовать тому же соглашению.

    Третий элемент существует, чтобы неопределённое направление не
    проходило молча: ошибка в знаке крупной суммы искажает агрегат вдвое.
    """
    flat = re.sub(r"\s+", " ", text)
    for m in re.finditer(re.escape(txn_id), flat):
        chunk = flat[m.start() : m.start() + window]
        # Если рядом сказано «фактическая сумма ... составляет», берём
        # величину после этих слов: в тексте может стоять и другая сумма
        # (например, ошибочно отражённая), а нужна именно фактическая.
        anchor = re.search(r"(?:фактическ\w*\s+сумм\w*|составляет|actual\s+amount)", chunk, re.I)
        search_from = anchor.start() if anchor else 0
        money = DISCLOSED_AMOUNT_RE.search(chunk, search_from)
        if not money:
            money = DISCLOSED_AMOUNT_RE.search(chunk)
        if not money:
            continue
        value = _to_float(money.group(1))
        low = chunk.lower()
        exp_at = min((low.find(w) for w in EXPENSE_WORDS if w in low), default=-1)
        inc_at = min((low.find(w) for w in INCOME_WORDS if w in low), default=-1)
        direction_known = exp_at >= 0 or inc_at >= 0
        if exp_at >= 0 and (inc_at < 0 or exp_at < inc_at):
            value = -value
        return value, chunk[:220].strip(), direction_known
    return None


# --------------------------------------------------------------------------- #
# Курсы валют
# --------------------------------------------------------------------------- #


#: Валюты, которые вообще могут встретиться в этом наборе.
#:
#: ЗАЧЕМ СПИСОК, А НЕ «ЛЮБЫЕ ТРИ ЗАГЛАВНЫЕ БУКВЫ». Регулярное выражение
#: принимало за код валюты любое трёхбуквенное сочетание. Пока тексты
#: приходили из текстового слоя PDF, это было безобидно. Как только
#: нарисованные страницы пошли на распознавание, из плохого OCR
#: посыпались «курсы» вымышленных валют YPE и CMJ — и встали в один ряд
#: с настоящим EUR.
#:
#: Курс — это множитель для сумм. Выдуманный курс не падает: он молча
#: пересчитывает деньги. Поэтому валюта обязана быть настоящей, а список
#: держится широким (все, что могут появиться в регионе и в отчётности),
#: но конечным.
KNOWN_CURRENCIES = frozenset({
    "USD", "EUR", "KZT", "RUB", "GBP", "CHF", "CNY", "JPY", "TRY",
    "AED", "KGS", "UZS", "AZN", "GEL", "BYN", "PLN", "SEK", "NOK",
    "DKK", "CAD", "AUD", "SGD", "HKD", "INR", "KRW",
})


def find_fx_rates(
    text: str, problems: list[str] | None = None
) -> dict[str, tuple[float, str]]:
    """Выводит курсы из раскрытых пар «сумма в валюте → платёж в долларах».

    Внешние котировки не используются принципиально: ковенант проверяется
    по курсу фактического расчёта, а расхождение с рынком легко превысит
    порог в 5%.

    `problems` — сток для отбраковки: отброшенный кандидат на курс обязан
    попасть в отчёт шага, а не только в лог. На приватном наборе валюта
    вне словаря может оказаться настоящей — и тогда её пропуск это тихая
    потеря пересчёта, которую надо увидеть в первые минуты.
    """
    flat = re.sub(r"\s+", " ", text)
    out: dict[str, tuple[float, str]] = {}
    for m in FX_PAIR_RE.finditer(flat):
        foreign, currency, usd = m.group(1), m.group(2).upper(), m.group(3)
        if currency == BASE_CURRENCY:
            continue
        if currency not in KNOWN_CURRENCIES:
            # Не настоящая валюта — почти наверняка мусор распознавания.
            context = flat[max(0, m.start() - 60):m.end() + 20].strip()[:160]
            log.warning("Курс для неизвестной валюты %r пропущен: %s", currency, context)
            if problems is not None:
                problems.append(
                    f"валюта {currency!r} вне словаря KNOWN_CURRENCIES — кандидат "
                    f"на курс отброшен; если это настоящая валюта, пополните "
                    f"словарь ({context[:100]})"
                )
            continue
        f, u = _to_float(foreign), _to_float(usd)
        if f <= 0 or u <= 0:
            continue
        rate = u / f
        if not (0.01 <= rate <= 100):
            continue
        out[currency] = (rate, flat[max(0, m.start() - 90) : m.end() + 30].strip())
    return out


# --------------------------------------------------------------------------- #
# Прогон
# --------------------------------------------------------------------------- #


def run(dataset: DatasetPaths, paths: RunPaths) -> tuple[list[Txn], LedgerReport]:
    from . import classify

    docs: dict[str, DocClass] = classify.load(paths)
    texts_dir = paths.artifacts / A.TEXTS_DIR
    template = json.loads(dataset.template_json.read_text(encoding="utf-8"))
    scenarios = set(template.get("answers", {}))

    # Тексты авторитетных документов, сгруппированные по заёмщику.
    by_scenario: dict[str, list[tuple[str, str]]] = {}
    for doc_id, d in docs.items():
        if d.scenario_id and d.type in AUTHORITATIVE:
            p = texts_dir / f"{doc_id}.txt"
            if p.exists():
                by_scenario.setdefault(d.scenario_id, []).append(
                    (doc_id, p.read_text(encoding="utf-8"))
                )

    report = LedgerReport()
    txns: list[Txn] = []
    seen_ids: Counter = Counter()

    with dataset.ledger_csv.open(newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            report.total_rows += 1
            txn_id = (row.get("txn_id") or "").strip()
            parts = txn_id.split("-")
            scenario = parts[1] if len(parts) >= 3 else ""
            if scenario not in scenarios:
                report.dropped_noise += 1
                continue

            seen_ids[txn_id] += 1
            raw_amount = (row.get("amount") or "").strip()
            try:
                amount = float(raw_amount) if raw_amount else None
            except ValueError:
                amount = None

            t = Txn(
                txn_id=txn_id,
                scenario_id=scenario,
                date=(row.get("date") or "").strip(),
                account_id=(row.get("account_id") or "").strip(),
                counterparty=(row.get("counterparty") or "").strip(),
                description=(row.get("description") or "").strip(),
                amount=amount,
                currency=(row.get("currency") or BASE_CURRENCY).strip().upper(),
            )
            if raw_amount and amount is None:
                t.problems.append(f"amount нечисловой: {raw_amount!r}")
            txns.append(t)

    report.kept_rows = len(txns)
    for txn_id, n in seen_ids.items():
        if n > 1:
            report.problems.append(f"дубль txn_id: {txn_id} встречается {n} раз")

    # --- восстановление пропущенных сумм ---
    for t in txns:
        if t.amount is not None:
            continue
        t.problems.append("сумма отсутствует в реестре")
        for doc_id, text in by_scenario.get(t.scenario_id, []):
            found = find_disclosed_amount(t.txn_id, text)
            if found:
                value, evidence, direction_known = found
                t.amount, t.evidence = value, f"{doc_id}: {evidence}"
                t.recovered = True
                if not direction_known:
                    t.problems.append(
                        "направление операции не указано в документе — знак суммы "
                        "принят положительным, проверьте вручную"
                    )
                    report.problems.append(
                        f"{t.txn_id}: направление восстановленной суммы не определено"
                    )
                report.recovered_amounts.append(
                    f"{t.txn_id} = {t.amount:,.2f} (источник {doc_id})"
                )
                break
        if t.amount is None:
            report.unresolved.append(
                f"{t.txn_id}: сумма отсутствует и не найдена в документах заёмщика"
            )

    # --- курсы валют ---
    rates: dict[str, tuple[float, str]] = {}
    fx_drops: list[str] = []
    for scenario_docs in by_scenario.values():
        for doc_id, text in scenario_docs:
            for currency, (rate, ev) in find_fx_rates(text, problems=fx_drops).items():
                if currency not in rates:
                    rates[currency] = (rate, f"{doc_id}: {ev}")
    # Одна и та же мусорная «валюта» встречается в нескольких документах —
    # в отчёт по разу.
    report.problems.extend(dict.fromkeys(fx_drops))
    report.fx_rates = {c: round(r, 6) for c, (r, _) in rates.items()}
    report.fx_evidence = {c: ev for c, (_, ev) in rates.items()}

    for t in txns:
        if t.amount is None:
            continue
        if t.currency == BASE_CURRENCY:
            t.amount_usd = t.amount
            continue
        if t.currency in rates:
            t.fx_rate = rates[t.currency][0]
            t.amount_usd = round(t.amount * t.fx_rate, 2)
            if not any(t.currency in e for e in report.problems):
                pass
        else:
            t.problems.append(f"курс {t.currency}→USD не раскрыт ни в одном документе")
            # Валюта вне словаря — отдельная беда: раскрытие могло быть,
            # но его кандидат отброшен как мусор (см. find_fx_rates).
            suffix = (
                " (валюта вне словаря KNOWN_CURRENCIES)"
                if t.currency not in KNOWN_CURRENCIES else ""
            )
            report.unresolved.append(
                f"{t.txn_id}: {t.amount:,.2f} {t.currency} без раскрытого курса{suffix}"
            )

    # Курс раскрыт у одного заёмщика, а операции в этой валюте есть у других.
    # Заимствование в пределах датасета допустимо, но обязано быть заметным.
    for currency in {t.currency for t in txns if t.currency != BASE_CURRENCY}:
        if currency in rates:
            holders = {t.scenario_id for t in txns if t.currency == currency}
            if len(holders) > 1:
                report.problems.append(
                    f"курс {currency}={report.fx_rates[currency]} раскрыт в документах "
                    f"одного заёмщика, но применён к {len(holders)} сценариям — "
                    f"проверьте, нет ли отдельного раскрытия у каждого"
                )

    report.per_scenario = dict(Counter(t.scenario_id for t in txns))

    # --- вывод ---
    out_csv = paths.artifacts / A.LEDGER_CLEAN
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow([
            "txn_id", "scenario_id", "date", "account_id", "counterparty",
            "description", "amount", "currency", "amount_usd", "recovered",
            "fx_rate", "evidence", "problems",
        ])
        for t in txns:
            w.writerow([
                t.txn_id, t.scenario_id, t.date, t.account_id, t.counterparty,
                t.description, t.amount, t.currency, t.amount_usd,
                int(t.recovered), t.fx_rate, t.evidence or "", "; ".join(t.problems),
            ])
    (paths.artifacts / A.LEDGER_REPORT).write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    for p in report.problems + report.unresolved:
        log.warning("РЕЕСТР: %s", p)
    log.info(
        "Реестр: %d строк всего, %d целевых, %d шума, восстановлено сумм %d, курсы %s",
        report.total_rows, report.kept_rows, report.dropped_noise,
        len(report.recovered_amounts), report.fx_rates,
    )
    return txns, report
