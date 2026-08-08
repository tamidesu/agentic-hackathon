"""Шаг 10: категоризация транзакций реестра.

Вход:  реестр (после очистки от шума на шаге 9), 02_doc_index.json
Выход: <run>/artifacts/07_categories.json

ЗАЧЕМ ЭТОТ ШАГ

Все агрегаты ковенантов — это `AGG(category)`. В реестре категории нет:
есть дата, контрагент, описание и сумма. Категория выводится из описания,
и без этого шага КАЖДЫЙ агрегат вернёт ноль. Это последний разрыв между
извлечением и расчётом.

ГЛАВНАЯ ЛОВУШКА: ИМЯ КОНТРАГЕНТА ПРОТИВОРЕЧИТ ОПИСАНИЮ

В публичном наборе 96 строк из 673 — четырнадцать процентов — устроены так,
что название контрагента указывает на одну статью, а описание на другую:

    Cedarville Payroll LP                 Revolver interest — November
    Glenwood Property Trust               Franchise tax filing
    Bridgeport Utility Co                 Land tax instalment — Almaty office
    Sentinel Insurance Group              Interest credited on current account

Правильный ответ везде даёт ОПИСАНИЕ: это interest, taxes, taxes, interest.
Контрагент по фамилии «Payroll» может оказывать любые услуги; в описании
сказано, за что заплатили. Четырнадцать процентов строк, разложенных по
названиям вместо описаний, сдвинут агрегаты так, что часть вердиктов
перевернётся — а выглядеть это будет как обычный расчёт.

ПОЧЕМУ ПАКЕТАМИ, А НЕ ПО ОДНОЙ И НЕ ЦЕЛИКОМ

По одной — 673 вызова: дорого и медленно. Целиком — один ответ на 673
строки, где потеря десятка строк в середине незаметна, а повтор стоит
всего прогона. Пакет по несколько десятков строк даёт и обозримый ответ,
и дешёвый повтор при сбое.

НИ ОДНА СТРОКА НЕ ИМЕЕТ ПРАВА ИСЧЕЗНУТЬ

Пропущенная строка — это не ошибка, а недостача в агрегате: она молча
уменьшает сумму. Поэтому состав ответа сверяется с составом запроса
поимённо, пропуски дозапрашиваются, а то, что не удалось получить и со
второй попытки, попадает в `other` с громкой отметкой — но не исчезает.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Sequence

from . import artifacts as A
from .config import RunPaths
from .covenant_types import CATEGORIES
from .llm import LLMClient, LLMRequest, ValidationFailed
from .schemas import TXN_CATEGORY_SCHEMA, make_txn_category_validator

log = logging.getLogger(__name__)

PROMPT_VERSION = "categorize-v1"

#: Строк в одном запросе. Меньше — больше вызовов, больше — дороже повтор
#: при сбое и выше риск, что модель потеряет строку в середине списка.
BATCH_SIZE = 40

#: Категория, в которую попадает всё, что не удалось разложить. Ноль
#: агрегата по ней безопаснее, чем случайная статья: `other` не входит
#: ни в один ковенант публичного набора.
FALLBACK_CATEGORY = "other"

_PROMPT = """Разложи банковские операции по статьям учёта. Описания могут быть \
на русском или на английском языке.

САМОЕ ВАЖНОЕ: КАТЕГОРИЮ ОПРЕДЕЛЯЕТ ОПИСАНИЕ, А НЕ НАЗВАНИЕ КОНТРАГЕНТА.

Названия контрагентов в этом реестре систематически вводят в заблуждение. \
Организация с «Payroll» в названии может получать проценты по кредиту, \
а «Water Works» — арендную плату. Смотри, ЗА ЧТО заплачено:

    Cedarville Payroll LP      | Revolver interest — November        → interest
    Glenwood Property Trust    | Franchise tax filing                → taxes
    Bridgeport Utility Co      | Land tax instalment                 → taxes
    Sentinel Insurance Group   | Interest credited on current account → interest
    Eastgate Water Works Ltd   | Rent for spare parts store          → lease

СТАТЬИ (только эти значения):
  revenue           выручка, поступления от основной деятельности
  opex              операционные расходы, не подходящие под статьи ниже
  capex             капитальные затраты: покупка и строительство основных средств
  payroll           оплата труда и связанные с ней выплаты персоналу
  utilities         электроэнергия, вода, тепло, связь и подобные поставки
  taxes             налоги, сборы, пошлины
  interest          проценты по займам и по остаткам на счетах
  lease             аренда и лизинговые платежи
  insurance         страховые премии и связанные с ними расчёты
  financing_inflow  поступления по финансированию: займы, взносы в капитал
  other             ничего из перечисленного

flow: outflow — расход, inflow — поступление, reversal — возврат или сторно \
ранее совершённой операции. Знак суммы подсказывает, но решает описание: \
возврат страховой премии это insurance с flow=reversal, а не выручка.

confidence — от 0 до 1. Ставь ниже 0.5, если описание допускает два прочтения.

ВЕРНИ РОВНО ПО ОДНОЙ ЗАПИСИ НА КАЖДУЮ ОПЕРАЦИЮ ИЗ СПИСКА, с тем же txn_id. \
Ни одной не пропусти и ни одной не придумай: пропущенная операция молча \
уменьшит итоговую сумму.

ОПЕРАЦИИ:
"""


def format_rows(rows: Sequence[dict]) -> str:
    """Строки для промпта. Сумма даётся, но описание идёт первым."""
    lines = []
    for r in rows:
        amount = r.get("amount")
        amount_text = f"{float(amount):,.2f}" if amount not in (None, "") else "СУММА ОТСУТСТВУЕТ"
        lines.append(
            f"{r['txn_id']} | {r.get('description', '')} | контрагент: "
            f"{r.get('counterparty', '')} | {amount_text} {r.get('currency', '')}"
        )
    return "\n".join(lines)


def build_prompt(rows: Sequence[dict]) -> str:
    return _PROMPT + format_rows(rows)


# --------------------------------------------------------------------------- #
# Результат
# --------------------------------------------------------------------------- #


@dataclass
class TxnCategory:
    txn_id: str
    category: str
    flow: str = "outflow"
    confidence: float = 0.0
    reason: str | None = None
    #: Проставляется кодом, когда категорию получить не удалось.
    fallback: bool = False


@dataclass
class CategoryReport:
    items: dict[str, TxnCategory] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)
    batches: int = 0
    retried: int = 0

    def low_confidence(self, threshold: float = 0.5) -> list[str]:
        return sorted(
            t for t, c in self.items.items() if not c.fallback and c.confidence < threshold
        )

    def fallbacks(self) -> list[str]:
        return sorted(t for t, c in self.items.items() if c.fallback)

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for c in self.items.values():
            out[c.category] = out.get(c.category, 0) + 1
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))

    def alarms(self, expected: int | None = None) -> list[str]:
        out: list[str] = []
        if expected is not None and len(self.items) != expected:
            out.append(
                f"размечено {len(self.items)} операций из {expected} — "
                f"недостающие молча уменьшат агрегаты"
            )
        if self.items:
            share = len(self.fallbacks()) / len(self.items)
            if share > 0.05:
                out.append(
                    f"{share:.0%} операций не удалось разложить — проверьте, "
                    f"доходят ли ответы модели"
                )
            counts = self.counts()
            top = next(iter(counts.values()), 0)
            if top / len(self.items) > 0.6:
                out.append(
                    f"более 60% операций попали в одну статью "
                    f"({next(iter(counts))}) — похоже на вырожденную разметку"
                )
        return out

    def to_dict(self, expected: int | None = None) -> dict:
        return {
            "alarms": self.alarms(expected),
            "problems": self.problems,
            "counts": self.counts(),
            "batches": self.batches,
            "retried": self.retried,
            "low_confidence": self.low_confidence(),
            "fallbacks": self.fallbacks(),
            "items": {t: asdict(c) for t, c in sorted(self.items.items())},
        }


# --------------------------------------------------------------------------- #
# Пакетная разметка
# --------------------------------------------------------------------------- #


def chunk(rows: Sequence[dict], size: int = BATCH_SIZE) -> list[list[dict]]:
    if size <= 0:
        raise ValueError("размер пакета должен быть положительным")
    return [list(rows[i : i + size]) for i in range(0, len(rows), size)]


def categorize_batch(
    rows: Sequence[dict], llm: LLMClient, model: str | None = None
) -> tuple[dict[str, TxnCategory], list[str]]:
    """Один пакет. Возвращает (разметка, замечания)."""
    ids = [r["txn_id"] for r in rows]
    validator = make_txn_category_validator(ids, list(CATEGORIES))

    kwargs: dict[str, Any] = {"model": model} if model else {}
    request = LLMRequest(
        prompt=build_prompt(rows),
        schema=TXN_CATEGORY_SCHEMA,
        prompt_version=PROMPT_VERSION,
        max_tokens=8000,
        **kwargs,
    )

    notes: list[str] = []
    try:
        payload = llm.extract(request, validator=validator).data
    except ValidationFailed as exc:
        # Частичный ответ ценен: разложенные строки уже верны, недостающие
        # дозапросим отдельно. Бросить всё значило бы потерять и то, что есть.
        payload = exc.last_payload if isinstance(exc.last_payload, dict) else {}
        notes.append(f"пакет не прошёл проверку: {'; '.join(exc.problems)[:200]}")

    wanted = set(ids)
    out: dict[str, TxnCategory] = {}
    for raw in payload.get("items", []):
        txn_id = raw.get("txn_id")
        if txn_id not in wanted or txn_id in out:
            # Придуманная или повторная строка молча искажает счёт.
            continue
        category = raw.get("category")
        if category not in CATEGORIES:
            notes.append(f"{txn_id}: статья {category!r} вне словаря — строка не размечена")
            continue
        out[txn_id] = TxnCategory(
            txn_id=txn_id,
            category=category,
            flow=raw.get("flow", "outflow"),
            confidence=float(raw.get("confidence") or 0.0),
            reason=raw.get("reason"),
        )
    return out, notes


def run(
    rows: Sequence[dict],
    paths: RunPaths,
    llm: LLMClient,
    model: str | None = None,
    workers: int = 6,
    batch_size: int = BATCH_SIZE,
) -> CategoryReport:
    """Размечает все строки, не теряя ни одной.

    Стратегия в двух проходах. Первый — пакетами. Второй — только по тем
    строкам, которых не хватило: они собираются заново и запрашиваются
    маленькими пакетами, где ответ обозрим. Что не пришло и во второй раз,
    получает `other` с отметкой `fallback` — молчаливого исчезновения нет
    ни на одном пути.
    """
    report = CategoryReport()
    batches = chunk(rows, batch_size)
    report.batches = len(batches)

    def work(batch: list[dict]):
        return categorize_batch(batch, llm, model)

    for batch, result in zip(batches, LLMClient.map_parallel(work, batches, workers=workers)):
        if isinstance(result, Exception):
            report.problems.append(
                f"пакет из {len(batch)} операций упал — "
                f"{type(result).__name__}: {str(result)[:160]}"
            )
            continue
        marked, notes = result
        report.items.update(marked)
        report.problems.extend(notes)

    missing = [r for r in rows if r["txn_id"] not in report.items]
    if missing:
        report.retried = len(missing)
        log.warning("КАТЕГОРИИ: не размечено %d операций, дозапрашиваю", len(missing))
        small = chunk(missing, max(5, batch_size // 4))
        for batch, result in zip(small, LLMClient.map_parallel(work, small, workers=workers)):
            if isinstance(result, Exception):
                report.problems.append(
                    f"повторный пакет упал — {type(result).__name__}: {str(result)[:160]}"
                )
                continue
            marked, notes = result
            report.items.update(marked)
            report.problems.extend(notes)

    still_missing = [r["txn_id"] for r in rows if r["txn_id"] not in report.items]
    for txn_id in still_missing:
        report.items[txn_id] = TxnCategory(
            txn_id=txn_id, category=FALLBACK_CATEGORY, confidence=0.0,
            reason="не удалось получить категорию", fallback=True,
        )
    if still_missing:
        report.problems.append(
            f"не размечено даже со второй попытки ({len(still_missing)}): "
            f"{still_missing[:10]} — отнесены к {FALLBACK_CATEGORY!r}"
        )

    out = paths.artifacts / A.TXN_CATEGORIES
    out.write_text(
        json.dumps(report.to_dict(expected=len(rows)), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    for alarm in report.alarms(expected=len(rows)):
        log.warning("КАТЕГОРИИ: %s", alarm)
    log.info(
        "Размечено %d операций за %d пакетов; распределение: %s",
        len(report.items), report.batches, report.counts(),
    )
    return report


def load(paths: RunPaths) -> dict[str, TxnCategory]:
    data = json.loads((paths.artifacts / A.TXN_CATEGORIES).read_text(encoding="utf-8"))
    return {t: TxnCategory(**c) for t, c in data["items"].items()}
