"""JSON Schema + содержательные валидаторы для каждого типа извлечения.

Схема проверяет ФОРМУ. Валидатор проверяет СМЫСЛ. Разделение принципиальное:
модель легко вернёт синтаксически безупречный JSON с выдуманным порогом,
и схема это пропустит.

Ключевой антигаллюцинационный механизм — поле `quote`: модель обязана
привести дословную цитату из документа, а `make_quote_validator` проверяет,
что цитата действительно встречается в исходном тексте. Выдумать число,
не выдумав при этом цитату, модель не может.

Реестр EXTRACTORS в конце файла — точка расширения: новый тип извлечения
добавляется одной записью.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Callable

# --------------------------------------------------------------------------- #
# Общие помощники
# --------------------------------------------------------------------------- #


def _normalize(text: str) -> str:
    """Нормализация для сверки цитат.

    Гасит различия, безразличные для смысла, но ломающие точное сравнение:
    неразрывные пробелы и переносы строк из PDF, разные виды тире и кавычек,
    юникодные формы. Регистр НЕ гасим — он может нести смысл в названиях.
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("­", "")  # мягкий перенос
    for dash in "‐‑‒–—―−":
        text = text.replace(dash, "-")
    for quote in "«»“”„‘’":
        text = text.replace(quote, '"')
    return re.sub(r"\s+", " ", text).strip()


def make_quote_validator(
    source_text: str, field: str = "quote", min_len: int = 12, path: tuple = ()
) -> Callable[[dict], list[str]]:
    """Проверяет, что цитаты в ответе дословно встречаются в исходном тексте."""
    haystack = _normalize(source_text)

    def collect(obj, prefix: str = "") -> list[tuple[str, str]]:
        found = []
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k == field and isinstance(v, str):
                    found.append((prefix or "<root>", v))
                else:
                    found.extend(collect(v, f"{prefix}.{k}" if prefix else k))
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                found.extend(collect(v, f"{prefix}[{i}]"))
        return found

    def validator(payload: dict) -> list[str]:
        problems: list[str] = []
        for where, quote in collect(payload):
            q = _normalize(quote)
            if len(q) < min_len:
                problems.append(f"{where}: цитата слишком короткая для проверки: {quote!r}")
            elif q not in haystack:
                problems.append(
                    f"{where}: цитата отсутствует в документе дословно — "
                    f"перенеси её точно как в тексте: {quote[:90]!r}"
                )
        return problems

    return validator


def combine(*validators: Callable[[dict], list[str]] | None) -> Callable[[dict], list[str]]:
    def combined(payload: dict) -> list[str]:
        out: list[str] = []
        for v in validators:
            if v is not None:
                out.extend(v(payload))
        return out

    return combined


_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


# --------------------------------------------------------------------------- #
# 1. Спецификация ковенанта (шаг 5)
# --------------------------------------------------------------------------- #

COVENANT_SPEC_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["covenants"],
    "properties": {
        "covenants": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "point", "title", "direction", "threshold", "unit",
                    "period_start", "period_end", "metric_definition", "quote",
                ],
                "properties": {
                    "point": {"type": "string", "description": "Номер пункта, например '6.1'"},
                    "title": {"type": "string"},
                    "direction": {
                        "type": "string", "enum": ["max", "min"],
                        "description": "max — показатель не должен превышать порог; min — не должен опускаться ниже",
                    },
                    "threshold": {"type": "number", "description": "Порог. Всегда положительное число"},
                    "unit": {
                        "type": "string", "enum": ["amount", "ratio"],
                        "description": "amount — сумма в долларах; ratio — коэффициент или доля",
                    },
                    "period_start": {"type": "string", "description": "ISO-дата YYYY-MM-DD"},
                    "period_end": {"type": "string", "description": "ISO-дата YYYY-MM-DD"},
                    "metric_definition": {
                        "type": "string",
                        "description": "Что именно измеряется, дословно по смыслу договора",
                    },
                    "numerator_definition": {"type": ["string", "null"]},
                    "denominator_definition": {"type": ["string", "null"]},
                    "is_conditional": {
                        "type": "boolean",
                        "description": "true для springing-тестов, применяемых только при срабатывании условия",
                    },
                    "condition_description": {"type": ["string", "null"]},
                    "carve_outs": {
                        "type": "array", "items": {"type": "string"},
                        "description": "Оговорки и исключения, при которых превышение допускается",
                    },
                    "quote": {
                        "type": "string",
                        "description": "Дословная цитата пункта из договора, не менее одного предложения",
                    },
                },
            },
        }
    },
}


def validate_covenant_spec(payload: dict) -> list[str]:
    problems: list[str] = []
    covenants = payload.get("covenants", [])
    seen: set[str] = set()
    for c in covenants:
        p = c.get("point", "?")
        if p in seen:
            problems.append(f"пункт {p} встречается дважды")
        seen.add(p)
        if not isinstance(c.get("threshold"), (int, float)):
            problems.append(f"пункт {p}: порог не число")
        elif c["threshold"] <= 0:
            problems.append(f"пункт {p}: порог должен быть положительным, получено {c['threshold']}")
        for fld in ("period_start", "period_end"):
            if not _DATE_RE.match(str(c.get(fld, ""))):
                problems.append(f"пункт {p}: {fld} не в формате YYYY-MM-DD: {c.get(fld)!r}")
        if _DATE_RE.match(str(c.get("period_start", ""))) and _DATE_RE.match(str(c.get("period_end", ""))):
            if c["period_start"] >= c["period_end"]:
                problems.append(f"пункт {p}: период пуст или перевёрнут")
        if c.get("unit") == "ratio" and c.get("threshold", 0) > 1000:
            problems.append(
                f"пункт {p}: unit=ratio, но порог {c['threshold']} выглядит как сумма — перепроверь"
            )
        if c.get("is_conditional") and not c.get("condition_description"):
            problems.append(f"пункт {p}: is_conditional=true, но условие не описано")
    return problems


# --------------------------------------------------------------------------- #
# 2. Аудиторские корректировки (шаг 7)
# --------------------------------------------------------------------------- #

ADJUSTMENT_KINDS = [
    "reclassification",      # переклассификация между статьями
    "fx_translation",        # пересчёт валютной операции, раскрыт курс/сумма расчёта
    "cutoff",                # отсечение: операция относится к другому периоду
    "off_ledger",            # сумма для агрегирования, не отражённая операцией
    "missing_amount",        # сумма операции отсутствует в выгрузке реестра
    "revenue_recognition",   # правило признания выручки
    "ebitda_adjustment",     # корректировка EBITDA
    "no_effect",             # примечание прочитано, влияния на ковенанты нет
    "other",
]

AUDIT_ADJUSTMENTS_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["notes"],
    "properties": {
        "notes": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["note_id", "kind", "description", "quote"],
                "properties": {
                    "note_id": {"type": "string", "description": "Номер примечания, например '8.1'"},
                    "kind": {"type": "string", "enum": ADJUSTMENT_KINDS},
                    "target_txn_id": {"type": ["string", "null"]},
                    "target_counterparty": {"type": ["string", "null"]},
                    "from_category": {"type": ["string", "null"]},
                    "to_category": {"type": ["string", "null"]},
                    "value_usd": {
                        "type": ["number", "null"],
                        "description": "Сумма в USD, положительная. Направление — в поле sign",
                    },
                    "sign": {"type": ["string", "null"], "enum": ["expense", "income", None]},
                    "period_start": {"type": ["string", "null"]},
                    "period_end": {"type": ["string", "null"]},
                    "description": {"type": "string"},
                    "quote": {"type": "string"},
                },
            },
        }
    },
}


def validate_audit_adjustments(payload: dict) -> list[str]:
    problems: list[str] = []
    for n in payload.get("notes", []):
        nid = n.get("note_id", "?")
        kind = n.get("kind")
        if kind in {"missing_amount", "off_ledger"} and n.get("value_usd") is None:
            problems.append(f"примечание {nid}: kind={kind} требует value_usd, получено null")
        if kind == "missing_amount" and not n.get("target_txn_id"):
            problems.append(f"примечание {nid}: kind=missing_amount требует target_txn_id")
        if kind == "reclassification" and not (n.get("from_category") or n.get("to_category")):
            problems.append(f"примечание {nid}: переклассификация без указания статей")
        v = n.get("value_usd")
        if v is not None and v < 0:
            problems.append(
                f"примечание {nid}: value_usd должно быть положительным, направление — в sign"
            )
    return problems


# --------------------------------------------------------------------------- #
# 3. Связанные стороны (шаг 8)
# --------------------------------------------------------------------------- #

RELATED_PARTIES_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["threshold_pct", "parties", "quote"],
    "properties": {
        "threshold_pct": {
            "type": "number",
            "description": "Порог доли голосующих прав для признания связанной стороной, в процентах",
        },
        "parties": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["name", "ownership_pct", "is_related"],
                "properties": {
                    "name": {"type": "string"},
                    "ownership_pct": {"type": ["number", "null"]},
                    "is_related": {"type": "boolean"},
                    "basis": {"type": ["string", "null"]},
                },
            },
        },
        "quote": {"type": "string"},
    },
}


def validate_related_parties(payload: dict) -> list[str]:
    problems: list[str] = []
    th = payload.get("threshold_pct")
    if not isinstance(th, (int, float)) or not (0 < th <= 100):
        problems.append(f"threshold_pct вне диапазона (0, 100]: {th!r}")
        return problems
    for p in payload.get("parties", []):
        pct = p.get("ownership_pct")
        name = p.get("name", "?")
        if pct is None:
            continue
        if not (0 <= pct <= 100):
            problems.append(f"{name}: доля вне диапазона [0, 100]: {pct}")
            continue
        expected = pct >= th
        if bool(p.get("is_related")) != expected and not p.get("basis"):
            problems.append(
                f"{name}: доля {pct}% при пороге {th}% даёт is_related={expected}, "
                f"а указано {p.get('is_related')}. Если основание иное — заполни basis"
            )
    return problems


# --------------------------------------------------------------------------- #
# 4. Категоризация транзакций (шаг 10)
# --------------------------------------------------------------------------- #

TXN_CATEGORY_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["items"],
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["txn_id", "category", "flow", "confidence"],
                "properties": {
                    "txn_id": {"type": "string"},
                    "category": {"type": "string"},
                    "flow": {
                        "type": "string", "enum": ["outflow", "inflow", "reversal"],
                        "description": (
                            "outflow — расход; inflow — поступление; "
                            "reversal — возврат/сторно ранее совершённой операции"
                        ),
                    },
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "reason": {"type": ["string", "null"]},
                },
            },
        }
    },
}


def make_txn_category_validator(
    expected_ids: list[str], allowed_categories: list[str]
) -> Callable[[dict], list[str]]:
    expected = set(expected_ids)
    allowed = set(allowed_categories)

    def validator(payload: dict) -> list[str]:
        problems: list[str] = []
        got = [i.get("txn_id") for i in payload.get("items", [])]
        got_set = set(got)
        if len(got) != len(got_set):
            dupes = [t for t in got_set if got.count(t) > 1]
            problems.append(f"дубли txn_id в ответе: {sorted(dupes)[:5]}")
        missing = expected - got_set
        if missing:
            problems.append(f"пропущены транзакции ({len(missing)}): {sorted(missing)[:5]}")
        extra = got_set - expected
        if extra:
            problems.append(f"придуманы транзакции, которых не было: {sorted(extra)[:5]}")
        for i in payload.get("items", []):
            if i.get("category") not in allowed:
                problems.append(
                    f"{i.get('txn_id')}: категория {i.get('category')!r} вне таксономии"
                )
        return problems

    return validator


# --------------------------------------------------------------------------- #
# Реестр — точка расширения
# --------------------------------------------------------------------------- #

EXTRACTORS: dict[str, dict] = {
    "covenant_spec": {
        "schema": COVENANT_SPEC_SCHEMA,
        "validator": validate_covenant_spec,
        "quote_checked": True,
    },
    "audit_adjustments": {
        "schema": AUDIT_ADJUSTMENTS_SCHEMA,
        "validator": validate_audit_adjustments,
        "quote_checked": True,
    },
    "related_parties": {
        "schema": RELATED_PARTIES_SCHEMA,
        "validator": validate_related_parties,
        "quote_checked": True,
    },
    "txn_category": {
        "schema": TXN_CATEGORY_SCHEMA,
        "validator": None,  # строится динамически: нужен список ожидаемых id
        "quote_checked": False,
    },
}


# --------------------------------------------------------------------------- #
# 5. Транскрипция страницы скана (шаг 2)
# --------------------------------------------------------------------------- #

PAGE_TRANSCRIPTION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["text"],
    "properties": {
        "text": {
            "type": "string",
            "description": "Весь видимый текст страницы, дословно, с сохранением порядка",
        },
        "uncertain": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Фрагменты, в распознавании которых нет уверенности",
        },
        "has_tables": {"type": "boolean"},
    },
}

EXTRACTORS["page_transcription"] = {
    "schema": PAGE_TRANSCRIPTION_SCHEMA,
    "validator": None,
    "quote_checked": False,
}
