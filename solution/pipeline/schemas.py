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


#: Номер страницы, вклинившийся между строками. В извлечённом тексте PDF
#: колонтитул попадает прямо в середину предложения:
#:
#:     «...в адрес аффилированных и связанных\n5\nсторон не должны...»
#:
#: Модель цитирует ОСМЫСЛЕННО и цифру опускает — и совершенно правильно
#: делает. Первая версия сверки объявляла такую цитату выдуманной и
#: выбрасывала верный ковенант (реальный случай: пункт 6.3 у P5).
#: Требовать от модели воспроизводить артефакты вёрстки бессмысленно:
#: проверка должна ловить ВЫДУМКУ, а не различия в раскладке.
_PAGE_NUMBER_LINE = re.compile(r"\n\s*\d{1,4}\s*\n")


def _normalize(text: str) -> str:
    """Нормализация для сверки цитат.

    Гасит различия, безразличные для смысла, но ломающие точное сравнение:
    неразрывные пробелы и переносы строк из PDF, разные виды тире и кавычек,
    юникодные формы, номера страниц посреди предложения. Регистр НЕ гасим —
    он может нести смысл в названиях.
    """
    text = _PAGE_NUMBER_LINE.sub("\n", text)
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

from .covenant_types import (  # noqa: E402
    ANY_CATEGORY,
    CATEGORIES,
    PARTY_KINDS,
    REGISTRY,
    SCOPES,
)

#: ПЛОСКОЕ ПРЕДСТАВЛЕНИЕ ДЕРЕВА — БЕЗ РЕКУРСИИ
#:
#: Первая версия описывала дерево рекурсивно: `args` содержал `$ref` на сам
#: узел. Схема валидна, SDK её принимал, а на живом прогоне выяснилось, что
#: gemini-3.6-flash РЕКУРСИЮ НЕ РАЗВОРАЧИВАЕТ: все узлы-операции приходили
#: как `{"op": "DIV"}` с пустым `args`. Уцелели только листья — голые `AGG`.
#: Это стоило бы половины ковенантов: из 36 семнадцать являются отношениями.
#:
#: Отказ был не громким, а коварным: JSON валиден по схеме, поля на месте,
#: и лишь семантический валидатор поймал «DIV без аргументов».
#:
#: Поэтому рекурсия убрана вовсе. Дерево передаётся ПЛОСКИМ СПИСКОМ узлов
#: с идентификаторами, а связи задаются ссылками по имени:
#:
#:     nodes: [{id: "root", op: "DIV", args: ["a", "b"]},
#:             {id: "a", op: "AGG", category: "revenue"},
#:             {id: "b", op: "AGG", category: "opex"}]
#:     root:  "root"
#:
#: `args` здесь — массив СТРОК, никаких вложенных объектов и `$ref`.
#: Такую форму поддерживает любой провайдер, и она не зависит от того,
#: насколько добросовестно он реализует JSON Schema. В дерево это
#: разворачивает код (`nest_metric`), где ошибки видны и проверяемы.
_METRIC_NODE = {
    "type": "object",
    "additionalProperties": False,
    "required": ["id", "op"],
    "properties": {
        "id": {
            "type": "string",
            "description": "Короткое уникальное имя узла, например 'root', 'a', 'b'",
        },
        "op": {
            "type": "string",
            "enum": sorted(REGISTRY),
            "description": (
                "AGG — сумма по статье (лист дерева, args не нужен); "
                "ADD/SUB/MUL/DIV/MAX/MIN — операции, args ОБЯЗАТЕЛЕН; "
                "DISCLOSED — величина, раскрытая в документах и отсутствующая "
                "в реестре; CONST — константа"
            ),
        },
        "category": {
            "type": "string",
            "enum": [*CATEGORIES, ANY_CATEGORY],
            "description": (
                f"Только для AGG. {ANY_CATEGORY!r} — любая статья: так задаются "
                "ковенанты, где важен получатель платежа, а не статья расхода"
            ),
        },
        "scope": {
            "type": "string", "enum": list(SCOPES),
            "description": "borrower по умолчанию; group — показатель всей Группы",
        },
        "party": {
            "type": "string", "enum": list(PARTY_KINDS),
            "description": "Ограничение по типу контрагента, если ковенант его требует",
        },
        "period": {
            "type": "array", "items": {"type": "string"},
            "minItems": 2, "maxItems": 2,
            "description": "Свой период узла (квартальный срез), если отличается от общего",
        },
        "key": {"type": "string", "description": "Только для DISCLOSED: имя раскрытой величины"},
        "value": {"type": "number", "description": "Только для CONST"},
        "args": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "ИМЕНА (id) дочерних узлов из этого же списка, по порядку. "
                "Для DIV: [числитель, знаменатель]. Для SUB: [уменьшаемое, "
                "вычитаемое]. Не объекты, а строки-ссылки"
            ),
        },
    },
}

COVENANT_SPEC_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["covenants"],
    "$defs": {"metric_node": _METRIC_NODE},  # оставлен для совместимости схем
    "properties": {
        "covenants": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "point", "title", "direction", "threshold", "unit",
                    "period_start", "period_end", "metric_definition",
                    "metric_nodes", "metric_root", "quote",
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
                        "description": "Что измеряется — словами, для прослеживаемости",
                    },
                    "metric_nodes": {
                        "type": "array",
                        "minItems": 1,
                        "items": _METRIC_NODE,
                        "description": (
                            "Плоский список узлов дерева. Связи — ссылками по id, "
                            "вложенных объектов быть не должно"
                        ),
                    },
                    "metric_root": {
                        "type": "string",
                        "description": "id корневого узла — с него начинается вычисление",
                    },
                    "is_conditional": {
                        "type": "boolean",
                        "description": "true для springing-тестов, применяемых только при срабатывании условия",
                    },
                    "condition_nodes": {
                        "type": "array", "items": _METRIC_NODE,
                        "description": "Узлы дерева условия springing-теста",
                    },
                    "condition_root": {"type": "string"},
                    "condition_direction": {"type": "string", "enum": ["max", "min"]},
                    "condition_threshold": {"type": ["number", "null"]},
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


class MetricGraphError(ValueError):
    """Плоский список узлов не складывается в дерево."""


def nest_metric(nodes: list, root_id: str, _depth: int = 0) -> dict:
    """Плоский список узлов → вложенное дерево выражений.

    ЗАЧЕМ ЭТОТ ПЕРЕХОД СУЩЕСТВУЕТ. Модель отдаёт плоский список, потому что
    рекурсивные схемы провайдер не разворачивает (см. комментарий к
    `_METRIC_NODE`). Движок шага 12, наоборот, работает с вложенным деревом
    и ничего о плоской форме не знает. Стык между ними — здесь, в одном
    месте и под тестами.

    ЧТО ЗДЕСЬ ПРОВЕРЯЕТСЯ И ПОЧЕМУ ИМЕННО ЭТО. Ссылки по имени открывают
    три способа сломать дерево, которых при вложенности физически не было:
    ссылка в никуда, цикл и потерянный корень. Каждый из них — не
    исключение при вычислении, а неверное ЧИСЛО или зависание, поэтому
    ловятся они здесь, до расчёта.
    """
    if _depth > 12:
        raise MetricGraphError("дерево глубже 12 уровней — похоже на цикл")

    index = {}
    for node in nodes:
        if not isinstance(node, dict):
            continue
        node_id = node.get("id")
        if node_id in index:
            raise MetricGraphError(f"узел {node_id!r} объявлен дважды")
        index[node_id] = node

    if root_id not in index:
        raise MetricGraphError(
            f"корневой узел {root_id!r} отсутствует в списке; есть: {sorted(index)}"
        )

    def build(node_id: str, seen: tuple, depth: int) -> dict:
        if depth > 12:
            raise MetricGraphError("дерево глубже 12 уровней — похоже на цикл")
        if node_id in seen:
            raise MetricGraphError(
                f"цикл в дереве: {' -> '.join((*seen, node_id))}"
            )
        node = index.get(node_id)
        if node is None:
            raise MetricGraphError(
                f"узел {node_id!r} упомянут в args, но не объявлен; есть: {sorted(index)}"
            )
        out = {k: v for k, v in node.items() if k not in ("id", "args")}
        children = node.get("args") or []
        if children:
            out["args"] = [build(str(c), (*seen, node_id), depth + 1) for c in children]
        return out

    return build(root_id, (), _depth)


def flatten_metric(tree: dict, prefix: str = "n") -> tuple[list, str]:
    """Дерево → плоский список. Обратная операция, нужна тестам и отладке."""
    nodes: list = []
    counter = [0]

    def walk(node: dict) -> str:
        counter[0] += 1
        node_id = f"{prefix}{counter[0]}"
        flat = {k: v for k, v in node.items() if k != "args"}
        flat["id"] = node_id
        children = [walk(a) for a in node.get("args", [])]
        if children:
            flat["args"] = children
        nodes.append(flat)
        return node_id

    root = walk(tree)
    return nodes, root


def metric_from_payload(covenant: dict) -> dict:
    """Достаёт дерево из ковенанта в любой из двух форм.

    Плоская форма — основная. Вложенная поддерживается, потому что
    провайдеры различаются: если какой-то из них рекурсию всё же отдаёт,
    незачем ломать работающий ответ.
    """
    if isinstance(covenant.get("metric"), dict):
        return covenant["metric"]
    nodes = covenant.get("metric_nodes")
    if not nodes:
        raise MetricGraphError("ни metric, ни metric_nodes не заполнены")
    return nest_metric(nodes, str(covenant.get("metric_root", "")))


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
        # Поля условия переехали в плоскую форму вместе с основным деревом.
        # Проверка осталась на старом имени и объявляла верный springing-тест
        # неполным (реальный случай: пункт 6.1 у P3).
        if c.get("is_conditional") and not (
            c.get("condition_metric") or c.get("condition_nodes")
        ):
            problems.append(
                f"пункт {p}: is_conditional=true, но дерево условия "
                f"(condition_nodes/condition_root) не задано — "
                f"springing-тест невозможно проверить"
            )
        try:
            metric = metric_from_payload(c)
        except MetricGraphError as exc:
            problems.append(f"пункт {p}: {exc}")
        else:
            problems.extend(f"пункт {p}: {m}" for m in _validate_metric(metric))
    return problems


def _validate_metric(node, depth: int = 0) -> list[str]:
    """Дерево обязано быть исполнимым движком шага 12."""
    if depth > 12:
        return ["дерево слишком глубокое"]
    if not isinstance(node, dict):
        return ["metric отсутствует или не является узлом"]
    op = node.get("op")
    if op not in REGISTRY:
        return [f"операция {op!r} не зарегистрирована; известны {sorted(REGISTRY)}"]

    out: list[str] = []
    if op == "AGG":
        category = node.get("category")
        if category not in (*CATEGORIES, ANY_CATEGORY):
            out.append(
                f"AGG с категорией {category!r} вне словаря — агрегат окажется пустым"
            )
        if node.get("args"):
            out.append("AGG не принимает вложенных узлов")
    elif op == "DISCLOSED":
        if not node.get("key"):
            out.append("DISCLOSED без key")
    elif op == "CONST":
        if not isinstance(node.get("value"), (int, float)):
            out.append("CONST без числового value")
    else:
        args = node.get("args") or []
        if not args:
            out.append(f"{op} без аргументов")
        if op == "DIV" and len(args) != 2:
            out.append(f"DIV ожидает ровно два аргумента, получено {len(args)}")
        for a in args:
            out.extend(_validate_metric(a, depth + 1))
    return out


# --------------------------------------------------------------------------- #
# 2. Аудиторские корректировки (шаг 7)
# --------------------------------------------------------------------------- #

#: Что документ ГОВОРИТ о корректировке. Отдельно от того, что она собой
#: представляет.
#:
#: ДВЕ ЛОВУШКИ, ОБЕ ВСТРЕТИЛИСЬ В ПУБЛИЧНОМ НАБОРЕ. Приложение аудитора
#: упоминает суммы, которые применять НЕЛЬЗЯ, и выглядят они как обычные
#: корректировки:
#:
#:   * B1, п. 9.1: «Сумма $592,296.10 … была ОТОБРАНА ДЛЯ ПРОВЕРКИ
#:     классификации. Вывод изложен в отчёте № AR-2025-0634 и в настоящих
#:     примечаниях НЕ ПОВТОРЯЕТСЯ». Вывода здесь нет — применять нечего;
#:   * P10, п. 7.2: операция «рассматривалась на предмет возможной
#:     переклассификации; по итогам разъяснений ПЕРВОНАЧАЛЬНАЯ
#:     КЛАССИФИКАЦИЯ СОХРАНЯЕТСЯ».
#:
#: В обоих случаях есть сумма, контрагент и слово «переклассификация».
#: Применить их — значит переложить сотни тысяч долларов между статьями
#: без основания, и результат будет выглядеть совершенно нормально.
#: Поэтому статус извлекается явным полем, а код применяет только `applied`.
ADJUSTMENT_STATUS = [
    "applied",                  # корректировка сделана и описана здесь
    "considered_but_rejected",  # рассматривалась, первоначальный учёт сохранён
    "referred_elsewhere",       # вывод вынесен в другой документ, здесь его нет
    "informational",            # описание правила, без конкретной суммы
]

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
                "required": ["note_id", "kind", "status", "description", "quote"],
                "properties": {
                    "note_id": {"type": "string", "description": "Номер примечания, например '8.1'"},
                    "kind": {"type": "string", "enum": ADJUSTMENT_KINDS},
                    "status": {
                        "type": "string",
                        "enum": ADJUSTMENT_STATUS,
                        "description": (
                            "applied — корректировка сделана; "
                            "considered_but_rejected — рассматривалась, но первоначальная "
                            "классификация сохранена; "
                            "referred_elsewhere — вывод вынесен в другой документ и здесь "
                            "не повторяется; "
                            "informational — описание правила без конкретной суммы"
                        ),
                    },
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
        },
        "materiality_threshold_usd": {
            "type": ["number", "null"],
            "description": (
                "Порог существенности, если он назван: «Разовыми для целей ковенантов "
                "признаются статьи в сумме не менее $300,000.00». Сравнение с ним "
                "делает КОД — не отбрасывай статьи сам, перечисли все"
            ),
        },
        "materiality_quote": {"type": ["string", "null"]},
        "no_adjustments_stated": {
            "type": "boolean",
            "description": (
                "true, если сказано «Переклассификаций за ковенантный период "
                "не требовалось» и иных корректировок нет"
            ),
        },
    },
}


def validate_audit_adjustments(payload: dict) -> list[str]:
    problems: list[str] = []
    threshold = payload.get("materiality_threshold_usd")
    if threshold is not None and (not isinstance(threshold, (int, float)) or threshold <= 0):
        problems.append(f"порог существенности должен быть положительным: {threshold!r}")

    if payload.get("no_adjustments_stated"):
        applied = [n.get("note_id", "?") for n in payload.get("notes", [])
                   if n.get("status") == "applied"]
        if applied:
            problems.append(
                f"сказано, что корректировок не требовалось, но примечания {applied} "
                f"помечены как применённые — определись"
            )

    for n in payload.get("notes", []):
        nid = n.get("note_id", "?")
        kind = n.get("kind")
        status = n.get("status")
        if status not in ADJUSTMENT_STATUS:
            problems.append(f"примечание {nid}: неизвестный статус {status!r}")
            continue
        if status != "applied":
            # У неприменяемых примечаний не требуем полноты полей: суммы
            # и статьи там могут быть названы вскользь.
            continue
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

#: Досье KYC: кто связанная сторона и по какому признаку.
#:
#: ПОРОГ УЧАСТИЯ РАЗНЫЙ У КАЖДОГО ЗАЁМЩИКА. В публичном наборе встретились
#: значения от 20.0% до 40.0%, и записаны они в тексте самого досье:
#: «Организации, в которых Группа владеет 20.0% и более голосующих прав,
#: признаются связанными сторонами для целей Договора». Взять 40% как
#: универсальное правило означало бы у восьми заёмщиков из двенадцати
#: недосчитать связанные стороны — и получить заниженный, но правдоподобный
#: `actual`. Поэтому порог ИЗВЛЕКАЕТСЯ, а не предполагается.
#:
#: РЕШЕНИЕ ПРИНИМАЕТ КОД, А НЕ МОДЕЛЬ. Признание связанной стороной — это
#: сравнение `доля >= порог`, то есть арифметика. Мнение модели (`is_related`)
#: всё же запрашивается, но не как решение, а как независимая сверка: если
#: её вердикт расходится с расчётом, значит одно из двух чисел прочитано
#: неверно, и это стоит увидеть. Проверка бесплатна.
RELATED_PARTIES_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["has_ownership_section", "threshold_pct", "parties"],
    "properties": {
        "has_ownership_section": {
            "type": "boolean",
            "description": (
                "Есть ли в досье раздел о бенефициарном владении с долями "
                "участия. false — если такого раздела нет вовсе. Это НЕ то же "
                "самое, что пустой список: отсутствие раздела и отсутствие "
                "связанных сторон различаются"
            ),
        },
        "threshold_pct": {
            "type": ["number", "null"],
            "description": (
                "Порог доли голосующих прав для признания связанной стороной, "
                "в процентах, как написано в досье. null — если порог не указан"
            ),
        },
        "threshold_quote": {
            "type": ["string", "null"],
            "description": "Дословная фраза, где назван порог",
        },
        "parties": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["name", "ownership_pct", "is_related"],
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Название организации ТОЧНО как в досье",
                    },
                    "ownership_pct": {
                        "type": ["number", "null"],
                        "description": "Доля голосующих прав в процентах",
                    },
                    "is_related": {
                        "type": "boolean",
                        "description": (
                            "Твоё мнение: связанная ли сторона. Решение всё равно "
                            "принимает код сравнением доли с порогом — это поле "
                            "нужно для сверки"
                        ),
                    },
                    "basis": {
                        "type": ["string", "null"],
                        "description": "Признак связанности, если он не доля участия",
                    },
                    "quote": {
                        "type": "string",
                        "description": "Дословная строка досье с названием и долей",
                    },
                },
            },
        },
    },
}


def validate_related_parties(payload: dict) -> list[str]:
    problems: list[str] = []
    th = payload.get("threshold_pct")

    if "has_ownership_section" not in payload:
        # ОТСУТСТВИЕ ПОЛЯ И ЗНАЧЕНИЕ false — не одно и то же. Раньше здесь
        # стоял `payload.get(...)`, и ответ без этого поля молча пропускал
        # ВСЕ последующие проверки: и порог, и доли, и сверку с мнением
        # модели. Схема поле требует, но валидатор не вправе полагаться
        # на то, что схему кто-то применил.
        return ["has_ownership_section не заполнено — неясно, есть ли раздел о владении"]

    if not payload["has_ownership_section"]:
        # Досье без раздела о владении — законный случай (встретился у P2).
        # Требовать от него порог и список участников бессмысленно.
        if payload.get("parties"):
            problems.append(
                "has_ownership_section=false, но участники перечислены — "
                "определись: либо раздел есть, либо участников нет"
            )
        return problems

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
                    "category": {
                        "type": "string",
                        "enum": list(CATEGORIES),
                        "description": (
                            "Статья расхода или поступления. ТОЛЬКО из перечисления: "
                            "выдуманная статья не упадёт, она молча выпадет из агрегата"
                        ),
                    },
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

# --------------------------------------------------------------------------- #
# Тип документа — запасной путь шага 3
# --------------------------------------------------------------------------- #

#: Используется ТОЛЬКО когда правила видимо промахнулись (см. classify.alarms).
#: Модель здесь не «понимает документ», а отвечает на один вопрос с закрытым
#: списком ответов, и обязана привести дословную строку-основание: без неё
#: ответ невозможно отличить от догадки.
DOC_TYPE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["doc_type", "evidence_quote", "confidence"],
    "properties": {
        "doc_type": {
            "type": "string",
            "enum": ["LOAN", "AUDIT_DRAFT", "AUDIT_FINAL", "KYC",
                     "TREASURY_MEMO", "BACKGROUND"],
        },
        "evidence_quote": {
            "type": "string",
            "description": "Дословная строка из документа, по которой определён тип. "
                           "Для BACKGROUND допустима пустая строка.",
        },
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
}
