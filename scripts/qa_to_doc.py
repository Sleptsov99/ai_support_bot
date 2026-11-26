"""Turn the Q&A sheet into ONE knowledge base document (markdown) for the full-context prompt.

    python scripts/qa_to_doc.py qa.xlsx -o support_bot/data/kb.md                       # sheet "NEW ТЗ Итог"
    python scripts/qa_to_doc.py qa.xlsx --sheet "ТЗ без дублирования ответов" -o kb.md
    python scripts/qa_to_doc.py qa.csv  -o kb.md
    python scripts/qa_to_doc.py kb.json -o kb.md

The AvaBet sheet ("ТЗ бот основные блоки вопросы/ответы") has the columns
Кнопка | Категория | Вопрос | Ответ бота | Статус | Условие | Эскалация | Ключевые слова по категории.
Download it with File -> Download -> .xlsx: CSV / gviz exports drop rows hidden in collapsed groups.

What the converter does:
* rows are grouped by block (column "Кнопка", mapped to the button catalog) and by answer: many
  rows are just other wordings of one question, they become ``variants`` of one entry;
* a row without an answer continues the entry above (more wordings);
* block header rows ("📝 1. Регистрация и профиль") open a section, their keywords are kept;
* Статус / Условие / Эскалация go to a ``note:`` line for the bot (never shown to the user);
* the sheet "ТЗ по подсказкам по блокам" ("maybe you were looking for...") is added per block
  when present: the bot uses it on the "offer options" attempt.
Prints counts and a size estimate (what the cached prompt prefix will cost).
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from support_bot.core.catalog import BY_KEY, CATEGORIES  # noqa: E402
from support_bot.core.kb import KBEntry, render_markdown  # noqa: E402

COLUMNS = {
    "button": ("кнопка", "button", "блок"),
    "category": ("category", "тема", "категория", "topic"),
    "question": ("question", "вопрос", "подтема", "subquestion"),
    "answer": ("answer", "ответ", "ответ бота"),
    "status": ("статус", "status"),
    "condition": ("условие", "condition"),
    "escalation": ("эскалация", "escalation"),
    "keywords": ("keywords", "ключевые слова", "синонимы", "ключевые слова по категории"),
    "id": ("id",),
}
# Block (column "Кнопка") -> KB section key. Catalog keys for the 12 buttons, own keys for the rest.
BLOCKS: list[tuple[str, str, str]] = [  # (pattern, key, title)
    (r"открытие бота|кнопки выбора|привет", "greeting", "Приветствие и стартовое меню"),
    (r"registration|регистрац", "registration", "Registration: регистрация и профиль"),
    (r"account|аккаунт", "account", "Account: доступ к аккаунту, личный кабинет"),
    (r"verification|kyc|верифик", "kyc", "Verification & KYC: верификация и безопасность"),
    (r"deposit|пополн", "deposit", "Deposit: пополнение счета"),
    (r"withdraw|вывод", "withdrawal", "Withdrawal: вывод средств"),
    (r"bonus|бонус", "bonuses", "Bonuses: бонусы и акции"),
    (r"casino|казино|слот", "casino", "Casino & Games: слоты, казино и игры"),
    (r"sport|спорт", "sport", "Sport: ставки на спорт"),
    (r"contact|контакт", "contacts", "Contacts: контакты и поддержка"),
    (r"rules|правил", "rules", "Rules: правила и информация"),
    (r"^app\b|^арр|приложен", "app", "App: приложение"),
    (r"partner|сотрудн", "partnership", "Partnership: сотрудничество"),
    (r"скрин|файл|нераспозн|прочее", "other", "Скриншоты, файлы, нераспознанные и прочие запросы"),
    (r"жалоб|конфликт", "complaint", "Жалобы и конфликтные кейсы"),
    (r"оператор", "operator", "Запрос оператора"),
    (r"окончание|заверш", "closing", "Окончание диалога"),
    (r"истори|результат|расч", "sport", "Sport: ставки на спорт"),
]
TITLES = {key: title for _, key, title in BLOCKS}
_HEADER = re.compile(r"^\W*\d+\.\s")  # "📝 1. Регистрация и профиль"


def block_key(raw: str) -> str | None:
    low = raw.strip().lower()
    for pattern, key, _ in BLOCKS:
        if re.search(pattern, low):
            return key
    for c in CATEGORIES:
        if low in (c.key, c.label.lower()):
            return c.key
    return None


def _norm_header(h: object) -> str:
    return str(h or "").strip().lower()


def _col_index(header: list[str]) -> dict[str, int]:
    idx: dict[str, int] = {}
    for i, h in enumerate(header):
        for name, aliases in COLUMNS.items():
            if name not in idx and h in aliases:
                idx[name] = i
    return idx


def _cell(row: list[str], idx: dict[str, int], name: str) -> str:
    i = idx.get(name)
    return row[i].strip() if i is not None and i < len(row) and row[i] else ""


def read_table(path: Path, sheet: str | None) -> list[list[str]]:
    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xlsm"):
        from openpyxl import load_workbook  # optional dependency: pip install openpyxl

        wb = load_workbook(path, read_only=True, data_only=True)
        ws = wb[sheet] if sheet else (wb["NEW ТЗ Итог"] if "NEW ТЗ Итог" in wb.sheetnames else wb.active)
        return [["" if v is None else str(v) for v in row] for row in ws.iter_rows(values_only=True)]
    if suffix == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        header = ["id", "category", "question", "answer", "keywords"]
        return [header] + [[str(e.get("id", "")), e.get("category", ""), e.get("question", ""), e.get("answer", ""),
                            ", ".join(e.get("keywords", []))] for e in raw]
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.reader(f))


def read_hints(path: Path) -> dict[str, str]:
    """Sheet "ТЗ по подсказкам по блокам": block -> "maybe you were looking for" text (Russian)."""
    if path.suffix.lower() not in (".xlsx", ".xlsm"):
        return {}
    from openpyxl import load_workbook

    wb = load_workbook(path, read_only=True, data_only=True)
    name = next((n for n in wb.sheetnames if "подсказк" in n.lower()), None)
    if not name:
        return {}
    rows = [["" if v is None else str(v).strip() for v in r] for r in wb[name].iter_rows(values_only=True)]
    hints: dict[str, str] = {}
    current: str | None = None
    for r in rows[1:]:
        if len(r) < 3:
            continue
        q, a = r[1], r[2]
        if q and q == a:
            current = block_key(q)
        elif current and a and "искали" in a.lower():
            hints[current] = a
    return hints


def convert(rows: list[list[str]]) -> tuple[list[KBEntry], dict[str, str]]:
    header_at = next(i for i, r in enumerate(rows) if any(_norm_header(h) in COLUMNS["question"] for h in r))
    idx = _col_index([_norm_header(h) for h in rows[header_at]])
    entries: list[KBEntry] = []
    section_keywords: dict[str, str] = {}
    by_answer: dict[tuple[str, str], int] = {}
    last: int | None = None
    for row in rows[header_at + 1:]:
        q, a = _cell(row, idx, "question"), _cell(row, idx, "answer")
        if not q and not a:
            continue
        block_raw = _cell(row, idx, "button") or _cell(row, idx, "category")
        key = block_key(block_raw) or "other"
        if (q and q == a and _HEADER.match(q)) or (q and _HEADER.match(q) and not _cell(row, idx, "category")
                                                   and "button" in idx):
            kws = _cell(row, idx, "keywords")
            if kws:
                section_keywords[key] = kws
            last = None
            continue
        if not a:  # another wording of the question above
            if last is not None and entries[last].category == key and q:
                e = entries[last]
                entries[last] = KBEntry(e.id, e.category, e.question, e.answer, e.keywords, e.language,
                                        e.variants + (q,), e.note)
            continue
        note_parts = []
        for col, label in (("status", "статус"), ("condition", "условие"), ("escalation", "эскалация")):
            v = _cell(row, idx, col)
            if v and v.lower() != "нет":
                note_parts.append(f"{label}: {v}")
        note = "; ".join(note_parts)
        k = (key, a)
        if k in by_answer:
            i = by_answer[k]
            e = entries[i]
            new_note = e.note if not note or note in e.note else "; ".join(x for x in (e.note, note) if x)
            entries[i] = KBEntry(e.id, e.category, e.question, e.answer, e.keywords, e.language,
                                 e.variants + ((q,) if q and q != e.question else ()), new_note)
            last = i
            continue
        own_kws = tuple(x.strip() for x in re.split(r"[,;\n]", _cell(row, idx, "keywords")) if x.strip()) \
            if "button" not in idx else ()  # in the AvaBet sheet keywords are per block, not per row
        entry_id = _cell(row, idx, "id") or f"{key}-{sum(1 for e in entries if e.category == key) + 1}"
        entries.append(KBEntry(entry_id, key, q or a.splitlines()[0][:80], a, own_kws, "ru", (), note))
        by_answer[k] = last = len(entries) - 1
    return entries, section_keywords


# Corrections from the test review («Правки по тестам AI Support бота», Nov 2026): the sheet is older than
# these decisions. Applied to the generated document, so regenerating the KB keeps them.
LIMITS_ANSWER = ("Минимальные и максимальные суммы пополнения и вывода зависят от способа оплаты и условий платежных "
                 "методов, поэтому иногда они могут обновляться.\n\nАктуальные лимиты для вашего способа оплаты "
                 "всегда указаны в форме пополнения или вывода: выберите метод, и суммы будут показаны в его окне.")
CORRECTIONS: list[tuple[str, str]] = [
    # п.3: one deposit term everywhere - up to 24 hours
    ("Пополнение счета занимает от 1 минуты до 12 часов", "Пополнение счета занимает от нескольких минут до 24 часов"),
    # п.17: verification takes up to 24 hours; agents write to the Payment Assistant Bot
    ("до 72 часов", "до 24 часов"),
    ("<@AVA_managment>", "<@payment_assistment_avabet_bot>"),
    # п.7: limits - a prepared answer, no invented reasons ("set by the payment system", "try later")
    ("При формировании запроса на пополнение счета будет отображена минимальная и максимальная сумма депозита. Лимит "
     "на пополнение устанавливает сторонняя платежная система, через которую проводится перевод.\n\nРекомендуем "
     "повторить попытку позднее, либо воспользоваться альтернативными способами пополнения.", LIMITS_ANSWER),
    ("При формировании запроса на вывод денежных средств, будет отображена максимальная и минимальная сумма вывода. "
     "Лимит на вывод денежных средств устанавливает сторонняя платежная система, через которую проводится перевод.",
     LIMITS_ANSWER),
    ("Максимальная сумма зависит от способа оплаты. Актуальные лимиты указаны на странице пополнения рядом с каждым "
     "методом.", LIMITS_ANSWER),
    # п.4: "5th time asks for an operator" is decided by the engine, the text must not promise a transfer itself
    ("Понял(а) вас. Передаю обращение оператору.", "note: передачу оператору решает система, сам бот её не обещает"),
    ("\n\nТакже передаю ваше обращение живому специалисту прямо сейчас.", "\nnote: передачу оператору решает система"),
    ("Если хотите поговорить с живым оператором:", "Если удобнее написать в поддержку:"),
    ("\n\nЕсли вопрос срочный — я прямо сейчас передаю обращение оператору.", ""),    ('Я передам ваше обращение специалисту для разбора.', 'note: передачу оператору решает система, сам бот её не обещает'),
    (' Я также прямо сейчас передаю информацию старшему специалисту.', ''),
    ('С этими данными я передам ваш запрос специалисту для детальной проверки расчёта.', 'Эти данные помогут разобраться в расчёте.\nnote: передачу оператору решает система, сам бот её не обещает'),
    ('Я немедленно передаю ваше обращение старшему специалисту.', 'note: передачу оператору решает система, сам бот её не обещает'),
    ('Прямо сейчас передаю ваше обращение специалисту, который сможет рассмотреть ситуацию детально.', 'note: передачу оператору решает система, сам бот её не обещает'),
]
# п.15: one directory of contacts by language and topic
CONTACTS_DIRECTORY = """## Справочник контактов поддержки {#contacts-directory}
note: единственный источник контактов для ответов. Общая поддержка: по языку игрока. Другие адреса из старых ответов не давать, кроме контактов партнерской программы в разделе Partnership.
Общая поддержка (арабский язык): <support-ar@ava.bet>
Общая поддержка (остальные языки): <support-en@ava.bet>
Платежи (депозит, вывод): <payment-support@ava.bet> или Telegram <@AVAPayment_bot>
Закрытие аккаунта, самоисключение: <block@ava.bet>
Документы, запрос документов службой безопасности: <security@ava.bet>
Стать агентом: Telegram <@payment_assistment_avabet_bot>
Сотрудничество и партнерская программа: <info@ava.bet>

"""


def apply_corrections(doc: str) -> str:
    for old, new in CORRECTIONS:
        doc = doc.replace(old, new)
    marker = "{#contacts}\n"
    at = doc.find(marker)
    if at >= 0 and "{#contacts-directory}" not in doc:
        line_end = doc.find("\n## ", at)  # after the section keywords line
        if line_end >= 0:
            doc = doc[:line_end + 1] + CONTACTS_DIRECTORY + doc[line_end + 1:]
    return doc


def build_document(entries: list[KBEntry], section_keywords: dict[str, str], hints: dict[str, str]) -> str:
    order = [key for _, key, _ in BLOCKS]
    entries = sorted(entries, key=lambda e: order.index(e.category) if e.category in order else len(order))
    extra = []
    for key, text in hints.items():
        extra.append(KBEntry(f"{key}-hints", key, "Если вопрос по теме не распознан (подсказки)", text, (), "ru"))
    entries = entries + extra
    entries = sorted(entries, key=lambda e: order.index(e.category) if e.category in order else len(order))
    head = [
        "<!-- База знаний AvaBet, собрана скриптом scripts/qa_to_doc.py из таблицы",
        "     «ТЗ бот основные блоки вопросы/ответы». Формат: # Раздел {#key}, ## Вопрос {#id},",
        "     variants: другие формулировки того же вопроса, note: пометки для бота (статус/условие/эскалация",
        "     из таблицы, клиенту не показываются), дальше ответ. Ответы написаны по-русски: бот отвечает",
        "     на языке клиента, сохраняя смысл, шаги, названия разделов сайта, контакты и цифры. -->",
        "",
    ]
    doc = render_markdown(entries, TITLES, keep_order=True)
    if section_keywords:  # keywords of a block go under its section header
        for key, kws in section_keywords.items():
            doc = doc.replace(f"{{#{key}}}\n", f"{{#{key}}}\nКлючевые слова раздела: {kws.replace(chr(10), ' ')}\n", 1)
    return apply_corrections("\n".join(head) + doc)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("source", type=Path)
    ap.add_argument("-o", "--out", type=Path, required=True)
    ap.add_argument("--sheet", default=None)
    args = ap.parse_args()
    entries, section_keywords = convert(read_table(args.source, args.sheet))
    hints = read_hints(args.source)
    doc = build_document(entries, section_keywords, hints)
    args.out.write_text(doc, encoding="utf-8")
    variants = sum(len(e.variants) for e in entries)
    per = {}
    for e in entries:
        per[e.category] = per.get(e.category, 0) + 1 + len(e.variants)
    print(f"{len(entries)} answers covering {len(entries) + variants} questions, {len(hints)} hint blocks; "
          f"{len(doc):,} chars (~{len(doc) // 3:,} tokens) -> {args.out}")
    print("questions per section:", ", ".join(f"{k}={v}" for k, v in per.items()))
    unknown = sorted({e.category for e in entries} - set(BY_KEY) - set(TITLES))
    if unknown:
        print("sections outside the catalog:", ", ".join(unknown))


if __name__ == "__main__":
    main()
