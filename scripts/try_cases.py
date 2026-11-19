"""Run a few real user cases end-to-end on a real model and print the dialogues.

    pip install -r requirements.txt
    DEEPSEEK_API_KEY=sk-... python scripts/try_cases.py            # all cases
    DEEPSEEK_API_KEY=sk-... python scripts/try_cases.py 2          # only case 2
    DEEPSEEK_API_KEY=sk-... python scripts/try_cases.py --compare  # every case in 3 language modes side by side
    ANTHROPIC_API_KEY=sk-ant-... python scripts/try_cases.py        # the same on Claude

Language modes (--mode, or all three with --compare):
    native    the model reads and writes the client's language itself, KB stays Russian
    pivot-ru  every message -> Russian -> answer from the Russian KB -> back (Kirill's variant)
    pivot-en  every message -> English -> answer -> back (machine translation is strongest via English)

The whole engine runs (webhook events -> dispatcher -> engine -> Consultant reactions), with the
real knowledge base (support_bot/data/kb.md) and every language going through the translator into
Russian (PIVOT_LANGUAGES="*", PIVOT_TARGET=ru). Consultant is replaced by a recorder, nothing is
sent anywhere except the model API. For each turn you see what the client wrote, what the bot
understood in Russian, the reply the client gets, buttons, and hidden messages for L1.
"""
from __future__ import annotations

import asyncio
import itertools
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from support_bot.app import build_engine, llm_provider  # noqa: E402
from support_bot.consultant.models import parse_event  # noqa: E402
from support_bot.core.kb import InMemoryKB  # noqa: E402
from support_bot.core.personas import PersonaRegistry  # noqa: E402
from support_bot.demo.server import TIMEZONES, RecordingConsultant  # noqa: E402
from support_bot.dispatcher import Dispatcher  # noqa: E402

CASES = [
    {
        "title": "1. Египет, арабский (диалект): депозит не пришёл, прошёл час -> обычный срок, без эскалации",
        "geo": "EG", "lang": "ar", "channel": "App_Android",
        "messages": [
            "عملت ايداع من ساعة بفودافون كاش ولسه موصلش",  # made a deposit an hour ago via Vodafone Cash, not arrived
            "طب ليه بياخد وقت كده؟ عايز اكلم حد",  # why does it take so long? I want to talk to someone
            "تمام شكرا",  # ok thanks
        ],
    },
    {
        "title": "2. Кения, суахили: не может войти в аккаунт -> ответ из базы, 'не помогло' -> следующая попытка",
        "geo": "KE", "lang": "sw", "channel": "WebMobi",
        "messages": [
            "Nimesahau nenosiri langu na siwezi kuingia kwenye akaunti",  # forgot my password, can't log in
            "Bado haifanyi kazi, SMS haifiki",  # still not working, the SMS doesn't arrive
            "Sawa, asante sana",  # ok, thank you very much
        ],
    },
    {
        "title": "3. Нигерия, английский: депозит не пришёл больше суток -> сбор данных -> передача на L1",
        "geo": "NG", "lang": "en", "channel": "Web",
        "messages": [
            "I deposited 5000 naira with my bank card yesterday morning and it's still not in my balance!!",
            "Visa card",
            "I don't have a screenshot right now",
            "yesterday around 9am",
            "I already gave you everything, just call a manager",
            "this is ridiculous, operator NOW",
        ],
    },
    {
        "title": "4. Эфиопия, амхарский: как вывести деньги -> слабый для моделей язык, тут перевод нужнее всего",
        "geo": "ET", "lang": "am", "channel": "App_Android",
        "messages": [
            "ገንዘብ እንዴት ማውጣት እችላለሁ?",  # how can I withdraw money?
            "አልገባኝም፣ የትኛውን ቁልፍ ልጫን?",  # I don't understand, which button should I press?
        ],
    },
]


MODES = {"hybrid": ("sw,am,zu,ky,af", "en"), "native": ("", "ru"), "pivot-ru": ("*", "ru"), "pivot-en": ("*", "en")}


async def run_case(case: dict, idx: int) -> None:
    kb = InMemoryKB.from_path(Path(os.environ.get("KB_PATH", ROOT / "support_bot/data/kb.md")))
    engine = build_engine(kb, personas=PersonaRegistry.from_file(rng=random.Random(idx)))
    consultant = RecordingConsultant()
    disp = Dispatcher(engine, consultant)
    chat = f"case{idx}"
    ids = itertools.count(1)

    async def deliver(ev: dict) -> list[dict]:
        before = len(consultant.sent)
        disp.submit(parse_event(ev))
        await disp.drain()
        return consultant.sent[before:]

    print("\n" + "=" * 100 + f"\n{case['title']}\n" + "=" * 100)
    await deliver({"$type": "SetChat", "chatId": chat, "conversationId": f"conv-{chat}"})
    for text in case["messages"]:
        ev = {"$type": "ClientMessage", "chatId": chat, "conversationId": f"conv-{chat}", "projectId": 1,
              "messageId": next(ids), "client": {"id": chat}, "text": text, "languageCode": case["lang"],
              "channel": case["channel"], "geo": {"country": case["geo"], "timezone": TIMEZONES.get(case["geo"])}}
        print(f"\n👤 КЛИЕНТ: {text}")
        t0 = time.monotonic()
        reactions = await deliver(ev)
        dt = time.monotonic() - t0
        d = await engine.store.get(chat)
        if d and d.history:
            users = [t.text for t in d.history if t.role == "user"]
            if users and users[-1] != text:
                print(f"   ↳ бот прочитал: {users[-1]}")
        for r in reactions:
            kind = r["$type"]
            if kind in ("BotMessage", "BotReplyMessage"):
                print(f"🤖 БОТ [{r.get('result', '')}]: {r['text']}")
                markup = r.get("replyMarkup") or {}
                labels = [b["text"] for b in markup.get("buttons", []) + markup.get("rows", [])]
                if labels:
                    print("   кнопки: " + " · ".join(labels))
            elif kind == "SupportBotMessage":
                print("🔒 ЗАМЕТКА ДЛЯ L1:\n   " + r["text"].replace("\n", "\n   "))
            elif kind == "InvokeOperator":
                print("📞 ПЕРЕДАНО ОПЕРАТОРУ L1")
            elif kind == "ChangeChatLanguage":
                print(f"   (язык чата -> {r['languageCode']})")
            elif kind == "ResolveChat":
                print("✅ ДИАЛОГ ЗАКРЫТ")
        if d:
            print(f"   [{dt:.1f}s | тема: {d.issue.key if d.current else '-'} | сигнал: {d.last_signal} | "
                  f"неудач: {d.issue.failures} | язык: {d.language} | персона: {d.persona.name if d.persona else '-'}]")
        if d and d.payment:
            p = d.payment
            print(f"   [платёж: {p.kind}, ждёт {p.wait_hours}ч, собрано: {p.details or '-'}, спросили: {p.details_asked}]")
        if d and not d.active:
            break


async def main() -> None:
    provider = llm_provider()
    if provider == "fake":
        sys.exit("Нужен ключ: DEEPSEEK_API_KEY=... или ANTHROPIC_API_KEY=...")
    os.environ.setdefault("PAYMENT_SILENCE_SECONDS", "0")
    args = sys.argv[1:]
    only = {int(a) for a in args if a.isdigit()}
    if "--compare" in args:
        modes = list(MODES)
    elif "--mode" in args:
        modes = [args[args.index("--mode") + 1]]
    else:
        modes = [None]  # whatever PIVOT_LANGUAGES / PIVOT_TARGET say (default: the app default)
    for i, case in enumerate(CASES, 1):
        if only and i not in only:
            continue
        for mode in modes:
            if mode:
                os.environ["PIVOT_LANGUAGES"], os.environ["PIVOT_TARGET"] = MODES[mode]
            label = mode or f"PIVOT_LANGUAGES={os.environ.get('PIVOT_LANGUAGES', '(default)')}"
            print(f"\n### Модель: {provider}; режим языков: {label}")
            await run_case(case, i)


if __name__ == "__main__":
    asyncio.run(main())
