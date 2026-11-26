"""End-to-end scenarios: Consultant events -> engine -> reactions to the fake Consultant."""
from __future__ import annotations

import json
from pathlib import Path

from support_bot.core.engine import GREETING

from .conftest import Harness
from .fakes import client_message as msg
from .fakes import set_chat, unset_chat

PERSONAS = json.loads((Path(__file__).parents[1] / "support_bot/data/personas.json").read_text())


# ------------------------------------------------------------------ greeting


async def test_set_chat_sends_english_greeting_with_12_category_buttons(h: Harness):
    await h.send(set_chat())
    [greet] = h.consultant.sent
    assert greet["$type"] == "BotMessage"
    assert greet["text"] == GREETING
    buttons = greet["replyMarkup"]["buttons"]
    assert [b["text"] for b in buttons] == [
        "Deposit", "Withdrawal", "Registration", "Bonuses", "Account", "Sport", "App", "Rules",
        "Contacts", "Partnership", "Casino & Games", "Verification & KYC",
    ]
    assert all(b["$type"] == "ButtonVariant" and len(b["text"]) <= 20 for b in buttons)


# ------------------------------------------------------------------- persona


async def test_persona_from_geo_pool_introduces_itself_once(h: Harness):
    await h.send(set_chat(), msg("forgot password", country="EG"))
    d = await h.dialogue()
    assert d.persona.name in PERSONAS["geo"]["EG"]["names"]
    reply = h.consultant.for_chat("chat1")[-1]
    assert reply["$type"] == "BotReplyMessage"
    assert f"My name is {d.persona.name}" in reply["text"]
    assert reply["result"] == "Full"

    await h.send(msg("minimum deposit", country="EG"))
    assert "My name is" not in h.consultant.last_text("chat1")
    assert (await h.dialogue()).persona == d.persona  # fixed for the dialogue


async def test_unknown_geo_uses_base_pool_and_is_not_guessed(h: Harness):
    await h.send(set_chat(), msg("forgot password", country=None))
    d = await h.dialogue()
    assert d.country is None
    assert d.persona.pool.geo is None
    assert d.persona.name in PERSONAS["base"]["names"]


async def test_geo_without_pool_uses_base_pool(h: Harness):
    await h.send(set_chat(), msg("forgot password", country="FR"))
    assert (await h.dialogue()).persona.name in PERSONAS["base"]["names"]


async def test_new_dialogue_gets_new_state(h: Harness):
    await h.send(set_chat(), msg("forgot password"))
    first = await h.dialogue()
    await h.send(unset_chat(), set_chat(conversation_id="conv2"))
    second = await h.dialogue()
    assert second is not first
    assert second.persona is None  # chosen again on the first message
    assert second.conversation_id == "conv2"


# ------------------------------------------------------------------ language


async def test_reply_language_follows_last_message_and_updates_chat_language(h: Harness):
    await h.send(set_chat(), msg("забыл пароль", language="en"))
    sent = h.consultant.for_chat("chat1")
    assert {"$type": "ChangeChatLanguage", "chatId": "chat1", "languageCode": "ru"} in sent
    assert h.consultant.last_text("chat1").startswith("[ru]")

    h.consultant.clear()
    await h.send(msg("minimum deposit", language="ru"))
    assert h.consultant.last_text("chat1").startswith("[en]")
    assert (await h.dialogue()).language == "en"


async def test_change_chat_language_is_sent_once_per_change(h: Harness):
    await h.send(set_chat(), msg("забыл пароль", language="ky"), msg("спасибо", language="ky"),
                 msg("minimum deposit", language="ky"))
    changes = [p["languageCode"] for p in h.consultant.sent if p["$type"] == "ChangeChatLanguage"]
    assert changes == ["ru", "en"]


async def test_language_fallback_is_english(h: Harness):
    await h.send(set_chat(), msg("12345", language=None))
    assert h.consultant.last_text("chat1").startswith("[en]")


async def test_platform_language_code_is_normalized(h: Harness):
    await h.send(set_chat(), msg("12345", language="fr-FR"))
    assert h.consultant.last_text("chat1").startswith("[fr]")
    assert "ChangeChatLanguage" not in h.consultant.types("chat1")


# --------------------------------------------------------- buttons / topics


async def test_category_click_shows_five_subquestions(h: Harness):
    await h.send(set_chat(), msg("Account", variant="cat:account"))
    reply = h.consultant.for_chat("chat1")[-1]
    rows = reply["replyMarkup"]["rows"]  # some labels are > 20 chars -> rows, not buttons
    assert [r["text"] for r in rows] == [
        "Login issues", "Forgot password", "Restore access", "Verification code not received", "Account blocked",
    ]
    assert rows[1]["id"] == "q:account:1"


async def test_subquestion_click_answers_from_kb(h: Harness):
    await h.send(set_chat(), msg("Account", variant="cat:account"), msg("Forgot password", variant="q:account:1"))
    reply = h.consultant.for_chat("chat1")[-1]
    assert reply["$type"] == "BotReplyMessage"
    assert "Forgot password?" in reply["text"]
    assert reply["result"] == "Full"


async def test_unrecognized_request_shows_categories(h: Harness):
    await h.send(set_chat(), msg("hmm"))
    reply = h.consultant.for_chat("chat1")[-1]
    assert reply["result"] == "NotFound"
    assert len(reply["replyMarkup"]["buttons"]) == 12


# --------------------------------------------------------- operator request


async def test_operator_request_is_not_escalated_immediately(h: Harness):
    await h.send(set_chat(), msg("call operator please"))
    assert "InvokeOperator" not in h.consultant.types()
    assert "help you myself" in h.consultant.last_text("chat1")


# ------------------------------------------------------------ group A cycle


async def test_group_a_escalates_after_five_unsuccessful_attempts(h: Harness):
    """ТЗ 14.3: an attempt counts only when it did not help; 5 failed attempts -> L1."""
    await h.send(set_chat(), msg("my bonus problem"))
    assert (await h.dialogue()).issue.failures == 0  # describing the problem is not a failure
    for i, text in enumerate(["didn't help", "operator", "still not working", "doesn't work", "operator please"]):
        await h.send(msg(text))
        assert "InvokeOperator" not in h.consultant.types(), f"escalated too early at failure {i + 1}"
    assert (await h.dialogue()).issue.failures == 5

    await h.send(msg("still the bonus problem, didn't help"))
    tail = h.consultant.types("chat1")[-3:]
    # SupportBotMessage and the goodbye MUST go before InvokeOperator (403 after it)
    assert tail == ["SupportBotMessage", "BotReplyMessage", "InvokeOperator"]
    note = h.consultant.for_chat("chat1")[-3]["text"]
    assert "bonuses" in note and "attempts: 5" in note

    # after hand-off the bot stays silent
    before = len(h.consultant.sent)
    await h.send(msg("hello?"))
    assert len(h.consultant.sent) == before
    assert h.consultant.rejected == []


async def test_long_productive_dialogue_is_not_escalated(h: Harness):
    """Answers to clarifying questions are progress, not attempts (reference dialogues 1-3)."""
    await h.send(set_chat(), msg("how to make a deposit"))
    for text in ["OPay", "I opened the cashier", "the amount field", "found it", "entered 300", "now what",
                 "pressed confirm", "it asks for a code"]:
        await h.send(msg(text))
    assert "InvokeOperator" not in h.consultant.types()
    assert (await h.dialogue()).issue.failures == 0


async def test_counters_are_kept_per_problem(h: Harness):
    await h.send(set_chat(), msg("bonus problem"))
    for _ in range(3):
        await h.send(msg("didn't help"))
    await h.send(msg("registration problem"))
    d = await h.dialogue()
    assert d.issue.key == "registration" and d.issue.failures == 0
    assert d.issues["bonuses"].failures == 3

    await h.send(msg("and the bonus thing still doesn't work"))  # back to the old problem: counter continues
    d = await h.dialogue()
    assert d.issue.key == "bonuses" and d.issue.failures == 4
    assert "InvokeOperator" not in h.consultant.types()


async def test_group_d_never_escalates(h: Harness):
    await h.send(set_chat())
    for _ in range(12):
        await h.send(msg("I want to become a partner, operator"))
    assert "InvokeOperator" not in h.consultant.types()


async def test_complaint_is_handled_by_the_bot_first(h: Harness):
    # test review п.1/12: accusations of fraud are not a reason for L1 at once, the usual cycle applies
    await h.send(set_chat())
    for _ in range(5):
        await h.send(msg("this is a scam, I will complain"))
    assert "InvokeOperator" not in h.consultant.types()
    await h.send(msg("complain!"))
    assert h.consultant.types()[-1] == "InvokeOperator"


# ----------------------------------------------------------- group B payments


async def test_payment_without_time_asks_for_time_then_collects_details_over_24h(h: Harness):
    await h.send(set_chat(), msg("my deposit did not arrive"))
    reply = h.consultant.for_chat("chat1")[-1]
    assert "How much time" in reply["text"]
    assert [b["id"] for b in reply["replyMarkup"]["buttons"]] == ["wait:lt2", "wait:2to24", "wait:gt24"]
    assert "InvokeOperator" not in h.consultant.types()

    await h.send(msg("3 days already"))
    reply = h.consultant.for_chat("chat1")[-1]
    assert "method" in reply["text"]  # dialogue 4: collect data for L1 first
    labels = [b["text"] for b in reply["replyMarkup"]["buttons"]]
    assert labels[:5] == ["Vodafone Cash", "Etisalat Cash", "Orange Cash", "InstaPay", "Crypto"]  # п.6: EG list
    await h.send(msg("Vodafone Cash"), msg("1000, yesterday at 8:40"), msg("01001982785"))
    assert "InvokeOperator" not in h.consultant.types()
    receipt = [{"$type": "MediaPhoto", "id": "r1", "mimeType": "image/jpeg", "size": 10}]
    await h.send(msg(None, media=receipt))
    assert h.consultant.types("chat1")[-3:] == ["SupportBotMessage", "BotReplyMessage", "InvokeOperator"]
    note = h.consultant.for_chat("chat1")[-3]["text"]
    assert "deposit" in note and "72h" in note
    assert "method: Vodafone Cash" in note and "receipt: received" in note
    assert "amount: 1000" in note and "wallet: 01001982785" in note
    assert "recipient" not in note  # п.8: never asked
    asked = " ".join(p.get("text") or "" for p in h.consultant.for_chat("chat1"))
    assert "recipient" not in asked.lower()


async def test_payment_over_24h_given_upfront_does_not_ask_time_again(h: Harness):
    await h.send(set_chat(), msg("сделал депозит 3 дня назад на 1000, до сих пор не пришел", language="ru"))
    d = await h.dialogue()
    assert d.payment.over_threshold and d.payment.details["amount"] == "1000"
    assert d.awaiting == "payment_details"
    assert not any("How much time" in (p.get("text") or "") for p in h.consultant.sent)


async def test_irritation_while_collecting_is_explained_then_second_pushback_escalates(h: Harness):
    await h.send(set_chat(), msg("withdrawal not received for more than a day"), msg("Vodafone Cash"))
    await h.send(msg("I already sent everything, operator!"))
    assert "InvokeOperator" not in h.consultant.types()  # dialogue 4: explain why we need it
    assert (await h.dialogue()).payment.pushback == 1
    await h.send(msg("operator now!!!"))
    assert h.consultant.types()[-1] == "InvokeOperator"
    note = [p for p in h.consultant.sent if p["$type"] == "SupportBotMessage"][-1]["text"]
    assert "method: Vodafone Cash" in note and "upset" in note


async def test_payment_within_24h_is_not_escalated_even_on_operator_requests(h: Harness):
    await h.send(set_chat(), msg("deposit not arrived"), msg("40 minutes"))
    assert "deposits are credited within 24 hours" in h.consultant.last_text("chat1")
    for _ in range(8):
        await h.send(msg("I want operator"))
    assert "InvokeOperator" not in h.consultant.types()


async def test_payment_wait_button_click(h: Harness):
    await h.send(set_chat(), msg("deposit not arrived"), msg("> 24 hours", variant="wait:gt24"))
    d = await h.dialogue()
    assert d.payment.over_threshold and d.awaiting == "payment_details"


async def test_ambiguous_yesterday_is_clarified_once(h: Harness):
    await h.send(set_chat(), msg("deposit not arrived"), msg("yesterday"))
    assert "more precisely" in h.consultant.last_text("chat1")
    await h.send(msg("about 5 hours"))
    d = await h.dialogue()
    assert d.payment.within_norm and d.payment.wait_hours == 5


async def test_payment_user_does_not_answer_wait_question_escalates(h: Harness):
    # test review п.2: only after 3 messages in a row without an answer about the time
    await h.send(set_chat(), msg("deposit not arrived"), msg("call operator"), msg("operator!!!"))
    assert "InvokeOperator" not in h.consultant.types()
    await h.send(msg("OPERATOR"))
    assert h.consultant.types()[-1] == "InvokeOperator"


# ------------------------------------------------------------------- closing


async def test_resolution_confirmation_and_goodbye(h: Harness):
    await h.send(set_chat(), msg("forgot password"), msg("ok thanks"))
    reply = h.consultant.for_chat("chat1")[-1]
    assert "Did that solve" in reply["text"]
    assert [b["id"] for b in reply["replyMarkup"]["buttons"]] == ["confirm:yes", "confirm:no"]

    await h.send(msg("Yes", variant="confirm:yes"))
    types = h.consultant.types("chat1")
    assert types[-2:] == ["BotReplyMessage", "ResolveChat"]
    assert "Glad I could help" in h.consultant.last_text("chat1")
    assert "InvokeOperator" not in types


async def test_not_resolved_continues_dialogue(h: Harness):
    await h.send(set_chat(), msg("forgot password"), msg("thanks"), msg("No", variant="confirm:no"))
    assert "ResolveChat" not in h.consultant.types()
    d = await h.dialogue()
    assert d.active and d.issue.failures == 1  # "no" = the answer did not help


# --------------------------------------------------------------------- media


async def test_file_without_text_asks_to_describe(h: Harness):
    media = [{"$type": "MediaDocument", "id": "m1", "mimeType": "image/jpeg", "size": 10}]
    await h.send(set_chat(), msg(None, media=media))
    assert "describe the problem in text" in h.consultant.last_text("chat1")


# ------------------------------------------------------------- robustness


async def test_message_without_set_chat_still_answered(h: Harness):
    await h.send(msg("forgot password", chat_id="orphan"))
    assert h.consultant.types("orphan") == ["BotReplyMessage"]


async def test_chats_are_independent(h: Harness):
    await h.send(set_chat("a"), set_chat("b"), msg("forgot password", chat_id="a", country="NG"),
                 msg("forgot password", chat_id="b", country="KE"))
    a, b = await h.dialogue("a"), await h.dialogue("b")
    assert a.persona.name in PERSONAS["geo"]["NG"]["names"]
    assert b.persona.name in PERSONAS["geo"]["KE"]["names"]


async def test_403_from_consultant_stops_sending_to_chat(h: Harness):
    await h.send(set_chat())
    h.consultant.forbidden_chats.add("chat1")
    await h.send(msg("forgot password"))
    assert not (await h.dialogue()).active
    rejected = len(h.consultant.rejected)
    await h.send(msg("hello"))
    assert len(h.consultant.rejected) == rejected  # no more attempts


async def test_unset_chat_drops_state(h: Harness):
    await h.send(set_chat(), msg("forgot password"), unset_chat(reason="Timeout"))
    assert await h.dialogue() is None


async def test_consecutive_messages_are_glued_with_debounce():
    h = Harness(debounce=0.05)
    from support_bot.consultant.models import parse_event

    await h.send(set_chat())
    for text in ("sir", "sir", "forgot password"):
        h.dispatcher.submit(parse_event(msg(text)))
    await h.dispatcher.drain()
    replies = [p for p in h.consultant.sent if p["$type"] == "BotReplyMessage"]
    assert len(replies) == 1
    assert h.llm.analyze_calls[-1].text == "sir\nsir\nforgot password"
