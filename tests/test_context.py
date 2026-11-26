"""Dialogue context: glueing, supersede, platform history, live operator, silence timer, corner cases."""
from __future__ import annotations

import asyncio

from support_bot.consultant.models import parse_event
from support_bot.core.engine import EngineConfig

from .conftest import Harness
from .fakes import client_message as msg
from .fakes import set_chat


class SlowLLM:
    """Wraps FakeLLM: analyze takes a while, so the user can write in between."""

    def __init__(self, inner, delay: float):
        self.inner, self.delay = inner, delay

    async def analyze(self, req):
        await asyncio.sleep(self.delay)
        return await self.inner.analyze(req)

    def __getattr__(self, name):
        return getattr(self.inner, name)


async def test_message_during_thinking_supersedes_the_unsent_answer():
    h = Harness()
    h.engine.llm = SlowLLM(h.llm, 0.05)
    await h.send(set_chat(), msg("bonus problem"))
    h.consultant.clear()
    h.dispatcher.submit(parse_event(msg("didn't help")))
    await asyncio.sleep(0.01)  # the bot is thinking about "didn't help"...
    h.dispatcher.submit(parse_event(msg("I mean the welcome bonus is not credited")))
    await h.dispatcher.drain()
    replies = [p for p in h.consultant.sent if p["$type"] == "BotReplyMessage"]
    assert len(replies) == 1  # one answer to both messages
    assert h.llm.analyze_calls[-1].text == "didn't help\nI mean the welcome bonus is not credited"
    assert (await h.dialogue()).issue.failures == 1  # counted once, not twice


async def test_fragments_wait_longer_than_full_messages():
    h = Harness(debounce=0.0)
    h.dispatcher.fragment_debounce = 0.05
    await h.send(set_chat())
    h.dispatcher.submit(parse_event(msg("sir")))
    await asyncio.sleep(0.01)
    h.dispatcher.submit(parse_event(msg("sir ?")))
    await asyncio.sleep(0.01)
    h.dispatcher.submit(parse_event(msg("my deposit did not arrive")))
    await h.dispatcher.drain()
    assert len([p for p in h.consultant.sent if p["$type"] == "BotReplyMessage"]) == 1
    assert "How much time" in h.consultant.last_text("chat1")


class HistoryConsultant:
    def __init__(self, consultant, items):
        self.c, self.items = consultant, items

    async def get_chat_messages(self, chat_id, skip=0, limit=100):
        return {"items": self.items, "total": len(self.items)}

    def __getattr__(self, name):
        return getattr(self.c, name)


async def test_live_operator_in_chat_silences_the_bot():
    h = Harness()
    items = []
    h.dispatcher.client = HistoryConsultant(h.consultant, items)
    h.dispatcher.fetch_history = True
    await h.send(set_chat(), msg("forgot password"))
    bot_text = h.consultant.last_text("chat1")
    items += [{"Id": 1, "text": "forgot password", "fromId": None},
              {"Id": 2, "text": bot_text, "fromId": "bot-7"},  # our own message: not an operator
              {"Id": 3, "text": "Hello, I'm Anna from support, let me check", "fromId": "op-42"}]
    before = len(h.consultant.sent)
    await h.send(msg("thank you Anna"))
    assert len(h.consultant.sent) == before
    d = await h.dialogue()
    assert d.operator_joined and not d.active
    assert d.history[-1].role == "operator"


async def test_context_restored_from_platform_history_after_restart():
    h = Harness()
    h.dispatcher.client = HistoryConsultant(h.consultant, [
        {"Id": 1, "text": "my withdrawal did not arrive", "fromId": None},
        {"Id": 2, "text": "How much time has passed since the request?", "fromId": "bot-7"},
    ])
    h.dispatcher.fetch_history = True
    await h.send(msg("ok"))  # no SetChat, empty store: the process was restarted
    d = await h.dialogue()
    assert [t.role for t in d.history[:2]] == ["user", "bot"]
    assert h.llm.analyze_calls[-1].history[0].text == "my withdrawal did not arrive"


async def test_silence_after_wait_question_hands_off():
    h = Harness()
    h.engine.cfg = EngineConfig(payment_silence_seconds=0.05)
    await h.send(set_chat(), msg("my deposit did not arrive"))
    assert "InvokeOperator" not in h.consultant.types()
    await h.dispatcher.drain(include_silence=True)
    assert h.consultant.types()[-3:] == ["SupportBotMessage", "BotMessage", "InvokeOperator"]
    assert "no answer about the waiting time" in h.consultant.for_chat("chat1")[-3]["text"]


async def test_answer_before_silence_timer_cancels_it():
    h = Harness()
    h.engine.cfg = EngineConfig(payment_silence_seconds=0.05)
    await h.send(set_chat(), msg("my deposit did not arrive"), msg("40 minutes"))
    await h.dispatcher.drain(include_silence=True)
    assert "InvokeOperator" not in h.consultant.types()


# ---------------------------------------------------------------- corner cases


async def test_noise_never_counts_and_never_escalates(h: Harness):
    await h.send(set_chat())
    for _ in range(10):
        await h.send(msg("sir sir ?"))
    assert "InvokeOperator" not in h.consultant.types()
    d = await h.dialogue()
    assert d.noise_count == 10 and all(i.failures == 0 for i in d.issues.values())
    texts = [p["text"] for p in h.consultant.sent if p["$type"] == "BotReplyMessage"]
    assert "How can I help" in texts[0]  # test review п.18: the first one is greeted, with top topics
    first = [p for p in h.consultant.sent if p["$type"] == "BotReplyMessage"][0]
    assert [b["id"] for b in first["replyMarkup"]["rows"]] == ["top:deposit_missing", "top:withdrawal_missing",
                                                                  "top:login"]


async def test_off_topic_is_redirected_and_not_escalated(h: Harness):
    await h.send(set_chat())
    for _ in range(8):
        await h.send(msg("what's the weather in Cairo"))
    assert "InvokeOperator" not in h.consultant.types()
    assert "only help with AvaBet" in h.consultant.last_text("chat1")


async def test_aggression_is_a_tone_not_a_topic(h: Harness):
    await h.send(set_chat(), msg("my deposit did not arrive you idiots"))
    d = await h.dialogue()
    assert d.payment and d.payment.kind == "deposit" and d.current == "deposit"
    assert h.llm.compose_calls[-1].aggressive


async def test_gambling_harm_gets_care_and_a_human(h: Harness):
    await h.send(set_chat(), msg("I lost everything and can't stop playing"))
    assert h.consultant.types()[-3:] == ["SupportBotMessage", "BotReplyMessage", "InvokeOperator"]
    assert "risk signal: gambling_harm" in h.consultant.for_chat("chat1")[-3]["text"]


async def test_short_reply_does_not_flip_language(h: Harness):
    await h.send(set_chat(), msg("забыл пароль от аккаунта", language="ru"), msg("ok", language="ru"))
    assert (await h.dialogue()).language == "ru"
    assert h.consultant.last_text("chat1").startswith("[ru]")


async def test_previous_contact_note_is_optional():
    from support_bot.core.state import ContactMemory
    from .fakes import unset_chat

    h = Harness()
    h.engine.memory = ContactMemory(ttl_seconds=3600)
    await h.send(set_chat(), msg("forgot password"), unset_chat(reason="Timeout"), set_chat(conversation_id="c2"))
    d = await h.dialogue()
    assert d.previous_contact and "account" in d.previous_contact
