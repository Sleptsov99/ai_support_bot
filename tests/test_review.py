"""Test review (итерация «правки по тесту»): one test per item of the review document."""
from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

from support_bot.core.engine import Engine
from support_bot.core.guard import promises_handoff
from support_bot.core.llm import Composed, Task
from support_bot.core.actions import Option
from support_bot.core.timeparse import hours_since

from .conftest import Harness
from .fakes import client_message as msg
from .fakes import set_chat

CAIRO = ZoneInfo("Africa/Cairo")
TUE = dt.datetime(2026, 11, 24, 15, 0, tzinfo=CAIRO)  # Tuesday 24.11, 15:00


def at(h: Harness, when: dt.datetime) -> Harness:
    h.engine._clock = lambda: when
    return h


def buttons(reply: dict) -> list[str]:
    m = reply.get("replyMarkup") or {}
    return [b["id"] for b in m.get("buttons", []) + m.get("rows", [])]


def labels(reply: dict) -> list[str]:
    m = reply.get("replyMarkup") or {}
    return [b["text"] for b in m.get("buttons", []) + m.get("rows", [])]


# ------------------------------------------------------------- п.1 risk / losses

async def test_loss_complaint_and_scam_claim_are_not_l1(h: Harness):
    await h.send(set_chat(), msg("I lost everything, this casino is a scam"))
    assert "InvokeOperator" not in h.consultant.types()


async def test_chasing_losses_gets_responsible_gaming_answer(h: Harness):
    await h.send(set_chat(), msg("пополнить, чтобы отыграться?", language="ru"))
    assert "InvokeOperator" not in h.consultant.types()
    assert h.llm.compose_calls[-1].task is Task.RESPONSIBLE_GAMING


async def test_explicit_addiction_or_self_exclusion_goes_to_l1(h: Harness):
    await h.send(set_chat(), msg("I want self-exclusion, I am addicted"))
    assert h.consultant.types()[-1] == "InvokeOperator"


# --------------------------------------------------- п.2 no answer about the time

async def test_noise_while_waiting_for_time_escalates_only_after_three(h: Harness):
    await h.send(set_chat(), msg("deposit not arrived"), msg("الو"), msg("?"))
    assert "InvokeOperator" not in h.consultant.types()
    last = h.llm.compose_calls[-1]
    assert last.task is Task.ASK_WAIT_TIME and any("ONE short sentence" in f for f in last.facts)
    assert buttons(h.consultant.for_chat("chat1")[-1]) == ["wait:lt2", "wait:2to24", "wait:gt24"]
    await h.send(msg("??"))
    assert h.consultant.types()[-1] == "InvokeOperator"


# ------------------------------------------------------ п.3/5 24 hours, no clock

async def test_within_norm_says_24h_without_a_clock_time(h: Harness):
    at(h, TUE)
    await h.send(set_chat(), msg("deposit not arrived"), msg("40 minutes"))
    facts = " ".join(h.llm.compose_calls[-1].facts)
    assert "24 hours" in facts and "15:" not in facts and "16:" not in facts


def test_hours_since_weekday_and_dates():
    # "on Saturday" said on Tuesday 24.11 = 21.11, the whole day
    lo, hi = hours_since({"weekday": "saturday"}, TUE)
    assert (lo, hi) == (63.0, 87.0)
    # "22/11" without an hour: the whole day, not from midnight
    assert hours_since({"date": "2026-11-22"}, TUE) == (39.0, 63.0)
    # "yesterday evening" at 20:00 -> 20..26 h: straddles 24h -> the engine asks for the hour
    lo, hi = hours_since({"days_ago": 1, "part_of_day": "evening"}, TUE.replace(hour=20))
    assert lo < 24 < hi
    # a date in the future is last year
    assert hours_since({"date": "2026-12-01"}, TUE)[0] > 24 * 300


async def test_yesterday_evening_is_clarified_by_hour(h: Harness):
    at(h, TUE.replace(hour=20))
    await h.send(set_chat(), msg("deposit not arrived"), msg("yesterday evening"))
    last = h.llm.compose_calls[-1]
    assert last.task is Task.CLARIFY_WAIT_TIME and any("hour" in f for f in last.facts)


async def test_no_geo_asks_how_many_hours(h: Harness):
    await h.send(set_chat(), msg("deposit not arrived", country=None), msg("yesterday", country=None))
    last = h.llm.compose_calls[-1]
    assert last.task is Task.CLARIFY_WAIT_TIME and any("time zone is unknown" in f for f in last.facts)


# ------------------------------------------------------------ п.4 no operator

async def test_operator_buttons_from_the_model_are_dropped(h: Harness):
    real = h.llm.compose

    async def compose(req):
        c = await real(req)
        return Composed(c.text, c.result, (Option("free:0", "Talk to operator"), Option("free:1", "Bonus rules")))
    h.llm.compose = compose  # type: ignore[method-assign]
    await h.send(set_chat(), msg("bonus problem"))
    assert labels(h.consultant.for_chat("chat1")[-1]) == ["Bonus rules"]


def test_handoff_promises_are_detected():
    assert promises_handoff("Хорошо, сейчас соединю вас с оператором")
    assert promises_handoff("OK, I'll connect you to a specialist")
    assert promises_handoff("سأحولك إلى الدعم")
    assert not promises_handoff("I can help you right now. What happened?")


async def test_promise_of_operator_is_rewritten(h: Harness):
    texts = iter(["Sure, I'll connect you to an operator now.", "I can help right away. What happened?"])
    real = h.llm.compose

    async def compose(req):
        c = await real(req)
        return Composed(next(texts, c.text), c.result, c.options) if req.task is Task.DEFLECT_OPERATOR else c
    h.llm.compose = compose  # type: ignore[method-assign]
    await h.send(set_chat(), msg("operator"))
    assert "connect you" not in h.consultant.last_text("chat1")


# ------------------------------------------------------------ п.6 GEO methods

async def test_method_buttons_follow_the_geo(h: Harness):
    await h.send(set_chat(), msg("deposit not arrived", country="NG"), msg("3 days already", country="NG"))
    assert labels(h.consultant.for_chat("chat1")[-1]) == ["OPay", "PalmPay", "Bank Transfer", "Crypto", "Other method"]
    facts = " ".join(h.llm.compose_calls[-1].facts)
    assert "OPay" in facts and "has NOT named a payment method" in facts


async def test_unknown_geo_gets_no_method_buttons(h: Harness):
    await h.send(set_chat(), msg("deposit not arrived", country="TR"), msg("3 days already", country="TR"))
    assert buttons(h.consultant.for_chat("chat1")[-1]) == []
    assert any("not known" in f for f in h.llm.compose_calls[-1].facts)


async def test_words_not_matching_a_method_are_clarified_with_buttons(h: Harness):
    await h.send(set_chat(), msg("deposit not arrived"), msg("3 days already"), msg("visa"))
    assert "Vodafone Cash" in labels(h.consultant.for_chat("chat1")[-1])
    assert (await h.dialogue()).payment.details["method_user_words"].lower() == "visa"


# --------------------------------------------------- п.8/9/10 data to collect

async def test_withdrawal_asks_for_request_screenshot_not_receipt(h: Harness):
    await h.send(set_chat(), msg("withdrawal not received for more than a day"), msg("Vodafone Cash"),
                 msg("500, yesterday at 10:00"), msg("01001982785"))
    req = h.llm.compose_calls[-1]
    assert req.missing_detail == "receipt" and req.payment_kind == "withdrawal"
    from support_bot.core.prompts import detail_hint
    assert "AvaBet history" in detail_hint(req.payment_kind, req.missing_detail)
    assert "wallet" in detail_hint("deposit", "receipt")


async def test_receipt_is_asked_twice_then_handed_over_without_it(h: Harness):
    await h.send(set_chat(), msg("deposit not arrived"), msg("3 days already"), msg("Vodafone Cash"),
                 msg("1000, yesterday at 8:40"), msg("01001982785"), msg("I don't have it"))
    assert "InvokeOperator" not in h.consultant.types()
    assert any("NOT been received" in f for f in h.llm.compose_calls[-1].facts)
    await h.send(msg("no"))
    assert h.consultant.types()[-1] == "InvokeOperator"
    handoff = h.llm.compose_calls[-1]
    assert handoff.task is Task.HANDOFF and any("NOT received" in f for f in handoff.facts)
    note = [p for p in h.consultant.sent if p["$type"] == "SupportBotMessage"][-1]["text"]
    assert "receipt / screenshot: not received" in note


async def test_at_most_four_questions_and_known_data_not_asked_again(h: Harness):
    await h.send(set_chat(), msg("deposit not arrived"), msg("3 days already"),
                 msg("Vodafone Cash for 1000 egp"))  # method and amount at once
    asked = [r.missing_detail for r in h.llm.compose_calls if r.task is Task.COLLECT_PAYMENT_DETAILS]
    assert asked == ["method", "paid_at"]


# ------------------------------------------------------- п.13 handoff text

async def test_handoff_message_has_no_trailing_question(h: Harness):
    real = h.llm.compose

    async def compose(req):
        c = await real(req)
        if req.task is Task.HANDOFF:
            return Composed("I'm passing this to a colleague, they will reply here. Anything else?", c.result)
        return c
    h.llm.compose = compose  # type: ignore[method-assign]
    await h.send(set_chat(), msg("I want self-exclusion"))
    assert h.consultant.types()[-1] == "InvokeOperator"
    text = [p for p in h.consultant.sent if p["$type"] == "BotReplyMessage"][-1]["text"]
    assert not text.rstrip().endswith("?")


# ------------------------------------------------- п.15/16/25 text finishing

def _finish(text: str, task=Task.ANSWER, language="en") -> str:
    h = Harness()

    class D:  # the fields _finish_text reads
        pass
    d = D()
    d.language = language
    return Engine._finish_text(h.engine, text, task, d)  # type: ignore[arg-type]


def test_markdown_is_stripped():
    assert _finish("Open **Profile** and __Cashier__") == "Open Profile and Cashier"


def test_arabic_users_get_the_arabic_support_address():
    assert "support-ar@ava.bet" in _finish("اكتب إلى support-en@ava.bet", language="ar")
    assert "support-en@ava.bet" in _finish("Write to support-en@ava.bet")


def test_question_goes_to_its_own_paragraph():
    assert _finish("Deposits take up to 24 hours. Which method did you use?") == \
        "Deposits take up to 24 hours.\n\nWhich method did you use?"


async def test_support_contact_fact_depends_on_language(h: Harness):
    await h.send(set_chat(), msg("مرحبا كيف أسحب", language="ar"))
    facts = " ".join(h.llm.compose_calls[-1].facts)
    assert "support-ar@ava.bet" in facts


# ---------------------------------------------- п.18/20/21/22 buttons & cycle

async def test_greeting_gets_greeting_and_top_three(h: Harness):
    await h.send(set_chat(), msg("الو", language="ar"))
    assert h.llm.compose_calls[-1].task is Task.GREET
    assert buttons(h.consultant.for_chat("chat1")[-1]) == ["top:deposit_missing", "top:withdrawal_missing", "top:login"]


async def test_failed_payment_is_clarified_first(h: Harness):
    await h.send(set_chat(), msg("my deposit failed"))
    assert h.llm.compose_calls[-1].task is Task.CLARIFY_PAYMENT_KIND
    assert buttons(h.consultant.for_chat("chat1")[-1]) == ["pay:failed", "pay:charged"]
    await h.send(msg("Charged, not on balance", variant="pay:charged"))
    assert h.llm.compose_calls[-1].task is Task.ASK_WAIT_TIME


async def test_bare_account_number_is_recognized(h: Harness):
    await h.send(set_chat(), msg("12345678"))
    assert h.llm.compose_calls[-1].task is Task.ACCOUNT_NUMBER
    assert (await h.dialogue()).account_number == "12345678"


async def test_yes_does_not_start_a_new_cycle(h: Harness):
    await h.send(set_chat(), msg("bonus problem"), msg("didn't help"), msg("yes"))
    d = await h.dialogue()
    assert d.issue.key == "bonuses" and d.issue.failures >= 1


async def test_button_ladder_categories_then_top_six_then_model(h: Harness):
    await h.send(set_chat(), msg("operator"))
    first = buttons(h.consultant.for_chat("chat1")[-1])
    await h.send(msg("operator!"))
    second = buttons(h.consultant.for_chat("chat1")[-1])
    assert len(first) == 12 and all(b.startswith("cat:") for b in first)
    assert second == ["top:deposit_missing", "top:how_deposit", "top:payment_status", "top:withdrawal_missing",
                      "top:login", "top:bonus"]


async def test_top_request_button_starts_the_right_flow(h: Harness):
    await h.send(set_chat(), msg("Deposit not received", variant="top:deposit_missing"))
    assert h.llm.compose_calls[-1].task is Task.ASK_WAIT_TIME


async def test_day_from_the_previous_answer_is_kept(h: Harness):
    at(h, TUE.replace(hour=20))
    await h.send(set_chat(), msg("deposit not arrived"), msg("yesterday evening"), msg("at 9 pm"))
    d = await h.dialogue()
    assert d.payment.within_norm and d.payment.wait_hours == 23.0


async def test_mixed_language_model_buttons_are_dropped(h: Harness):
    real = h.llm.compose

    async def compose(req):
        c = await real(req)
        return Composed(c.text, c.result, (Option("free:0", "مش عارف أف open السجل"), Option("free:1", "فتحت السجل")))
    h.llm.compose = compose  # type: ignore[method-assign]
    await h.send(set_chat(), msg("bonus مشكلة", language="ar"))
    assert labels(h.consultant.for_chat("chat1")[-1]) == ["فتحت السجل"]


async def test_which_methods_question_gets_the_geo_list(h: Harness):
    await h.send(set_chat(), msg("what deposit methods are there?", country="NG"))
    facts = " ".join(h.llm.compose_calls[-1].facts)
    assert "OPay, PalmPay, Bank Transfer, Crypto" in facts and "Vodafone" not in facts
