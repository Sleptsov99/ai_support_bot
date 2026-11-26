"""Claude adapter (mocked SDK), fact check, KB document, translation pivot, escalation budget."""
from __future__ import annotations

import json
import random
from types import SimpleNamespace

from support_bot.core.actions import Option
from support_bot.core.budget import InMemoryBudget
from support_bot.core.claude_llm import ClaudeConfig, ClaudeLLM, ClaudeTranslator
from support_bot.core.engine import Engine, EngineConfig
from support_bot.core.guard import check_reply
from support_bot.core.kb import InMemoryKB, KBEntry, parse_markdown, render_markdown
from support_bot.core.llm import AnalyzeRequest, Composed, ComposeRequest, FakeLLM, Task, Turn
from support_bot.core.personas import PersonaRegistry
from support_bot.core.state import InMemoryStore
from support_bot.core.translation import LanguagePolicy
from support_bot.dispatcher import Dispatcher

from .conftest import TEST_KB, Harness
from .fakes import FakeConsultant, set_chat
from .fakes import client_message as msg


class FakeAnthropic:
    """Records requests, returns queued JSON answers like the SDK does."""

    def __init__(self, *answers, stop_reason="end_turn"):
        self.answers = list(answers)
        self.calls: list[dict] = []
        self.stop_reason = stop_reason
        self.messages = SimpleNamespace(create=self._create)
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    async def _create(self, **kw):
        self.calls.append(kw)
        text = json.dumps(self.answers.pop(0)) if self.answers else "{}"
        return SimpleNamespace(stop_reason=self.stop_reason, usage=None,
                               content=[SimpleNamespace(type="thinking"), SimpleNamespace(type="text", text=text)])


ANALYSIS = {"language": "ar", "language_confident": True, "category": "deposit", "signal": "new_request",
            "understood": True, "wants_operator": False, "payment_issue": "deposit", "wait_hours": 29,
            "wait_hours_max": 33, "confirmation": None, "gratitude": False, "aggressive": False, "complaint": False,
            "off_topic": False, "noise": False, "risk": None, "media_understood": True,
            "payment_details": {"method": None, "amount": "1000", "paid_at": None, "sender": None, "recipient": None}}


async def test_claude_analyze_uses_cached_kb_prefix_and_json_schema():
    client = FakeAnthropic(ANALYSIS)
    llm = ClaudeLLM(client, "## Deposit time {#dep-time}\nup to 24 hours", ClaudeConfig(analyze_model="m-fast"))
    a = await llm.analyze(AnalyzeRequest(text="عملت ايداع امبارح الصبح ب 1000", history=(Turn("bot", "hi"),),
                                         now_local="2026-11-18 14:00 (Africa/Cairo)"))
    assert a.payment_issue == "deposit" and a.wait_hours == 29 and a.payment_details == {"amount": "1000"}
    call = client.calls[0]
    assert call["model"] == "m-fast"
    assert call["system"][0]["cache_control"] == {"type": "ephemeral"} and "dep-time" in call["system"][0]["text"]
    assert "Africa/Cairo" in call["system"][1]["text"]
    assert call["output_config"]["format"]["type"] == "json_schema"
    assert call["messages"][0]["role"] == "user"  # history starts with the user


async def test_claude_compose_free_buttons_and_refusal_fallback():
    persona = PersonaRegistry.from_file(rng=random.Random(1)).assign("EG")
    req = ComposeRequest(task=Task.ANSWER, persona=persona, language="en", user_text="how to bet",
                         history=(Turn("user", "how to bet"),))
    llm = ClaudeLLM(FakeAnthropic({"text": "Open Sport 👍", "result": "Partial", "options": ["Done", "Can't find it"]}),
                    "kb")
    c = await llm.compose(req)
    assert c.text == "Open Sport 👍" and [o.id for o in c.options] == ["free:0", "free:1"]

    refusing = ClaudeLLM(FakeAnthropic({}, stop_reason="refusal"), "kb")
    a = await refusing.analyze(AnalyzeRequest(text="hello"))
    assert a.signal is None and not a.language_confident  # neutral, never crashes the turn


async def test_engine_survives_compose_failure_and_free_button_click():
    class Broken(FakeLLM):
        async def compose(self, req):
            raise RuntimeError("overloaded")

    h = Harness()
    h.engine.llm = Broken()
    await h.send(set_chat(), msg("forgot password"))
    assert "Forgot password?" in h.consultant.last_text("chat1")  # the KB answer as is

    class Buttons(FakeLLM):
        async def compose(self, req):
            return Composed("Did it work?", "Partial", (Option("free:0", "It worked"),))

    h2 = Harness()
    h2.engine.llm = Buttons()
    await h2.send(set_chat(), msg("forgot password"), msg("It worked", variant="free:0"))
    assert h2.engine.llm.analyze_calls[-1].text == "It worked"


def test_fact_check_catches_invented_numbers_links_claims_and_forbidden_words():
    src = ["Deposits are credited within 24 hours. Minimum 50 EGP. https://avabet.com/help"]
    assert check_reply("Usually it takes up to 24 hours, minimum 50 EGP 👍", sources=src).ok
    assert check_reply("See https://avabet.com/help", sources=src).ok
    bad = check_reply("It takes 48 hours. I checked your payment, it is a freebie, see bit.ly.com",
                      sources=src)
    text = " ".join(bad.problems)
    assert "48" in text and "claims a check" in text and "freebie" in text and "link" in text


def test_kb_markdown_round_trip():
    md = render_markdown(TEST_KB)
    assert "{#dep-time}" in md
    back = parse_markdown(md)
    by_id = {e.id: e for e in back}
    assert set(by_id) == {e.id for e in TEST_KB}
    assert by_id["dep-time"].answer == TEST_KB[0].answer and by_id["dep-time"].category == "deposit"


class FakeTranslator:
    def __init__(self):
        self.to_calls, self.from_calls = [], []

    async def to_pivot(self, text, pivot, hint_language=None):
        self.to_calls.append((text, pivot))
        if text == "нууру унутуп калдым":
            return "забыл пароль", "ky"
        return text, "ru" if pivot == "ru" else None

    async def from_pivot(self, texts, pivot, target):
        self.from_calls.append((texts, pivot, target))
        return [f"<{target}> {t}" for t in texts]


async def test_pivot_language_goes_through_the_translator():
    tr = FakeTranslator()
    llm = FakeLLM()
    engine = Engine(llm, InMemoryKB(TEST_KB), PersonaRegistry.from_file(rng=random.Random(3)), InMemoryStore(),
                    EngineConfig(), translator=tr, language_policy=LanguagePolicy.from_env("*", "ru"))
    c = FakeConsultant()
    disp = Dispatcher(engine, c)
    from support_bot.consultant.models import parse_event
    for ev in (set_chat(), msg("нууру унутуп калдым", language="ky", country="KG")):
        disp.submit(parse_event(ev))
        await disp.drain()
    assert llm.analyze_calls[-1].text == "забыл пароль"  # the model reads Russian (the KB language)
    assert llm.compose_calls[-1].language == "ru"  # and writes Russian
    assert c.last_text("chat1").startswith("<ky> ")  # the user gets Kyrgyz
    assert tr.from_calls[-1][1:] == ("ru", "ky")

    for ev in (msg("спасибо большое, помогло", language="ky", country="KG"),):
        disp.submit(parse_event(ev))
        await disp.drain()
    assert tr.from_calls[-1][2] == "ky"  # Russian typed: no translation back
    assert not c.last_text("chat1").startswith("<")


def test_budget_needs_enough_data_and_compares_with_target():
    now = [0.0]
    b = InMemoryBudget(min_conversations=10, clock=lambda: now[0])
    for i in range(9):
        b.record("deposit", escalated=True)
    assert not b.over_target("deposit")  # too few conversations
    b.record("deposit", escalated=False)
    assert b.over_target("deposit")  # 90% > 5%
    assert not b.over_target("contacts")  # no target: never escalated anyway


async def test_over_budget_category_gets_extra_attempts():
    b = InMemoryBudget(min_conversations=1)
    b.record("bonuses", escalated=True)
    h = Harness()
    h.engine.budget = b
    await h.send(set_chat(), msg("bonus problem"))
    for _ in range(6):
        await h.send(msg("didn't help"))
    assert "InvokeOperator" not in h.consultant.types()  # 5 + 2 extra
    await h.send(msg("didn't help"), msg("didn't help"))
    assert h.consultant.types()[-1] == "InvokeOperator"


def test_qa_sheet_converter_groups_wordings_and_keeps_notes(tmp_path):
    import csv
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("qa_to_doc", Path(__file__).parents[1] / "scripts/qa_to_doc.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rows = [
        ["Кнопка", "Категория", "Вопрос", "Ответ бота", "Статус", "Условие", "Эскалация", "Ключевые слова по категории"],
        ["Account", "", "🧭 2. Доступ к аккаунту", "🧭 2. Доступ к аккаунту", "", "", "", "пароль, войти"],
        ["Account", "Общие", "Забыл пароль", "Нажмите **Забыли пароль**.", "", "", "Нет", ""],
        ["Account", "Общие", "Не помню пароль", "Нажмите **Забыли пароль**.", "", "", "Нет", ""],
        ["Account", "Общие", "Как восстановить пароль?", "", "", "", "", ""],
        ["Account", "Общие", "Подозреваю взлом", "Смените пароль.", "Вызов оператора", "Подозревает взлом", "", ""],
    ]
    src = tmp_path / "qa.csv"
    with src.open("w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerows(rows)
    entries, kws = mod.convert(mod.read_table(src, None))
    assert [e.question for e in entries] == ["Забыл пароль", "Подозреваю взлом"]
    assert entries[0].variants == ("Не помню пароль", "Как восстановить пароль?")
    assert "Вызов оператора" in entries[1].note and kws["account"] == "пароль, войти"
    doc = mod.build_document(entries, kws, {})
    back = InMemoryKB(parse_markdown(doc))
    assert back.search("не помню пароль")[0].id == "account-1"
    assert back.entries[1].note and "Вызов оператора" not in back.entries[1].answer


def test_real_kb_document_parses():
    from pathlib import Path

    kb = InMemoryKB.from_path(Path(__file__).parents[1] / "support_bot/data/kb.md")
    assert len(kb.entries) > 200
    assert sum(1 + len(e.variants) for e in kb.entries) >= 530
    assert kb.search("забыл пароль")[0].category == "account"


async def test_deepseek_adapter_json_mode_and_cached_prefix():
    import httpx

    from support_bot.core.deepseek_llm import DeepSeekLLM, DeepSeekTranslator

    seen: list[dict] = []
    answers = [ANALYSIS, {"text": "Нажмите **Забыли пароль** 🙂", "result": "Full", "options": ["Помогло"]},
               {"text": "забыл пароль", "language": "sw"}, {"items": ["Bonyeza **Umesahau nenosiri** 🙂"]}]

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        assert request.headers["Authorization"] == "Bearer k"
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(answers.pop(0))},
                                                      "finish_reason": "stop"}],
                                         "usage": {"prompt_tokens": 1200, "prompt_cache_hit_tokens": 1000,
                                                   "completion_tokens": 30}})

    from support_bot.monitoring import begin_turn

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    llm = DeepSeekLLM("k", "## Забыл пароль {#account-1}\nНажмите **Забыли пароль**.", http=http)
    turn = begin_turn()
    a = await llm.analyze(AnalyzeRequest(text="забыл пароль"))
    assert (turn.llm_calls, turn.tokens_in, turn.tokens_cached, turn.tokens_out) == (1, 1200, 1000, 30)
    assert a.payment_issue == "deposit"
    persona = PersonaRegistry.from_file(rng=random.Random(1)).assign("KE")
    c = await llm.compose(ComposeRequest(task=Task.ANSWER, persona=persona, language="ru", user_text="забыл пароль"))
    assert c.text.startswith("Нажмите") and c.options[0].label == "Помогло"
    sys0, sys1 = seen[0]["messages"][0]["content"], seen[1]["messages"][0]["content"]
    prefix = llm.prefix
    assert sys0.startswith(prefix) and sys1.startswith(prefix)  # identical prefix -> DeepSeek cache hit
    assert "response_format" not in seen[0] and "json" in sys0.lower()  # plain request first

    tr = DeepSeekTranslator("k", http=http)
    assert await tr.to_pivot("Nimesahau nenosiri", "ru") == ("забыл пароль", "sw")
    assert await tr.from_pivot(["Нажмите **Забыли пароль** 🙂"], "ru", "sw") == ["Bonyeza **Umesahau nenosiri** 🙂"]


def test_loose_json_from_non_strict_providers_is_validated():
    from support_bot.core.claude_llm import _to_analysis

    a = _to_analysis({**ANALYSIS, "payment_issue": "Deposit", "signal": "weird", "wait_hours": "25",
                      "risk": "null", "aggressive": "true", "category": "Deposit",
                      "payment_details": {"method": "Visa", "amount": "null", "receipt": ""}})
    assert a.payment_issue == "deposit" and a.signal is None and a.wait_hours == 25.0
    assert a.risk is None and a.aggressive is True and a.category == "deposit"
    assert a.payment_details == {"method": "Visa"}


async def test_deepseek_falls_back_to_json_mode_when_plain_answer_is_not_json():
    import httpx

    from support_bot.core.deepseek_llm import DeepSeekLLM
    from support_bot.monitoring import begin_turn

    modes: list[bool] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        modes.append("response_format" in body)
        content = json.dumps(ANALYSIS) if body.get("response_format") else "   "
        return httpx.Response(200, json={"choices": [{"message": {"content": content}, "finish_reason": "stop"}]})

    llm = DeepSeekLLM("k", "## Забыл пароль {#account-1}\nНажмите **Забыли пароль**.",
                      http=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    turn = begin_turn()
    assert (await llm.analyze(AnalyzeRequest(text="забыл пароль"))).payment_issue == "deposit"
    assert modes == [False, True] and turn.llm_calls == 2
    assert turn.llm_errors and "plain" in turn.llm_errors[0] and "finish=stop" in turn.llm_errors[0]
