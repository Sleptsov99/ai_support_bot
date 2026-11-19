"""Unit tests: models, personas, language, wait time parsing, prompts, HTTP client, webhook."""
from __future__ import annotations

import json
import random

import httpx
import pytest
from pydantic import ValidationError

from support_bot.app import build_app
from support_bot.consultant.client import ChatForbidden, ConsultantError, HttpConsultantClient
from support_bot.consultant.models import (
    BotMessage,
    ButtonVariant,
    ClientMessage,
    PingActivity,
    UnknownEvent,
    UnsetChat,
    parse_event,
)
from support_bot.core.catalog import CATEGORIES, resolve_variant
from support_bot.core.language import choose_language, normalize
from support_bot.core.llm import ComposeRequest, Task, parse_wait_hours
from support_bot.core.personas import PersonaRegistry
from support_bot.core.prompts import compose_instructions, static_prefix

from .conftest import Harness
from .fakes import PING, client_message, set_chat

# -------------------------------------------------------------------- models


def test_parse_client_message_ignores_unknown_fields_and_accepts_medias():
    ev = parse_event({
        "$type": "ClientMessage", "chatId": "c", "conversationId": "v", "messageId": 4, "text": "Hello",
        "languageCode": "en", "client": {"id": "u", "country": "ng"}, "brandNewField": {"x": 1},
        "medias": [{"$type": "MediaDocument", "id": "m", "mimeType": "image/png",
                    "attributes": [{"$type": "DocumentAttributeFileName", "name": "a.png"}]}],
        "command": {"$type": "SelectVariantCommand", "variantId": "cat:deposit"},
    })
    assert isinstance(ev, ClientMessage)
    assert ev.country == "NG"  # from client.country when geo is absent
    assert ev.all_media[0].file_name == "a.png"
    assert ev.command.variant_id == "cat:deposit"


def test_geo_country_has_priority_over_client_country():
    ev = parse_event(client_message("x", country="EG") | {"client": {"id": "u", "country": "KE"}})
    assert ev.country == "EG"


def test_parse_other_events():
    assert isinstance(parse_event(PING), PingActivity)
    assert isinstance(parse_event({"$type": "UnsetChat", "chatId": "c", "reasonType": "SomethingNew"}), UnsetChat)
    assert isinstance(parse_event({"$type": "AcknowledgeSomething", "foo": 1}), UnknownEvent)


def test_reaction_payload_shape():
    payload = BotMessage(chat_id="c", text="Hi", reply_markup={"buttons": [{"id": "b", "text": "Deposit"}]}).to_payload()
    assert payload == {
        "$type": "BotMessage", "chatId": "c", "text": "Hi",
        "replyMarkup": {"buttons": [{"$type": "ButtonVariant", "id": "b", "text": "Deposit"}], "rows": []},
    }


def test_button_text_limit_is_enforced():
    with pytest.raises(ValidationError):
        ButtonVariant(id="x", text="This label is way too long")


# ------------------------------------------------------------------ catalog


def test_catalog_has_12_categories_with_5_questions():
    assert len(CATEGORIES) == 12
    assert all(len(c.questions) == 5 for c in CATEGORIES)
    assert all(len(c.label) <= 20 for c in CATEGORIES)
    assert resolve_variant("q:sport:1")[1] == "How to place a bet"
    assert resolve_variant("q:sport:99") is None
    assert resolve_variant("garbage") is None


# ----------------------------------------------------------------- personas


def test_all_geo_pools_have_10_personas():
    reg = PersonaRegistry.from_file()
    assert len(reg.pools) == 18
    assert all(len(p.names) == 10 for p in reg.pools.values())
    assert len(reg.base.names) == 10


def test_persona_assignment():
    reg = PersonaRegistry.from_file(rng=random.Random(1))
    assert reg.assign("tr").pool.geo == "TR"
    assert reg.assign(None).pool is reg.base
    assert reg.assign("XX").pool is reg.base


# ----------------------------------------------------------------- language


@pytest.mark.parametrize("code,expected", [("en", "en"), ("en-US", "en"), ("ru_RU", "ru"), ("", None), (None, None)])
def test_normalize(code, expected):
    assert normalize(code) == expected


def test_choose_language():
    assert choose_language("ru", "en") == "ru"
    assert choose_language(None, "es-AR") == "es"
    assert choose_language(None, None) == "en"


# --------------------------------------------------------------- wait time


@pytest.mark.parametrize("text,hours", [
    ("40 minutes", 40 / 60), ("40 мин", 40 / 60), ("прошло 3 часа", 3), ("2 days", 48),
    ("больше суток", 25), ("третий день жду", 48), ("час назад", 1), ("hello", None),
])
def test_parse_wait_hours(text, hours):
    assert parse_wait_hours(text) == pytest.approx(hours) if hours is not None else parse_wait_hours(text) is None


# ------------------------------------------------------------------ prompts


def test_prompt_layout_static_prefix_and_dynamic_part():
    persona = PersonaRegistry.from_file(rng=random.Random(0)).assign("EG")
    prefix = static_prefix("## Minimum deposit {#min-dep}\nanswer text")
    assert "Never say you checked" in prefix and "min-dep" in prefix
    assert "Диалог 4" in prefix  # reference dialogues are part of the cached prefix
    dyn = compose_instructions(ComposeRequest(
        task=Task.ANSWER, persona=persona, language="ar", user_text="?", introduce=True, attempt=4))
    assert persona.name in dyn and "Arabic" in dyn
    assert "Attempt 4" in dyn
    assert "Алкогольные аналогии" in dyn  # GEO-specific restrictions
    assert "min-dep" not in dyn  # the KB is only in the cached part


# -------------------------------------------------------------- HTTP client


async def test_http_client_posts_reaction_to_messages_endpoint():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200)

    client = HttpConsultantClient("https://cons.test", "TOKEN", http=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    await client.send(BotMessage(chat_id="c", text="Hi"))
    assert str(seen[0].url) == "https://cons.test/bot/TOKEN/messages"
    assert json.loads(seen[0].content) == {"$type": "BotMessage", "chatId": "c", "text": "Hi"}


async def test_http_client_errors_and_retries():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if request.url.path.endswith("/messages"):
            return httpx.Response(403)
        return httpx.Response(500) if calls["n"] < 3 else httpx.Response(200, json=[{"id": 1, "tags": []}])

    client = HttpConsultantClient("https://cons.test", "T", http=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    with pytest.raises(ChatForbidden):
        await client.send(BotMessage(chat_id="c", text="Hi"))
    assert await client.get_projects() == [{"id": 1, "tags": []}]  # 2 x 500 then 200

    bad = HttpConsultantClient("https://cons.test", "T", http=httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(401))))
    with pytest.raises(ConsultantError) as exc:
        await bad.send(BotMessage(chat_id="c", text="Hi"))
    assert exc.value.status == 401


# ------------------------------------------------------------------ webhook


async def _post(app, path: str, body) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://bot") as c:
        content = body if isinstance(body, str) else json.dumps(body)
        return await c.post(path, content=content, headers={"Content-Type": "application/json"})


async def test_webhook_returns_200_empty_and_reacts_async():
    h = Harness()
    app = build_app(h.dispatcher)
    r = await _post(app, "/webhook", set_chat())
    assert r.status_code == 200 and r.content == b""
    r = await _post(app, "/webhook", PING)
    assert r.status_code == 200 and r.content == b""
    await h.dispatcher.drain()
    assert h.consultant.types() == ["BotMessage", "PongActivity"]


async def test_webhook_accepts_garbage_without_failing():
    h = Harness()
    app = build_app(h.dispatcher)
    assert (await _post(app, "/webhook", "not json")).status_code == 200
    assert (await _post(app, "/webhook", {"$type": "ClientMessage"})).status_code == 200  # missing chatId
    await h.dispatcher.drain()
    assert h.consultant.sent == []


async def test_webhook_secret_path():
    h = Harness()
    app = build_app(h.dispatcher, webhook_secret="s3cret")
    assert (await _post(app, "/webhook/wrong", PING)).status_code == 404
    assert (await _post(app, "/webhook/s3cret", PING)).status_code == 200
    await h.dispatcher.drain()
    assert h.consultant.types() == ["PongActivity"]
