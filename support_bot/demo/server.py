"""Local demo: chat widget in the browser on top of the real engine.

    python -m support_bot.demo            # http://127.0.0.1:8000/demo

The page plays the role of the Consultant chat: it sends the same webhook
events (SetChat / ClientMessage / PingActivity / UnsetChat) to the same
Dispatcher and Engine, and shows every reaction the bot sends back, including
those hidden from the client (SupportBotMessage, InvokeOperator, ...).
No network, no API keys: DemoLLM + demo knowledge base.
"""
from __future__ import annotations

import itertools
import json
import os
import random
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel

from ..app import build_app, build_engine, llm_provider
from ..consultant.client import ChatForbidden
from ..consultant.models import InvokeOperator, Reaction, parse_event
from ..core.catalog import BY_KEY, EscalationGroup
from ..core.engine import Engine, EngineConfig
from ..core.kb import InMemoryKB, KBEntry
from ..core.personas import PersonaRegistry
from ..core.state import InMemoryStore
from ..dispatcher import Dispatcher
from .demo_llm import DemoLLM

HERE = Path(__file__).resolve().parent


class RecordingConsultant:
    """Stands in for Consultant: stores reactions per chat, 403 after InvokeOperator like the real one."""

    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []
        self.closed: set[str] = set()

    async def send(self, reaction: Reaction) -> None:
        payload = reaction.to_payload()
        chat = payload.get("chatId")
        if chat in self.closed:
            raise ChatForbidden(403, "bot was removed from the chat")
        self.sent.append(payload)
        if isinstance(reaction, InvokeOperator):
            self.closed.add(chat)

    async def get_projects(self, language: str | None = None) -> list[dict[str, Any]]:
        return []

    async def get_download_link(self, media_id: str, image_size: str | None = None) -> dict[str, Any]:
        return {}

    async def get_chat_messages(self, chat_id: str, skip: int = 0, limit: int = 100) -> dict[str, Any]:
        return {"items": [], "total": 0}


def load_demo_kb() -> tuple[InMemoryKB, dict[str, str]]:
    raw = json.loads((HERE / "demo_kb.json").read_text(encoding="utf-8"))
    entries, ru = [], {}
    for e in raw:
        ru[e["id"]] = e.pop("answer_ru")
        entries.append(KBEntry(**{**e, "keywords": tuple(e.get("keywords", ()))}))
    return InMemoryKB(entries), ru


TIMEZONES = {"EG": "Africa/Cairo", "NG": "Africa/Lagos", "KE": "Africa/Nairobi", "KG": "Asia/Bishkek",
             "AR": "America/Argentina/Buenos_Aires", "CI": "Africa/Abidjan", "JO": "Asia/Amman",
             "TZ": "Africa/Dar_es_Salaam", "ET": "Africa/Addis_Ababa", "ZA": "Africa/Johannesburg"}


class NewChat(BaseModel):
    geo: str | None = None
    language: str | None = None
    channel: str | None = None


class ChatMessage(NewChat):
    chat_id: str
    text: str | None = None
    variant_id: str | None = None
    with_file: bool = False


class ChatRef(BaseModel):
    chat_id: str


def create_demo_app(seed: int | None = None) -> FastAPI:
    kb, ru_answers = load_demo_kb()
    store = InMemoryStore()
    personas = PersonaRegistry.from_file(rng=random.Random(seed))
    if llm_provider() != "fake":  # the same engine on the real model and the real KB from the Q&A sheet
        real_kb = InMemoryKB.from_path(Path(os.environ.get("KB_PATH", HERE.parent / "data" / "kb.md")))
        engine = build_engine(real_kb, personas=personas)
        store = engine.store
    else:
        engine = Engine(DemoLLM(ru_answers), kb, personas, store, EngineConfig())
    consultant = RecordingConsultant()
    dispatcher = Dispatcher(engine, consultant, debounce_seconds=0)
    app = build_app(dispatcher)  # /webhook works too, against the recording Consultant
    msg_ids = itertools.count(1)

    async def deliver(event: dict, chat_id: str | None) -> dict:
        before = len(consultant.sent)
        dispatcher.submit(parse_event(event))
        await dispatcher.drain()
        new = consultant.sent[before:]
        if chat_id:
            new = [r for r in new if r.get("chatId") in (chat_id, None)]
        return {"event": event, "reactions": new, "state": await state_of(chat_id)}

    async def state_of(chat_id: str | None) -> dict | None:
        d = await store.get(chat_id) if chat_id else None
        if d is None:
            return None
        cat = BY_KEY.get(d.issue.category or "")
        group = cat.group.value if cat else EscalationGroup.A.value
        p = d.payment
        if p:
            group = ("B (< 24ч, без эскалации)" if p.within_norm else
                     "B (> 24ч, сбор данных для L1)" if p.over_threshold else "B")
        signal_ru = {"progress": "прогресс", "new_request": "новый запрос", "not_helped": "не помогло",
                     "not_understood": "не понял", "repeat": "повтор", "dissatisfied": "недоволен",
                     "closing": "завершение", "noise": "шум"}
        status = ("передан L1" if d.handed_off else "оператор в чате, бот молчит" if d.operator_joined
                  else "закрыт" if d.closed else "активен")
        return {
            "persona": f"{d.persona.name} ({d.persona.pool.geo_name})" if d.persona else None,
            "geo": d.country,
            "language": d.language,
            "device": d.channel,
            "topic": d.issue.category,
            "group": group,
            "signal": signal_ru.get(d.last_signal or "", d.last_signal),
            "attempts": f"{d.issue.failures} из {engine.cfg.max_attempts}" if d.current else "0",
            "issues": ", ".join(f"{k}: {i.failures}" for k, i in d.issues.items()) or None,
            "payment_wait_hours": p.wait_hours if p else None,
            "payment_details": ", ".join(f"{k}={v}" for k, v in p.details.items()) if p and p.details else None,
            "awaiting": d.awaiting,
            "escalation_reason": d.escalation_reason,
            "status": status,
        }

    def client_message(m: ChatMessage) -> dict:
        ev: dict[str, Any] = {
            "$type": "ClientMessage",
            "chatId": m.chat_id,
            "conversationId": f"conv-{m.chat_id}",
            "projectId": 1,
            "messageId": next(msg_ids),
            "client": {"id": f"client-{m.chat_id}"},
        }
        if m.channel:
            ev["channel"] = m.channel
        if m.language:
            ev["languageCode"] = m.language
        if m.geo:
            ev["geo"] = {"country": m.geo, "timezone": TIMEZONES.get(m.geo.upper())}
        if m.text:
            ev["text"] = m.text
        if m.variant_id:
            ev["command"] = {"$type": "SelectVariantCommand", "variantId": m.variant_id}
        if m.with_file:
            ev["medias"] = [{"$type": "MediaDocument", "id": uuid.uuid4().hex, "mimeType": "image/jpeg", "size": 48213,
                             "attributes": [{"$type": "DocumentAttributeFileName", "name": "screenshot.jpg"}]}]
        return ev

    @app.get("/", include_in_schema=False)
    async def root() -> RedirectResponse:
        return RedirectResponse("/demo")

    @app.get("/demo", response_class=HTMLResponse, include_in_schema=False)
    async def demo_page() -> str:
        return (HERE / "demo.html").read_text(encoding="utf-8")

    @app.post("/demo/api/new")
    async def new_chat(_: NewChat) -> dict:
        chat_id = uuid.uuid4().hex[:12]
        res = await deliver({"$type": "SetChat", "chatId": chat_id, "conversationId": f"conv-{chat_id}"}, chat_id)
        return {"chat_id": chat_id, **res}

    @app.post("/demo/api/message")
    async def message(m: ChatMessage) -> dict:
        return await deliver(client_message(m), m.chat_id)

    @app.post("/demo/api/ping")
    async def ping() -> dict:
        return await deliver({"$type": "PingActivity"}, None)

    @app.post("/demo/api/close")
    async def close(c: ChatRef) -> dict:
        return await deliver({"$type": "UnsetChat", "chatId": c.chat_id, "conversationId": f"conv-{c.chat_id}",
                              "reasonType": "Timeout"}, c.chat_id)

    return app
