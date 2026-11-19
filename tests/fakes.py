"""Fake Consultant API + event builders for tests."""
from __future__ import annotations

import itertools
from typing import Any

from support_bot.consultant.client import ChatForbidden
from support_bot.consultant.models import InvokeOperator, Reaction


class FakeConsultant:
    """In-memory Consultant: records reactions, behaves like the real one after InvokeOperator (403)."""

    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []
        self.forbidden_chats: set[str] = set()
        self.rejected: list[dict[str, Any]] = []

    async def send(self, reaction: Reaction) -> None:
        payload = reaction.to_payload()
        chat = payload.get("chatId")
        if chat in self.forbidden_chats:
            self.rejected.append(payload)
            raise ChatForbidden(403, "chat is closed for the bot")
        self.sent.append(payload)
        if isinstance(reaction, InvokeOperator):
            self.forbidden_chats.add(chat)

    async def get_projects(self, language: str | None = None) -> list[dict[str, Any]]:
        return [{"id": 1, "tags": [{"id": 545, "name": "Deposit"}]}]

    async def get_download_link(self, media_id: str, image_size: str | None = None) -> dict[str, Any]:
        return {"uri": f"https://files.example/{media_id}", "expiryIn": 1700000000}

    async def get_chat_messages(self, chat_id: str, skip: int = 0, limit: int = 100) -> dict[str, Any]:
        return {"items": [], "total": 0}

    # ---------------------------------------------------------------- helpers

    def for_chat(self, chat_id: str) -> list[dict[str, Any]]:
        return [p for p in self.sent if p.get("chatId") == chat_id]

    def types(self, chat_id: str | None = None) -> list[str]:
        items = self.sent if chat_id is None else self.for_chat(chat_id)
        return [p["$type"] for p in items]

    def last_text(self, chat_id: str) -> str:
        texts = [p["text"] for p in self.for_chat(chat_id) if p.get("text") and p["$type"] in ("BotMessage", "BotReplyMessage")]
        return texts[-1]

    def clear(self) -> None:
        self.sent.clear()


_ids = itertools.count(1)


def set_chat(chat_id: str = "chat1", conversation_id: str = "conv1") -> dict:
    return {"$type": "SetChat", "chatId": chat_id, "conversationId": conversation_id}


def unset_chat(chat_id: str = "chat1", reason: str = "Resolved") -> dict:
    return {"$type": "UnsetChat", "chatId": chat_id, "conversationId": "conv1", "reasonType": reason}


def client_message(
    text: str | None = None,
    chat_id: str = "chat1",
    *,
    country: str | None = "EG",
    language: str | None = "en",
    channel: str = "App_Android",
    variant: str | None = None,
    media: list[dict] | None = None,
) -> dict:
    ev: dict[str, Any] = {
        "$type": "ClientMessage",
        "chatId": chat_id,
        "conversationId": "conv1",
        "projectId": 1,
        "channel": channel,
        "messageId": next(_ids),
        "client": {"id": "client-1"},
    }
    if language:
        ev["languageCode"] = language
    if text is not None:
        ev["text"] = text
    if country:
        ev["geo"] = {"country": country, "timezone": "Africa/Cairo"}
    if variant:
        ev["command"] = {"$type": "SelectVariantCommand", "variantId": variant}
    if media:
        ev["medias"] = media
    return ev


PING = {"$type": "PingActivity"}
