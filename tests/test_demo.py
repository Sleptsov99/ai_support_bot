"""Smoke tests for the local demo server."""
from __future__ import annotations

import httpx

from support_bot.demo.server import create_demo_app


async def _client():
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=create_demo_app(seed=1)), base_url="http://demo")


async def test_demo_page_and_full_payment_escalation():
    async with await _client() as c:
        assert "AvaBet Support" in (await c.get("/demo")).text
        new = (await c.post("/demo/api/new", json={"geo": "EG", "language": "ru"})).json()
        chat = new["chat_id"]
        assert new["reactions"][0]["$type"] == "BotMessage"

        r = (await c.post("/demo/api/message", json={"chat_id": chat, "text": "депозит не пришел", "geo": "EG", "language": "ru"})).json()
        assert "сколько времени прошло" in r["reactions"][-1]["text"]
        assert r["state"]["group"] == "B"

        r = (await c.post("/demo/api/message", json={"chat_id": chat, "text": "уже 3 дня", "geo": "EG", "language": "ru"})).json()
        assert "каким способом" in r["reactions"][-1]["text"]  # dialogue 4: collect data before L1
        assert r["state"]["group"].startswith("B (> 24ч")
        for text in ["Vodafone Cash", "чека нет под рукой", "1000", "вчера в 8:40", "01001982785"]:
            r = (await c.post("/demo/api/message", json={"chat_id": chat, "text": text, "geo": "EG", "language": "ru"})).json()
            assert r["state"]["status"] == "активен"
        r = (await c.post("/demo/api/message", json={"chat_id": chat, "text": "01098765432", "geo": "EG", "language": "ru"})).json()
        assert [x["$type"] for x in r["reactions"]] == ["SupportBotMessage", "BotReplyMessage", "InvokeOperator"]
        assert "method: Vodafone Cash" in r["reactions"][0]["text"]
        assert r["state"]["status"] == "передан L1"


async def test_demo_russian_kb_answer_and_buttons():
    async with await _client() as c:
        chat = (await c.post("/demo/api/new", json={})).json()["chat_id"]
        r = (await c.post("/demo/api/message", json={"chat_id": chat, "text": "Sport", "variant_id": "cat:sport", "language": "ru"})).json()
        rows = r["reactions"][-1]["replyMarkup"]
        labels = [b["text"] for b in rows["buttons"] + rows["rows"]]
        assert "Как сделать ставку" in labels
        r = (await c.post("/demo/api/message", json={"chat_id": chat, "text": "Как сделать ставку", "variant_id": "q:sport:1", "language": "ru"})).json()
        assert "купон" in r["reactions"][-1]["text"]
        assert "Меня зовут" in r["reactions"][-1]["text"]
