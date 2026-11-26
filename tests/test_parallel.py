"""Several testers at once: one slow chat must not hold the others."""
import asyncio
import time

import httpx

from support_bot.demo.server import create_demo_app
from support_bot.dispatcher import Dispatcher


async def test_drain_waits_only_for_its_chat():
    d = Dispatcher(engine=None, client=None, debounce_seconds=0)
    slow = d._spawn(asyncio.sleep(1.0), "slow")
    d._spawn(asyncio.sleep(0.01), "fast")
    t = time.monotonic()
    await d.drain(chat="fast")
    assert time.monotonic() - t < 0.5
    assert not slow.done()
    await d.drain()
    assert slow.done() and not d._task_chat


async def test_parallel_demo_chats_are_isolated():
    app = create_demo_app(seed=1)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
        a, b = await asyncio.gather(c.post("/demo/api/new", json={}), c.post("/demo/api/new", json={}))
        ids = a.json()["chat_id"], b.json()["chat_id"]
        assert ids[0] != ids[1]
        texts = ["не могу войти", "депозит не пришел"]
        res = await asyncio.gather(*(c.post("/demo/api/message", json={"chat_id": i, "text": t, "language": "ru"})
                                     for i, t in zip(ids, texts)))
        for chat_id, r in zip(ids, res):
            reactions = r.json()["reactions"]
            assert reactions and all(x.get("chatId") in (chat_id, None) for x in reactions)
        assert res[0].json()["state"]["topic"] != res[1].json()["state"]["topic"]
