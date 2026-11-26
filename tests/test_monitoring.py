"""Turn log, LLM usage accounting and the /admin/stats page."""
import json

import httpx

from support_bot import monitoring
from support_bot.demo.server import create_demo_app
from support_bot.monitoring import Monitor, begin_turn, mask, note, record_llm


def test_mask_hides_long_numbers_but_keeps_short_ones():
    assert mask("карта 4276 1234 5678 9012, сумма 5000") == "карта ***9012, сумма 5000"
    assert mask("тел +254712345678") == "тел +***5678"
    assert mask("x" * 600).endswith("…")


def test_turn_stats_collect_llm_usage_and_events(monkeypatch):
    mon = Monitor()
    monkeypatch.setattr(monitoring, "MONITOR", mon)
    stats = begin_turn()
    record_llm("deepseek-chat", 800, tokens_in=40000, tokens_cached=38000, tokens_out=200)
    record_llm("deepseek-chat", 50, error="HTTP 503")
    note("fallback")
    note("fact_check_failed")
    assert (stats.llm_calls, stats.tokens_in, stats.tokens_cached, stats.tokens_out) == (2, 40000, 38000, 200)
    assert stats.llm_errors == ["HTTP 503"] and stats.fallbacks == 1 and stats.fact_check_failed == 1
    assert mon.llm_failed == 1 and mon.errors[-1]["where"] == "llm"


def test_monitor_writes_jsonl_and_aggregates(tmp_path):
    mon = Monitor(log_dir=tmp_path, log_texts=False)
    (tmp_path / "turns-2000-01-01.jsonl").write_text("{}\n")  # older than LOG_KEEP_DAYS -> removed
    mon.turn({"chat_id": "a", "handoff": True, "escalation_reason": "5 неудач", "latency_ms": 3000,
              "tokens_in": 1_000_000, "tokens_cached": 900_000, "tokens_out": 1000, "user_text": "секрет"})
    mon.turn({"chat_id": "b", "result": "Full", "latency_ms": 1000})
    files = sorted(p.name for p in tmp_path.iterdir())
    assert len(files) == 1 and files[0].startswith("turns-")
    lines = [json.loads(x) for x in (tmp_path / files[0]).read_text().splitlines()]
    assert len(lines) == 2 and "user_text" not in lines[0]
    s = mon.stats()
    assert s["chats"] == 2 and s["handoffs"] == 1 and s["handoff_rate"] == 0.5
    assert s["handoff_reasons"] == [("5 неудач", 1)] and s["results"] == {"Full": 1}
    assert s["tokens"]["cache_hit"] == 0.9 and s["cost_usd"] > 0


async def test_demo_turns_show_up_on_the_stats_page(monkeypatch):
    mon = Monitor()
    monkeypatch.setattr(monitoring, "MONITOR", mon)
    monkeypatch.setattr("support_bot.dispatcher.MONITOR", mon)
    monkeypatch.setattr("support_bot.app.MONITOR", mon)
    app = create_demo_app(seed=1)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
        chat = (await c.post("/demo/api/new", json={})).json()["chat_id"]
        await c.post("/demo/api/message", json={"chat_id": chat, "text": "не могу войти, карта 4276123456789012",
                                               "language": "ru"})
        page = await c.get("/admin/stats")
        assert page.status_code == 200 and "Мониторинг" in page.text
        snap = (await c.get("/admin/stats.json")).json()
    kinds = [t["kind"] for t in snap["recent_turns"]]
    assert kinds == ["message", "start"]
    last = snap["recent_turns"][0]
    assert last["chat_id"] == chat and last["topic"] and last["bot_text"] and last["latency_ms"] >= 0
    assert "4276123456789012" not in last["user_text"] and "***9012" in last["user_text"]
    assert snap["day"]["turns"] == 2 and snap["day"]["chats"] == 1
