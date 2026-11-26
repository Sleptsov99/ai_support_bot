"""Logs and monitoring: one JSON record per bot turn, LLM usage, errors, aggregated stats.

* Every turn of every chat -> one JSON line in LOG_DIR/turns-YYYY-MM-DD.jsonl and in stdout
  (``docker compose logs``): chat, language, topic, signal, what the bot did, escalation reason,
  latency, LLM calls / tokens / errors, fallbacks, fact-check failures, texts (long digit runs masked).
* Errors (exceptions in a chat, failed LLM calls) -> LOG_DIR/errors-YYYY-MM-DD.jsonl.
* In memory: the last MONITOR_KEEP turns for the /admin/stats page (1 h / 24 h / since start).

LLM usage is attributed to the turn through a context variable: the dispatcher opens a turn,
the adapters report every HTTP call, the engine reports fallbacks and fact-check failures.

Env: LOG_DIR (no files if unset; /app/logs in Docker), LOG_KEEP_DAYS (30), LOG_TEXTS (1 = keep user/bot texts),
MONITOR_KEEP (20000 turns in memory),
PRICE_INPUT_PER_M / PRICE_CACHED_PER_M / PRICE_OUTPUT_PER_M: USD per 1M tokens for the cost estimate
(defaults: DeepSeek deepseek-chat list prices, check them).
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from collections import Counter, deque
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger("support_bot.turns")

_DIGITS = re.compile(r"\d[\d \-]{6,}\d")


def mask(text: str | None, limit: int = 500) -> str | None:
    """Long digit runs (card / phone / account numbers) -> last 4 digits only."""
    if text is None:
        return None
    text = _DIGITS.sub(lambda m: "***" + re.sub(r"\D", "", m.group())[-4:], text)
    return text if len(text) <= limit else text[:limit] + "…"


@dataclass
class TurnStats:
    """Filled while one turn is computed (LLM calls of this turn, engine events)."""
    llm_calls: int = 0
    llm_errors: list[str] = field(default_factory=list)
    llm_ms: float = 0.0
    tokens_in: int = 0
    tokens_cached: int = 0
    tokens_out: int = 0
    fallbacks: int = 0
    fact_check_failed: int = 0


_TURN: ContextVar[TurnStats | None] = ContextVar("support_bot_turn", default=None)


def begin_turn() -> TurnStats:
    stats = TurnStats()
    _TURN.set(stats)
    return stats


def record_llm(model: str, ms: float, *, tokens_in: int | None = None, tokens_cached: int | None = None,
               tokens_out: int | None = None, error: str | None = None) -> None:
    """Called by the LLM adapters for every HTTP call (also the failed ones)."""
    t = _TURN.get()
    if t is not None:
        t.llm_calls += 1
        t.llm_ms += ms
        t.tokens_in += tokens_in or 0
        t.tokens_cached += tokens_cached or 0
        t.tokens_out += tokens_out or 0
        if error:
            t.llm_errors.append(error[:200])
    MONITOR.llm_call(model, ms, error)


def note(event: str) -> None:
    """Engine events: 'fallback' (a template answer instead of the model), 'fact_check_failed'."""
    t = _TURN.get()
    if t is None:
        return
    if event == "fallback":
        t.fallbacks += 1
    elif event == "fact_check_failed":
        t.fact_check_failed += 1


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    return values[min(len(values) - 1, int(round(q * (len(values) - 1))))]


class Monitor:
    def __init__(self, log_dir: str | Path | None = None, keep: int = 20000, keep_days: int = 30,
                 log_texts: bool = True, prices: tuple[float, float, float] = (0.28, 0.028, 0.42)):
        self.log_dir = Path(log_dir) if log_dir else None
        self.keep_days = keep_days
        self.log_texts = log_texts
        self.prices = prices  # USD per 1M tokens: input (cache miss), cached input, output
        self.started = time.time()
        self.turns: deque[dict] = deque(maxlen=keep)
        self.errors: deque[dict] = deque(maxlen=500)
        self.llm_total = 0
        self.llm_failed = 0
        self.llm_streak = 0  # consecutive failed LLM calls (model down?)
        self._cleaned_for: str | None = None

    @classmethod
    def from_env(cls) -> "Monitor":
        return cls(
            log_dir=os.environ.get("LOG_DIR") or None,
            keep=int(_env_float("MONITOR_KEEP", 20000)),
            keep_days=int(_env_float("LOG_KEEP_DAYS", 30)),
            log_texts=os.environ.get("LOG_TEXTS", "1").strip().lower() in ("1", "true", "yes", "on"),
            prices=(_env_float("PRICE_INPUT_PER_M", 0.28), _env_float("PRICE_CACHED_PER_M", 0.028),
                    _env_float("PRICE_OUTPUT_PER_M", 0.42)),
        )

    # ------------------------------------------------------------- recording

    def cost(self, tokens_in: int, cached: int, out: int) -> float:
        p_in, p_cached, p_out = self.prices
        return ((tokens_in - cached) * p_in + cached * p_cached + out * p_out) / 1_000_000

    def turn(self, record: dict[str, Any]) -> None:
        record = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), **record}
        if not self.log_texts:
            record.pop("user_text", None)
            record.pop("bot_text", None)
        record["cost_usd"] = round(self.cost(record.get("tokens_in", 0), record.get("tokens_cached", 0),
                                             record.get("tokens_out", 0)), 6)
        record["_t"] = time.time()
        self.turns.append(record)
        line = {k: v for k, v in record.items() if k != "_t"}
        log.info("turn %s", json.dumps(line, ensure_ascii=False))
        self._write("turns", line)

    def error(self, where: str, message: str, chat_id: str | None = None) -> None:
        record = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "where": where,
                  "chat_id": chat_id, "error": message[:1000], "_t": time.time()}
        self.errors.append(record)
        self._write("errors", {k: v for k, v in record.items() if k != "_t"})

    def llm_call(self, model: str, ms: float, error: str | None) -> None:
        self.llm_total += 1
        if error:
            self.llm_failed += 1
            self.llm_streak += 1
            self.error("llm", f"{model}: {error}")
        else:
            self.llm_streak = 0

    def _write(self, kind: str, record: dict) -> None:
        if self.log_dir is None:
            return
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            with open(self.log_dir / f"{kind}-{day}.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            if self._cleaned_for != day:
                self._cleaned_for = day
                self._cleanup()
        except OSError as exc:  # a full disk must not stop the bot
            logging.getLogger(__name__).warning("cannot write %s log: %s", kind, exc)

    def _cleanup(self) -> None:
        border = (datetime.now(timezone.utc) - timedelta(days=self.keep_days)).strftime("%Y-%m-%d")
        for path in self.log_dir.glob("*.jsonl"):
            day = path.stem.rsplit("-", 3)[-3:]
            if "-".join(day) < border:
                path.unlink(missing_ok=True)

    # ----------------------------------------------------------------- stats

    def stats(self, window_seconds: float | None = None) -> dict[str, Any]:
        since = time.time() - window_seconds if window_seconds else 0.0
        turns = [t for t in self.turns if t["_t"] >= since]
        errors = [e for e in self.errors if e["_t"] >= since]
        n = len(turns)
        latencies = [t["latency_ms"] for t in turns if t.get("latency_ms") is not None]
        handoffs = [t for t in turns if t.get("handoff")]
        tokens_in = sum(t.get("tokens_in", 0) for t in turns)
        cached = sum(t.get("tokens_cached", 0) for t in turns)
        out = sum(t.get("tokens_out", 0) for t in turns)
        chats = {t.get("chat_id") for t in turns}
        handed_chats = {t.get("chat_id") for t in handoffs}
        return {
            "turns": n,
            "chats": len(chats),
            "handoffs": len(handoffs),
            "handoff_rate": round(len(handed_chats) / len(chats), 3) if chats else None,
            "handoff_reasons": Counter(t.get("escalation_reason") or "?" for t in handoffs).most_common(),
            "resolved": sum(1 for t in turns if t.get("resolved")),
            "results": dict(Counter(t["result"] for t in turns if t.get("result"))),
            "latency_ms": {"avg": round(sum(latencies) / len(latencies)) if latencies else None,
                           "p50": _percentile(latencies, 0.5), "p95": _percentile(latencies, 0.95),
                           "max": max(latencies) if latencies else None},
            "llm_calls": sum(t.get("llm_calls", 0) for t in turns),
            "llm_errors": sum(len(t.get("llm_errors") or ()) for t in turns),
            "fallbacks": sum(t.get("fallbacks", 0) for t in turns),
            "fact_check_failed": sum(t.get("fact_check_failed", 0) for t in turns),
            "superseded": sum(1 for t in turns if t.get("superseded")),
            "errors": len(errors),
            "tokens": {"in": tokens_in, "cached": cached, "out": out,
                       "cache_hit": round(cached / tokens_in, 3) if tokens_in else None},
            "cost_usd": round(sum(t.get("cost_usd", 0.0) for t in turns), 4),
            "cost_per_chat_usd": round(sum(t.get("cost_usd", 0.0) for t in turns) / len(chats), 5) if chats else None,
            "languages": Counter(t.get("language") or "?" for t in turns).most_common(),
            "topics": Counter(t.get("topic") or "—" for t in turns).most_common(12),
            "signals": Counter(t.get("signal") or "—" for t in turns).most_common(),
        }

    def snapshot(self, recent: int = 50) -> dict[str, Any]:
        strip = lambda r: {k: v for k, v in r.items() if k != "_t"}  # noqa: E731
        return {
            "started": datetime.fromtimestamp(self.started, timezone.utc).isoformat(timespec="seconds"),
            "uptime_seconds": round(time.time() - self.started),
            "llm": {"calls": self.llm_total, "failed": self.llm_failed, "failing_now": self.llm_streak >= 3},
            "hour": self.stats(3600),
            "day": self.stats(86400),
            "all": self.stats(None),
            "recent_turns": [strip(t) for t in list(self.turns)[-recent:]][::-1],
            "recent_errors": [strip(e) for e in list(self.errors)[-20:]][::-1],
        }


MONITOR = Monitor.from_env()
