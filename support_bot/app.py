"""FastAPI web service: Consultant webhook.

Run:  uvicorn support_bot.app:create_app --factory --host 0.0.0.0 --port 8000

Environment:
  CONSULTANT_BASE_URL   default https://cons-suph.top
  CONSULTANT_BOT_TOKEN  bot token (required for real traffic)
  WEBHOOK_SECRET        optional; if set, webhook URL is /webhook/<secret>
                        (Consultant has no webhook authentication yet)
  DEBOUNCE_SECONDS      glue consecutive client messages, default 2.0
  FRAGMENT_DEBOUNCE_SECONDS  longer wait after a short fragment ("sir"), default 5.0
  FETCH_HISTORY         read the chat history from Consultant every turn, default 1
  KB_PATH               knowledge base: .md document (preferred) or .json, default data/kb.md (from the Q&A sheet)

  LLM_PROVIDER          fake | claude | deepseek (default: whichever API key is set, else fake)
  DEEPSEEK_API_KEY      DeepSeek key; DEEPSEEK_MODEL (deepseek-chat), DEEPSEEK_BASE_URL
  ANTHROPIC_API_KEY     Claude API key
  CLAUDE_ANALYZE_MODEL  model for the NLU step, default claude-opus-5-5
  CLAUDE_COMPOSE_MODEL  model that writes the replies, default claude-opus-5-5
  CLAUDE_EFFORT         low | medium | high | none, default low
  CLAUDE_CACHE_TTL      "" (5 min) or 1h
  CLAUDE_FALLBACKS      1 = server-side refusal fallback (beta), default 0
  PIVOT_LANGUAGES       languages answered through the translator, default "sw,am,zu,ky,af" (weak for models);
                        "*" = all, "" = none (the model reads and writes every language itself)
  PIVOT_TARGET          the language translated messages are handled in: en (default) or ru

  PAYMENT_SILENCE_SECONDS  silence after "how long ago?" -> L1, default 0 = off (test review: silence is not a reason)
  BOT_MARKDOWN             1 = keep **bold** in replies (only if Consultant renders it), default 0
  CONTACT_MEMORY_HOURS     keep a short note about the previous conversation, default 0 (off)
  RISK_ESCALATES           gambling harm / self-harm -> L1 at once, default 1

  LOG_DIR               write turns-*.jsonl / errors-*.jsonl there (off if unset); LOG_KEEP_DAYS 30
  LOG_TEXTS             keep user/bot texts in the turn log (long digit runs masked), default 1
  LOG_LEVEL             INFO; PRICE_INPUT_PER_M / PRICE_CACHED_PER_M / PRICE_OUTPUT_PER_M for the cost estimate
  Monitoring page: /admin/stats (JSON: /admin/stats.json)
"""
from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import HTMLResponse
from pydantic import ValidationError

from .consultant.client import HttpConsultantClient
from .consultant.models import parse_event
from .core.budget import InMemoryBudget
from .core.engine import Engine, EngineConfig
from .core.kb import InMemoryKB
from .core.llm import FakeLLM
from .core.personas import PersonaRegistry
from .core.state import ContactMemory, InMemoryStore
from .core.translation import LanguagePolicy
from .dispatcher import Dispatcher
from .monitoring import MONITOR

log = logging.getLogger(__name__)
DATA = Path(__file__).resolve().parent / "data"


def build_app(dispatcher: Dispatcher, webhook_secret: str | None = None) -> FastAPI:
    app = FastAPI(title="AI Support Bot")
    app.state.dispatcher = dispatcher

    async def _handle(request: Request) -> Response:
        try:
            payload = json.loads(await request.body())
            event = parse_event(payload)
        except (ValueError, ValidationError) as exc:
            # 4xx would make Consultant retry and then send the client to an operator,
            # so log loudly but still accept.
            log.error("bad webhook payload: %s", exc)
            return Response(status_code=200)
        dispatcher.submit(event)
        return Response(status_code=200)  # empty body; reactions go via POST /messages

    if webhook_secret:
        @app.post("/webhook/{secret}")
        async def webhook_secret_route(secret: str, request: Request) -> Response:
            if secret != webhook_secret:
                raise HTTPException(status_code=404)
            return await _handle(request)
    else:
        @app.post("/webhook")
        async def webhook(request: Request) -> Response:
            return await _handle(request)

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    # Monitoring (closed by the nginx password, like the demo).
    @app.get("/admin/stats", response_class=HTMLResponse, include_in_schema=False)
    async def admin_stats() -> str:
        return (Path(__file__).resolve().parent / "admin_stats.html").read_text(encoding="utf-8")

    @app.get("/admin/stats.json", include_in_schema=False)
    async def admin_stats_json(recent: int = 50) -> dict:
        return MONITOR.snapshot(recent=max(1, min(recent, 500)))

    return app


# Strong languages (en, ar, fr, es, ru, tr...) are read and written by the model itself; languages where
# models are weak go through a translator via English (machine translation is strongest paired with English).
DEFAULT_PIVOT_LANGUAGES = "sw,am,zu,ky,af"


def _env_bool(name: str, default: bool) -> bool:
    return os.environ.get(name, "1" if default else "0").strip().lower() in ("1", "true", "yes", "on")


def llm_provider() -> str:
    explicit = os.environ.get("LLM_PROVIDER")
    if explicit:
        return explicit
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "claude"
    if os.environ.get("DEEPSEEK_API_KEY"):
        return "deepseek"
    return "fake"


def build_llm(kb: InMemoryKB):
    """Returns (llm, translator or None)."""
    provider = llm_provider()
    if provider == "fake":
        log.warning("Using FakeLLM: keyword NLU and template answers, not for production")
        return FakeLLM(), None
    if provider == "deepseek":
        from .core.deepseek_llm import DEFAULT_BASE_URL, DEFAULT_MODEL as DS_MODEL, DeepSeekLLM, DeepSeekTranslator

        key = os.environ["DEEPSEEK_API_KEY"]
        base = os.environ.get("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL)
        model = os.environ.get("DEEPSEEK_MODEL", DS_MODEL)
        return (DeepSeekLLM(key, kb.document(), model=model, base_url=base),
                DeepSeekTranslator(key, model=model, base_url=base))
    if provider != "claude":
        raise ValueError(f"unknown LLM_PROVIDER {provider!r}")
    from anthropic import AsyncAnthropic

    from .core.claude_llm import DEFAULT_MODEL, ClaudeConfig, ClaudeLLM, ClaudeTranslator

    effort = os.environ.get("CLAUDE_EFFORT", "low")
    cfg = ClaudeConfig(
        analyze_model=os.environ.get("CLAUDE_ANALYZE_MODEL", DEFAULT_MODEL),
        compose_model=os.environ.get("CLAUDE_COMPOSE_MODEL", DEFAULT_MODEL),
        effort=None if effort in ("", "none") else effort,
        cache_ttl=os.environ.get("CLAUDE_CACHE_TTL") or None,
        fallbacks=_env_bool("CLAUDE_FALLBACKS", False),
    )
    client = AsyncAnthropic()
    return ClaudeLLM(client, kb.document(), cfg), ClaudeTranslator(client, cfg=cfg)


def build_engine(kb: InMemoryKB, llm=None, translator=None, personas: PersonaRegistry | None = None) -> Engine:
    if llm is None:
        llm, translator = build_llm(kb)
    policy = LanguagePolicy.from_env(os.environ.get("PIVOT_LANGUAGES", DEFAULT_PIVOT_LANGUAGES),
                                     os.environ.get("PIVOT_TARGET", "en"))
    if policy.modes and translator is None:
        log.info("no translator (fake LLM): every language is answered natively")
        policy = LanguagePolicy()
    memory_hours = float(os.environ.get("CONTACT_MEMORY_HOURS", "0"))
    config = EngineConfig(
        payment_silence_seconds=float(os.environ.get("PAYMENT_SILENCE_SECONDS", "0")),
        markdown=_env_bool("BOT_MARKDOWN", False),
        risk_escalates=_env_bool("RISK_ESCALATES", True),
    )
    return Engine(
        llm=llm,
        kb=kb,
        personas=personas or PersonaRegistry.from_file(),
        store=InMemoryStore(),
        config=config,
        budget=InMemoryBudget(),
        translator=translator,
        language_policy=policy,
        contact_memory=ContactMemory(memory_hours * 3600) if memory_hours > 0 else None,
    )


def setup_logging() -> None:
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO").upper(),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)


def create_app() -> FastAPI:
    setup_logging()
    kb = InMemoryKB.from_path(Path(os.environ.get("KB_PATH", DATA / "kb.md")))
    engine = build_engine(kb)
    client = HttpConsultantClient(
        os.environ.get("CONSULTANT_BASE_URL", "https://cons-suph.top"),
        os.environ.get("CONSULTANT_BOT_TOKEN", "missing-token"),
    )
    dispatcher = Dispatcher(
        engine, client,
        debounce_seconds=float(os.environ.get("DEBOUNCE_SECONDS", "2.0")),
        fragment_debounce_seconds=float(os.environ.get("FRAGMENT_DEBOUNCE_SECONDS", "5.0")),
        fetch_history=_env_bool("FETCH_HISTORY", True),
    )
    app = build_app(dispatcher, os.environ.get("WEBHOOK_SECRET") or None)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        await dispatcher.drain()
        await client.aclose()

    app.router.lifespan_context = lifespan
    return app
