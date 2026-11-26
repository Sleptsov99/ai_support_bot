"""Claude adapter for the LLM protocol (analyze / compose / localize) and a Claude translator.

* Structured output (output_config.format = json_schema): the engine always
  gets valid JSON, no parsing of free text.
* Prompt caching: system = [static prefix (rules + whole KB + reference
  dialogues), cached] + [dynamic per-turn instructions]. The prefix is
  identical for every chat, so after the first call it is read from the
  cache (~10% of the input price).
* Separate models for analyze (cheap/fast is fine) and compose (style
  matters): CLAUDE_ANALYZE_MODEL / CLAUDE_COMPOSE_MODEL.
* Low effort: support replies do not need long reasoning; keeps latency down.
* Refusals (stop_reason == "refusal") and API errors never reach the user as
  is: analyze falls back to a neutral analysis, compose raises LLMUnavailable
  and the engine uses its fallback text.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any

from ..monitoring import record_llm
from .actions import Option
from .language import SUPPORTED_LANGUAGES
from .llm import Analysis, AnalyzeRequest, Composed, ComposeRequest
from .prompts import analyze_instructions, compose_instructions, history_messages, static_prefix

log = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-opus-5-5"


class LLMUnavailable(RuntimeError):
    pass


PAYMENT_FIELDS = ("method", "amount", "paid_at", "wallet", "receipt")
WHEN_FIELDS = {"date": "string", "days_ago": "number", "weekday": "string", "time": "string", "part_of_day": "string"}


def _nullable(t: str) -> dict:
    return {"type": [t, "null"]}


ANALYSIS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "language": _nullable("string"),
        "language_confident": {"type": "boolean"},
        "category": _nullable("string"),
        "signal": {"type": "string", "enum": ["new_request", "progress", "not_helped", "not_understood", "repeat",
                                              "dissatisfied", "noise", "closing"]},
        "understood": {"type": "boolean"},
        "wants_operator": {"type": "boolean"},
        "payment_issue": {"type": ["string", "null"], "enum": ["deposit", "withdrawal", None]},
        "payment_unclear": {"type": "boolean"},
        "payment_when": {"anyOf": [{
            "type": "object",
            "properties": {k: _nullable(t) for k, t in WHEN_FIELDS.items()},
            "required": list(WHEN_FIELDS),
            "additionalProperties": False,
        }, {"type": "null"}]},
        "wait_hours": _nullable("number"),
        "wait_hours_max": _nullable("number"),
        "confirmation": {"type": ["string", "null"], "enum": ["yes", "no", None]},
        "gratitude": {"type": "boolean"},
        "aggressive": {"type": "boolean"},
        "complaint": {"type": "boolean"},
        "off_topic": {"type": "boolean"},
        "noise": {"type": "boolean"},
        "risk": {"type": ["string", "null"], "enum": ["gambling_harm", "self_harm", "minor", None]},
        "chasing_losses": {"type": "boolean"},
        "greeting": {"type": "boolean"},
        "account_number": _nullable("string"),
        "media_understood": {"type": "boolean"},
        "payment_details": {
            "type": "object",
            "properties": {k: _nullable("string") for k in PAYMENT_FIELDS},
            "required": list(PAYMENT_FIELDS),
            "additionalProperties": False,
        },
    },
    "additionalProperties": False,
}
ANALYSIS_SCHEMA["required"] = list(ANALYSIS_SCHEMA["properties"])

COMPOSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "text": {"type": "string"},
        "result": {"type": "string", "enum": ["Full", "Partial", "NotFound"]},
        "options": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["text", "result", "options"],
    "additionalProperties": False,
}

LABELS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"items": {"type": "array", "items": {"type": "string"}}},
    "required": ["items"],
    "additionalProperties": False,
}

TO_PIVOT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"text": {"type": "string"}, "language": _nullable("string")},
    "required": ["text", "language"],
    "additionalProperties": False,
}


@dataclass
class ClaudeConfig:
    analyze_model: str = DEFAULT_MODEL
    compose_model: str = DEFAULT_MODEL
    effort: str | None = "low"  # None = do not send (models without the effort setting)
    max_tokens: int = 4000
    cache_ttl: str | None = None  # None = 5 min default; "1h" for low-traffic hours
    fallbacks: bool = False  # server-side refusal fallback (beta)


class _Claude:
    def __init__(self, client: Any, cfg: ClaudeConfig, prefix: str):
        self.client = client
        self.cfg = cfg
        self.prefix = prefix

    def _system(self, dynamic: str) -> list[dict]:
        cache: dict[str, Any] = {"type": "ephemeral"}
        if self.cfg.cache_ttl:
            cache["ttl"] = self.cfg.cache_ttl
        return [{"type": "text", "text": self.prefix, "cache_control": cache}, {"type": "text", "text": dynamic}]

    async def _json(self, model: str, system: list[dict] | str, messages: list[dict], schema: dict) -> dict:
        output_config: dict[str, Any] = {"format": {"type": "json_schema", "schema": schema}}
        if self.cfg.effort:
            output_config["effort"] = self.cfg.effort
        kwargs: dict[str, Any] = dict(model=model, max_tokens=self.cfg.max_tokens, system=system, messages=messages,
                                      output_config=output_config)
        started = time.monotonic()
        try:
            if self.cfg.fallbacks:
                resp = await self.client.beta.messages.create(
                    **kwargs, betas=["server-side-fallback-2026-07-01"], fallbacks="default")
            else:
                resp = await self.client.messages.create(**kwargs)
        except Exception as exc:  # network, rate limit, overload: the engine has fallbacks
            record_llm(model, (time.monotonic() - started) * 1000, error=f"{type(exc).__name__}: {exc}")
            raise LLMUnavailable(str(exc)) from exc
        usage = getattr(resp, "usage", None)
        cached = getattr(usage, "cache_read_input_tokens", None) or 0
        record_llm(model, (time.monotonic() - started) * 1000,
                   tokens_in=((getattr(usage, "input_tokens", None) or 0) + cached
                              + (getattr(usage, "cache_creation_input_tokens", None) or 0)) if usage else None,
                   tokens_cached=cached, tokens_out=getattr(usage, "output_tokens", None) if usage else None,
                   error=resp.stop_reason if resp.stop_reason in ("refusal", "max_tokens") else None)
        if usage is not None:
            log.debug("claude %s: in=%s cache_read=%s cache_write=%s out=%s", model,
                      getattr(usage, "input_tokens", None), getattr(usage, "cache_read_input_tokens", None),
                      getattr(usage, "cache_creation_input_tokens", None), getattr(usage, "output_tokens", None))
        if resp.stop_reason == "refusal":
            raise LLMUnavailable("refusal")
        if resp.stop_reason == "max_tokens":
            raise LLMUnavailable("max_tokens")
        text = next((b.text for b in resp.content if getattr(b, "type", None) == "text"), None)
        if text is None:
            raise LLMUnavailable("no text block")
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise LLMUnavailable(f"bad json: {exc}") from exc


_NULLISH = {"", "null", "none", "n/a", "unknown"}


def _to_when(value: Any) -> dict | None:
    if not isinstance(value, dict):
        return None
    when: dict[str, Any] = {}
    for k, t in WHEN_FIELDS.items():
        v = value.get(k)
        if v is None or str(v).strip().lower() in _NULLISH:
            continue
        if t == "number":
            try:
                v = int(float(str(v).replace(",", ".")))
            except ValueError:
                continue
        else:
            v = str(v).strip().lower() if k in ("weekday", "part_of_day") else str(v).strip()
        when[k] = v
    return when or None


def _to_analysis(data: dict) -> Analysis:
    """Validate the model's JSON (providers without strict schemas may send "Deposit", "null", "2", ...)."""
    props = ANALYSIS_SCHEMA["properties"]
    raw_details = data.get("payment_details") or {}
    details = {k: str(v).strip() for k, v in raw_details.items()
               if k in PAYMENT_FIELDS and v is not None and str(v).strip().lower() not in _NULLISH}
    out: dict[str, Any] = {}
    for key, spec in props.items():
        if key == "payment_details" or key not in data:
            continue
        value = data[key]
        if key == "payment_when":
            out[key] = _to_when(value)
            continue
        types = spec["type"] if isinstance(spec["type"], list) else [spec["type"]]
        if isinstance(value, str) and value.strip().lower() in _NULLISH and "null" in types:
            value = None
        elif "enum" in spec and value is not None:
            value = str(value).strip().lower()
            if value not in spec["enum"]:
                value = None
        elif "boolean" in types and not isinstance(value, bool) and value is not None:
            value = str(value).strip().lower() in ("true", "1", "yes")
        elif "number" in types and value is not None and not isinstance(value, (int, float)):
            try:
                value = float(str(value).replace(",", "."))
            except ValueError:
                value = None
        elif key in ("language", "category") and isinstance(value, str):
            value = value.strip().lower() or None
        out[key] = value
    if "signal" in out and out["signal"] is None:
        out.pop("signal")
    return Analysis(**out, payment_details=details)


class ClaudeLLM(_Claude):
    def __init__(self, client: Any, kb_document: str, cfg: ClaudeConfig | None = None,
                 references: str | None = None):
        super().__init__(client, cfg or ClaudeConfig(), static_prefix(kb_document, references))

    async def analyze(self, req: AnalyzeRequest) -> Analysis:
        note = f"\n[{len(req.media)} file(s) attached: " + ", ".join(
            m.mime_type or "file" for m in req.media) + "]" if req.media else ""
        msgs = history_messages(req.history, req.text, note)
        try:
            data = await self._json(self.cfg.analyze_model, self._system(analyze_instructions(req)), msgs,
                                    ANALYSIS_SCHEMA)
        except LLMUnavailable as exc:
            log.warning("analyze failed (%s): neutral analysis", exc)
            return Analysis(language=None, language_confident=False, signal=None, understood=bool(req.text))
        return _to_analysis(data)

    async def compose(self, req: ComposeRequest) -> Composed:
        history = req.history
        current = req.user_text
        if history and history[-1].role == "user":
            history, current = history[:-1], history[-1].text
        msgs = history_messages(history, current)
        data = await self._json(self.cfg.compose_model, self._system(compose_instructions(req)), msgs, COMPOSE_SCHEMA)
        labels = [o.strip()[:60] for o in data.get("options", []) if o and o.strip()][:4]  # > 20 chars -> rows
        options = tuple(Option(f"free:{i}", lbl) for i, lbl in enumerate(labels))
        return Composed(data["text"].strip(), data.get("result", "Full"), options)

    async def localize(self, labels: list[str], language: str) -> list[str]:
        if not labels:
            return []
        lang = SUPPORTED_LANGUAGES.get(language, language)
        prompt = (f"Translate these support chat UI labels to {lang}. Keep them short (<= 20 characters where "
                  f"possible), keep the order, return exactly {len(labels)} items.\n" + json.dumps(labels, ensure_ascii=False))
        try:
            data = await self._json(self.cfg.analyze_model, self._system("Task: translate UI labels."),
                                    [{"role": "user", "content": prompt}], LABELS_SCHEMA)
        except LLMUnavailable:
            return list(labels)
        items = data.get("items", [])
        return items if len(items) == len(labels) else list(labels)


class ClaudeTranslator(_Claude):
    """Translator for the pivot mode (see translation.py): any language <-> the pivot language."""

    SYSTEM = ("You translate customer support chat messages of an iGaming company (AvaBet). Keep the meaning, tone, "
              "emoji, numbers, amounts, times, names of payment methods, site sections, e-mails and links exactly. "
              "Keep markdown (**bold**) and line breaks. Do not answer the message, only translate it.")

    def __init__(self, client: Any, model: str = DEFAULT_MODEL, cfg: ClaudeConfig | None = None):
        super().__init__(client, cfg or ClaudeConfig(analyze_model=model, compose_model=model), self.SYSTEM)

    def _sys(self):
        return self.SYSTEM

    async def to_pivot(self, text: str, pivot: str, hint_language: str | None = None) -> tuple[str, str | None]:
        lang = SUPPORTED_LANGUAGES.get(pivot, pivot)
        prompt = (f"Translate the user's message to {lang}. If it is already in {lang}, return it unchanged. Also "
                  f"return the ISO 639-1 code of the source language, or null if the message is too short or "
                  f"neutral to tell (\"ok\", \"sir\", a number). Arabic written in Latin letters is 'ar'. "
                  f"Probable language: {hint_language or 'unknown'}. Return JSON {{\"text\": ..., \"language\": ...}}."
                  f"\n\nMessage:\n{text}")
        try:
            data = await self._json(self.cfg.analyze_model, self._sys(), [{"role": "user", "content": prompt}],
                                    TO_PIVOT_SCHEMA)
        except LLMUnavailable:
            return text, None
        return data.get("text") or text, data.get("language")

    async def from_pivot(self, texts: list[str], pivot: str, target: str) -> list[str]:
        src = SUPPORTED_LANGUAGES.get(pivot, pivot)
        lang = SUPPORTED_LANGUAGES.get(target, target)
        prompt = (f"Translate each item from {src} to {lang}. Natural chat style of a friendly support operator, same "
                  f"tone and emoji. Button labels must stay short (<= 20 characters). Return JSON "
                  f"{{\"items\": [...]}} with exactly {len(texts)} items in the same order.\n"
                  + json.dumps(texts, ensure_ascii=False))
        try:
            data = await self._json(self.cfg.compose_model, self._sys(), [{"role": "user", "content": prompt}],
                                    LABELS_SCHEMA)
        except LLMUnavailable:
            return list(texts)
        items = data.get("items", [])
        return items if len(items) == len(texts) else list(texts)
