"""DeepSeek adapter (OpenAI-compatible chat API) for checking the engine on another model.

Same prompts, same JSON contracts and same engine as the Claude adapter; only
the transport differs:

* POST {base_url}/chat/completions, ``response_format = {"type": "json_object"}``
  (DeepSeek's JSON mode: the expected JSON shape is described in the prompt);
* system prompt = static prefix (rules + whole KB + reference dialogues) +
  dynamic part, in this order: DeepSeek caches identical prompt prefixes
  automatically, so the KB is billed at the cache price after the first call.

Env: DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL (default https://api.deepseek.com),
DEEPSEEK_MODEL (default deepseek-chat).
"""
from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from .claude_llm import ClaudeConfig, ClaudeLLM, ClaudeTranslator, LLMUnavailable

log = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"


def _shape(schema: dict) -> str:
    """Compact description of the expected JSON for JSON mode."""
    return json.dumps(schema, ensure_ascii=False)


def _example(schema: dict) -> Any:
    t = schema.get("type")
    t = t[0] if isinstance(t, list) else t
    if t == "object":
        return {k: _example(v) for k, v in schema.get("properties", {}).items()}
    if t == "array":
        return [_example(schema.get("items", {"type": "string"}))]
    if "enum" in schema:
        return schema["enum"][0]
    return {"string": "...", "number": 0, "boolean": False}.get(t, None)


def _parse(content: str) -> dict | None:
    text = content.strip()
    if not text:
        return None
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            value = json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return None
    return value if isinstance(value, dict) else None


class _DeepSeekTransport:
    http: httpx.AsyncClient
    base_url: str
    api_key: str
    prefix: str
    cfg: ClaudeConfig

    def _setup(self, api_key: str, base_url: str, http: httpx.AsyncClient | None, timeout: float) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.http = http or httpx.AsyncClient(timeout=timeout)

    def _system(self, dynamic: str) -> str:
        return self.prefix + "\n\n" + dynamic

    async def _json(self, model: str, system: Any, messages: list[dict], schema: dict) -> dict:
        system_text = system if isinstance(system, str) else "\n\n".join(b["text"] for b in system)
        system_text += ("\n\n# Response format\nReply with ONE json object only, no other text. It must match this "
                        "JSON schema (all listed keys present, null where unknown):\n" + _shape(schema)
                        + "\nExample of the shape:\n" + json.dumps(_example(schema), ensure_ascii=False))
        last_error = ""
        # DeepSeek's JSON mode sometimes returns only whitespace: retry cooler, then without JSON mode.
        for attempt, (temperature, json_mode) in enumerate(((0.7, True), (0.3, True), (0.3, False))):
            body: dict[str, Any] = {
                "model": model,
                "messages": [{"role": "system", "content": system_text}, *messages],
                "max_tokens": min(self.cfg.max_tokens, 8000),
                "temperature": temperature,
            }
            if json_mode:
                body["response_format"] = {"type": "json_object"}
            else:
                body["messages"].append({"role": "user", "content": "Return the json object now, nothing else."})
            try:
                resp = await self.http.post(f"{self.base_url}/chat/completions", json=body,
                                            headers={"Authorization": f"Bearer {self.api_key}"})
            except httpx.HTTPError as exc:
                raise LLMUnavailable(f"network: {exc}") from exc
            if resp.status_code != 200:
                raise LLMUnavailable(f"HTTP {resp.status_code}: {resp.text[:300]}")
            data = resp.json()
            usage = data.get("usage") or {}
            log.debug("deepseek %s: in=%s cache_hit=%s out=%s", model, usage.get("prompt_tokens"),
                      usage.get("prompt_cache_hit_tokens"), usage.get("completion_tokens"))
            choice = (data.get("choices") or [{}])[0]
            content = (choice.get("message") or {}).get("content") or ""
            if choice.get("finish_reason") == "length":
                raise LLMUnavailable("max_tokens")
            parsed = _parse(content)
            if parsed is not None:
                return parsed
            last_error = f"bad json on attempt {attempt + 1}: {content[:80]!r}"
            log.info("deepseek: %s", last_error)
        raise LLMUnavailable(last_error or "empty answer")

    async def aclose(self) -> None:
        await self.http.aclose()


class DeepSeekLLM(_DeepSeekTransport, ClaudeLLM):
    def __init__(self, api_key: str, kb_document: str, *, model: str = DEFAULT_MODEL,
                 base_url: str = DEFAULT_BASE_URL, http: httpx.AsyncClient | None = None, timeout: float = 60.0,
                 references: str | None = None):
        ClaudeLLM.__init__(self, None, kb_document,
                           ClaudeConfig(analyze_model=model, compose_model=model, effort=None), references)
        self._setup(api_key, base_url, http, timeout)


class DeepSeekTranslator(_DeepSeekTransport, ClaudeTranslator):
    def __init__(self, api_key: str, *, model: str = DEFAULT_MODEL, base_url: str = DEFAULT_BASE_URL,
                 http: httpx.AsyncClient | None = None, timeout: float = 60.0):
        ClaudeTranslator.__init__(self, None, cfg=ClaudeConfig(analyze_model=model, compose_model=model, effort=None))
        self._setup(api_key, base_url, http, timeout)

    def _sys(self) -> str:
        return self.SYSTEM
