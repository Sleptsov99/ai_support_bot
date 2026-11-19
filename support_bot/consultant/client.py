"""HTTP client for the Consultant bot API (POST /bot/{token}/messages etc.)."""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Protocol

import httpx

from .models import Reaction

log = logging.getLogger(__name__)


class ConsultantError(Exception):
    def __init__(self, status: int, body: str = ""):
        super().__init__(f"Consultant API error {status}: {body[:200]}")
        self.status = status
        self.body = body


class ChatForbidden(ConsultantError):
    """403: the bot may no longer write to this chat (e.g. after InvokeOperator)."""


class ConsultantClient(Protocol):
    async def send(self, reaction: Reaction) -> None: ...

    async def get_projects(self, language: str | None = None) -> list[dict[str, Any]]: ...

    async def get_download_link(self, media_id: str, image_size: str | None = None) -> dict[str, Any]: ...

    async def get_chat_messages(self, chat_id: str, skip: int = 0, limit: int = 100) -> dict[str, Any]: ...


class HttpConsultantClient:
    def __init__(
        self,
        base_url: str,
        token: str,
        *,
        timeout: float = 5.0,
        retries: int = 2,
        http: httpx.AsyncClient | None = None,
    ):
        self._base = f"{base_url.rstrip('/')}/bot/{token}"
        self._retries = retries
        self._http = http or httpx.AsyncClient(timeout=timeout)

    async def aclose(self) -> None:
        await self._http.aclose()

    async def send(self, reaction: Reaction) -> None:
        await self._request("POST", "/messages", json=reaction.to_payload())

    async def get_projects(self, language: str | None = None) -> list[dict[str, Any]]:
        headers = {"Accept-Language": language} if language else None
        return (await self._request("GET", "/projects", headers=headers)).json()

    async def get_download_link(self, media_id: str, image_size: str | None = None) -> dict[str, Any]:
        params = {"mediaId": media_id}
        if image_size:
            params["imageSize"] = image_size
        return (await self._request("GET", "/media/download-link", params=params)).json()

    async def get_chat_messages(self, chat_id: str, skip: int = 0, limit: int = 100) -> dict[str, Any]:
        params = {"skip": skip, "limit": limit}
        return (await self._request("GET", f"/chats/{chat_id}/messages", params=params)).json()

    async def _request(self, method: str, path: str, **kw: Any) -> httpx.Response:
        last_exc: Exception | None = None
        for attempt in range(self._retries + 1):
            try:
                resp = await self._http.request(method, self._base + path, **kw)
            except httpx.TransportError as exc:  # network problem -> retry
                last_exc = exc
                log.warning("Consultant %s %s transport error (attempt %d): %s", method, path, attempt + 1, exc)
                await asyncio.sleep(0.2 * (attempt + 1))
                continue
            if resp.status_code == 200:
                return resp
            if resp.status_code == 403:
                raise ChatForbidden(resp.status_code, resp.text)
            if resp.status_code >= 500 and attempt < self._retries:
                await asyncio.sleep(0.2 * (attempt + 1))
                continue
            # 400 validation / 401 token / 404 bot / 410 blocked -> no point retrying
            raise ConsultantError(resp.status_code, resp.text)
        raise ConsultantError(0, f"transport error: {last_exc}")
