"""Event dispatcher: webhook events -> engine -> Consultant reactions.

* The webhook must answer 200 with an empty body within 3s; all work is done
  here in background tasks, reactions are POSTed to /messages separately.
* Events of one chat are processed strictly one after another.
* Glueing: people often write a question in pieces ("sir", "sir", "?", "my
  deposit..."). Messages are collected for a short window (adaptive: longer
  after a short fragment) and answered once.
* Supersede: if the user writes again while the bot is still thinking, the
  unfinished answer is thrown away (its state is not saved) and a new one is
  computed for all unanswered messages, so the user gets one coherent reply
  and the attempt counter moves once.
* Platform history: before each turn the open dialogue is read from
  Consultant (GET /chats/{id}/messages). It restores the context after a
  restart and shows if a live operator wrote in the chat: then the bot steps
  back.
* Silence: after "how long ago was the payment?" a timer runs; no answer ->
  Engine.on_silence (ТЗ 14.4: do not leave a payment case hanging).
* PingActivity -> PongActivity (must arrive within 10s, 3 misses disable the bot).
"""
from __future__ import annotations

import asyncio
import logging
from collections import defaultdict

from .consultant.adapter import to_incoming, to_reactions
from .consultant.client import ChatForbidden, ConsultantClient, ConsultantError
from .consultant.models import (
    ClientMessage,
    Event,
    PingActivity,
    PongActivity,
    SetChat,
    UnknownEvent,
    UnsetChat,
)
from .core.actions import Outcome
from .core.engine import Engine, IncomingMessage, PlatformMessage

log = logging.getLogger(__name__)

MAX_SUPERSEDES = 3  # after that an answer is sent even if the user keeps typing


def _is_fragment(msg: IncomingMessage) -> bool:
    """A short piece ("sir", "hello?", "deposit") - the rest of the question is probably coming."""
    if msg.variant_id or msg.media:
        return False
    return len((msg.text or "").split()) <= 2


class Dispatcher:
    def __init__(
        self,
        engine: Engine,
        client: ConsultantClient,
        *,
        debounce_seconds: float = 0.0,
        fragment_debounce_seconds: float | None = None,
        fetch_history: bool = False,
        tag_map: dict[str, list[int]] | None = None,
    ):
        self.engine = engine
        self.client = client
        self.debounce = debounce_seconds
        self.fragment_debounce = debounce_seconds if fragment_debounce_seconds is None else fragment_debounce_seconds
        self.fetch_history = fetch_history
        self.tag_map = tag_map or {}
        self._locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._pending: dict[str, list[IncomingMessage]] = {}  # received, not answered yet
        self._gen: dict[str, int] = defaultdict(int)  # +1 on every new message of the chat
        self._supersedes: dict[str, int] = defaultdict(int)
        self._timers: dict[str, asyncio.Task] = {}
        self._silence: dict[str, asyncio.Task] = {}
        self._tasks: set[asyncio.Task] = set()

    # ------------------------------------------------------------------ API

    def submit(self, event: Event) -> None:
        """Called from the webhook handler; never blocks."""
        if isinstance(event, PingActivity):
            self._spawn(self._pong())
        elif isinstance(event, SetChat):
            self._spawn(self._in_chat(event.chat_id, self._set_chat(event)))
        elif isinstance(event, UnsetChat):
            self._cancel_pending(event.chat_id)
            self._spawn(self._in_chat(event.chat_id, self.engine.end_dialogue(event.chat_id, event.reason_type)))
        elif isinstance(event, ClientMessage):
            self._buffer(to_incoming(event))
        elif isinstance(event, UnknownEvent):
            log.info("ignoring unknown event type %r", event.type_)

    async def drain(self, include_silence: bool = False) -> None:
        """Wait until all scheduled work is finished (tests, graceful shutdown).

        Silence timers (minutes long) are not waited for unless asked.
        """
        while True:
            waiting = set(self._silence.values())
            tasks = [t for t in self._tasks if include_silence or t not in waiting]
            if not tasks:
                return
            await asyncio.gather(*tasks, return_exceptions=True)

    # ------------------------------------------------------------ internals

    def _spawn(self, coro) -> asyncio.Task:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    async def _in_chat(self, chat_id: str, coro) -> None:
        async with self._locks[chat_id]:
            try:
                await coro
            except Exception:  # never let one chat break the dispatcher
                log.exception("error while processing chat %s", chat_id)

    async def _pong(self) -> None:
        try:
            await self.client.send(PongActivity())
        except ConsultantError:
            log.exception("failed to send PongActivity")

    async def _set_chat(self, ev: SetChat) -> None:
        outcome = await self.engine.start_dialogue(ev.chat_id, ev.conversation_id)
        await self._send(ev.chat_id, outcome)

    def _buffer(self, msg: IncomingMessage) -> None:
        chat = msg.chat_id
        self._pending.setdefault(chat, []).append(msg)
        self._gen[chat] += 1
        self._cancel_task(self._silence, chat)  # the user answered
        self._cancel_task(self._timers, chat)
        delay = self.fragment_debounce if _is_fragment(msg) else self.debounce
        self._timers[chat] = self._spawn(self._flush_later(chat, delay))

    def _cancel_task(self, tasks: dict[str, asyncio.Task], chat: str) -> None:
        task = tasks.pop(chat, None)
        if task and task is not asyncio.current_task():
            task.cancel()
            self._tasks.discard(task)

    async def _flush_later(self, chat: str, delay: float) -> None:
        if delay > 0:
            await asyncio.sleep(delay)
        if self._timers.get(chat) is asyncio.current_task():
            self._timers.pop(chat, None)
        await self._in_chat(chat, self._process(chat))

    def _cancel_pending(self, chat: str) -> None:
        self._cancel_task(self._timers, chat)
        self._cancel_task(self._silence, chat)
        self._pending.pop(chat, None)

    async def _history(self, chat: str) -> list[PlatformMessage] | None:
        if not self.fetch_history:
            return None
        try:
            page = await self.client.get_chat_messages(chat, 0, 100)
        except (ConsultantError, ChatForbidden) as exc:
            log.warning("chat %s: could not read the chat history: %s", chat, exc)
            return None
        items = page.get("items") or []
        return [PlatformMessage(i.get("Id", i.get("id")), i.get("text") or "", i.get("fromId"), i.get("createdAt"))
                for i in items]

    async def _process(self, chat: str) -> None:
        msgs = list(self._pending.get(chat, []))
        if not msgs:
            return
        gen = self._gen[chat]
        result = await self.engine.run_turn(msgs, await self._history(chat))
        if self._gen[chat] != gen and self._supersedes[chat] < MAX_SUPERSEDES:
            # The user wrote again while we were thinking: drop this answer, the next flush
            # answers all messages at once. Nothing was saved, so nothing is counted twice.
            self._supersedes[chat] += 1
            log.info("chat %s: answer superseded by a newer message", chat)
            return
        await self.engine.commit(result)
        self._supersedes[chat] = 0
        answered = len(msgs)
        rest = self._pending.get(chat, [])[answered:]
        if rest:
            self._pending[chat] = rest
        else:
            self._pending.pop(chat, None)
        await self._send(chat, result.outcome)
        if result.silence_timeout:
            self._arm_silence(chat, result.silence_timeout)

    def _arm_silence(self, chat: str, seconds: float) -> None:
        self._cancel_task(self._silence, chat)

        async def wait() -> None:
            await asyncio.sleep(seconds)
            if self._silence.get(chat) is asyncio.current_task():
                self._silence.pop(chat, None)
            await self._in_chat(chat, self._on_silence(chat))

        self._silence[chat] = self._spawn(wait())

    async def _on_silence(self, chat: str) -> None:
        if self._pending.get(chat):
            return  # the user did write
        d = await self.engine.store.get(chat)
        if d is None:
            return
        await self._send(chat, await self.engine.on_silence(chat, d.awaiting_since))

    async def _send(self, chat: str, outcome: Outcome) -> None:
        for reaction in to_reactions(chat, outcome.actions, self.tag_map):
            try:
                await self.client.send(reaction)
            except ChatForbidden:
                log.warning("chat %s: Consultant returned 403, bot can no longer write there", chat)
                await self.engine.mark_unreachable(chat)
                return
