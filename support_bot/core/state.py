"""Dialogue state.

chatId is the client's chat (lives forever); one SetChat..UnsetChat cycle is
one dialogue (conversationId). Persona, language and escalation counters are
per dialogue (ТЗ 10: a new dialogue picks a new persona).

Escalation counters are kept PER PROBLEM (``issues``): a problem is a topic
category; if the user jumps to another topic and comes back, the old counter
continues (ТЗ 14.3: repeats do not start a new cycle).

Optionally a short note about the previous conversation of the same chatId
is kept for a limited time (``ContactMemory``, off by default: the ТЗ does not
ask for state between conversations).

InMemoryStore is enough for one process; swap for Redis/DB (same Protocol).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol

from .llm import Turn
from .personas import Persona

UNKNOWN_ISSUE = "unknown"  # unrecognized requests are one "problem" for the cycle (ТЗ 14.3, row 1)


@dataclass
class PaymentCase:
    kind: str  # "deposit" | "withdrawal"
    wait_hours: float | None = None
    wait_questions_asked: int = 0  # "how long?" asked and not answered
    clarify_asked: int = 0  # answer was ambiguous around 24h and we asked again
    within_norm: bool = False  # < 24h -> must not be escalated
    over_threshold: bool = False  # > 24h -> collect details, then L1
    details: dict[str, str] = field(default_factory=dict)  # method / amount / paid_at ... for the L1 card
    details_asked: list[str] = field(default_factory=list)
    pushback: int = 0  # "I already sent everything!" while we collect details


@dataclass
class Issue:
    key: str  # category key, "complaint" or UNKNOWN_ISSUE
    failures: int = 0  # unsuccessful attempts on this problem (ТЗ 14.3 cycle)
    last_result: str | None = None
    turns: int = 0

    @property
    def attempt(self) -> int:
        """Strategy number for the next answer (1..5): the n-th failure gets the n-th strategy."""
        return max(1, self.failures)

    # backwards-compatible name used by the demo panel / older tests
    @property
    def attempts(self) -> int:
        return self.failures

    @property
    def category(self) -> str | None:
        return None if self.key == UNKNOWN_ISSUE else self.key


@dataclass
class Dialogue:
    chat_id: str
    conversation_id: str | None = None
    persona: Persona | None = None
    introduced: bool = False
    language: str | None = None  # current reply language
    platform_language: str | None = None  # languageCode as Consultant knows it
    chat_language_set: str | None = None  # last language sent via ChangeChatLanguage
    country: str | None = None
    timezone: str | None = None  # IANA, from geo.timezone
    channel: str | None = None
    client_id: str | None = None
    history: list[Turn] = field(default_factory=list)
    issues: dict[str, Issue] = field(default_factory=dict)
    current: str | None = None  # key of the problem we are working on
    payment: PaymentCase | None = None
    awaiting: str | None = None  # "wait_time" | "confirmation" | "payment_details" | None
    awaiting_since: float | None = None  # monotonic time when the bot asked (silence timer)
    offered: dict[str, str] = field(default_factory=dict)  # free-form buttons of the last answer: id -> label
    noise_count: int = 0
    off_topic_count: int = 0
    previous_contact: str | None = None
    seen_platform_ids: set[int] = field(default_factory=set)
    handed_off: bool = False
    operator_joined: bool = False  # a live operator wrote in this chat: the bot stays silent
    closed: bool = False
    escalation_reason: str | None = None
    last_signal: str | None = None  # for logs / the demo panel

    @property
    def active(self) -> bool:
        return not (self.handed_off or self.closed or self.operator_joined)

    @property
    def issue(self) -> Issue:
        """The problem we are working on (an empty one if none yet)."""
        if self.current and self.current in self.issues:
            return self.issues[self.current]
        return Issue(key=UNKNOWN_ISSUE)

    def switch_issue(self, key: str) -> Issue:
        self.current = key
        return self.issues.setdefault(key, Issue(key=key))

    def bot_texts(self, n: int = 5) -> tuple[str, ...]:
        return tuple(t.text for t in self.history if t.role == "bot")[-n:]


class DialogueStore(Protocol):
    async def get(self, chat_id: str) -> Dialogue | None: ...

    async def save(self, d: Dialogue) -> None: ...

    async def delete(self, chat_id: str) -> None: ...


class InMemoryStore:
    def __init__(self) -> None:
        self.items: dict[str, Dialogue] = {}

    async def get(self, chat_id: str) -> Dialogue | None:
        return self.items.get(chat_id)

    async def save(self, d: Dialogue) -> None:
        self.items[d.chat_id] = d

    async def delete(self, chat_id: str) -> None:
        self.items.pop(chat_id, None)


class ContactMemory:
    """Short note about the previous conversation of a chatId, kept for ``ttl_seconds``."""

    def __init__(self, ttl_seconds: float, clock=time.time) -> None:
        self.ttl = ttl_seconds
        self._clock = clock
        self._items: dict[str, tuple[float, str]] = {}

    def remember(self, chat_id: str, note: str) -> None:
        if self.ttl > 0:
            self._items[chat_id] = (self._clock() + self.ttl, note)

    def recall(self, chat_id: str) -> str | None:
        item = self._items.get(chat_id)
        if not item:
            return None
        expires, note = item
        if self._clock() > expires:
            self._items.pop(chat_id, None)
            return None
        return note
