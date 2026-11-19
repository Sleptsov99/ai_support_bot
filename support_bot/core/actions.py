"""Platform-neutral outputs of the engine.

The engine never builds Consultant payloads directly (ТЗ п.2: Consultant is
only the first integration and must not define the core logic). An adapter
(support_bot/consultant/adapter.py) maps these to Consultant reactions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

AnswerResult = Literal["Full", "Partial", "NotFound"]


@dataclass(frozen=True)
class Option:
    id: str
    label: str


@dataclass(frozen=True)
class Say:
    text: str
    options: tuple[Option, ...] = ()
    reply_to: int | None = None  # id of the user message we answer (-> BotReplyMessage)
    result: AnswerResult | None = None
    category: str | None = None


@dataclass(frozen=True)
class NoteToOperator:
    """Hidden from the user, visible to L1 (SupportBotMessage)."""
    text: str


@dataclass(frozen=True)
class HandOff:
    """Transfer to L1 (InvokeOperator). Must be the LAST action: after it the chat is read-only for the bot."""


@dataclass(frozen=True)
class Resolve:
    """Mark the dialogue as resolved (ResolveChat)."""
    close_after_seconds: int | None = None


@dataclass(frozen=True)
class SetLanguage:
    language: str


Action = Say | NoteToOperator | HandOff | Resolve | SetLanguage


@dataclass
class Outcome:
    actions: list[Action] = field(default_factory=list)
    silence_timeout: float | None = None  # seconds: call Engine.on_silence if the user stays silent

    def add(self, *actions: Action) -> "Outcome":
        self.actions.extend(actions)
        return self
