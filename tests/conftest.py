from __future__ import annotations

import random

import pytest

from support_bot.consultant.models import parse_event
from support_bot.core.engine import Engine, EngineConfig
from support_bot.core.kb import InMemoryKB, KBEntry
from support_bot.core.llm import FakeLLM
from support_bot.core.personas import PersonaRegistry
from support_bot.core.state import InMemoryStore
from support_bot.dispatcher import Dispatcher

from .fakes import FakeConsultant

TEST_KB = [
    KBEntry("dep-time", "deposit", "How long does a deposit take?",
            "KB: deposits are credited within 24 hours.", ("deposit processing time",)),
    KBEntry("wd-time", "withdrawal", "How long does a withdrawal take?",
            "KB: withdrawals are processed within 24 hours.", ("withdrawal processing time",)),
    KBEntry("min-dep", "deposit", "Minimum deposit", "KB: minimum deposit is shown on the cashier page.",
            ("minimum deposit",)),
    KBEntry("forgot", "account", "Forgot password", "KB: use the 'Forgot password?' link.",
            ("forgot password", "забыл пароль")),
]


class Harness:
    """Engine + dispatcher + fake Consultant wired together."""

    def __init__(self, debounce: float = 0.0, seed: int = 42):
        self.llm = FakeLLM()
        self.store = InMemoryStore()
        self.engine = Engine(
            llm=self.llm,
            kb=InMemoryKB(TEST_KB),
            personas=PersonaRegistry.from_file(rng=random.Random(seed)),
            store=self.store,
            config=EngineConfig(),
        )
        self.consultant = FakeConsultant()
        self.dispatcher = Dispatcher(self.engine, self.consultant, debounce_seconds=debounce)

    async def send(self, *events: dict) -> None:
        """Deliver events one by one like Consultant does (sequentially, waiting for processing)."""
        for ev in events:
            self.dispatcher.submit(parse_event(ev))
            await self.dispatcher.drain()

    async def dialogue(self, chat_id: str = "chat1"):
        return await self.store.get(chat_id)


@pytest.fixture
def h() -> Harness:
    return Harness()
