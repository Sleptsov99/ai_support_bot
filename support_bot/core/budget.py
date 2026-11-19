"""Escalation budget per category (ТЗ 14.1 target model, 14.6 extra cycle).

If the actual share of escalated conversations in a category is above the
target, the engine gives the user extra attempts (extended search + options)
before escalating. Shares are computed over a sliding window and only once
there are enough conversations, so a rare category does not jump around
from 0% to 100%.

In memory for the demo; production needs a shared store (Redis), because
the share must be computed over all instances.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Protocol

# ТЗ 14.1, upper bound of the allowed L1 share. None = escalation not allowed.
TARGET_ESCALATION_SHARE: dict[str, float | None] = {
    "deposit": 0.05,
    "withdrawal": 0.05,
    "registration": 0.10,
    "bonuses": 0.10,
    "account": 0.10,
    "kyc": 0.10,
    "sport": 0.05,
    "casino": 0.05,
    "app": 0.05,
    "rules": 0.02,
    "contacts": None,
    "partnership": None,
    "complaint": 0.70,
    "unknown": 0.05,
}


class EscalationBudget(Protocol):
    def record(self, category: str, escalated: bool) -> None: ...

    def over_target(self, category: str) -> bool: ...


@dataclass
class InMemoryBudget:
    window_seconds: float = 7 * 24 * 3600
    min_conversations: int = 50
    targets: dict[str, float | None] = field(default_factory=lambda: dict(TARGET_ESCALATION_SHARE))
    clock: object = time.time
    _events: dict[str, deque] = field(default_factory=dict)

    def record(self, category: str, escalated: bool) -> None:
        self._events.setdefault(category, deque()).append((self.clock(), escalated))  # type: ignore[operator]

    def share(self, category: str) -> float | None:
        q = self._events.get(category)
        if not q:
            return None
        cutoff = self.clock() - self.window_seconds  # type: ignore[operator]
        while q and q[0][0] < cutoff:
            q.popleft()
        if len(q) < self.min_conversations:
            return None
        return sum(1 for _, e in q if e) / len(q)

    def over_target(self, category: str) -> bool:
        target = self.targets.get(category)
        share = self.share(category)
        return target is not None and share is not None and share > target
