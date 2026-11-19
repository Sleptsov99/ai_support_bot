"""Offline demo: plays a few dialogues against the fake Consultant and prints every reaction.

    python scripts/demo.py
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.conftest import Harness  # noqa: E402
from tests.fakes import client_message as msg, set_chat  # noqa: E402

SCENARIOS = {
    "Deposit not arrived, > 24h -> collect details -> L1 (dialogue 4)": [
        set_chat("d1"), msg("my deposit did not arrive", "d1"), msg("call operator", "d1"), msg("3 days already", "d1"),
        msg("Vodafone Cash", "d1"), msg("I already sent everything!", "d1"), msg("receipt is not here", "d1"),
        msg("1000", "d1"), msg("yesterday 08:40", "d1"), msg("01001982785", "d1"), msg("01098765432", "d1"),
    ],
    "Russian, forgot password, resolved": [
        set_chat("d2"), msg("забыл пароль", "d2", country="KG", language="ky"), msg("спасибо", "d2"),
        msg("Да", "d2", variant="confirm:yes"),
    ],
    "Category buttons": [
        set_chat("d3"), msg("Sport", "d3", variant="cat:sport", country="NG"),
    ],
}


async def main() -> None:
    for title, events in SCENARIOS.items():
        h = Harness()
        print(f"\n=== {title} ===")
        for ev in events:
            if ev["$type"] == "ClientMessage":
                print(f"\n>>> CLIENT: {ev.get('text')!r}" + (f" [click {ev['command']['variantId']}]" if "command" in ev else ""))
            else:
                print(f"\n>>> EVENT: {ev['$type']}")
            before = len(h.consultant.sent)
            await h.send(ev)
            for p in h.consultant.sent[before:]:
                print("<<< " + json.dumps(p, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
