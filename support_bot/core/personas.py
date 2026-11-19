"""AI personas per GEO (ТЗ п.10).

Pools live in data/personas.json, so a new GEO is added by editing data only.
If GEO is unknown or has no pool, the base pool is used. A persona is chosen
randomly once per dialogue and kept until the dialogue ends.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "personas.json"


@dataclass(frozen=True)
class PersonaPool:
    geo: str | None  # ISO 3166 alpha-2, None for the base pool
    geo_name: str
    names: tuple[str, ...]
    archetype: str
    tone: str
    money: str
    forbidden: str
    female: frozenset[str] = frozenset()  # names that need feminine grammar (ru "рада", ar "سعيدة"...)


@dataclass(frozen=True)
class Persona:
    name: str
    pool: PersonaPool

    @property
    def geo(self) -> str | None:
        return self.pool.geo

    @property
    def gender(self) -> str:
        """Grammatical gender for languages that need it; not in the ТЗ, kept in personas.json."""
        return "female" if self.name in self.pool.female else "male"


def _pool(geo: str | None, raw: dict) -> PersonaPool:
    return PersonaPool(
        geo=geo,
        geo_name=raw["geo_name"],
        names=tuple(raw["names"]),
        archetype=raw["archetype"],
        tone=raw["tone"],
        money=raw.get("money", ""),
        forbidden=raw["forbidden"],
        female=frozenset(raw.get("female", ())),
    )


class PersonaRegistry:
    def __init__(self, base: PersonaPool, pools: dict[str, PersonaPool], rng: random.Random | None = None):
        self.base = base
        self.pools = {k.upper(): v for k, v in pools.items()}
        self._rng = rng or random.Random()

    @classmethod
    def from_file(cls, path: Path = DATA_FILE, rng: random.Random | None = None) -> "PersonaRegistry":
        raw = json.loads(path.read_text(encoding="utf-8"))
        pools = {geo: _pool(geo, p) for geo, p in raw["geo"].items()}
        return cls(_pool(None, raw["base"]), pools, rng)

    def pool_for(self, country: str | None) -> PersonaPool:
        if country and country.upper() in self.pools:
            return self.pools[country.upper()]
        return self.base

    def assign(self, country: str | None) -> Persona:
        pool = self.pool_for(country)
        return Persona(name=self._rng.choice(pool.names), pool=pool)
