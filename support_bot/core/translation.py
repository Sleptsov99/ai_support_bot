"""Translation layer ("pivot") per language.

Two ways to serve a language:

* ``native``: the model reads and writes the user's language directly
  (keeps the persona's tone and dialects; one call less);
* ``pivot``: the user's text is translated into the PIVOT language (Russian by
  default, because the knowledge base is written in Russian), the bot
  understands it, finds the answer and writes the reply in Russian, and the
  reply plus button labels are translated back into the user's language.

``PIVOT_LANGUAGES=*`` sends every language except the pivot one through the
translator ("all languages"); ``PIVOT_LANGUAGES=ky,am,zu`` only those.
``PIVOT_TARGET=ru`` (default) chooses the pivot language. The translator is
pluggable: an LLM (Claude, DeepSeek), Google Translate, NLLB...
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol

from .language import normalize

Mode = Literal["native", "pivot"]
ALL = "*"


class Translator(Protocol):
    async def to_pivot(self, text: str, pivot: str, hint_language: str | None = None) -> tuple[str, str | None]:
        """Return (text in the pivot language, detected source language code or None)."""
        ...

    async def from_pivot(self, texts: list[str], pivot: str, target: str) -> list[str]: ...


@dataclass
class LanguagePolicy:
    modes: dict[str, Mode] = field(default_factory=dict)  # language -> mode; ALL -> default for the others
    pivot: str = "en"

    def mode(self, language: str | None) -> Mode:
        lang = normalize(language)
        if lang == self.pivot:
            return "native"
        if lang and lang in self.modes:
            return self.modes[lang]
        return self.modes.get(ALL, "native")

    @property
    def translate_everything(self) -> bool:
        """With "*" every typed message goes through the translator: it detects the language itself,
        and a message already in the pivot language comes back unchanged."""
        return self.modes.get(ALL) == "pivot"

    @property
    def enabled(self) -> bool:
        return any(m == "pivot" for m in self.modes.values())

    @classmethod
    def from_env(cls, value: str | None, pivot: str | None = None) -> "LanguagePolicy":
        """PIVOT_LANGUAGES="*" -> every language via the translator; "ky,am,zu" -> only those."""
        target = normalize(pivot) or "en"
        items = [x.strip() for x in (value or "").split(",") if x.strip()]
        modes: dict[str, Mode] = {}
        for item in items:
            if item in (ALL, "all"):
                modes[ALL] = "pivot"
            elif lang := normalize(item):
                modes[lang] = "pivot"
        return cls(modes, target)


class NoTranslator:
    """Used when no language is in pivot mode."""

    async def to_pivot(self, text: str, pivot: str, hint_language: str | None = None) -> tuple[str, str | None]:
        return text, None

    async def from_pivot(self, texts: list[str], pivot: str, target: str) -> list[str]:
        return list(texts)
