"""Knowledge base.

The source of truth is ONE markdown document (the "ТЗ бот основные блоки
вопросы/ответы" sheet converted with scripts/qa_to_doc.py, later + ava.bet
pages). People edit the document; the engine loads it at start.

    # Deposit {#deposit}
    ## How long does a deposit take? {#deposit-time}
    keywords: deposit processing time, сколько идет депозит
    Deposit crediting takes from a few minutes up to 24 hours...

The whole document goes into the cached part of the compose prompt (full
context, no RAG). ``search`` is a cheap keyword lookup used to point the
model at the most relevant sections and by the offline Fake/Demo LLMs.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

_WORD = re.compile(r"\w+", re.UNICODE)
_STOP = {"не", "на", "в", "и", "я", "как", "что", "мне", "мой", "моя", "the", "a", "my", "to", "is", "how", "of"}


def _stems(text: str) -> set[str]:
    """Crude stemming for keyword lookup: first 5 letters ("депозита" ~ "депозит", "пришел" ~ "пришёл")."""
    return {w[:5].replace("ё", "е") for w in _WORD.findall(text.lower()) if w not in _STOP and len(w) > 1}


@dataclass(frozen=True)
class KBEntry:
    id: str
    category: str
    question: str
    answer: str
    keywords: tuple[str, ...] = ()
    language: str = "en"
    variants: tuple[str, ...] = ()  # other wordings of the same question (they share this answer)
    note: str = ""  # for the bot only (status / condition / escalation from the sheet), never shown as is


class KnowledgeBase(Protocol):
    def search(self, query: str, category: str | None = None, limit: int = 3) -> list[KBEntry]: ...

    def document(self) -> str: ...

_CAT = re.compile(r"^#\s+(.+?)\s*\{#([\w-]+)\}\s*$")
_Q = re.compile(r"^##\s+(.+?)\s*(?:\{#([\w-]+)\})?\s*$")


def parse_markdown(text: str) -> list[KBEntry]:
    entries: list[KBEntry] = []
    category = "general"
    q: str | None = None
    qid: str | None = None
    keywords: tuple[str, ...] = ()
    variants: tuple[str, ...] = ()
    note = ""
    body: list[str] = []

    def flush() -> None:
        if q is not None:
            entries.append(KBEntry(qid or f"{category}-{len(entries) + 1}", category, q,
                                   "\n".join(body).strip(), keywords, variants=variants, note=note))

    for line in text.splitlines():
        if m := _CAT.match(line):
            flush()
            q, body, keywords, variants, note = None, [], (), (), ""
            category = m.group(2)
        elif m := _Q.match(line):
            flush()
            q, qid, body, keywords, variants, note = m.group(1), m.group(2), [], (), (), ""
        elif q is not None and line.lower().startswith("keywords:") and not body:
            keywords = tuple(k.strip() for k in line.split(":", 1)[1].split(",") if k.strip())
        elif q is not None and line.lower().startswith("variants:") and not body:
            variants = tuple(v.strip() for v in line.split(":", 1)[1].split(" | ") if v.strip())
        elif q is not None and line.lower().startswith("note:") and not body:
            note = line.split(":", 1)[1].strip()
        elif q is not None:
            body.append(line)
    flush()
    return entries


def render_markdown(entries: list[KBEntry], titles: dict[str, str] | None = None, keep_order: bool = False) -> str:
    out: list[str] = []
    last_cat = None
    for e in (entries if keep_order else sorted(entries, key=lambda e: e.category)):
        if e.category != last_cat:
            out.append(f"\n# {(titles or {}).get(e.category, e.category.capitalize())} {{#{e.category}}}")
            last_cat = e.category
        out.append(f"## {e.question} {{#{e.id}}}")
        if e.keywords:
            out.append("keywords: " + ", ".join(e.keywords))
        if e.variants:
            out.append("variants: " + " | ".join(v.replace("\n", " ") for v in e.variants))
        if e.note:
            out.append("note: " + e.note.replace("\n", " "))
        out.append(e.answer + "\n")
    return "\n".join(out).strip() + "\n"


class InMemoryKB:
    def __init__(self, entries: list[KBEntry]):
        self.entries = list(entries)

    @classmethod
    def from_json(cls, path: Path) -> "InMemoryKB":
        raw = json.loads(path.read_text(encoding="utf-8"))
        fields = set(KBEntry.__dataclass_fields__)
        return cls([KBEntry(**{**{k: v for k, v in e.items() if k in fields},
                               "keywords": tuple(e.get("keywords", ()))}) for e in raw])

    @classmethod
    def from_markdown(cls, path: Path) -> "InMemoryKB":
        kb = cls(parse_markdown(path.read_text(encoding="utf-8")))
        kb._doc = path.read_text(encoding="utf-8")
        return kb

    @classmethod
    def from_path(cls, path: Path) -> "InMemoryKB":
        return cls.from_markdown(path) if path.suffix.lower() in (".md", ".markdown") else cls.from_json(path)

    def document(self) -> str:
        return getattr(self, "_doc", None) or render_markdown(self.entries)

    def search(self, query: str, category: str | None = None, limit: int = 3) -> list[KBEntry]:
        q = query.lower()
        stems = _stems(q)
        scored: list[tuple[float, KBEntry]] = []
        for e in self.entries:
            score = 0.0
            if e.question.lower() == q.strip():
                score += 10
            if any(v.lower() == q.strip() for v in e.variants):
                score += 10
            score += sum(2 for k in e.keywords if k.lower() in q)
            score += max(len(stems & _stems(t)) for t in (e.question, *e.variants)) * 0.75
            if category and e.category == category:
                score += 1
            if score >= 1.5:
                scored.append((score, e))
        scored.sort(key=lambda s: -s[0])
        return [e for _, e in scored[:limit]]
