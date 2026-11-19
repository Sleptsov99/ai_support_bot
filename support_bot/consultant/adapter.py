"""Consultant <-> engine mapping."""
from __future__ import annotations

from ..core.actions import Action, HandOff, NoteToOperator, Option, Resolve, Say, SetLanguage
from ..core.engine import IncomingMessage
from ..core.llm import Media
from .models import (
    BotMessage,
    BotReplyMessage,
    ButtonVariant,
    ChangeChatLanguage,
    ChatReplyMarkup,
    ClientMessage,
    InvokeOperator,
    Reaction,
    ResolveChat,
    RowVariant,
    SupportBotMessage,
)

BUTTON_MAX = 20  # Consultant limit for ButtonVariant.text; longer labels go to rows
TEXT_MAX = 4096


def to_incoming(ev: ClientMessage) -> IncomingMessage:
    return IncomingMessage(
        chat_id=ev.chat_id,
        text=ev.text,
        media=tuple(Media(m.id, m.mime_type, m.file_name) for m in ev.all_media),
        variant_id=ev.command.variant_id if ev.command else None,
        message_id=ev.message_id,
        conversation_id=ev.conversation_id,
        language_code=ev.language_code,
        country=ev.country,
        timezone=ev.geo.timezone if ev.geo else None,
        channel=ev.channel,
        client_id=ev.client.id if ev.client else None,
    )


def _markup(options: tuple[Option, ...]) -> ChatReplyMarkup | None:
    if not options:
        return None
    if all(len(o.label) <= BUTTON_MAX for o in options):
        return ChatReplyMarkup(buttons=[ButtonVariant(id=o.id, text=o.label) for o in options])
    return ChatReplyMarkup(rows=[RowVariant(id=o.id, text=o.label[:2048]) for o in options])


def to_reactions(chat_id: str, actions: list[Action], tag_map: dict[str, list[int]] | None = None) -> list[Reaction]:
    """tag_map: category key -> Consultant tag ids (from GET /projects)."""
    out: list[Reaction] = []
    for a in actions:
        if isinstance(a, Say):
            text = a.text[:TEXT_MAX]
            if a.reply_to is not None:
                tags = (tag_map or {}).get(a.category or "")
                out.append(BotReplyMessage(chat_id=chat_id, reply_message_id=a.reply_to, text=text,
                                           reply_markup=_markup(a.options), result=a.result, tag=tags or None))
            else:
                out.append(BotMessage(chat_id=chat_id, text=text, reply_markup=_markup(a.options)))
        elif isinstance(a, NoteToOperator):
            out.append(SupportBotMessage(chat_id=chat_id, text=a.text[:TEXT_MAX]))
        elif isinstance(a, SetLanguage):
            out.append(ChangeChatLanguage(chat_id=chat_id, language_code=a.language))
        elif isinstance(a, Resolve):
            out.append(ResolveChat(chat_id=chat_id, time_seconds=a.close_after_seconds))
        elif isinstance(a, HandOff):
            out.append(InvokeOperator(chat_id=chat_id))
    # Safety net: nothing may follow InvokeOperator (Consultant returns 403 after it).
    for i, r in enumerate(out):
        if isinstance(r, InvokeOperator):
            return out[: i + 1]
    return out
