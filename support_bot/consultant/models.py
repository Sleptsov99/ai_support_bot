"""Consultant webhook events and bot reactions.

Based on "Интеграция чат-бота в систему Консультант" + answers from the
Consultant team:
  * unknown fields and enum values must be ignored (forward compatibility);
  * conversationId is present in all events;
  * media list is called ``medias`` in practice and ``media`` in the docs,
    both are accepted.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class _Model(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)


# --------------------------------------------------------------------- events


class ChatClient(_Model):
    id: str | None = None
    country: str | None = None


class ChatGeo(_Model):
    country: str | None = None
    timezone: str | None = None


class MediaDocument(_Model):
    type_: str = Field(default="MediaDocument", alias="$type")
    id: str
    size: int | None = None
    mime_type: str | None = Field(default=None, alias="mimeType")
    attributes: list[dict[str, Any]] = Field(default_factory=list)

    @property
    def file_name(self) -> str | None:
        for attr in self.attributes:
            if attr.get("$type") == "DocumentAttributeFileName":
                return attr.get("name")
        return None


class SelectVariantCommand(_Model):
    type_: str = Field(default="SelectVariantCommand", alias="$type")
    variant_id: str = Field(alias="variantId")


class ClientMessage(_Model):
    type_: Literal["ClientMessage"] = Field(alias="$type")
    chat_id: str = Field(alias="chatId")
    conversation_id: str | None = Field(default=None, alias="conversationId")
    project_id: int | None = Field(default=None, alias="projectId")
    channel: str | None = None  # Web / WebMobi / App_Windows / App_iOs / App_Android / unknown
    language_code: str | None = Field(default=None, alias="languageCode")
    message_id: int | None = Field(default=None, alias="messageId")
    reply_message_id: int | None = Field(default=None, alias="replyMessageId")
    text: str | None = None
    media: list[MediaDocument] = Field(default_factory=list)
    medias: list[MediaDocument] = Field(default_factory=list)
    command: SelectVariantCommand | None = None
    client: ChatClient | None = None
    geo: ChatGeo | None = None

    @property
    def all_media(self) -> list[MediaDocument]:
        return [*self.media, *self.medias]

    @property
    def country(self) -> str | None:
        """GEO country; geo.country first, then client.country. Never guessed."""
        if self.geo and self.geo.country:
            return self.geo.country.upper()
        if self.client and self.client.country:
            return self.client.country.upper()
        return None


class SetChat(_Model):
    type_: Literal["SetChat"] = Field(alias="$type")
    chat_id: str = Field(alias="chatId")
    conversation_id: str | None = Field(default=None, alias="conversationId")


class UnsetChat(_Model):
    type_: Literal["UnsetChat"] = Field(alias="$type")
    chat_id: str = Field(alias="chatId")
    conversation_id: str | None = Field(default=None, alias="conversationId")
    reason_type: str | None = Field(default=None, alias="reasonType")  # Resolved/OperatorInvoked/Timeout/...


class PingActivity(_Model):
    type_: Literal["PingActivity"] = Field(alias="$type")


class UnknownEvent(_Model):
    model_config = ConfigDict(extra="allow")
    type_: str | None = Field(default=None, alias="$type")


Event = ClientMessage | SetChat | UnsetChat | PingActivity | UnknownEvent

_EVENT_TYPES: dict[str, type[_Model]] = {
    "ClientMessage": ClientMessage,
    "SetChat": SetChat,
    "UnsetChat": UnsetChat,
    "PingActivity": PingActivity,
}


def parse_event(payload: dict[str, Any]) -> Event:
    cls = _EVENT_TYPES.get(str(payload.get("$type")))
    if cls is None:
        return UnknownEvent.model_validate(payload)
    return cls.model_validate(payload)  # type: ignore[return-value]


# ------------------------------------------------------------------ reactions


class ButtonVariant(_Model):
    type_: str = Field(default="ButtonVariant", alias="$type")
    id: str = Field(max_length=64)
    text: str = Field(min_length=1, max_length=20)


class RowVariant(_Model):
    type_: str = Field(default="RowVariant", alias="$type")
    id: str = Field(max_length=64)
    text: str = Field(min_length=1, max_length=2048)


class ChatReplyMarkup(_Model):
    buttons: list[ButtonVariant] = Field(default_factory=list)
    rows: list[RowVariant] = Field(default_factory=list)


class _Reaction(_Model):
    def to_payload(self) -> dict[str, Any]:
        return self.model_dump(by_alias=True, exclude_none=True)


class BotMessage(_Reaction):
    type_: Literal["BotMessage"] = Field(default="BotMessage", alias="$type")
    chat_id: str = Field(alias="chatId")
    reply_message_id: int | None = Field(default=None, alias="replyMessageId")
    text: str | None = Field(default=None, max_length=4096)
    media_id: list[str] | None = Field(default=None, alias="mediaId")
    reply_markup: ChatReplyMarkup | None = Field(default=None, alias="replyMarkup")


class BotReplyMessage(_Reaction):
    type_: Literal["BotReplyMessage"] = Field(default="BotReplyMessage", alias="$type")
    chat_id: str = Field(alias="chatId")
    reply_message_id: int | None = Field(default=None, alias="replyMessageId")
    text: str | None = Field(default=None, max_length=4096)
    media_id: list[str] | None = Field(default=None, alias="mediaId")
    reply_markup: ChatReplyMarkup | None = Field(default=None, alias="replyMarkup")
    result: Literal["Full", "Partial", "NotFound"] | None = None
    tag: list[int] | None = None


class SupportBotMessage(_Reaction):
    type_: Literal["SupportBotMessage"] = Field(default="SupportBotMessage", alias="$type")
    chat_id: str = Field(alias="chatId")
    text: str = Field(max_length=4096)


class InvokeOperator(_Reaction):
    type_: Literal["InvokeOperator"] = Field(default="InvokeOperator", alias="$type")
    chat_id: str = Field(alias="chatId")


class ResolveChat(_Reaction):
    type_: Literal["ResolveChat"] = Field(default="ResolveChat", alias="$type")
    chat_id: str = Field(alias="chatId")
    time_seconds: int | None = Field(default=None, alias="timeSeconds")


class ChangeChatLanguage(_Reaction):
    type_: Literal["ChangeChatLanguage"] = Field(default="ChangeChatLanguage", alias="$type")
    chat_id: str = Field(alias="chatId")
    language_code: str = Field(alias="languageCode")


class PongActivity(_Reaction):
    type_: Literal["PongActivity"] = Field(default="PongActivity", alias="$type")


Reaction = (
    BotMessage
    | BotReplyMessage
    | SupportBotMessage
    | InvokeOperator
    | ResolveChat
    | ChangeChatLanguage
    | PongActivity
)
