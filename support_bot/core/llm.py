"""LLM boundary.

The engine (deterministic policy: languages, personas, escalation, payment
24h rule, closing) talks to the model only through ``LLM``:

  * analyze()  - NLU: language (+ confidence), topic, what kind of turn this
                 is (progress / not helped / not understood / noise ...),
                 operator request, payment issue and waiting time as a range,
                 tone, complaint, risk signals.
  * compose()  - write the user-facing text for a task chosen by the engine.
  * localize() - translate short UI labels (buttons) to the user language.

``FakeLLM`` is a keyword-based implementation used in tests and for local
runs without an API key. ``ClaudeLLM`` (claude_llm.py) is the real adapter.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal, Protocol

from .actions import AnswerResult, Option
from .catalog import BY_KEY, CATEGORIES, COMPLAINT
from .kb import KBEntry
from .personas import Persona


@dataclass(frozen=True)
class Media:
    id: str
    mime_type: str | None = None
    file_name: str | None = None


@dataclass(frozen=True)
class Turn:
    role: Literal["user", "bot", "operator"]
    text: str


# What the last user turn means for the attempt cycle (ТЗ 14.3). Only the
# "failure" signals count as an unsuccessful attempt; answering a clarifying
# question or clicking a button is progress, not a failed attempt.
Signal = Literal[
    "new_request",     # a (new) question or problem
    "progress",        # answers the bot's question / did the step / adds details
    "not_helped",      # "still doesn't work", "that didn't help", "No" to "did it solve?"
    "not_understood",  # has content but the bot cannot tell what the user wants
    "repeat",          # repeats the same question / complaint without new info
    "dissatisfied",    # unhappy with the answer or the service (without new info)
    "noise",           # no content at all: "sir", "?", "hello", emoji
    "closing",         # thanks / ok / got it
]
FAILURE_SIGNALS = frozenset({"not_helped", "not_understood", "repeat", "dissatisfied"})
Risk = Literal["gambling_harm", "self_harm", "minor"]


@dataclass(frozen=True)
class AnalyzeRequest:
    text: str  # several consecutive messages already glued together
    media: tuple[Media, ...] = ()
    history: tuple[Turn, ...] = ()
    current_category: str | None = None
    awaiting: str | None = None  # what the bot asked last: "wait_time" | "confirmation" | "payment_details" | None
    current_language: str | None = None
    now_local: str | None = None  # "2026-11-18 14:52 (Africa/Cairo)" to resolve "since yesterday morning"
    previous_contact: str | None = None  # short note about the client's previous conversation, if enabled
    asked_detail: str | None = None  # awaiting == "payment_details": which item the bot asked for


@dataclass(frozen=True)
class Analysis:
    language: str | None = None  # ISO 639-1 of the message, None if unknown
    language_confident: bool = True  # False for "ok", "sir", "40 min": do not switch the dialogue language
    category: str | None = None  # catalog key, "complaint", or None
    signal: Signal | None = None  # None -> the engine derives it from the other fields
    understood: bool = True  # False: the bot cannot tell what the user wants
    wants_operator: bool = False
    payment_issue: Literal["deposit", "withdrawal"] | None = None  # money did not arrive
    wait_hours: float | None = None  # lower bound of how long the user waits
    wait_hours_max: float | None = None  # upper bound; None = same as wait_hours
    confirmation: Literal["yes", "no"] | None = None
    gratitude: bool = False  # "ok thanks", "got it"
    aggressive: bool = False  # rude tone, insults: a TONE modifier, not a topic
    complaint: bool = False  # an actual complaint about the service / an operator / fraud claim
    off_topic: bool = False  # weather, jokes, politics, "who are you"...
    noise: bool = False  # no content at all
    risk: Risk | None = None  # responsible gaming / self-harm / minor
    media_understood: bool = True
    payment_details: dict[str, str] = field(default_factory=dict)  # method / amount / paid_at ... found in text
    english_text: str | None = None  # set by the engine in pivot mode: the translated text the model read

    @property
    def wait_ambiguous(self) -> bool:
        """'since yesterday evening' at 10 am: maybe < 24h, maybe more -> ask once more."""
        if self.wait_hours is None or self.wait_hours_max is None:
            return False
        return self.wait_hours <= 24.0 < self.wait_hours_max


class Task(str, Enum):
    ANSWER = "answer"  # answer the question from KB / clarify
    CLARIFY_UNRECOGNIZED = "clarify_unrecognized"  # nothing recognized, ask to describe
    NUDGE = "nudge"  # "sir?" without content: one short line, no repetition of the last answer
    OFF_TOPIC = "off_topic"  # politely bring the user back to support topics
    DEFLECT_OPERATOR = "deflect_operator"  # "call an operator" -> find out the problem first
    ASK_WAIT_TIME = "ask_wait_time"  # group B: how long since deposit / withdrawal request
    CLARIFY_WAIT_TIME = "clarify_wait_time"  # group B: answer was ambiguous around 24h
    PAYMENT_WITHIN_NORM = "payment_within_norm"  # group B, < 24h
    COLLECT_PAYMENT_DETAILS = "collect_payment_details"  # group B, > 24h: data for L1 (dialogue 4 of the ТЗ)
    ASK_RESOLVED = "ask_resolved"  # "Did that solve your question?"
    GOODBYE = "goodbye"  # dialogue resolved
    HANDOFF = "handoff"  # message to the user before transfer to L1
    RISK_SUPPORT = "risk_support"  # responsible gaming / self-harm: care first, then L1
    MEDIA_UNREADABLE = "media_unreadable"  # ask to describe the file content in text


@dataclass(frozen=True)
class ComposeRequest:
    task: Task
    persona: Persona
    language: str
    user_text: str
    introduce: bool = False  # first answer after persona assignment -> say the name
    attempt: int = 1  # 1..5, strategy of the attempt cycle (ТЗ 14.3)
    category: str | None = None
    payment_kind: str | None = None
    wait_hours: float | None = None
    channel: str | None = None  # device: Web / WebMobi / App_Android ...
    country: str | None = None
    kb: tuple[KBEntry, ...] = ()  # most relevant entries (the full document is in the cached prompt)
    history: tuple[Turn, ...] = ()
    previous_bot_texts: tuple[str, ...] = ()  # to avoid repeating answers verbatim
    aggressive: bool = False  # user is angry: calm, facts only, no arguing
    facts: tuple[str, ...] = ()  # engine-computed facts the reply may use ("deadline: 14:12 tomorrow")
    missing_detail: str | None = None  # COLLECT_PAYMENT_DETAILS: which field to ask for
    fix_hint: str | None = None  # set on a retry after the fact check failed
    previous_contact: str | None = None


@dataclass(frozen=True)
class Composed:
    text: str
    result: AnswerResult = "Full"
    options: tuple[Option, ...] = ()


class LLM(Protocol):
    async def analyze(self, req: AnalyzeRequest) -> Analysis: ...

    async def compose(self, req: ComposeRequest) -> Composed: ...

    async def localize(self, labels: list[str], language: str) -> list[str]: ...


# =========================================================================
# Fake implementation (deterministic, offline)
# =========================================================================

_OPERATOR = re.compile(r"operator|оператор|human|живого|живой человек|real person|agent please", re.I)
_YES = re.compile(r"^\s*(yes|yep|yeah|да|ага|конечно|resolved|решено)\b", re.I)
_NO = re.compile(r"^\s*(no|nope|нет|не решено|not resolved)\b", re.I)
_THANKS = re.compile(r"thank|thanks|спасибо|понятно|got it|ок\b|ok\b|okay|ладно", re.I)
_NOT_ARRIVED = re.compile(
    r"not (arrive|received|credited|come)|didn'?t (arrive|come|get|receive)|haven'?t (received|got)|"
    r"не (пришл|пришел|поступ|зачисл|получил|дошл)|нету|не вижу|missing",
    re.I,
)
_NOT_HELPED = re.compile(r"didn'?t help|doesn'?t work|still (not|doesn)|не помогло|не работает|все равно|всё равно|"
                         r"не получается|не входит", re.I)
_AGGRESSIVE = re.compile(r"idiot|stupid|идиот|дурак|туп|убью|kill", re.I)
_FED_UP = re.compile(r"already (sent|told|did)|how many times|уже (все |всё )?(отправил|писал|сказал)|сколько можно", re.I)
_COMPLAINT = re.compile(r"complain|жалоб|пожалуюсь|scam|мошенн|обман|fraud", re.I)
_NOISE = re.compile(r"^[\s?!.,)(]*((sir|сэр|hello|hi|привет|алло|hey|ау)[\s?!.,]*)*$", re.I)
_OFF_TOPIC = re.compile(r"weather|погод|joke|анекдот|шутк|politic|политик|who are you|ты кто|ты бот", re.I)
_RISK_GAMBLING = re.compile(r"addict|can'?t stop|lost everything|лудоман|зависим|не могу остановиться|проиграл все", re.I)
_RISK_SELF = re.compile(r"kill myself|suicide|покончу|не хочу жить|убить себя", re.I)
_MINOR = re.compile(r"i'?m (1[0-7]|[5-9]) years|мне (1[0-7]|[5-9]) ?(лет)?\b", re.I)
_AMOUNT = re.compile(r"(?:\bна|\bfor|\bof)\s+(\d{2,}(?:[.,]\d+)?)(?!\s*(?:min|h|час|мин|д|day))|"
                     r"(\d{2,}(?:[.,]\d+)?)\s*(?:egp|usd|kgs|ngn|kes|руб|сом|фунт|pounds?|naira)", re.I)
_METHOD = re.compile(r"vodafone cash|etisalat cash|orange cash|instapay|opay|m-?pesa|visa|mastercard|карт[аой]|"
                     r"qiwi|usdt|bitcoin|crypto", re.I)
_NUM = r"(\d+(?:[.,]\d+)?)"
_DURATIONS: list[tuple[re.Pattern[str], float]] = [
    (re.compile(_NUM + r"\s*(?:min|mins|minutes?|мин|минут\w*)", re.I), 1 / 60),
    (re.compile(_NUM + r"\s*(?:h|hrs?|hours?|час\w*|ч\b)", re.I), 1.0),
    (re.compile(_NUM + r"\s*(?:d|days?|дн\w*|день|суток|сут\w*)", re.I), 24.0),
    (re.compile(r"(?:минут\w*)\s+" + _NUM, re.I), 1 / 60),  # "минут 40 назад"
    (re.compile(r"(?:час\w*)\s+" + _NUM, re.I), 1.0),  # "часов 5 назад"
    (re.compile(r"(?:дн\w*|сут\w*)\s+" + _NUM, re.I), 24.0),
]
# (pattern, min hours, max hours)
_PHRASES: list[tuple[re.Pattern[str], float, float]] = [
    (re.compile(r"just now|только что|сейчас", re.I), 0.1, 0.1),
    (re.compile(r"an hour ago|час назад", re.I), 1.0, 1.0),
    (re.compile(r"more than (a|one) day|больше суток|более суток|over 24", re.I), 25.0, 25.0),
    (re.compile(r"yesterday morning|вчера утром|со вчерашнего утра", re.I), 25.0, 34.0),
    (re.compile(r"yesterday|вчера", re.I), 12.0, 36.0),  # ambiguous around 24h -> the engine asks once more
    (re.compile(r"several days|few days|несколько дней|третий день|третьи сутки|неделю|week", re.I), 48.0, 48.0),
]


def parse_wait_range(text: str) -> tuple[float, float] | None:
    for rx, mult in _DURATIONS:
        m = rx.search(text)
        if m:
            v = float(m.group(1).replace(",", ".")) * mult
            return v, v
    for rx, lo, hi in _PHRASES:
        if rx.search(text):
            return lo, hi
    return None


def parse_wait_hours(text: str) -> float | None:
    r = parse_wait_range(text)
    return r[0] if r else None


def detect_script_language(text: str) -> str | None:
    """Very rough: only scripts we can tell apart without a model."""
    if re.search(r"[؀-ۿ]", text):
        return "ar"
    if re.search(r"[ሀ-፿]", text):
        return "am"
    if re.search(r"[өүңӨҮҢ]", text):
        return "ky"
    if re.search(r"[а-яА-ЯёЁ]", text):
        return "ru"
    if re.search(r"[a-zA-Z]", text):
        return "en"
    return None


def language_confident(text: str) -> bool:
    """Enough real words to trust the detected language (2+ words of 2+ letters)."""
    words = [w for w in re.findall(r"[^\W\d_]+", text) if len(w) >= 2]
    return len(words) >= 2


def detect_category(text: str) -> str | None:
    low = text.lower()
    for cat in (COMPLAINT, *CATEGORIES):
        if any(k in low for k in cat.keywords):
            return cat.key
    return None


class FakeLLM:
    """Keyword NLU + template texts. Good enough to drive the engine in tests."""

    def __init__(self) -> None:
        self.compose_calls: list[ComposeRequest] = []
        self.analyze_calls: list[AnalyzeRequest] = []

    async def analyze(self, req: AnalyzeRequest) -> Analysis:
        self.analyze_calls.append(req)
        text = req.text or ""
        complaint = bool(_COMPLAINT.search(text))
        category = detect_category(text)
        if category == COMPLAINT.key and not complaint:
            category = None
        payment: Literal["deposit", "withdrawal"] | None = None
        if _NOT_ARRIVED.search(text) or (req.awaiting == "wait_time"):
            cat = category if category in ("deposit", "withdrawal") else req.current_category
            if cat in ("deposit", "withdrawal"):
                payment = cat  # type: ignore[assignment]
        wants_operator = bool(_OPERATOR.search(text))
        confirmation = "yes" if _YES.search(text) else "no" if _NO.search(text) else None
        wait = parse_wait_range(text)
        noise = (not text.strip() and not req.media) or bool(text.strip() and _NOISE.match(text))
        off_topic = bool(_OFF_TOPIC.search(text)) and category is None
        risk: Risk | None = ("self_harm" if _RISK_SELF.search(text) else
                             "gambling_harm" if _RISK_GAMBLING.search(text) else
                             "minor" if _MINOR.search(text) else None)
        gratitude = bool(_THANKS.search(text)) and confirmation is None
        understood = bool(category or wants_operator or confirmation or gratitude or wait is not None
                          or off_topic or noise or risk or complaint)
        if not understood and req.current_category and req.awaiting is None and len(text.split()) <= 6:
            understood = True  # answer in context ("OPay", "I opened the cashier") -> continues the current problem
        signal: Signal
        if noise:
            signal = "noise"
        elif _NOT_HELPED.search(text) or confirmation == "no":
            signal = "not_helped"
        elif _FED_UP.search(text):
            signal = "dissatisfied"
        elif not understood:
            signal = "not_understood"
        elif gratitude and not category:
            signal = "closing"
        elif complaint or wants_operator:
            signal = "dissatisfied"
        elif category and category != req.current_category:
            signal = "new_request"
        elif any(t.role == "user" and t.text.strip().lower() == text.strip().lower() for t in req.history):
            signal = "repeat"
        else:
            signal = "progress"
        details: dict[str, str] = {}
        if m := _AMOUNT.search(text):
            details["amount"] = m.group(1) or m.group(2)
        if m := _METHOD.search(text):
            details["method"] = m.group(0)
        return Analysis(
            payment_details=details,
            language=detect_script_language(text),
            language_confident=language_confident(text),
            category=COMPLAINT.key if complaint and not payment and category in (None, COMPLAINT.key) else category,
            signal=signal,
            understood=understood,
            wants_operator=wants_operator,
            payment_issue=payment,
            wait_hours=wait[0] if wait else None,
            wait_hours_max=wait[1] if wait else None,
            confirmation=confirmation,  # type: ignore[arg-type]
            gratitude=gratitude,
            aggressive=bool(_AGGRESSIVE.search(text)),
            complaint=complaint,
            off_topic=off_topic,
            noise=noise,
            risk=risk,
            media_understood=not req.media or bool(text.strip()),
        )

    async def compose(self, req: ComposeRequest) -> Composed:
        self.compose_calls.append(req)
        intro = f"Hi! My name is {req.persona.name}. " if req.introduce else ""
        lang = f"[{req.language}] "
        t = req.task
        if t is Task.ANSWER:
            if req.kb:
                return Composed(lang + intro + req.kb[0].answer, "Full")
            cat = BY_KEY.get(req.category or "")
            opts = tuple(Option(cat.question_id(i), q) for i, q in enumerate(cat.questions)) if cat else ()
            return Composed(
                lang + intro + f"(attempt {req.attempt}) Could you tell me a bit more about the problem?",
                "NotFound",
                opts if req.attempt >= 4 else (),
            )
        texts = {
            Task.CLARIFY_UNRECOGNIZED: ("Sorry, I didn't quite get your question. Could you describe it in more detail?", "NotFound"),
            Task.NUDGE: ("I'm here 🙂 Please pick one of the options or tell me what happened.", "Partial"),
            Task.OFF_TOPIC: ("I can only help with AvaBet questions 🙂 What can I help you with?", "NotFound"),
            Task.DEFLECT_OPERATOR: ("I'll try to help you myself. What problem did you face?", "Partial"),
            Task.ASK_WAIT_TIME: ("How much time has passed since the payment / withdrawal request?", "Partial"),
            Task.CLARIFY_WAIT_TIME: ("Could you say more precisely how many hours have passed?", "Partial"),
            Task.PAYMENT_WITHIN_NORM: ("This is still within the usual processing time.", "Full"),
            Task.COLLECT_PAYMENT_DETAILS: (f"To pass it to a specialist I need one more detail: {req.missing_detail}.", "Partial"),
            Task.ASK_RESOLVED: ("Did that solve your question?", "Full"),
            Task.GOODBYE: ("Glad I could help 😊 If you have other questions, just write.", "Full"),
            Task.HANDOFF: ("I'm passing your request to a support specialist, they will reply here.", "Partial"),
            Task.RISK_SUPPORT: ("I hear you, and I'm sorry it's so hard right now. A colleague will join this chat.", "Partial"),
            Task.MEDIA_UNREADABLE: ("I couldn't read the file. Could you describe the problem in text?", "NotFound"),
        }
        text, result = texts[t]
        if t is Task.PAYMENT_WITHIN_NORM and req.kb:
            text = req.kb[0].answer
        return Composed(lang + intro + text, result)  # type: ignore[arg-type]

    async def localize(self, labels: list[str], language: str) -> list[str]:
        return list(labels)
