"""Dialogue engine: deterministic policy around the LLM.

One turn = (several glued user messages) -> analyze (LLM, strict JSON) ->
policy (this file, plain code) -> compose (LLM) -> fact check (guard.py) ->
actions for the platform adapter.

What the code decides (never the model):
  ТЗ 8    reply language (switch only on a confident detection, see below)
  ТЗ 10   persona from the GEO pool, fixed per dialogue, introduces itself once
  ТЗ 13   "call an operator" is not an escalation by itself
  ТЗ 14.3 group A: counters PER PROBLEM; only failure signals count as an
          attempt (not helped / not understood / repeat / dissatisfied /
          operator request). Answering a clarifying question is progress.
          The 6th failure (= 5 unsuccessful attempts were made) -> L1.
  ТЗ 14.4 group B: waiting time as a range; ambiguous around 24h -> ask once
          more; > 24h -> collect payment details (dialogue 4) -> L1;
          < 24h -> standard terms, never escalated; time question not
          answered twice or silence after it -> L1
  ТЗ 14.5 group D and off-topic never escalate; aggression is a tone, not a
          topic; an actual complaint goes to the complaint rules
  ТЗ 14.6 category above its target escalation share -> extra attempts
  ТЗ 16   "did that solve your question?" -> goodbye -> resolve
  (extra) responsible gaming / self-harm signals -> caring reply + L1 at once
"""
from __future__ import annotations

import copy
import datetime as dt
import json
import logging
import re
from pathlib import Path
from dataclasses import dataclass, field, replace
from zoneinfo import ZoneInfo

from .actions import HandOff, NoteToOperator, Option, Outcome, Resolve, Say, SetLanguage
from .budget import EscalationBudget
from .catalog import BY_KEY, CATEGORIES, COMPLAINT, EscalationGroup, resolve_variant
from .guard import check_reply, promises_handoff
from ..monitoring import note
from .kb import KnowledgeBase
from .language import DEFAULT_LANGUAGE, normalize
from .llm import (
    FAILURE_SIGNALS,
    LLM,
    Analysis,
    AnalyzeRequest,
    Composed,
    ComposeRequest,
    Media,
    Task,
    Turn,
)
from .personas import PersonaRegistry
from .state import UNKNOWN_ISSUE, ContactMemory, Dialogue, DialogueStore, Issue, PaymentCase
from .timeparse import COUNTRY_TZ, hours_since
from .translation import LanguagePolicy, NoTranslator, Translator

log = logging.getLogger(__name__)

GREETING = (
    "Welcome to AvaBet! 👋 I'll be happy to help you with any questions 😊\n"
    "Please choose a category below or type your question."
)

WAIT_OPTIONS = (
    (Option("wait:lt2", "< 2 hours"), 1.0),
    (Option("wait:2to24", "2-24 hours"), 12.0),
    (Option("wait:gt24", "> 24 hours"), 25.0),
)
CONFIRM_OPTIONS = (Option("confirm:yes", "Yes"), Option("confirm:no", "No"))
# Test review п.18/22: the most frequent requests. (id, English label, category, sub-question)
TOP_REQUESTS = (
    ("top:deposit_missing", "Deposit not received", "deposit", "Deposit not received"),
    ("top:how_deposit", "How to top up", "deposit", "How to make deposit?"),
    ("top:payment_status", "Payment request status", "deposit", "Payment request status"),
    ("top:withdrawal_missing", "Withdrawal not received", "withdrawal", "Withdrawal not received"),
    ("top:login", "Can't log in", "account", "Login issues"),
    ("top:bonus", "Bonuses & promo codes", "bonuses", "Bonuses and promo codes"),
)
TOP_BY_ID = {t[0]: t for t in TOP_REQUESTS}
GREETING_TOPICS = ("top:deposit_missing", "top:withdrawal_missing", "top:login")
PAYMENT_KIND_OPTIONS = (Option("pay:failed", "Payment failed"), Option("pay:charged", "Charged, not on balance"))
OTHER_METHOD = "Other method"
# Test review п.4: the bot never offers "operator" / "supervisor" buttons itself.
_OPERATOR_LABEL = re.compile(
    r"operat|supervis|оператор|супервайз|сотрудник|живой человек|human|real person|مشرف|موظف|opérateur|"
    r"superviseur|humano|operador|operatör|yetkili|temsilci|msimamizi|mhudumu", re.I)
_MARKDOWN = re.compile(r"\*\*(.+?)\*\*|__(.+?)__", re.S)
_SENTENCE_END = re.compile(r"(?<=[.!…。])\s+|\n+")
DATA = Path(__file__).resolve().parents[1] / "data"


def _split_sentences(text: str) -> list[str]:
    return _SENTENCE_END.split(text)


def load_payment_methods(path: Path = DATA / "payment_methods.json") -> dict[str, list[str]]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {geo.upper(): list(v.get("deposit", [])) for geo, v in raw.get("geo", {}).items()}
# "Complain about operator" / "File a complaint" live under Contacts (group D) but are complaints (14.1).
COMPLAINT_SUBQUESTIONS = {("contacts", 3), ("contacts", 4)}
PAYMENT_KINDS = ("deposit", "withdrawal")
UNLIMITED = 10**9
FALLBACK_TEXTS = {
    Task.ANSWER: "Could you tell me a bit more about your question?",
    Task.HANDOFF: "I'm passing your request to a support colleague, they will continue in this chat.",
    Task.RISK_SUPPORT: "Thank you for telling me. A support colleague will join this chat now to help you.",
    Task.GREET: "Hello! How can I help you?",
    Task.RESPONSIBLE_GAMING: "Please don't try to win losses back with a new deposit. You can set limits or a "
                             "self-exclusion in the Responsible Gaming section of your profile.",
    Task.CLARIFY_PAYMENT_KIND: "Did the payment fail with an error, or was the money charged but not credited?",
    Task.ACCOUNT_NUMBER: "Thank you! What is your question about this account?",
    Task.ASK_WAIT_TIME: "How much time has passed since the payment?",
    Task.CLARIFY_WAIT_TIME: "Roughly how many hours have passed since the payment?",
    Task.ASK_RESOLVED: "Did that solve your question?",
    Task.GOODBYE: "Glad I could help! Write any time if you have other questions.",
    Task.COLLECT_PAYMENT_DETAILS: "Could you share the payment details: method, amount and time?",
}


@dataclass(frozen=True)
class IncomingMessage:
    chat_id: str
    text: str | None = None
    media: tuple[Media, ...] = ()
    variant_id: str | None = None  # button / row click
    message_id: int | None = None
    conversation_id: str | None = None
    language_code: str | None = None
    country: str | None = None
    timezone: str | None = None
    channel: str | None = None
    client_id: str | None = None


@dataclass(frozen=True)
class PlatformMessage:
    """A message from the platform's chat history (Consultant GET /chats/{id}/messages)."""
    id: int | None
    text: str
    from_id: str | None  # None = the client; otherwise an operator or the bot itself
    created_at: str | None = None


@dataclass
class EngineConfig:
    max_attempts: int = 5  # group A, ТЗ 14.3
    complaint_attempts: int = 5  # test review п.1: complaints go through the normal cycle, no early L1
    extra_attempts_over_budget: int = 2  # ТЗ 14.6
    payment_threshold_hours: float = 24.0  # group B, ТЗ 14.4
    max_unanswered_wait: int = 3  # test review п.2: 3 messages in a row without the time -> L1
    usual_payment_hours: dict = field(default_factory=lambda: {"deposit": 24.0})  # test review п.3: one term, 24h
    payment_silence_seconds: float = 0.0  # test review п.2: silence is not a reason for L1 (the timer is off)
    # test review «Какие данные бот собирает по платежам»: at most 4 questions, amount and time asked together
    payment_detail_groups: tuple[tuple[str, ...], ...] = (("method",), ("amount", "paid_at"), ("wallet",),
                                                          ("receipt",))
    markdown: bool = False  # test review п.16: Consultant shows "**" as is -> plain text
    payment_methods: dict = field(default_factory=load_payment_methods)  # GEO -> methods (Google sheet)
    risk_escalates: bool = True  # explicit addiction / self-exclusion / self-harm -> L1 at once (test review п.1)
    resolve_after_seconds: int | None = None
    history_limit: int = 30
    fact_check: bool = True


@dataclass
class TurnResult:
    outcome: Outcome
    dialogue: Dialogue | None  # the new state; commit only if the result is actually sent

    @property
    def silence_timeout(self) -> float | None:
        return self.outcome.silence_timeout


@dataclass
class _Ctx:
    d: Dialogue
    text: str  # what the model reads (in the pivot language in pivot mode)
    media: tuple[Media, ...]
    reply_to: int | None
    analysis: Analysis
    pivot: bool = False
    out: Outcome = field(default_factory=Outcome)


class Engine:
    def __init__(
        self,
        llm: LLM,
        kb: KnowledgeBase,
        personas: PersonaRegistry,
        store: DialogueStore,
        config: EngineConfig | None = None,
        *,
        budget: EscalationBudget | None = None,
        translator: Translator | None = None,
        language_policy: LanguagePolicy | None = None,
        contact_memory: ContactMemory | None = None,
        clock=None,
    ):
        self.llm = llm
        self.kb = kb
        self.personas = personas
        self.store = store
        self.cfg = config or EngineConfig()
        self.budget = budget
        self.translator = translator or NoTranslator()
        self.policy = language_policy or LanguagePolicy()
        self.memory = contact_memory
        self._clock = clock or (lambda: dt.datetime.now(dt.timezone.utc))

    # ------------------------------------------------------------ lifecycle

    async def start_dialogue(self, chat_id: str, conversation_id: str | None = None) -> Outcome:
        d = Dialogue(chat_id=chat_id, conversation_id=conversation_id)
        if self.memory:
            d.previous_contact = self.memory.recall(chat_id)
        await self.store.save(d)
        options = tuple(Option(c.variant_id, c.label) for c in CATEGORIES)
        return Outcome().add(Say(GREETING, options))

    async def end_dialogue(self, chat_id: str, reason: str | None = None) -> None:
        log.info("dialogue %s ended: %s", chat_id, reason)
        d = await self.store.get(chat_id)
        if d is not None:
            if self.budget and d.current and not d.handed_off:
                self.budget.record(d.current, escalated=False)
            if self.memory and d.history:
                self.memory.remember(chat_id, self._contact_note(d, reason))
        await self.store.delete(chat_id)

    async def mark_unreachable(self, chat_id: str) -> None:
        """Consultant answered 403: we can no longer write to the chat."""
        d = await self.store.get(chat_id)
        if d:
            d.handed_off = True
            await self.store.save(d)

    async def commit(self, result: TurnResult) -> None:
        if result.dialogue is not None:
            await self.store.save(result.dialogue)

    # ------------------------------------------------------------- messages

    async def handle_messages(self, msgs: list[IncomingMessage],
                              platform_history: list[PlatformMessage] | None = None) -> Outcome:
        """run_turn + commit (callers that never discard a turn)."""
        result = await self.run_turn(msgs, platform_history)
        await self.commit(result)
        return result.outcome

    async def run_turn(self, msgs: list[IncomingMessage],
                       platform_history: list[PlatformMessage] | None = None) -> TurnResult:
        """Compute the reply to one or more consecutive user messages WITHOUT saving the state.

        The dispatcher commits only if no newer message arrived meanwhile; otherwise it
        throws the result away and recomputes with all messages (one answer, counted once).
        """
        if not msgs:
            return TurnResult(Outcome(), None)
        last = msgs[-1]
        stored = await self.store.get(last.chat_id)
        d = copy.deepcopy(stored) if stored is not None else None
        if d is None:  # no SetChat seen (e.g. restart) - start a dialogue silently
            d = Dialogue(chat_id=last.chat_id, conversation_id=last.conversation_id)
            if self.memory:
                d.previous_contact = self.memory.recall(last.chat_id)
        if not d.active:
            return TurnResult(Outcome(), None)

        self._update_user_data(d, msgs)
        if platform_history:
            self._sync_platform_history(d, platform_history, {m.message_id for m in msgs})
            if d.operator_joined:
                log.info("chat %s: a live operator is in the chat, the bot steps back", d.chat_id)
                return TurnResult(Outcome(), d)

        texts = [m.text.strip() for m in msgs if m.text and m.text.strip()]
        media = tuple(x for m in msgs for x in m.media)
        variant = next((m.variant_id for m in reversed(msgs) if m.variant_id), None)
        typed_text = "\n".join(texts)

        forced: dict = {}
        if variant:
            forced = self._parse_variant(variant, d)
            if "show_category" in forced:
                out = await self._show_category(d, forced["show_category"])
                return TurnResult(out, d)

        text = forced.get("question") or forced.get("label") or typed_text
        current_lang = d.language or d.platform_language
        pivot = bool(text) and (self.policy.translate_everything or self.policy.mode(current_lang) == "pivot")
        model_text, detected = text, None
        if pivot:
            model_text, detected = await self.translator.to_pivot(text, self.policy.pivot, current_lang)

        analysis = await self.llm.analyze(AnalyzeRequest(
            text=model_text, media=media, history=tuple(d.history[-self.cfg.history_limit:]),
            current_category=d.issue.category if d.current else None, awaiting=d.awaiting,
            current_language=d.language, now_local=self._now_local(d), previous_contact=d.previous_contact,
            asked_detail=(d.payment.details_asked[-1] if d.awaiting == "payment_details" and d.payment
                          and d.payment.details_asked else None),
            geo_methods=tuple(self._geo_methods(d)), timezone_known=self._tz(d) is not None,
        ))
        if pivot:
            analysis = replace(analysis, language=detected or current_lang, language_confident=detected is not None,
                               english_text=model_text)
            if detected == self.policy.pivot and not self.policy.translate_everything:
                pivot = False
        analysis = self._apply_forced(analysis, forced)

        d.last_signal = analysis.signal
        ctx = _Ctx(d=d, text=model_text, media=media, reply_to=last.message_id, analysis=analysis, pivot=pivot)
        self._update_language(ctx, from_text=bool(typed_text) and not variant)
        if d.persona is None:
            d.persona = self.personas.assign(d.country)
        d.history.append(Turn("user", model_text or ("[file]" if media else "")))
        d.offered = {}

        await self._route(ctx)
        return TurnResult(ctx.out, d)

    async def on_silence(self, chat_id: str, asked_at: float | None) -> Outcome:
        """The user did not answer "how long ago?" (ТЗ 14.4: do not leave a payment case unattended)."""
        d = await self.store.get(chat_id)
        if d is None or not d.active or d.awaiting != "wait_time" or d.awaiting_since != asked_at:
            return Outcome()
        ctx = _Ctx(d=d, text="", media=(), reply_to=None, analysis=Analysis())
        await self._escalate(ctx, reason=f"{d.payment.kind if d.payment else 'payment'} not received, "
                                         "no answer about the waiting time")
        await self.store.save(d)
        return ctx.out

    # ---------------------------------------------------------------- policy

    async def _route(self, ctx: _Ctx) -> None:
        d, a = ctx.d, ctx.analysis

        # Safety first. Test review п.1: only explicit addiction / self-exclusion / self-harm goes to L1 at once;
        # complaints about losses and "deposit to win it back?" are answered by the bot.
        if a.risk in ("gambling_harm", "self_harm") and self.cfg.risk_escalates:
            await self._escalate(ctx, reason=f"risk signal: {a.risk}", task=Task.RISK_SUPPORT)
            return
        if a.risk == "minor":
            d.switch_issue("rules")
            await self._say(ctx, Task.ANSWER, kb=self._kb("age restrictions minimum age", "rules"))
            return
        if a.chasing_losses:
            await self._say(ctx, Task.RESPONSIBLE_GAMING,
                            kb=self._kb("ответственная игра лимиты самоисключение", "rules"))
            return

        # ТЗ 16: closing the dialogue
        if d.awaiting == "confirmation":
            d.awaiting = None
            if a.confirmation == "yes" or (a.gratitude and not a.wants_operator and not a.category):
                await self._say(ctx, Task.GOODBYE)
                ctx.out.add(Resolve(self.cfg.resolve_after_seconds))
                d.closed = True
                if self.budget and d.current:
                    self.budget.record(d.current, escalated=False)
                return
            if a.confirmation == "no":
                ctx.analysis = a = replace(a, signal="not_helped")
        if self._is_closing(a) and d.current and d.issue.last_result in ("Full", "Partial"):
            d.awaiting = "confirmation"
            await self._say(ctx, Task.ASK_RESOLVED, options=CONFIRM_OPTIONS)
            return

        # Test review п.19: "my deposit failed" - an error, or charged but not credited? Ask first.
        if a.payment_unclear and not a.payment_issue and (d.payment is None or not d.payment.over_threshold):
            d.switch_issue(a.category if a.category in PAYMENT_KINDS else "deposit")
            await self._say(ctx, Task.CLARIFY_PAYMENT_KIND, options=PAYMENT_KIND_OPTIONS)
            return

        # ТЗ 14.4: group B, payment cases
        if a.payment_issue and (d.payment is None or d.payment.kind != a.payment_issue):
            d.payment = PaymentCase(kind=a.payment_issue)
            d.switch_issue(a.payment_issue)
        if self._in_payment_flow(d, a):
            await self._payment_flow(ctx)
            return

        # Off-topic and pure noise never count as attempts and never escalate.
        if a.off_topic and not a.category and not a.wants_operator:
            d.off_topic_count += 1
            await self._say(ctx, Task.OFF_TOPIC, options=self._ladder_options(d) if not d.current else ())
            return
        if a.account_number and not a.category and not a.wants_operator:  # п.20: only an account number
            d.account_number = a.account_number
            await self._say(ctx, Task.ACCOUNT_NUMBER, options=self._top_options(GREETING_TOPICS))
            return
        if self._is_noise(a) or (a.greeting and not (a.category or a.wants_operator or a.payment_issue)):
            d.noise_count += 1
            if d.noise_count == 1 and not d.current:  # п.18: a greeting gets a greeting, top topics as buttons
                await self._say(ctx, Task.GREET, options=self._top_options(GREETING_TOPICS))
            else:
                await self._say(ctx, Task.NUDGE, options=self._ladder_options(d) if not d.current else ())
            return

        # Which problem is this about? (ТЗ 15.2 / test review п.21: only a NEW problem named by the user starts a
        # new cycle; "yes" / "no" / a short answer to the bot's question continues the current one)
        new_topic = a.category and a.category != d.current and (
            d.current in (None, UNKNOWN_ISSUE) or a.signal == "new_request" or a.category in d.issues
            or a.confirmation is None
            and a.signal not in ("progress", "not_helped", "repeat", "dissatisfied", "not_understood"))
        if new_topic:
            d.switch_issue(a.category)
        elif d.current is None:
            d.switch_issue(UNKNOWN_ISSUE)
        issue = d.issue
        if d.payment and issue.key != d.payment.kind and issue.key in PAYMENT_KINDS:
            d.payment = None
        d.awaiting = None

        # ТЗ 14.3 cycle: only failures count
        if self._is_failure(a):
            if issue.failures >= self._attempt_limit(d, issue):
                await self._escalate(ctx, reason=f"{issue.failures} unsuccessful attempts on '{issue.key}'")
                return
            issue.failures += 1
        issue.turns += 1

        if ctx.media and not a.media_understood:
            await self._say(ctx, Task.MEDIA_UNREADABLE)
        elif a.wants_operator and issue.key == UNKNOWN_ISSUE:
            await self._say(ctx, Task.DEFLECT_OPERATOR, options=self._ladder_options(d))
        elif not a.understood and issue.key == UNKNOWN_ISSUE:
            await self._say(ctx, Task.CLARIFY_UNRECOGNIZED, options=self._ladder_options(d))
        else:
            await self._say(ctx, Task.ANSWER, kb=self._kb(ctx.text, issue.category))

    def _ladder_options(self, d: Dialogue) -> tuple[Option, ...]:
        """Test review п.22: buttons change from attempt to attempt while the topic is unknown.

        1st time: the 12 categories of the ТЗ; 2nd: the 6 most popular requests; then () - the model offers
        3-4 likely requests from what the user already wrote.
        """
        d.unknown_prompts += 1
        if d.unknown_prompts == 1:
            return self._category_options()
        if d.unknown_prompts == 2:
            return self._top_options(tuple(t[0] for t in TOP_REQUESTS))
        return ()

    @staticmethod
    def _top_options(ids: tuple[str, ...]) -> tuple[Option, ...]:
        return tuple(Option(i, TOP_BY_ID[i][1]) for i in ids)

    def _geo_methods(self, d: Dialogue) -> list[str]:
        return list(self.cfg.payment_methods.get((d.country or "").upper(), ()))

    def _method_options(self, d: Dialogue) -> tuple[Option, ...]:
        methods = self._geo_methods(d)
        if not methods:
            return ()  # unknown GEO list: no generic "card / bank / crypto" buttons, the user types it
        return tuple(Option(f"method:{i}", m) for i, m in enumerate(methods)) + (Option("method:other", OTHER_METHOD),)

    def _in_payment_flow(self, d: Dialogue, a: Analysis) -> bool:
        p = d.payment
        if p is None or p.within_norm:
            return False
        switched = a.category is not None and a.category not in (p.kind, COMPLAINT.key) and a.signal == "new_request"
        if switched:
            return False
        return bool(a.payment_issue) or d.awaiting in ("wait_time", "payment_details") or p.over_threshold

    def _wait_range(self, d: Dialogue, a: Analysis) -> tuple[float, float] | None | str:
        """Hours since the payment as (min, max); "no_tz" if a calendar answer cannot be converted."""
        if a.payment_when:
            tz = self._tz(d)
            if tz is None:
                return "no_tz"
            when = dict(a.payment_when)
            p = d.payment
            if p is not None:
                if not any(when.get(k) is not None for k in ("date", "days_ago", "weekday")):
                    # "yesterday" ... "at 9 am": the day from the previous answer, the hour from this one
                    when = {**{k: v for k, v in p.when.items() if k in ("date", "days_ago", "weekday")}, **when}
                p.when = when
            rng = hours_since(when, self._clock().astimezone(tz))
            if rng is not None:
                return rng
        if a.wait_hours is not None:
            return a.wait_hours, a.wait_hours_max if a.wait_hours_max is not None else a.wait_hours
        return None

    async def _payment_flow(self, ctx: _Ctx) -> None:
        d, a = ctx.d, ctx.analysis
        p = d.payment
        assert p is not None
        d.switch_issue(p.kind)
        self._merge_details(ctx)

        if p.over_threshold:  # > 24h: collecting details for L1 (dialogue 4)
            await self._collect_details(ctx)
            return

        wait = self._wait_range(d, a)
        if wait == "no_tz":  # п.5: no GEO -> no time zone -> ask for hours, not a clock time
            self._await(d, "wait_time")
            await self._say(ctx, Task.CLARIFY_WAIT_TIME, options=tuple(o for o, _ in WAIT_OPTIONS),
                            facts=("the user's time zone is unknown: ask roughly how many hours have passed since "
                                   "the payment (not the date or the clock time)",))
            return
        if isinstance(wait, tuple):
            lo, hi = wait
            p.unanswered = 0
            if lo <= self.cfg.payment_threshold_hours < hi:  # both sides of 24h: ask for the hour
                if p.clarify_asked < 1:
                    p.clarify_asked += 1
                    self._await(d, "wait_time")
                    await self._say(ctx, Task.CLARIFY_WAIT_TIME, options=tuple(o for o, _ in WAIT_OPTIONS),
                                    facts=(f"from the user's answer it is between {lo:g} and {hi:g} hours: ask the "
                                           "approximate hour of the payment, or how many hours have passed",))
                    return
                lo = hi  # still unclear: in favour of the client
            p.wait_hours = lo
            d.awaiting = None
            if lo > self.cfg.payment_threshold_hours:
                p.over_threshold = True
                await self._collect_details(ctx, first=True)
                return
            p.within_norm = True
            await self._say(ctx, Task.PAYMENT_WITHIN_NORM, kb=self._kb(f"{p.kind} processing time", p.kind),
                            facts=self._deadline_facts(d, lo, p.kind))
            return

        if a.signal == "closing" and not a.payment_issue:  # "ok thanks" instead of the time: let them go
            d.awaiting = "confirmation"
            await self._say(ctx, Task.ASK_RESOLVED, options=CONFIRM_OPTIONS)
            return
        if a.wants_operator or d.awaiting == "wait_time":
            # п.2: "الو", "?", "operator" are not a refusal. Only 3 messages in a row without the time -> L1.
            if a.wants_operator:
                d.issue.failures += 1
            p.unanswered += 1
            if p.unanswered >= self.cfg.max_unanswered_wait:
                await self._escalate(ctx, reason=f"{p.kind} not received, {p.unanswered} messages in a row "
                                                 "without the waiting time")
                return
        p.wait_questions_asked += 1
        repeat = p.wait_questions_asked > 1
        self._await(d, "wait_time")
        await self._say(ctx, Task.ASK_WAIT_TIME, options=tuple(o for o, _ in WAIT_OPTIONS),
                        facts=("you already asked this: ask again in ONE short sentence and point to the buttons",)
                        if repeat else ())
        ctx.out.silence_timeout = self.cfg.payment_silence_seconds or None

    def _merge_details(self, ctx: _Ctx) -> None:
        """Whatever the user told so far, in any order (test review: answers are accepted in any order)."""
        d, a = ctx.d, ctx.analysis
        p = d.payment
        assert p is not None
        for k, v in a.payment_details.items():
            if v and k in ("method", "amount", "paid_at", "wallet"):
                p.details[k] = v
        if ctx.media:
            p.details["receipt"] = "received (attached in the chat)"

    async def _collect_details(self, ctx: _Ctx, first: bool = False) -> None:
        """> 24h: collect what L1 needs before the hand-off (test review «Какие данные бот собирает»).

        At most 4 questions: method (buttons of the GEO) / amount + date and time / wallet / receipt (deposit) or
        the request screenshot from the AvaBet history (withdrawal). A field the user already gave is skipped,
        answers are accepted in any order, the receipt is asked again once, then the case goes without it.
        """
        d, a = ctx.d, ctx.analysis
        p = d.payment
        assert p is not None
        self._merge_details(ctx)
        if first and p.wait_hours is not None and "paid_at" not in p.details and a.payment_when:
            p.details["paid_at"] = self._describe_when(a.payment_when)
        asked = p.details_asked[-1] if p.details_asked else None
        pushback = not first and (a.wants_operator or a.signal in ("dissatisfied", "repeat")
                                  or (a.aggressive and not ctx.media))
        if not first and asked and not pushback and not a.payment_details and not ctx.media and ctx.text:
            if asked == "receipt":
                p.details.setdefault("receipt_note", ctx.text[:150])  # "will send later", "I have none"
            else:  # the answer to what we just asked ("amount+paid_at" in one go: keep the words as the time)
                fields = asked.split("+")
                p.details.setdefault("paid_at" if "paid_at" in fields else fields[0], ctx.text[:200])
        if pushback:
            p.pushback += 1
            if p.pushback >= 2 or not asked:
                await self._escalate(ctx, reason=f"{p.kind} not received for ~{p.wait_hours:g}h (> 24h)")
                return
            self._await(d, "payment_details")
            await self._say(ctx, Task.COLLECT_PAYMENT_DETAILS, missing=asked, options=self._detail_options(d, asked),
                            facts=("the user is irritated and did not answer: acknowledge the feeling, explain in one "
                                   "sentence why exactly this item helps to find the payment faster, then ask it in "
                                   "OTHER words than your previous message",))  # п.11/12
            return

        methods = self._geo_methods(d)
        method = p.details.get("method")
        if method and methods and method not in methods and not p.method_clarified and method != OTHER_METHOD:
            # п.6: the user's words did not match a method of the GEO -> clarify with buttons, once
            p.method_clarified = True
            p.details["method_user_words"] = method
            del p.details["method"]
            self._await(d, "payment_details")
            p.details_asked.append("method")
            await self._say(ctx, Task.COLLECT_PAYMENT_DETAILS, missing="method", options=self._method_options(d),
                            facts=(f"the user described the method as «{method}»: ask which of the buttons it is",))
            return

        for group in self.cfg.payment_detail_groups:
            todo = [f for f in group if f not in p.details]
            if not todo:
                continue
            if group == ("receipt",):
                if p.receipt_asks >= 2 or (p.receipt_asks == 1 and asked != "receipt"):
                    continue  # asked twice: hand over without it (п.10)
                p.receipt_asks += 1
                again = p.receipt_asks == 2
                p.details_asked.append("receipt")
                self._await(d, "payment_details")
                await self._say(ctx, Task.COLLECT_PAYMENT_DETAILS, missing="receipt",
                                facts=("the receipt has NOT been received yet: ask for it once more, briefly; if the "
                                       "user cannot send it, the case goes to a colleague without it",) if again else ())
                return
            asked_before = {f for x in p.details_asked for f in x.split("+")}
            if any(f in asked_before for f in todo):
                continue  # every item is asked once
            missing = "+".join(todo)
            p.details_asked.append(missing)
            self._await(d, "payment_details")
            await self._say(ctx, Task.COLLECT_PAYMENT_DETAILS, missing=missing,
                            options=self._detail_options(d, missing))
            return
        await self._escalate(ctx, reason=f"{p.kind} not received for ~{p.wait_hours:g}h (> 24h)")

    def _detail_options(self, d: Dialogue, missing: str | None) -> tuple[Option, ...]:
        return self._method_options(d) if missing == "method" else ()

    @staticmethod
    def _describe_when(when: dict) -> str:
        parts = [str(when[k]) for k in ("date", "weekday", "time", "part_of_day") if when.get(k)]
        if when.get("days_ago") is not None:
            parts.insert(0, f"{when['days_ago']} day(s) ago")
        return " ".join(parts) + " (client's words)"

    def _attempt_limit(self, d: Dialogue, issue: Issue) -> int:
        if d.payment and d.payment.within_norm and issue.key == d.payment.kind:
            return UNLIMITED  # < 24h payment case is not escalated
        cat = BY_KEY.get(issue.key)
        group = cat.group if cat else EscalationGroup.A
        if group is EscalationGroup.D:
            return UNLIMITED
        limit = self.cfg.complaint_attempts if group is EscalationGroup.COMPLAINT else self.cfg.max_attempts
        if self.budget and self.budget.over_target(issue.key):
            limit += self.cfg.extra_attempts_over_budget
        return limit

    async def _escalate(self, ctx: _Ctx, reason: str, task: Task = Task.HANDOFF) -> None:
        """SupportBotMessage + goodbye BEFORE InvokeOperator (after it Consultant returns 403)."""
        d = ctx.d
        d.escalation_reason = reason
        ctx.out.add(NoteToOperator(self._operator_summary(d, reason, ctx.analysis)))
        facts: tuple[str, ...] = ()
        if d.payment and d.payment.over_threshold:
            got = [k for k in ("method", "amount", "paid_at", "wallet") if k in d.payment.details]
            receipt = "received" if "receipt" in d.payment.details else "NOT received - do not say it is attached"
            facts = (f"collected for the colleague: {', '.join(got) or 'nothing'}; receipt/screenshot: {receipt}",)
        await self._say(ctx, task, check=False, facts=facts)
        ctx.out.add(HandOff())
        d.handed_off = True
        d.awaiting = None
        if self.budget and d.current:
            self.budget.record(d.current, escalated=True)

    # --------------------------------------------------------------- helpers

    @staticmethod
    def _is_failure(a: Analysis) -> bool:
        if a.wants_operator:
            return True  # ТЗ 13 / 14.2: an operator request goes through the group A cycle
        if a.signal is not None:
            return a.signal in FAILURE_SIGNALS
        return not a.understood

    @staticmethod
    def _is_noise(a: Analysis) -> bool:
        return (a.noise or a.signal == "noise") and not (a.category or a.wants_operator or a.payment_issue)

    @staticmethod
    def _is_closing(a: Analysis) -> bool:
        if a.signal == "closing":  # the model may keep topic / payment flags from the context
            return not a.wants_operator and a.confirmation != "no"
        return a.gratitude and not a.category and not a.wants_operator and not a.payment_issue \
            and a.confirmation != "no"

    def _await(self, d: Dialogue, what: str) -> None:
        d.awaiting = what
        d.awaiting_since = self._clock().timestamp()

    def _kb(self, query: str, category: str | None) -> tuple:
        return tuple(self.kb.search(query, category)) if query else ()

    @staticmethod
    def _category_options() -> tuple[Option, ...]:
        return tuple(Option(c.variant_id, c.label) for c in CATEGORIES)

    async def _say(self, ctx: _Ctx, task: Task, options: tuple[Option, ...] = (), kb: tuple = (),
                   facts: tuple[str, ...] = (), missing: str | None = None, check: bool = True) -> None:
        d, a = ctx.d, ctx.analysis
        assert d.persona is not None and d.language is not None
        lang = self.policy.pivot if ctx.pivot else d.language
        req = ComposeRequest(
            task=task,
            persona=d.persona,
            language=lang,
            user_text=ctx.text,
            introduce=not d.introduced,
            attempt=max(1, min(d.issue.attempt, self.cfg.max_attempts)),
            category=d.issue.category if d.current else None,
            payment_kind=d.payment.kind if d.payment else None,
            wait_hours=d.payment.wait_hours if d.payment else None,
            channel=d.channel,
            country=d.country,
            kb=kb,
            history=tuple(d.history[-self.cfg.history_limit:]),
            previous_bot_texts=d.bot_texts(),
            aggressive=a.aggressive,
            facts=facts + self._context_facts(d, task),
            missing_detail=missing,
            previous_contact=d.previous_contact,
        )
        try:
            composed = await self.llm.compose(req)
            if check and self.cfg.fact_check:
                composed = await self._checked(req, composed)
        except Exception:  # model down / refusal: a plain but correct message instead of silence
            log.exception("compose failed for task %s", task.value)
            composed = await self._fallback(req)
        d.introduced = True
        if task in (Task.ANSWER, Task.PAYMENT_WITHIN_NORM) and d.current:
            d.issues[d.current].last_result = composed.result
        opts = options or tuple(o for o in composed.options if not _OPERATOR_LABEL.search(o.label)
                                and not self._mixed_script(o.label, d))  # п.4, п.24
        if task in (Task.HANDOFF, Task.GOODBYE, Task.RISK_SUPPORT):
            opts = ()  # the bot leaves the chat / the dialogue is over: nothing to tap
        d.offered = {o.id: o.label for o in opts if not o.id.startswith(("cat:", "q:", "wait:", "confirm:"))}
        text = self._finish_text(composed.text, task, d)
        d.history.append(Turn("bot", text))
        if ctx.pivot and d.language != self.policy.pivot:
            translated = await self.translator.from_pivot([text, *[o.label for o in opts]], self.policy.pivot,
                                                          d.language)
            text, labels = self._finish_text(translated[0], task, d), translated[1:]
            opts = tuple(Option(o.id, lbl) for o, lbl in zip(opts, labels))
        else:
            # model buttons are already localized; payment method names stay as they are
            fixed = tuple(o for o in opts if not o.id.startswith("free:")
                          and not (o.id.startswith("method:") and o.id != "method:other"))
            localized = dict(zip([o.id for o in fixed], await self._localize(fixed, d.language)))
            opts = tuple(localized.get(o.id, o) for o in opts)
        ctx.out.add(Say(text, opts, ctx.reply_to, composed.result, d.issue.category if d.current else None))

    def _mixed_script(self, label: str, d: Dialogue) -> bool:
        """п.24: a model button in a non-Latin reply language with Latin words inside ("مش عارف أف open السجل")."""
        if d.language not in ("ar", "am", "ky", "ru"):
            return False
        rest = label
        for m in self._geo_methods(d) + ["AvaBet", "Ava.bet", "ID", "SMS", "PIN", "KYC", "2FA", "Telegram", "e-mail"]:
            rest = re.sub(re.escape(m), "", rest, flags=re.I)
        return bool(re.search(r"[A-Za-z]{2,}", rest))

    def _context_facts(self, d: Dialogue, task: Task) -> tuple[str, ...]:
        """Facts every reply may need: GEO payment methods (п.6/24) and the support contact by language (п.15)."""
        facts: list[str] = []
        topic = d.payment.kind if d.payment else d.issue.category if d.current else None
        if topic in PAYMENT_KINDS or task in (Task.CLARIFY_PAYMENT_KIND, Task.COLLECT_PAYMENT_DETAILS):
            methods = self._geo_methods(d)
            if methods:
                facts.append(f"payment methods available in the user's GEO ({d.country}): {', '.join(methods)}. "
                             "List or suggest only these")
            else:
                facts.append("the payment methods of the user's GEO are not known: do not list or suggest specific "
                             "methods, if needed ask the user to write the method")
            named = d.payment.details.get("method") if d.payment else None
            facts.append(f"payment method named by the user: {named}" if named else
                         "the user has NOT named a payment method: do not name or assume one")
        support = "support-ar@ava.bet" if d.language == "ar" else "support-en@ava.bet"
        facts.append(f"general support e-mail for this user: {support}; payments: payment-support@ava.bet or "
                     "Telegram @AVAPayment_bot; closing the account / self-exclusion: block@ava.bet; documents: security@ava.bet; "
                     "becoming an agent: Telegram @payment_assistment_avabet_bot; partnership: info@ava.bet. "
                     "Give only the contact that fits the question, no other addresses")
        return tuple(facts)

    def _finish_text(self, text: str, task: Task, d: Dialogue) -> str:
        """Plain text for Consultant (п.16), contact by language (п.15), handoff without a question (п.13),
        the question to the user in its own paragraph (п.25)."""
        if not self.cfg.markdown:
            text = _MARKDOWN.sub(lambda m: m.group(1) or m.group(2), text)
        if d.language == "ar":
            text = text.replace("support-en@ava.bet", "support-ar@ava.bet")
        text = text.strip()
        sentences = [x for x in _split_sentences(text) if x.strip()]
        if task in (Task.HANDOFF, Task.RISK_SUPPORT):
            while len(sentences) > 1 and sentences[-1].rstrip().endswith(("?", "؟")):
                text = text[:text.rstrip().rfind(sentences[-1].strip())].rstrip()
                sentences = sentences[:-1]
            return text
        if len(sentences) > 1 and sentences[-1].rstrip().endswith(("?", "؟")):
            last = sentences[-1].strip()
            head = text[:text.rfind(last)].rstrip()
            if head and not text[:text.rfind(last)].endswith("\n\n"):
                text = head + "\n\n" + last
        return text

    async def _checked(self, req: ComposeRequest, composed):
        sources = [self.kb.document(), *(t.text for t in req.history if t.role in ("user", "operator")),
                   req.user_text, *req.facts, *(o.label for o in composed.options)]
        result = check_reply(composed.text, sources=sources)
        if req.task not in (Task.HANDOFF, Task.RISK_SUPPORT) and promises_handoff(composed.text):  # п.4
            result = replace(result, problems=result.problems + (
                "promises to connect / pass to an operator, but there is no hand-off: never say that",))
        if result.ok:
            return composed
        log.warning("fact check failed (%s): %s", req.task.value, "; ".join(result.problems))
        note("fact_check_failed")
        retry = await self.llm.compose(replace(req, fix_hint=result.hint()))
        if check_reply(retry.text, sources=sources).ok and not (
                req.task not in (Task.HANDOFF, Task.RISK_SUPPORT) and promises_handoff(retry.text)):
            return retry
        return await self._fallback(req)

    async def _fallback(self, req: ComposeRequest) -> Composed:
        """Last resort: the KB answer as is (no styling) or a fixed text for the task."""
        note("fallback")
        if req.kb and req.task in (Task.ANSWER, Task.PAYMENT_WITHIN_NORM):
            text, result = req.kb[0].answer, "Full"
        else:
            text, result = FALLBACK_TEXTS.get(req.task, FALLBACK_TEXTS[Task.ANSWER]), "Partial"
        try:
            text = (await self.llm.localize([text], req.language))[0]
        except Exception:
            log.exception("localize failed")
        return Composed(text, result)  # type: ignore[arg-type]

    async def _show_category(self, d: Dialogue, category_key: str) -> Outcome:
        cat = BY_KEY[category_key]
        if d.persona is None:
            d.persona = self.personas.assign(d.country)
        if d.language is None:
            d.language = normalize(d.platform_language) or DEFAULT_LANGUAGE
        d.switch_issue(cat.key)
        options = tuple(Option(cat.question_id(i), q) for i, q in enumerate(cat.questions))
        prompt = f"{cat.label}: please choose a question or type your own."
        if self.policy.mode(d.language) == "pivot":
            texts = await self.translator.from_pivot([prompt, *[o.label for o in options]], "en", d.language)
            text, options = texts[0], tuple(Option(o.id, lbl) for o, lbl in zip(options, texts[1:]))
        else:
            options = await self._localize(options, d.language)
            text = (await self.llm.localize([prompt], d.language))[0]
        d.history.append(Turn("user", f"[clicked {cat.label}]"))
        d.history.append(Turn("bot", text))
        return Outcome().add(Say(text, options, None, None, cat.key))

    async def _localize(self, options: tuple[Option, ...], language: str) -> tuple[Option, ...]:
        if not options or normalize(language) == "en":
            return tuple(options)
        labels = await self.llm.localize([o.label for o in options], language)
        return tuple(Option(o.id, lbl) for o, lbl in zip(options, labels))

    def _parse_variant(self, variant: str, d: Dialogue) -> dict:
        if variant in TOP_BY_ID:
            _, label, category, question = TOP_BY_ID[variant]
            forced = {"category": category, "question": question}
            if variant.endswith("_missing"):
                forced["payment_issue"] = category
            return forced
        if variant == "pay:charged":
            kind = d.current if d.current in PAYMENT_KINDS else "deposit"
            return {"category": kind, "payment_issue": kind, "question": f"{kind} charged but not credited"}
        if variant == "pay:failed":
            kind = d.current if d.current in PAYMENT_KINDS else "deposit"
            return {"category": kind, "question": f"{kind} payment failed with an error"}
        if variant.startswith("method:"):
            idx = variant.split(":", 1)[1]
            methods = self._geo_methods(d)
            method = OTHER_METHOD if idx == "other" else methods[int(idx)] if idx.isdigit() and int(idx) < len(methods) else None
            return {"method": method, "label": method} if method else {}
        if variant.startswith("wait:"):
            return {"wait_hours": {o.id: h for o, h in WAIT_OPTIONS}.get(variant)}
        if variant.startswith("confirm:"):
            return {"confirmation": variant.split(":", 1)[1]}
        if variant in d.offered:  # a free-form button the model offered in its last answer
            return {"label": d.offered[variant]}
        resolved = resolve_variant(variant)
        if resolved is None:
            return {}
        cat, question = resolved
        if question is None:
            return {"show_category": cat.key}
        idx = cat.questions.index(question)
        if (cat.key, idx) in COMPLAINT_SUBQUESTIONS:
            return {"category": COMPLAINT.key, "question": question}
        return {"category": cat.key, "question": question}

    @staticmethod
    def _apply_forced(a: Analysis, forced: dict) -> Analysis:
        changes: dict = {}
        if forced.get("category"):
            changes.update(category=forced["category"], understood=True, noise=False, off_topic=False)
            if a.signal in (None, "noise", "not_understood"):
                changes["signal"] = "new_request"
        if forced.get("wait_hours") is not None:
            changes.update(wait_hours=forced["wait_hours"], wait_hours_max=forced["wait_hours"], understood=True,
                           signal="progress", noise=False)
        if forced.get("confirmation") in ("yes", "no"):
            changes.update(confirmation=forced["confirmation"], gratitude=False, understood=True, noise=False,
                           signal="closing" if forced["confirmation"] == "yes" else "not_helped")
        if forced.get("payment_issue"):
            changes.update(payment_issue=forced["payment_issue"], payment_unclear=False)
        if forced.get("method"):
            changes.update(payment_details={**a.payment_details, "method": forced["method"]}, signal="progress")
        if forced.get("label"):
            changes.update(understood=True, noise=False)
            if a.signal in (None, "noise", "not_understood"):
                changes["signal"] = "progress"
        return replace(a, **changes) if changes else a

    def _update_user_data(self, d: Dialogue, msgs: list[IncomingMessage]) -> None:
        for m in msgs:  # take only what the platform sent, never guess (ТЗ 7)
            if m.country and not d.country:
                d.country = m.country.upper()
            if m.timezone:
                d.timezone = m.timezone
            if m.channel:
                d.channel = m.channel
            if m.language_code:
                d.platform_language = normalize(m.language_code)
            if m.client_id:
                d.client_id = m.client_id
            if m.conversation_id:
                d.conversation_id = m.conversation_id

    def _update_language(self, ctx: _Ctx, from_text: bool) -> None:
        """ТЗ 8: language of the last message, but "ok" / "sir" / "40 min" must not flip it."""
        d, a = ctx.d, ctx.analysis
        detected = normalize(a.language) if from_text else None
        if detected and (a.language_confident or (d.language is None and d.platform_language is None)):
            lang = detected
        else:
            lang = d.language or d.platform_language or detected or DEFAULT_LANGUAGE
        d.language = lang
        # Tell Consultant once per change (it then keeps the chat in this language for operators too).
        if lang != d.platform_language and lang != d.chat_language_set:
            ctx.out.add(SetLanguage(lang))
            d.chat_language_set = lang

    def _sync_platform_history(self, d: Dialogue, items: list[PlatformMessage], current_ids: set) -> None:
        """Use the platform's chat history: restore context after a restart, notice a live operator.

        Consultant: fromId = null is the client, anything else an operator - or the bot itself
        (how to tell them apart is an open question to Uplatform), so the bot's own texts are
        recognised by content.
        """
        ours = {t.text.strip()[:200] for t in d.history if t.role == "bot"}
        restoring = not d.history
        for it in items:
            if it.id is not None and (it.id in current_ids or it.id in d.seen_platform_ids):
                continue
            if it.id is not None:
                d.seen_platform_ids.add(it.id)
            if it.from_id is None:
                if restoring:
                    d.history.append(Turn("user", it.text))
            elif restoring:
                d.history.append(Turn("bot", it.text))
            elif it.text.strip()[:200] not in ours:
                d.history.append(Turn("operator", it.text))
                d.operator_joined = True

    def _tz(self, d: Dialogue) -> ZoneInfo | None:
        """The user's time zone: from the platform, else from the GEO; None without GEO (test review п.5)."""
        for name in (d.timezone, COUNTRY_TZ.get((d.country or "").upper())):
            if name:
                try:
                    return ZoneInfo(name)
                except Exception:  # unknown zone name
                    continue
        return None

    def _now_local(self, d: Dialogue) -> str | None:
        tz = self._tz(d)
        if tz is None:
            return None  # no GEO: the model must not resolve clock times (п.5)
        return self._clock().astimezone(tz).strftime(f"%A %Y-%m-%d %H:%M ({tz.key})")

    def _deadline_facts(self, d: Dialogue, hours: float, kind: str | None = None) -> tuple[str, ...]:
        """Facts computed by code, so the model never does date arithmetic.

        Test review п.3/п.5: one deposit term, "within 24 hours", and no exact clock time of the deadline.
        """
        facts = [f"time since the payment/request: about {hours:g} h"]
        usual = self.cfg.usual_payment_hours.get(kind or "")
        if usual is None:
            facts.append("use the usual term exactly as the knowledge base states it; do not name a clock time")
        else:
            facts.append(f"the usual term is within {usual:g} hours from the moment of the payment, and it has not "
                         f"passed yet. Say exactly that («within {usual:g} hours»). Never name a clock time or a date "
                         "when it ends and never another term")
        return tuple(facts)

    def _operator_summary(self, d: Dialogue, reason: str, a: Analysis) -> str:
        p = d.payment
        user_msgs = [t.text for t in d.history if t.role == "user"][-6:]
        lines = [
            "AI Support Bot -> L1",
            f"Reason: {reason}",
            f"Topic: {d.issue.key if d.current else 'unknown'}; unsuccessful attempts: {d.issue.failures}",
            f"Language: {d.language}; GEO: {d.country or 'unknown'}; device: {d.channel or 'unknown'}",
            f"Persona: {d.persona.name if d.persona else '-'}",
        ]
        if p:
            wait = f"{p.wait_hours:g}h" if p.wait_hours is not None else "not provided"
            lines.append(f"PAYMENT CASE: {p.kind}, waiting: {wait}")
            for k in ("method", "method_user_words", "amount", "paid_at", "wallet", "receipt", "receipt_note"):
                if k in p.details:
                    lines.append(f"  {k}: {p.details[k]}")
            if "receipt" not in p.details:
                lines.append("  receipt / screenshot: not received")
        if a.aggressive or a.complaint or (p and p.pushback):
            lines.append("Note: the client is upset")
        if d.account_number:
            lines.append(f"Account number given by the user: {d.account_number}")
        if d.previous_contact:
            lines.append(f"Previous contact: {d.previous_contact}")
        lines.append("Last user messages:")
        lines += [f"- {t[:300]}" for t in user_msgs]
        return "\n".join(lines)[:4096]

    @staticmethod
    def _contact_note(d: Dialogue, reason: str | None) -> str:
        outcome = "handed to L1" if d.handed_off else "resolved by the bot" if d.closed else f"closed ({reason})"
        parts = [f"topic: {d.issue.key if d.current else 'unknown'}", f"outcome: {outcome}"]
        if d.payment:
            parts.append(f"payment case: {d.payment.kind}, waiting {d.payment.wait_hours or '?'}h")
        if d.escalation_reason:
            parts.append(f"escalation reason: {d.escalation_reason}")
        return "; ".join(parts)
