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
import logging
from dataclasses import dataclass, field, replace
from zoneinfo import ZoneInfo

from .actions import HandOff, NoteToOperator, Option, Outcome, Resolve, Say, SetLanguage
from .budget import EscalationBudget
from .catalog import BY_KEY, CATEGORIES, COMPLAINT, EscalationGroup, resolve_variant
from .guard import check_reply
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
# "Complain about operator" / "File a complaint" live under Contacts (group D) but are complaints (14.1).
COMPLAINT_SUBQUESTIONS = {("contacts", 3), ("contacts", 4)}
PAYMENT_KINDS = ("deposit", "withdrawal")
UNLIMITED = 10**9
FALLBACK_TEXTS = {
    Task.ANSWER: "Could you tell me a bit more about your question?",
    Task.HANDOFF: "I'm passing your request to a support colleague, they will continue in this chat.",
    Task.RISK_SUPPORT: "Thank you for telling me. A support colleague will join this chat now to help you.",
    Task.ASK_WAIT_TIME: "How much time has passed since the payment?",
    Task.CLARIFY_WAIT_TIME: "Roughly how many hours have passed since the payment?",
    Task.ASK_RESOLVED: "Did that solve your question?",
    Task.GOODBYE: "Glad I could help! Write any time if you have other questions.",
    Task.COLLECT_PAYMENT_DETAILS: "To pass this to a specialist, could you share the payment details: method, amount "
                                  "and time?",
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
    complaint_attempts: int = 2  # complaints: "high escalation allowed"
    extra_attempts_over_budget: int = 2  # ТЗ 14.6
    payment_threshold_hours: float = 24.0  # group B, ТЗ 14.4
    max_wait_questions: int = 2  # asking "how long?" more times than this -> escalate
    usual_payment_hours: dict = field(default_factory=lambda: {"deposit": 12.0})  # from the KB; None = say as the KB does
    payment_silence_seconds: float = 300.0  # no answer to "how long?" -> L1 (0 = off), see open question 1
    # asked before L1 (ТЗ dialogue 4); "personal data" handling is out of scope for the demo
    payment_detail_fields: tuple[str, ...] = ("method", "receipt", "amount", "paid_at", "sender", "recipient")
    risk_escalates: bool = True  # gambling harm / self-harm -> L1 at once (not in the ТЗ, open question 3)
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

        # Safety first: responsible gaming / self-harm. Not in the ТЗ (Responsible gaming is group D there),
        # see open question 3. Minors get the age rules from the KB, no escalation.
        if a.risk in ("gambling_harm", "self_harm") and self.cfg.risk_escalates:
            await self._escalate(ctx, reason=f"risk signal: {a.risk}", task=Task.RISK_SUPPORT)
            return
        if a.risk == "minor":
            d.switch_issue("rules")
            await self._say(ctx, Task.ANSWER, kb=self._kb("age restrictions minimum age", "rules"))
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
            await self._say(ctx, Task.OFF_TOPIC, options=self._category_options() if not d.current else ())
            return
        if self._is_noise(a):
            d.noise_count += 1
            first = d.noise_count == 1 and not d.current
            await self._say(ctx, Task.CLARIFY_UNRECOGNIZED if first else Task.NUDGE,
                            options=self._category_options() if not d.current else ())
            return

        # Which problem is this about? (ТЗ 15.2: a short answer continues the current problem)
        if a.category and a.category != d.current:
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
            await self._say(ctx, Task.DEFLECT_OPERATOR)
        elif not a.understood and issue.key == UNKNOWN_ISSUE:
            await self._say(ctx, Task.CLARIFY_UNRECOGNIZED, options=self._category_options())
        else:
            await self._say(ctx, Task.ANSWER, kb=self._kb(ctx.text, issue.category))

    def _in_payment_flow(self, d: Dialogue, a: Analysis) -> bool:
        p = d.payment
        if p is None or p.within_norm:
            return False
        switched = a.category is not None and a.category not in (p.kind, COMPLAINT.key)
        if switched:
            return False
        return bool(a.payment_issue) or d.awaiting in ("wait_time", "payment_details") or p.over_threshold

    async def _payment_flow(self, ctx: _Ctx) -> None:
        d, a = ctx.d, ctx.analysis
        p = d.payment
        assert p is not None
        d.switch_issue(p.kind)
        p.details.update({k: v for k, v in a.payment_details.items() if v})  # whatever the client told so far

        if p.over_threshold:  # > 24h: collecting details for L1 (dialogue 4)
            await self._collect_details(ctx)
            return

        if a.wait_hours is not None:
            hours = a.wait_hours
            if a.wait_ambiguous:
                if p.clarify_asked < 1:
                    p.clarify_asked += 1
                    self._await(d, "wait_time")
                    await self._say(ctx, Task.CLARIFY_WAIT_TIME, options=tuple(o for o, _ in WAIT_OPTIONS))
                    ctx.out.silence_timeout = self.cfg.payment_silence_seconds or None
                    return
                hours = a.wait_hours_max or hours  # still unclear: in favour of the client
            p.wait_hours = hours
            d.awaiting = None
            if hours > self.cfg.payment_threshold_hours:
                p.over_threshold = True
                p.details.update(a.payment_details)
                await self._collect_details(ctx, first=True)
                return
            p.within_norm = True
            await self._say(ctx, Task.PAYMENT_WITHIN_NORM, kb=self._kb(f"{p.kind} processing time", p.kind),
                            facts=self._deadline_facts(d, hours, p.kind))
            return

        if a.signal == "closing" and not a.payment_issue:  # "ok thanks" instead of the time: let them go
            d.awaiting = "confirmation"
            await self._say(ctx, Task.ASK_RESOLVED, options=CONFIRM_OPTIONS)
            return
        if p.wait_questions_asked >= self.cfg.max_wait_questions:
            await self._escalate(ctx, reason=f"{p.kind} not received, the user did not tell the waiting time")
            return
        p.wait_questions_asked += 1
        self._await(d, "wait_time")
        await self._say(ctx, Task.ASK_WAIT_TIME, options=tuple(o for o, _ in WAIT_OPTIONS))
        ctx.out.silence_timeout = self.cfg.payment_silence_seconds or None

    async def _collect_details(self, ctx: _Ctx, first: bool = False) -> None:
        """> 24h: collect what L1 needs before the hand-off (ТЗ dialogue 4), then escalate.

        Every field is asked at most once; a field the user already mentioned is skipped;
        with a receipt attached amount and time are on it. Irritation is answered with an
        explanation (dialogue 4), a second push back hands off with what we have.
        """
        d, a = ctx.d, ctx.analysis
        p = d.payment
        assert p is not None
        p.details.update({k: v for k, v in a.payment_details.items() if v})
        asked = p.details_asked[-1] if p.details_asked else None
        pushback = not first and (a.wants_operator or a.signal in ("dissatisfied", "repeat")
                                  or (a.aggressive and not ctx.media))
        if not first and asked and asked not in p.details and not pushback and not a.payment_details:
            # the model found nothing structured: keep the raw answer under the asked item
            if ctx.media:
                p.details[asked if asked == "receipt" else "receipt"] = "attached (see the chat)"
            if ctx.text and asked != "receipt":
                p.details[asked] = ctx.text[:200]  # the answer to what we just asked
            elif asked == "receipt" and not ctx.media and ctx.text:
                p.details["receipt"] = f"not sent: {ctx.text[:150]}"
        elif ctx.media:
            p.details.setdefault("receipt", "attached (see the chat)")
        if pushback:
            p.pushback += 1
            if p.pushback >= 2 or not asked:
                await self._escalate(ctx, reason=f"{p.kind} not received for ~{p.wait_hours:g}h (> 24h)")
                return
            self._await(d, "payment_details")
            await self._say(ctx, Task.COLLECT_PAYMENT_DETAILS, missing=asked)  # explain why, ask again
            return
        on_receipt = ("amount", "paid_at") if p.details.get("receipt", "").startswith("attached") else ()
        missing = [f for f in self.cfg.payment_detail_fields
                   if f not in p.details and f not in p.details_asked and f not in on_receipt]
        if not missing:
            await self._escalate(ctx, reason=f"{p.kind} not received for ~{p.wait_hours:g}h (> 24h)")
            return
        p.details_asked.append(missing[0])
        self._await(d, "payment_details")
        await self._say(ctx, Task.COLLECT_PAYMENT_DETAILS, missing=missing[0])

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
        await self._say(ctx, task, check=False)
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
            facts=facts,
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
        opts = options or composed.options
        if task in (Task.HANDOFF, Task.GOODBYE, Task.RISK_SUPPORT):
            opts = ()  # the bot leaves the chat / the dialogue is over: nothing to tap
        d.offered = {o.id: o.label for o in opts if not o.id.startswith(("cat:", "q:", "wait:", "confirm:"))}
        d.history.append(Turn("bot", composed.text))
        text = composed.text
        if ctx.pivot and d.language != self.policy.pivot:
            translated = await self.translator.from_pivot([text, *[o.label for o in opts]], self.policy.pivot,
                                                          d.language)
            text, labels = translated[0], translated[1:]
            opts = tuple(Option(o.id, lbl) for o, lbl in zip(opts, labels))
        else:
            fixed = tuple(o for o in opts if not o.id.startswith("free:"))  # model buttons are already localized
            localized = dict(zip([o.id for o in fixed], await self._localize(fixed, d.language)))
            opts = tuple(localized.get(o.id, o) for o in opts)
        ctx.out.add(Say(text, opts, ctx.reply_to, composed.result, d.issue.category if d.current else None))

    async def _checked(self, req: ComposeRequest, composed):
        sources = [self.kb.document(), *(t.text for t in req.history if t.role in ("user", "operator")),
                   req.user_text, *req.facts, *(o.label for o in composed.options)]
        result = check_reply(composed.text, sources=sources)
        if result.ok:
            return composed
        log.warning("fact check failed (%s): %s", req.task.value, "; ".join(result.problems))
        retry = await self.llm.compose(replace(req, fix_hint=result.hint()))
        if check_reply(retry.text, sources=sources).ok:
            return retry
        return await self._fallback(req)

    async def _fallback(self, req: ComposeRequest) -> Composed:
        """Last resort: the KB answer as is (no styling) or a fixed text for the task."""
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

    def _now_local(self, d: Dialogue) -> str | None:
        now = self._clock()
        if d.timezone:
            try:
                return now.astimezone(ZoneInfo(d.timezone)).strftime(f"%Y-%m-%d %H:%M ({d.timezone})")
            except Exception:  # unknown zone name
                pass
        return now.strftime("%Y-%m-%d %H:%M (UTC)")

    def _deadline_facts(self, d: Dialogue, hours: float, kind: str | None = None) -> tuple[str, ...]:
        """Times computed by code, so the model never does date arithmetic.

        The usual term comes from the KB (deposit: "from 1 minute to 12 hours"); the 24h threshold of
        ТЗ 14.4 only decides about escalation and is never promised to the client.
        """
        now = self._clock()
        tz = None
        if d.timezone:
            try:
                tz = ZoneInfo(d.timezone)
            except Exception:
                tz = None
        local_now = now.astimezone(tz) if tz else now
        facts = [f"time since the payment/request: about {hours:g} h"]
        usual = self.cfg.usual_payment_hours.get(kind or "")
        if usual is None:
            facts.append("use the usual term exactly as the knowledge base states it")
        elif hours < usual:
            deadline = local_now - dt.timedelta(hours=hours) + dt.timedelta(hours=usual)
            day = "today" if deadline.date() == local_now.date() else "tomorrow"
            facts.append(f"usual term per the knowledge base: up to {usual:g} h, i.e. until about "
                         f"{deadline:%H:%M} {day} (client's local time)")
        else:
            facts.append(f"the usual term of the knowledge base (up to {usual:g} h) has passed, but it is still "
                         "within the time when the case is not handed over: say that at peak load processing can "
                         "take longer, ask to wait a little more, do not promise a time and do not say you checked")
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
            for k in self.cfg.payment_detail_fields:
                if k in p.details:
                    lines.append(f"  {k}: {p.details[k]}")
        if a.aggressive or a.complaint or (p and p.pushback):
            lines.append("Note: the client is upset")
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
