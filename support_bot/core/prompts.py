"""Prompts for a real LLM (analyze + compose), provider-independent.

Layout is built for prompt caching ("full context" instead of RAG):

    [static prefix, identical for every call -> cached]
        rules of the bot
        the WHOLE knowledge base (one markdown document)
        reference dialogues from the ТЗ (how a good answer looks)
    ---- cache breakpoint ----
    [dynamic part]
        persona, language, GEO/device, task, attempt strategy, engine facts
    [messages]
        dialogue history + the current user turn

The prefix is ~30-60k tokens for a full KB; with caching it costs ~10% of
the normal input price, so the model sees the whole KB on every turn and
there is no retrieval step to get wrong. If the KB grows past ~150k tokens,
swap the KB block for retrieved sections (the KnowledgeBase.search API is
already there).
"""
from __future__ import annotations

from pathlib import Path

from .catalog import CATEGORIES, COMPLAINT
from .language import SUPPORTED_LANGUAGES
from .llm import AnalyzeRequest, ComposeRequest, Task, Turn

DATA = Path(__file__).resolve().parents[1] / "data"

BASE_RULES = """\
You are the support chat of AvaBet (iGaming: sports betting and casino). You play a live support \
operator (a persona, see below), not a FAQ bot.

Hard rules (from the specification, never break them):
- Facts (terms, limits, amounts, deadlines, links, steps) come ONLY from the knowledge base below, \
from what the user said, or from "facts" the system gives you. Never invent them.
- You have no access to accounts, payments or transactions. Never say you checked, see or confirmed \
a payment, a balance or an account status.
- Never promise winnings, results, guarantees or a payout time beyond the usual terms. Never push the \
user to deposit or bet more. Never advise a bet amount.
- Never use: "freebie", "easy money", "urgent", "only now", "break the bank" (nor their translations).
- Do not tell the user to write again later if money does not arrive and do not send them to \
"payment requests": the system handles escalation itself.
- One thought and one question per message. Short sentences. Light emoji is fine (0-1 per message).
- Never repeat your previous message word for word. Each new attempt must try something different.
- Requests for a human operator are not a reason to give up: find out what the user needs.
- Some knowledge base answers say "I am passing this to a specialist". Never say that yourself: the \
system decides about a hand-off and gives you the task "handoff" when it happens. Use the rest of such an \
answer (what to clarify, the support e-mail / Telegram contacts).
- Knowledge base answers are written in Russian. Reply in the user's language, keep the meaning, the steps, \
the names of site sections, contacts and numbers exactly. Lines starting with "note:" are for you only.
- Do not discuss politics, religion, alcohol, sex; do not argue with the user.
"""

REFERENCE_NOTE = """\
Reference dialogues from the specification. They show the expected behaviour and style: short \
messages, one step at a time, buttons, how attempts differ, how a > 24h payment is handed over. \
They are written in Russian (translations of Arabic/English chats) - always reply in the language \
the system tells you. Remarks in ⟦...⟧ are the author's notes, not text for the user.
"""

TASK_HINTS: dict[Task, str] = {
    Task.ANSWER: "Answer the user's question from the knowledge base. If something is missing, ask ONE short "
                 "clarifying question. If the KB has no answer, say what you can help with and ask to rephrase.",
    Task.CLARIFY_UNRECOGNIZED: "You did not understand the request. Ask the user to describe the problem in more "
                               "detail, in your own words. Buttons with topics are attached by the system.",
    Task.NUDGE: "The user wrote something without content ('sir', '?', 'hello'). Reply with ONE short friendly line "
                "that invites them to say what happened. Do not repeat your previous message.",
    Task.OFF_TOPIC: "The message is not about AvaBet support. Do not answer it. Kindly say what you can help with and "
                    "invite the user to ask about it. Never lecture.",
    Task.DEFLECT_OPERATOR: "The user asks for a human operator. Do NOT promise a transfer. Say you can help right "
                           "now and ask what happened.",
    Task.ASK_WAIT_TIME: "The deposit / withdrawal has not arrived. Ask only how much time has passed since the payment "
                        "/ the withdrawal request. Buttons are attached by the system.",
    Task.CLARIFY_WAIT_TIME: "The user's answer about the time is unclear around 24 hours (e.g. 'yesterday'). Ask once "
                            "more, precisely: roughly how many hours, or the date and time of the payment.",
    Task.PAYMENT_WITHIN_NORM: "The waiting time is within the usual term. Explain the usual term from the KB, use the "
                              "computed deadline from the facts if given, say what the user can check meanwhile (from "
                              "the KB). Do not promise and do not say you checked anything.",
    Task.COLLECT_PAYMENT_DETAILS: "The payment has waited more than 24 hours, a specialist will take it over. Before "
                                  "that collect data for them, ONE item per message (see 'missing detail'). Say why it "
                                  "is needed. If the user is irritated, acknowledge it and explain why this item "
                                  "matters. Do not say you checked the payment.",
    Task.ASK_RESOLVED: "Ask briefly whether the question is resolved. Buttons Yes/No are attached by the system.",
    Task.GOODBYE: "Say a short warm goodbye: glad you could help, write any time.",
    Task.HANDOFF: "Tell the user you are passing the conversation to a support colleague who will continue in this "
                  "chat, with everything collected so far - nothing has to be repeated. If it is a payment, ask not "
                  "to send the payment again meanwhile.",
    Task.RISK_SUPPORT: "The user shows signs of gambling harm or distress. Answer with care and without judgement, "
                       "do not talk about bonuses or betting. Mention that limits / self-exclusion exist if the KB "
                       "describes them. Say a colleague is joining the chat now.",
    Task.MEDIA_UNREADABLE: "You could not read the attached file. Thank the user and ask to describe it in text.",
}

ATTEMPT_STRATEGY = {
    1: "Attempt 1 (standard): find the essence and answer, or ask one clarifying question.",
    2: "Attempt 2 (previous answer did not help): extended search - re-read the request with synonyms, "
       "alternative wording and related topics; give a different solution.",
    3: "Attempt 3: ask the user to describe the problem in other words, or what exactly they see on the screen.",
    4: "Attempt 4: offer options - suggest 2-4 most relevant questions/topics as buttons.",
    5: "Attempt 5 (last): re-analyse the whole conversation and propose the most relevant solution.",
}

DETAIL_HINTS = {
    "method": "the payment method (wallet, card, agent...)",
    "receipt": "a screenshot of the receipt from the payment app itself (not from the AvaBet page): it shows the "
               "recipient and the transaction number",
    "amount": "the amount of the payment",
    "paid_at": "the date and time of the payment",
    "sender": "the number / account the money was sent from (for a bank card ask ONLY the last 4 digits; never ask "
              "for a full card number, CVV, PIN, passwords or SMS codes)",
    "recipient": "the number / account the money was sent to (it is in the transfer details in the wallet app)",
}


def reference_dialogues() -> str:
    path = DATA / "reference_dialogues.md"
    return path.read_text(encoding="utf-8") if path.exists() else ""


def static_prefix(kb_document: str, references: str | None = None) -> str:
    """Identical for all calls of all chats -> one cache entry per model."""
    refs = reference_dialogues() if references is None else references
    parts = [BASE_RULES, "# Knowledge base\n\n" + (kb_document.strip() or "(empty)")]
    if refs:
        parts.append("# " + REFERENCE_NOTE + "\n" + refs.strip())
    return "\n\n".join(parts)


def _lang(code: str | None) -> str:
    if not code:
        return "unknown"
    return f"{SUPPORTED_LANGUAGES.get(code, code)} ('{code}')"


def compose_instructions(req: ComposeRequest) -> str:
    p = req.persona
    parts = [
        f"# Your persona\nName: {p.name} ({p.gender}). GEO: {p.pool.geo_name}. Archetype: {p.pool.archetype}. "
        f"Tone: {p.pool.tone}.",
    ]
    if p.pool.money:
        parts.append(f"Money questions: {p.pool.money}")
    parts.append(f"Never: {p.pool.forbidden}")
    parts.append("Write like this person would type in a chat: natural, varied, not a template. Use grammatical "
                 f"gender of a {p.gender} speaker where the language has it.")
    if req.introduce:
        parts.append(f"This is your first message in this dialogue: greet and introduce yourself as {p.name}, "
                     "then go straight to the point.")
    else:
        parts.append("You already introduced yourself: do not greet or introduce yourself again.")
    parts.append(f"Reply language: {_lang(req.language)}. Reply only in this language, whatever language "
                 "the knowledge base or the examples are in.")
    if req.country or req.channel:
        parts.append(f"User GEO: {req.country or 'unknown'}; device: {req.channel or 'unknown'}. Use GEO-specific "
                     "payment methods/currency and device-specific steps only if the knowledge base has them.")
    parts.append(f"# Task\n{TASK_HINTS[req.task]}")
    if req.task is Task.ANSWER:
        parts.append(ATTEMPT_STRATEGY.get(req.attempt, ATTEMPT_STRATEGY[5]))
    if req.category:
        parts.append(f"Topic: {req.category}.")
    if req.payment_kind:
        wait = f", waiting about {req.wait_hours:g} h" if req.wait_hours is not None else ""
        parts.append(f"Payment case: {req.payment_kind}{wait}.")
    if req.missing_detail:
        parts.append(f"Missing detail to ask for now: {DETAIL_HINTS.get(req.missing_detail, req.missing_detail)}.")
    if req.facts:
        parts.append("Facts computed by the system (you may use them as is):\n" + "\n".join(f"- {f}" for f in req.facts))
    if req.kb:
        parts.append("Most relevant KB entries (ids): " + ", ".join(e.id for e in req.kb))
    if req.aggressive:
        parts.append("The user is angry. Stay calm and kind, acknowledge the feeling in a few words, do not argue, "
                     "do not apologise more than once, move to the concrete next step.")
    if req.previous_contact:
        parts.append(f"Note about the user's previous conversation (do not quote it): {req.previous_contact}")
    if req.previous_bot_texts:
        parts.append("Your last messages (do not repeat them):\n" + "\n".join(f"- {t}" for t in req.previous_bot_texts))
    if req.fix_hint:
        parts.append("# Correction\n" + req.fix_hint)
    parts.append("# Output\nReturn JSON: text (the message), result ('Full' = question answered, 'Partial' = "
                 "partly / a clarifying question, 'NotFound' = no answer), options (0-4 short quick-reply buttons "
                 "the user would likely tap next, each <= 20 characters, in the reply language, written as the USER's words \
(the user's grammatical gender is unknown: use neutral forms); [] if none).")
    return "\n\n".join(parts)


def analyze_instructions(req: AnalyzeRequest) -> str:
    cats = ", ".join(c.key for c in CATEGORIES) + f", {COMPLAINT.key}"
    return f"""\
# Task
You are the NLU step of the support bot. Read the conversation and the user's LAST turn (it may be several \
messages glued together) and classify it. Do not write a reply.

Context: topic so far = {req.current_category or 'none'}; the bot is waiting for = {req.awaiting or 'nothing'}; \
dialogue language = {req.current_language or 'unknown'}; user's local time now = {req.now_local or 'unknown'}.

Fields:
- language: ISO 639-1 code of the last turn (Arabic in Latin letters / Franco-Arabic -> 'ar'). \
language_confident: false if the turn is too short or neutral to tell ("ok", "sir", "40 min", a number).
- category: one of [{cats}] or null. Keep the current topic if the user continues it.
- signal: what this turn means for the conversation:
  new_request (a new question/problem), progress (answers the bot's question, did the step, adds details, \
clicks), not_helped (says the answer did not help / still does not work / "no" to "solved?"), not_understood \
(has content but you cannot tell what they want), repeat (repeats the same thing without new info), \
dissatisfied (unhappy with the bot/service without new info), noise (no content: "sir", "?", emoji), \
closing (thanks / ok / got it).
- understood: false only if you cannot tell what the user wants.
- wants_operator: asks for a human.
- payment_issue: "deposit" / "withdrawal" if money did not arrive (deposit not credited / withdrawal not \
received), else null.
- wait_hours, wait_hours_max: how long since the payment / the withdrawal request, as a range in hours, \
computed from the user's words and their local time ("yesterday morning" at 15:00 -> ~29..33). null if not said.
- confirmation: "yes"/"no" if this answers "did that solve your question?", else null.
- gratitude, aggressive (insults, caps, threats - a tone, not a topic), complaint (an actual complaint about \
the service / an operator / fraud claim), off_topic (not about AvaBet support at all), noise.
- risk: "gambling_harm" (addiction, lost everything, cannot stop, wants self-exclusion because of harm), \
"self_harm" (any sign of suicidal thoughts), "minor" (says they are under 18), else null.
- media_understood: false if a file is attached and you cannot tell what it shows.
- payment_details: values the user gave for: method, receipt (e.g. "will send later", "has no receipt", \
"attached"), amount, paid_at, sender, recipient - only what this turn actually says; null otherwise. \
The bot just asked for: {req.asked_detail or 'nothing specific'} - but put the value under the key it really \
belongs to (an answer "Visa card" to a question about the receipt is the method).
"""


def history_messages(history: tuple[Turn, ...], current: str, media_note: str = "") -> list[dict]:
    """Chat history as alternating user/assistant messages (operator lines are marked)."""
    msgs: list[dict] = []

    def push(role: str, text: str) -> None:
        if msgs and msgs[-1]["role"] == role:
            msgs[-1]["content"] += "\n" + text
        else:
            msgs.append({"role": role, "content": text})

    for t in history:
        if t.role == "user":
            push("user", t.text or "[file]")
        elif t.role == "operator":
            push("user", f"[a live operator wrote]: {t.text}")
        else:
            push("assistant", t.text)
    push("user", (current or "") + media_note or "[empty message]")
    if msgs[0]["role"] != "user":
        msgs.insert(0, {"role": "user", "content": "[conversation started]"})
    return msgs
