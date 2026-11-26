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

import re
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

Rules from the test review (never break them either):
- Never promise or offer a human operator / supervisor / "connect you" / "pass you to a specialist", and \
never offer such buttons. Never say a colleague is joining unless your task is "handoff". Requests for an \
operator are answered by helping right away.
- Emotion first: if the user is upset, start by acknowledging it in your own words. No "but" right after it, \
no "let's stick to the facts". A complaint can be sent by e-mail: offer it as a normal option, not as a threat.
- Buttons are written in the PLAYER's voice ("I'll wait", "Leave a complaint"), in the reply language, \
<= 20 characters. Never put a payment method on a button or in the text that the user did not name, \
unless the system gives you the methods of the user's GEO.
- The user's gender is unknown: address them in masculine or neutral forms.
- Contacts: only the ones from the contacts directory of the knowledge base or from the facts. \
Never invent an e-mail, a bot or a link.
- Limits: never state minimum / maximum amounts and never invent reasons for a limit. Use the prepared KB \
answer: the limits depend on the method and are shown in the deposit / withdrawal form.
- Deposits are credited within 24 hours. Never name an exact clock time or date by which it will arrive.
- Never claim you received, saw or attached something (a receipt, a screenshot) before it actually arrived.
- Never ask for the recipient's number or the user's phone number.
- Never write "do not send the payment again".
- Plain text only: no markdown, no ** or __, no headings.
- Paragraphs by meaning, separated by an empty line. The question to the user goes LAST, in its own \
paragraph.
"""

REFERENCE_NOTE = """\
Reference dialogues from the specification. They show the expected behaviour and style: short \
messages, one step at a time, buttons, how attempts differ, how a > 24h payment is handed over. \
They are written in Russian (translations of Arabic/English chats) - always reply in the language \
the system tells you. Remarks in ⟦...⟧ are the author's notes, not text for the user. Where an example \
contradicts the rules above (operator buttons, an exact deadline time, "do not send the payment again", \
asking for the recipient number or checking it), the rules win.
"""

TASK_HINTS: dict[Task, str] = {
    Task.ANSWER: "Answer the user's question from the knowledge base. If something is missing, ask ONE short "
                 "clarifying question. If the KB has no answer, say what you can help with and ask to rephrase.",
    Task.CLARIFY_UNRECOGNIZED: "You did not understand the request. Ask the user to describe the problem in more "
                               "detail, in your own words. Buttons with topics are attached by the system.",
    Task.NUDGE: "The user wrote something without content ('sir', '?', 'hello'). Reply with ONE short friendly line "
                "that invites them to say what happened. Do not repeat your previous message. If the system gives "
                "no buttons, offer 3-4 most likely requests from what the user wrote so far as buttons.",
    Task.GREET: "The user only greeted you ('hello', 'الو'). Greet back, introduce yourself, ask how you can help. "
                "Buttons with the most popular requests are attached by the system.",
    Task.RESPONSIBLE_GAMING: "The user is upset about losses or wants to deposit to win it back. Acknowledge the "
                             "feeling first. Gently say that chasing losses usually makes it worse, and that limits, a "
                             "break and self-exclusion are in the 'Responsible gaming' section (from the KB). Never "
                             "push to deposit or bet, never promise anything. Do not hand over to anyone.",
    Task.CLARIFY_PAYMENT_KIND: "The user says the payment 'failed' / 'did not go through'. Ask what exactly "
                               "happened: did it fail with an error (the money was not charged), or was the money "
                               "charged but not credited to the balance? Buttons are attached by the system.",
    Task.ACCOUNT_NUMBER: "The user sent only an account number. Thank them, say you noted it, ask what the "
                         "question is. Buttons with popular requests are attached by the system.",
    Task.OFF_TOPIC: "The message is not about AvaBet support. Do not answer it. Kindly say what you can help with and "
                    "invite the user to ask about it. Never lecture.",
    Task.DEFLECT_OPERATOR: "The user asks for a human operator. Do NOT promise a transfer and never say you will "
                           "connect them. Say you can help right now, guess the likely problem from the conversation "
                           "('Is it about a deposit that did not arrive?') and ask.",
    Task.ASK_WAIT_TIME: "The deposit / withdrawal has not arrived. Ask only how much time has passed since the payment "
                        "/ the withdrawal request. Buttons are attached by the system. If you already asked, ask "
                        "again in ONE short sentence and point to the buttons. Do not say a colleague is joining.",
    Task.CLARIFY_WAIT_TIME: "The user's answer about the time is unclear around 24 hours (e.g. 'yesterday', a date "
                            "without the hour). Ask once more, precisely: the approximate hour of the payment, or how "
                            "many hours have passed (see the facts).",
    Task.PAYMENT_WITHIN_NORM: "The waiting time is within the usual term. Explain the usual term from the KB, use the "
                              "facts (within 24 hours, never a clock time), say what the user can check meanwhile (from "
                              "the KB). Do not promise and do not say you checked anything.",
    Task.COLLECT_PAYMENT_DETAILS: "The payment has waited more than 24 hours, a specialist will take it over. Before "
                                  "that collect data for them: ask ONLY the item(s) in 'missing detail' (amount and "
                                  "time can be one question). Say briefly why it is needed. If the user is irritated, "
                                  "acknowledge it and explain why this item matters. Do not say you checked the payment, "
                                  "do not say a colleague is joining yet.",
    Task.ASK_RESOLVED: "Ask briefly whether the question is resolved. Buttons Yes/No are attached by the system.",
    Task.GOODBYE: "Say a short warm goodbye: glad you could help, write any time.",
    Task.HANDOFF: "Tell the user you are passing the conversation to a support colleague who will continue in this "
                  "chat, with everything collected so far - nothing has to be repeated. Mention the receipt only "
                  "as the facts say (received or not). End WITHOUT a question.",
    Task.RISK_SUPPORT: "The user shows signs of gambling harm or distress. Answer with care and without judgement, "
                       "do not talk about bonuses or betting. Mention that limits / self-exclusion exist if the KB "
                       "describes them. Say a colleague is joining the chat now. End WITHOUT a question.",
    Task.MEDIA_UNREADABLE: "You could not read the attached file. Thank the user and ask to describe it in text.",
}

ATTEMPT_STRATEGY = {  # test review п.11
    1: "Attempt 1: find the essence and answer, or clarify the problem with ONE question.",
    2: "Attempt 2 (previous answer did not help): offer a hypothesis of what the problem is ('Did your deposit not "
       "arrive?') and a different solution; re-read the request with synonyms and related topics.",
    3: "Attempt 3: explain why you need the information and ask for it (what exactly they see on the screen).",
    4: "Attempt 4: offer 3-4 most likely topics as buttons, built from what the user already wrote.",
    5: "Attempt 5 (last): propose the solution for the most frequent scenario of this topic.",
}

DETAIL_HINTS = {  # test review п.8/9: the fields differ for a deposit and a withdrawal
    "deposit": {
        "method": "the payment method (buttons with the methods of the GEO may be attached)",
        "amount+paid_at": "the amount and the date / time of the payment, in ONE question",
        "wallet": "the number of the wallet / account the user paid FROM (for a bank card ask ONLY the last 4 digits; "
                  "never a full card number, CVV, PIN, passwords or SMS codes)",
        "receipt": "a screenshot of the receipt from the wallet / payment app itself (not from the AvaBet page)",
    },
    "withdrawal": {
        "method": "the withdrawal method",
        "amount+paid_at": "the amount and the date / time of the withdrawal request, in ONE question",
        "wallet": "the wallet / account number given in the withdrawal request",
        "receipt": "a screenshot of the withdrawal request from the AvaBet history (NOT a receipt from the wallet)",
    },
}


def detail_hint(kind: str | None, missing: str) -> str:
    table = DETAIL_HINTS.get(kind or "deposit", DETAIL_HINTS["deposit"])
    if missing in table:
        return table[missing]
    return "; ".join(table.get(m, m) for m in missing.split("+"))


# The reference dialogues were written before the test review: fix them when they are loaded (п.3, 4, 22).
REFERENCE_FIXES = [
    (re.compile(r"\s*·\s*(Хочу оператора|Связаться с оператором|Позвать оператора|Оператор|Супервайзер)(?=\s*[·\]])",
                re.I), ""),
    (re.compile(r",?\s*то есть до \d{1,2}:\d{2}( завтра)?"), ""),
    (re.compile(r",?\s*срок до \d{1,2}:\d{2}( завтра)?"), ""),
    (re.compile(r"\s*Пока жд[её]те ответа, пожалуйста, не отправляйте (этот )?платеж повторно\.?"), ""),
]


def reference_dialogues() -> str:
    path = DATA / "reference_dialogues.md"
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    for rx, new in REFERENCE_FIXES:
        text = rx.sub(new, text)
    return text


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
    if p.pool.example:
        parts.append("Sample of how personas of this GEO talk (written in Russian; copy only the manner: register, "
                     "warmth, length, formal or informal address. Not the facts, not the name, not the language): "
                     f"«{p.pool.example}»")
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
    if req.task in (Task.ANSWER, Task.DEFLECT_OPERATOR, Task.CLARIFY_UNRECOGNIZED, Task.NUDGE):
        parts.append(ATTEMPT_STRATEGY.get(req.attempt, ATTEMPT_STRATEGY[5]))
    if req.category:
        parts.append(f"Topic: {req.category}.")
    if req.payment_kind:
        wait = f", waiting about {req.wait_hours:g} h" if req.wait_hours is not None else ""
        parts.append(f"Payment case: {req.payment_kind}{wait}.")
    if req.missing_detail:
        parts.append(f"Missing detail to ask for now: {detail_hint(req.payment_kind, req.missing_detail)}.")
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
  "yes" / "no" / a short answer to the bot's question is progress (or not_helped), never new_request. \
new_request only when the user names a NEW problem.
- payment_issue: "deposit" / "withdrawal" if money did not arrive (deposit not credited / withdrawal not \
received), else null.
- payment_unclear: true if the user says the payment "failed" / "did not go through" and it is not clear \
whether it failed with an error or the money was charged but not credited.
- payment_when: WHEN the payment / withdrawal request was made, exactly as the user said it, as an object \
(all fields nullable): date (YYYY-MM-DD, for "22/11" or "24 November"; the year from the local time), \
days_ago (0 = today, 1 = yesterday, 2 = the day before yesterday), weekday ("monday".."sunday", for "on \
Saturday"), time ("HH:MM", 24h), part_of_day ("morning" / "afternoon" / "day" / "evening" / "night"). \
The same in every language: "yesterday morning" -> days_ago 1 + morning; "since the day before yesterday" -> \
days_ago 2; "since yesterday evening" -> days_ago 1 + evening; "24/11 at 10 am" -> date + time. \
Do NOT compute hours yourself. null if the user did not say when.
- wait_hours, wait_hours_max: ONLY for a duration the user said ("3 hours", "2 days", "40 minutes", "more than \
a day" -> 25..null, "the third day" -> 48..72, "several days" -> 48..null), in hours. \
null for calendar answers ("yesterday", "on Saturday") - those go to payment_when.
- confirmation: "yes"/"no" if this answers "did that solve your question?", else null.
- gratitude, aggressive (insults, caps, threats - a tone, not a topic), complaint (an actual complaint about \
the service / an operator / fraud claim), off_topic (not about AvaBet support at all), noise.
- risk: "gambling_harm" ONLY for an explicit addiction ("I am addicted", "I cannot stop playing") or a \
request for self-exclusion / to block themselves; "self_harm" (any sign of suicidal thoughts); "minor" (says \
they are under 18); else null. Complaints about losses or claims of fraud are NOT risk.
- chasing_losses: true if the user wants to deposit to win back losses ("should I deposit to win it back?") \
or is upset about losing, without an explicit addiction.
- greeting: true if the turn is only a greeting ("hello", "الو", "salam").
- account_number: the AvaBet account / player ID if the turn is only that number, else null.
- media_understood: false if a file is attached and you cannot tell what it shows.
- payment_details: values the user gave for: method, amount, paid_at (date / time in the user's words), \
wallet (the wallet / account number paid from, or given in the withdrawal request), receipt (e.g. "will send \
later", "has no receipt") - only what this turn actually says; null otherwise. The answers may come in any \
order. The bot just asked for: {req.asked_detail or 'nothing specific'} - but put the value under the key it \
really belongs to (an answer "Visa card" to a question about the receipt is the method).
- method: {('write it exactly as one of: ' + ', '.join(req.geo_methods) + ' if the user means one of them (e.g. ' +
             '"vodafone" -> "Vodafone Cash"), else the user words') if req.geo_methods else 'the user words'}.
{'' if req.timezone_known else 'The user time zone is unknown.'}
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
