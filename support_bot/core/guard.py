"""Cheap deterministic checks of a generated reply (ТЗ 5, 10.1, 15.3).

The reply is written by the LLM from scratch to sound alive; this module
catches the typical failures of that approach before the text reaches the
user:

* invented facts: every number (amounts, limits, hours, percentages, times)
  and every link in the reply must appear in the knowledge base, in what the
  user wrote, or in facts the engine computed itself (e.g. a deadline);
* claims the bot cannot make: "I checked your payment", "your deposit is on
  its way", "guaranteed win";
* words the ТЗ forbids ("freebie", "easy money", "urgent", "only now",
  "break the bank").

It is deliberately simple and language-agnostic for numbers; phrase lists
cover en/ru for now and are easy to extend per language.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_NUMBER = re.compile(r"(?<![\w.])(\d+(?:[.,:]\d+)*)(?![\w])")
_URL = re.compile(r"https?://\S+|www\.\S+|\b[\w-]+\.(?:com|bet|net|org|io)\b", re.I)
SMALL_NUMBERS = {str(i) for i in range(0, 11)}  # list numbering, "1 click", "2 minutes"... always allowed

FORBIDDEN = [
    r"\bfreebie", r"easy money", r"\burgent", r"only now", r"break the bank",
    r"халяв", r"л[её]гкие деньги", r"\bсрочно", r"только сейчас", r"взорви банк",
]
FALSE_CLAIMS = [
    r"\bI (have )?(checked|verified|see|can see) (your|the) (payment|transaction|deposit|withdrawal)",
    r"(payment|transaction|deposit|withdrawal) (is|was) (approved|confirmed|on (its|the) way|processed)",
    r"guarantee", r"you will (definitely|surely) win",
    r"я (проверил|проверила|вижу) (ваш|вашу|ваше|ваши)? ?(платеж|платёж|транзакц|депозит|вывод)",
    r"(платеж|платёж|депозит|вывод) (уже )?(одобрен|подтвержден|в пути)",
    r"гарантир", r"точно выиграете",
]
# Test review п.4: "I'll connect you", "OK, I'll do it" in reply to "call an operator" without a hand-off.
HANDOFF_PROMISES = [
    r"\b(i'?ll|i will|let me|i'?m going to|going to) (now )?(connect|transfer|pass|forward|put) you",
    r"\b(connect|transfer)(ing)? you (to|with)", r"\bpass(ing)? (you|your (request|case|chat)) (on |over )?to",
    r"соедин(ю|яю)", r"переда(м|ю|ла|л)\s+(ваш|вас|обращ|диалог|запрос|чат)", r"переключ(у|аю)",
    r"подключ(у|аю|ится|ается)\s+(оператор|специалист|коллег|сотрудник)", r"позову\s+(оператор|специалист)",
    r"سأحول|سأوصل|سأقوم بتحويل|سيتم تحويل|أحولك|سأنقل|سيتواصل معك (أحد|زميل|موظف)",
    r"vous (transférer|mettre en relation)|je vous (transf[eè]re|mets en relation)",
    r"te (transfiero|comunico|paso con)|voy a (transferir|pasar)",
    r"aktar(ıyorum|acağım)|bağl(ıyorum|ayacağım)",
]
_HANDOFF = [re.compile(p, re.I) for p in HANDOFF_PROMISES]


def promises_handoff(text: str) -> bool:
    return any(rx.search(text) for rx in _HANDOFF)


_FORBIDDEN = [re.compile(p, re.I) for p in FORBIDDEN]
_FALSE = [re.compile(p, re.I) for p in FALSE_CLAIMS]


@dataclass(frozen=True)
class GuardResult:
    problems: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.problems

    def hint(self) -> str:
        return ("Your previous draft had problems: " + "; ".join(self.problems)
                + ". Rewrite it: use only numbers and links from the knowledge base or the user's words, "
                  "and never claim checks you did not do.")


_THOUSANDS = re.compile(r"^\d{1,3}(?:[, \u00a0]\d{3})+$")


def _norm(num: str) -> str:
    if _THOUSANDS.match(num):  # "5,000" == "5000"
        return re.sub(r"[, \u00a0]", "", num)
    return num.replace(",", ".").rstrip(".")


def numbers_in(text: str) -> set[str]:
    out: set[str] = set()
    for m in _NUMBER.findall(text):
        n = _norm(m)
        out.add(n)
        out.update(_norm(p) for p in re.split(r"[:.]", n) if p)  # "14:12" also allows "14" and "12"
    return out


def check_reply(reply: str, *, sources: list[str]) -> GuardResult:
    allowed = set(SMALL_NUMBERS)
    src = "\n".join(sources)
    for s in sources:
        allowed |= numbers_in(s)
    problems: list[str] = []
    for m in _NUMBER.findall(reply):
        n = _norm(m)
        if n not in allowed:
            problems.append(f"number '{m}' is not in the sources")
    src_low = src.lower()
    for url in _URL.findall(reply):
        if url.lower().rstrip(".,)") not in src_low:
            problems.append(f"link '{url}' is not in the sources")
    for rx in _FORBIDDEN:
        if rx.search(reply):
            problems.append(f"forbidden wording '{rx.pattern}'")
    for rx in _FALSE:
        if rx.search(reply):
            problems.append("claims a check, a payment status or a guarantee")
    return GuardResult(tuple(dict.fromkeys(problems)))
