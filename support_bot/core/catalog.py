"""Topics (category buttons) and sub-questions from ТЗ п.6.2 / п.11.

Ids are stable and used as Consultant variant ids:
  cat:<category>            - category button
  q:<category>:<index>      - sub-question row
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class EscalationGroup(str, Enum):
    A = "A"  # standard: escalate after 5 unsuccessful attempts
    B = "B"  # payment cases (deposit/withdrawal not received): 24h rule
    D = "D"  # never escalate to L1 (contacts, partnership, rules, greeting...)
    COMPLAINT = "complaint"  # high escalation allowed


@dataclass(frozen=True)
class Category:
    key: str
    label: str  # English label, max 20 chars (Consultant button limit)
    questions: tuple[str, ...]
    group: EscalationGroup = EscalationGroup.A
    keywords: tuple[str, ...] = field(default_factory=tuple)

    @property
    def variant_id(self) -> str:
        return f"cat:{self.key}"

    def question_id(self, idx: int) -> str:
        return f"q:{self.key}:{idx}"


CATEGORIES: tuple[Category, ...] = (
    Category("deposit", "Deposit",
             ("Make a deposit", "How to make deposit?", "Minimum deposit", "Issues with deposit", "Deposit methods"),
             keywords=("deposit", "депозит", "пополн", "top up", "положил")),
    Category("withdrawal", "Withdrawal",
             ("Withdraw funds", "Minimum withdrawal", "Withdrawal methods", "Issues with withdrawal", "Withdrawal status"),
             keywords=("withdraw", "вывод", "вывел", "выве", "снял", "payout", "cash out")),
    Category("registration", "Registration",
             ("Register account", "Register in 1 click", "Verify phone / email", "Change personal data", "Registration issues"),
             keywords=("regist", "регистр", "sign up", "signup")),
    Category("bonuses", "Bonuses",
             ("Available bonuses", "How to get a bonus", "Bonus not credited", "Wagering requirement", "Use promo code"),
             keywords=("bonus", "бонус", "promo", "промо", "wager", "отыгр", "фриспин", "free spin")),
    Category("account", "Account",
             ("Login issues", "Forgot password", "Restore access", "Verification code not received", "Account blocked"),
             keywords=("login", "log in", "войти", "вход", "password", "пароль", "blocked", "заблок", "account", "аккаунт")),
    Category("sport", "Sport",
             ("Types of bets", "How to place a bet", "Bet not settled", "Bet history", "Minimum bet"),
             keywords=("bet", "ставк", "ставить", "постав", "match", "матч", "coupon", "купон")),
    Category("app", "App",
             ("Download app", "Get code from app", "Authenticator issues", "Android app", "Lost access to authenticator"),
             keywords=("app", "приложен", "android", "ios", "authenticator", "аутентификатор", "apk")),
    Category("rules", "Rules",
             ("Where to find rules", "Bonus terms", "License & legality", "Age restrictions", "Responsible gaming"),
             group=EscalationGroup.D,
             keywords=("rule", "правил", "licen", "лиценз", "age", "возраст", "responsible")),
    Category("contacts", "Contacts",
             ("Contact support", "Find contacts", "Hotline", "File a complaint", "Complain about operator"),
             group=EscalationGroup.D,
             keywords=("contact", "контакт", "hotline", "телефон поддержки", "email support")),
    Category("partnership", "Partnership",
             ("Become an agent", "Become a partner", "Affiliate program info", "Propose cooperation", "Track referral stats"),
             group=EscalationGroup.D,
             keywords=("partner", "партн", "affiliate", "агент", "referral", "реферал", "cooperation", "сотрудничеств")),
    Category("casino", "Casino & Games",
             ("Game issues", "Play without deposit", "Game history", "Slot rules", "RTP %"),
             keywords=("casino", "казино", "slot", "слот", "game", "игр", "rtp")),
    Category("kyc", "Verification & KYC",
             ("What is verification", "Documents required", "Submit documents", "Verification time", "2FA issues"),
             keywords=("verif", "верифик", "kyc", "document", "документ", "паспорт", "passport", "2fa")),
)

# Pseudo-categories used by the escalation policy (not shown as buttons).
COMPLAINT = Category("complaint", "Complaint", (), group=EscalationGroup.COMPLAINT,
                     keywords=("complain", "жалоб", "пожалуюсь", "scam", "мошенн", "обман", "fraud"))

BY_KEY: dict[str, Category] = {c.key: c for c in (*CATEGORIES, COMPLAINT)}


def resolve_variant(variant_id: str) -> tuple[Category, str | None] | None:
    """Map a clicked variant id to (category, sub-question or None)."""
    parts = variant_id.split(":")
    if parts[0] == "cat" and len(parts) == 2 and parts[1] in BY_KEY:
        return BY_KEY[parts[1]], None
    if parts[0] == "q" and len(parts) == 3 and parts[1] in BY_KEY:
        cat = BY_KEY[parts[1]]
        try:
            return cat, cat.questions[int(parts[2])]
        except (ValueError, IndexError):
            return None
    return None
