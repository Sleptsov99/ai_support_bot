"""DemoLLM: the same keyword NLU as FakeLLM, but readable Russian/English texts.

Only for the local demo. Not a real model: answers come from the demo KB,
phrasing comes from templates.
"""
from __future__ import annotations

from ..core.actions import Option
from ..core.catalog import BY_KEY
from ..core.llm import Composed, ComposeRequest, FakeLLM, Task

T = {
    "intro": {
        "ru": "Здравствуйте! Меня зовут {name}, я помогу вам разобраться 😊\n",
        "en": "Hi! My name is {name}, I'll help you sort this out 😊\n",
    },
    Task.CLARIFY_UNRECOGNIZED: {
        "ru": "Извините, я не совсем понял ваш вопрос. Можете, пожалуйста, детальнее описать проблему или выбрать тему ниже?",
        "en": "Sorry, I didn't quite get your question. Could you describe it in more detail or pick a topic below?",
    },
    Task.DEFLECT_OPERATOR: {
        "ru": "Я постараюсь помочь вам сам 😊 Расскажите, пожалуйста, с каким вопросом вы столкнулись?",
        "en": "I'll do my best to help you myself 😊 Could you tell me what happened?",
    },
    Task.ASK_WAIT_TIME: {
        "deposit": {
            "ru": "Понял, пополнение еще не пришло на баланс. Подскажите, пожалуйста, сколько времени прошло с момента оплаты?",
            "en": "Got it, the deposit hasn't reached your balance yet. How much time has passed since the payment?",
        },
        "withdrawal": {
            "ru": "Понял, вывод еще не получен. Подскажите, пожалуйста, сколько времени прошло с момента создания заявки на вывод?",
            "en": "Got it, the withdrawal hasn't arrived yet. How much time has passed since you created the request?",
        },
    },
    Task.PAYMENT_WITHIN_NORM: {
        "ru": "Спасибо! Это пока в пределах обычного срока. ",
        "en": "Thank you! This is still within the usual time. ",
    },
    Task.ASK_RESOLVED: {"ru": "Удалось ли решить ваш вопрос?", "en": "Did that solve your question?"},
    Task.GOODBYE: {
        "ru": "Рад, что смог вам помочь 😊 Если появятся другие вопросы, обращайтесь!",
        "en": "Glad I could help 😊 If you have any other questions, just write!",
    },
    Task.HANDOFF: {
        "ru": "Спасибо за терпение 🙏 Я передаю ваш вопрос специалисту поддержки, он продолжит диалог в этом чате.",
        "en": "Thank you for your patience 🙏 I'm passing your question to a support specialist, they will continue in this chat.",
    },
    Task.NUDGE: {
        "ru": "Я здесь 🙂 Напишите, пожалуйста, что случилось, или выберите тему ниже.",
        "en": "I'm here 🙂 Just tell me what happened, or pick a topic below.",
    },
    Task.OFF_TOPIC: {
        "ru": "С этим, к сожалению, не подскажу 🙂 Я помогаю с вопросами по AvaBet: депозиты, выводы, аккаунт, ставки. Чем могу помочь?",
        "en": "I can't help with that one, sorry 🙂 I'm here for AvaBet questions: deposits, withdrawals, account, bets. What can I do for you?",
    },
    Task.CLARIFY_WAIT_TIME: {
        "ru": "Уточните, пожалуйста, сколько примерно часов прошло: меньше суток или больше?",
        "en": "Could you tell me roughly how many hours it has been: less than a day or more?",
    },
    Task.COLLECT_PAYMENT_DETAILS: {
        "method": {
            "ru": "Понимаю, это действительно дольше обычного. Чтобы специалист сразу разобрался, подскажите, пожалуйста, каким способом вы платили?",
            "en": "I understand, that's longer than usual. So that a specialist can check it right away, which payment method did you use?",
        },
        "receipt": {
            "ru": "Спасибо. Теперь пришлите, пожалуйста, скриншот чека из приложения платежной системы, именно из кошелька, а не со страницы AvaBet. На нем видны номер получателя и номер операции, без него специалисты не найдут перевод.",
            "en": "Thank you. Now please send a screenshot of the receipt from the payment app itself, not from the AvaBet page. It shows the recipient and the transaction number, without it the specialists can't find the transfer.",
        },
        "amount": {
            "ru": "Спасибо! А на какую сумму был платеж?",
            "en": "Thank you! And what was the amount?",
        },
        "sender": {
            "ru": "Осталось два номера. С какого номера или счета вы переводили?",
            "en": "Two numbers left. Which number or account did you pay from?",
        },
        "recipient": {
            "ru": "Спасибо. А номер, на который вы переводили? Его можно найти в деталях перевода в приложении кошелька.",
            "en": "Thanks. And the number you sent it to? You can find it in the transfer details in the wallet app.",
        },
        "paid_at": {
            "ru": "Хорошо. Подскажите, пожалуйста, примерную дату и время платежа?",
            "en": "Okay. What was the approximate date and time of the payment?",
        },
    },
    Task.RISK_SUPPORT: {
        "ru": "Спасибо, что написали об этом 🙏 Мне очень жаль, что вам сейчас тяжело. Я подключаю коллегу, он поможет с ограничением игры и подскажет, куда можно обратиться за поддержкой.",
        "en": "Thank you for telling me 🙏 I'm really sorry you're going through this. I'm bringing in a colleague who can help you limit your play and point you to support.",
    },
    Task.MEDIA_UNREADABLE: {
        "ru": "Спасибо за файл! К сожалению, я не смог его разобрать. Опишите, пожалуйста, проблему текстом.",
        "en": "Thanks for the file! Unfortunately I couldn't read it. Could you describe the problem in text?",
    },
}

# ТЗ 14.3: every attempt must be different
ATTEMPTS = {
    "ru": [
        "Уточните, пожалуйста, что именно происходит? Расскажите немного подробнее, и я постараюсь помочь.",
        "Давайте разберемся. Что именно вы видите на экране или какой шаг не получается?",
        "Кажется, мне не хватает информации. Можете описать ситуацию другими словами?",
        "Возможно, ваш вопрос связан с одной из этих тем. Выберите подходящую 👇",
        "Попробую еще раз: выберите наиболее близкий вариант или опишите, что вы хотите сделать.",
    ],
    "en": [
        "Could you tell me what exactly is happening? A bit more detail will help me.",
        "Let's figure it out. What exactly do you see on the screen, or which step doesn't work?",
        "It seems I'm missing some information. Could you describe the situation in other words?",
        "Maybe your question is about one of these topics. Please pick one 👇",
        "Let me try once more: choose the closest option or describe what you want to do.",
    ],
}

RU_LABELS = {
    "Deposit": "Депозит", "Withdrawal": "Вывод", "Registration": "Регистрация", "Bonuses": "Бонусы",
    "Account": "Аккаунт", "Sport": "Спорт", "App": "Приложение", "Rules": "Правила", "Contacts": "Контакты",
    "Partnership": "Партнерство", "Casino & Games": "Казино и игры", "Verification & KYC": "Верификация",
    "Yes": "Да", "No": "Нет", "< 2 hours": "Меньше 2 часов", "2-24 hours": "2-24 часа", "> 24 hours": "Больше 24 часов",
    "Make a deposit": "Сделать депозит", "How to make deposit?": "Как пополнить счет?", "Minimum deposit": "Минимальный депозит",
    "Issues with deposit": "Проблема с депозитом", "Deposit methods": "Способы пополнения",
    "Withdraw funds": "Вывести средства", "Minimum withdrawal": "Минимальный вывод", "Withdrawal methods": "Способы вывода",
    "Issues with withdrawal": "Проблема с выводом", "Withdrawal status": "Статус вывода",
    "Register account": "Регистрация аккаунта", "Register in 1 click": "Регистрация в 1 клик",
    "Verify phone / email": "Подтвердить телефон / email", "Change personal data": "Изменить личные данные",
    "Registration issues": "Проблемы с регистрацией",
    "Login issues": "Проблемы со входом", "Forgot password": "Забыл пароль", "Restore access": "Восстановить доступ",
    "Verification code not received": "Не приходит код подтверждения", "Account blocked": "Аккаунт заблокирован",
    "What is verification": "Что такое верификация", "Documents required": "Какие нужны документы",
    "Submit documents": "Отправить документы", "Verification time": "Сроки верификации", "2FA issues": "Проблемы с 2FA",
    "Available bonuses": "Доступные бонусы", "How to get a bonus": "Как получить бонус", "Bonus not credited": "Бонус не начислен",
    "Wagering requirement": "Условия отыгрыша", "Use promo code": "Ввести промокод",
    "Game issues": "Проблема с игрой", "Play without deposit": "Играть без депозита", "Game history": "История игр",
    "Slot rules": "Правила слотов", "RTP %": "RTP %",
    "Types of bets": "Виды ставок", "How to place a bet": "Как сделать ставку", "Bet not settled": "Ставка не рассчитана",
    "Bet history": "История ставок", "Minimum bet": "Минимальная ставка",
    "Download app": "Скачать приложение", "Get code from app": "Получить код в приложении",
    "Authenticator issues": "Проблемы с аутентификатором", "Android app": "Приложение для Android",
    "Lost access to authenticator": "Нет доступа к аутентификатору",
    "Where to find rules": "Где найти правила", "Bonus terms": "Условия бонусов", "License & legality": "Лицензия",
    "Age restrictions": "Возрастные ограничения", "Responsible gaming": "Ответственная игра",
    "Contact support": "Связаться с поддержкой", "Find contacts": "Найти контакты", "Hotline": "Горячая линия",
    "File a complaint": "Подать жалобу", "Complain about operator": "Жалоба на оператора",
    "Become an agent": "Стать агентом", "Become a partner": "Стать партнером", "Affiliate program info": "Партнерская программа",
    "Propose cooperation": "Предложить сотрудничество", "Track referral stats": "Статистика рефералов",
}
CATEGORY_PROMPT = ": please choose a question or type your own."


def _lang(code: str) -> str:
    return "ru" if code in ("ru", "ky") else "en"


class DemoLLM(FakeLLM):
    def __init__(self, ru_answers: dict[str, str] | None = None):
        super().__init__()
        self.ru_answers = ru_answers or {}

    def _kb_answer(self, req: ComposeRequest) -> str | None:
        if not req.kb:
            return None
        e = req.kb[0]
        return self.ru_answers.get(e.id, e.answer) if _lang(req.language) == "ru" else e.answer

    async def compose(self, req: ComposeRequest) -> Composed:
        self.compose_calls.append(req)
        lg = _lang(req.language)
        intro = T["intro"][lg].format(name=req.persona.name) if req.introduce else ""
        if req.aggressive and req.task not in (Task.GOODBYE, Task.RISK_SUPPORT):
            intro += {"ru": "Понимаю, что ситуация неприятная, и мне жаль, что так вышло. ",
                      "en": "I understand this is frustrating, and I'm sorry about it. "}[lg]
        t = req.task
        if t is Task.ANSWER:
            answer = self._kb_answer(req)
            if answer:
                return Composed(intro + answer, "Full")
            cat = BY_KEY.get(req.category or "")
            opts: tuple[Option, ...] = ()
            if cat and cat.questions and req.attempt >= 4:
                opts = tuple(Option(cat.question_id(i), q) for i, q in enumerate(cat.questions))
            idx = max(1, min(req.attempt, 5)) - 1
            while idx < 4 and ATTEMPTS[lg][idx] in " ".join(req.previous_bot_texts):
                idx += 1  # never repeat the previous message word for word (ТЗ 14.3)
            text = ATTEMPTS[lg][idx]
            return Composed(intro + text, "NotFound" if req.attempt < 4 else "Partial", opts)
        if t is Task.ASK_WAIT_TIME:
            text = T[t][req.payment_kind or "deposit"][lg]
            return Composed(intro + text, "Partial")
        if t is Task.PAYMENT_WITHIN_NORM:
            return Composed(intro + T[t][lg] + (self._kb_answer(req) or ""), "Full")
        if t is Task.COLLECT_PAYMENT_DETAILS:
            text = T[t].get(req.missing_detail or "method", T[t]["method"])[lg]
            return Composed(intro + text, "Partial")
        result = {"CLARIFY_UNRECOGNIZED": "NotFound", "MEDIA_UNREADABLE": "NotFound", "OFF_TOPIC": "NotFound",
                  "NUDGE": "Partial", "CLARIFY_WAIT_TIME": "Partial", "RISK_SUPPORT": "Partial",
                  "DEFLECT_OPERATOR": "Partial", "HANDOFF": "Partial"}.get(t.name, "Full")
        return Composed(intro + T[t][lg], result)  # type: ignore[arg-type]

    async def localize(self, labels: list[str], language: str) -> list[str]:
        if _lang(language) != "ru":
            return list(labels)
        out = []
        for label in labels:
            if label.endswith(CATEGORY_PROMPT):
                cat = label[: -len(CATEGORY_PROMPT)]
                out.append(f"{RU_LABELS.get(cat, cat)}: выберите вопрос или напишите свой 👇")
            else:
                out.append(RU_LABELS.get(label, label))
        return out
