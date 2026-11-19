"""Language rules (ТЗ п.8).

* 12 supported languages; the bot also understands others if the model does.
* Reply language = language of the LAST user message.
* If the language cannot be determined -> English.
"""
from __future__ import annotations

SUPPORTED_LANGUAGES: dict[str, str] = {
    "ar": "Arabic",
    "en": "English",
    "ky": "Kyrgyz",
    "es": "Spanish",
    "fr": "French",
    "ru": "Russian",
    "sw": "Swahili",
    "am": "Amharic",
    "af": "Afrikaans",
    "zu": "Zulu",
    "tr": "Turkish",
    "az": "Azerbaijani",
}
DEFAULT_LANGUAGE = "en"


def normalize(code: str | None) -> str | None:
    """'en-US' / 'EN_us' / 'en' -> 'en'. Returns None for empty input."""
    if not code:
        return None
    return code.replace("_", "-").split("-")[0].strip().lower() or None


def choose_language(detected: str | None, platform_code: str | None) -> str:
    """Pick the reply language.

    detected      - language of the last message as detected by the LLM / detector
    platform_code - languageCode from the Consultant event (client UI language)

    Text wins over platform settings (the user may write in another language
    than the UI). Any language the model can handle is allowed, not only the
    12 supported ones; if nothing is known we fall back to English.
    """
    return normalize(detected) or normalize(platform_code) or DEFAULT_LANGUAGE
