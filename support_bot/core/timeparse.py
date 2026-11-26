"""How long ago was the payment: calendar arithmetic in code, not in the model (test review п.5).

The model only extracts WHAT the user said ("yesterday evening", "on Saturday", "22/11", "24/11 at 10 am")
as a small structure; this module turns it into a range of hours in the user's time zone:

* a clock time -> an exact moment;
* a part of the day -> a range (morning 06-12, afternoon 12-18, evening 18-24, night 00-06);
* a day without a time -> the whole day (00:00-24:00), so "22/11" is not counted from midnight;
* a weekday -> the last such day before today, by the calendar ("on Saturday" on Tuesday 24.11 = 21.11).

If the range lies on both sides of 24 hours the engine asks for the hour. Without a time zone (no GEO)
a calendar answer cannot be turned into hours: the engine asks how many hours have passed.
"""
from __future__ import annotations

import datetime as dt
import re

PARTS = {"morning": (6, 12), "afternoon": (12, 18), "day": (10, 18), "evening": (18, 24), "night": (0, 6)}
WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")

# Single-zone countries of the persona pools (Consultant normally sends geo.timezone itself).
COUNTRY_TZ = {
    "EG": "Africa/Cairo", "KG": "Asia/Bishkek", "AR": "America/Argentina/Buenos_Aires", "CI": "Africa/Abidjan",
    "JO": "Asia/Amman", "TZ": "Africa/Dar_es_Salaam", "NG": "Africa/Lagos", "ET": "Africa/Addis_Ababa",
    "KE": "Africa/Nairobi", "ZM": "Africa/Lusaka", "LR": "Africa/Monrovia", "GH": "Africa/Accra",
    "ZA": "Africa/Johannesburg", "TR": "Europe/Istanbul", "AZ": "Asia/Baku", "SA": "Asia/Riyadh",
    "CG": "Africa/Brazzaville", "CD": "Africa/Kinshasa",
}


def _day(when: dict, today: dt.date) -> dt.date | None:
    raw = when.get("date")
    if raw:
        try:
            day = dt.date.fromisoformat(str(raw)[:10])
        except ValueError:
            day = None
        if day is not None:
            if day > today:  # "22/11" said in January means last year's November
                try:
                    day = day.replace(year=day.year - 1)
                except ValueError:
                    return None
            return day
    if when.get("days_ago") is not None:
        try:
            return today - dt.timedelta(days=int(float(when["days_ago"])))
        except (TypeError, ValueError):
            return None
    weekday = str(when.get("weekday") or "").strip().lower()
    if weekday in WEEKDAYS:
        back = (today.weekday() - WEEKDAYS.index(weekday)) % 7 or 7  # the last such day BEFORE today
        return today - dt.timedelta(days=back)
    return None


def _clock(raw) -> tuple[int, int] | None:
    m = re.match(r"^\s*(\d{1,2})(?::(\d{2}))?\s*$", str(raw or ""))
    if not m:
        return None
    h, mi = int(m.group(1)), int(m.group(2) or 0)
    return (h, mi) if h < 24 and mi < 60 else None


def hours_since(when: dict | None, now_local: dt.datetime) -> tuple[float, float] | None:
    """(min hours, max hours) since the moment described by ``when``; None if nothing usable."""
    if not when:
        return None
    today = now_local.date()
    day = _day(when, today)
    clock = _clock(when.get("time"))
    part = str(when.get("part_of_day") or "").strip().lower()
    if day is None:
        if clock is None:
            return None
        day = today  # "at 10 am" -> today (or yesterday if that is still in the future)
        if dt.datetime.combine(day, dt.time(*clock), now_local.tzinfo) > now_local:
            day = today - dt.timedelta(days=1)
    tz = now_local.tzinfo
    if clock is not None:
        start = end = dt.datetime.combine(day, dt.time(*clock), tz)
    elif part in PARTS:
        lo, hi = PARTS[part]
        start = dt.datetime.combine(day, dt.time(lo), tz)
        end = dt.datetime.combine(day, dt.time(0), tz) + dt.timedelta(hours=hi)
    else:
        start = dt.datetime.combine(day, dt.time(0), tz)
        end = start + dt.timedelta(days=1)
    end = min(end, now_local)
    if start > now_local:
        return None
    to_h = lambda t: round((now_local - t).total_seconds() / 3600, 1)  # noqa: E731
    return to_h(end), to_h(start)
