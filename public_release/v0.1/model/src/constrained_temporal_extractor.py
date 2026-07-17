#!/usr/bin/env python3
"""Role-aware deterministic temporal candidate extraction and normalization.

The extractor never accepts a class as an input.  It accepts two independently
predicted temporal states (has_start, has_end), chooses role-appropriate
temporal candidates, emits normalized endpoint values, and only then derives
classification from the endpoints.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score


RELEASE_ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = RELEASE_ROOT / "dataset" / "announcement_item_marks_public_v0.1.csv"
DEFAULT_PRESENCE = MODEL_ROOT / "training" / "temporal_presence_predictions.csv"
DEFAULT_OUTPUT = MODEL_ROOT / "training" / "constrained_temporal_pipeline"
TORONTO = ZoneInfo("America/Toronto")
LABELS = ["not related", "notice", "deadline", "event"]

MONTHS = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}
MONTH_PATTERN = r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
WEEKDAYS = {
    "mon": 0, "monday": 0,
    "tue": 1, "tues": 1, "tuesday": 1,
    "wed": 2, "weds": 2, "wednesday": 2,
    "thu": 3, "thur": 3, "thurs": 3, "thursday": 3,
    "fri": 4, "friday": 4,
    "sat": 5, "saturday": 5,
    "sun": 6, "sunday": 6,
}
WORD_NUMBERS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "twenty": 20,
}

MONTH_FIRST_RE = re.compile(
    rf"\b(?P<month>{MONTH_PATTERN})\.?\s*(?P<day>\d{{1,2}})\s*(?:st|nd|rd|th)?(?:\s*,?\s*(?P<year>20\d{{2}}))?\b",
    re.I,
)
DAY_FIRST_RE = re.compile(
    rf"\b(?P<day>\d{{1,2}})\s*(?:st|nd|rd|th)?\s+(?:of\s+)?(?P<month>{MONTH_PATTERN})\.?(?:\s*,?\s*(?P<year>20\d{{2}}))?\b",
    re.I,
)
ISO_RE = re.compile(r"\b(?P<year>20\d{2})[-/](?P<month>\d{1,2})[-/](?P<day>\d{1,2})\b")
NUMERIC_RE = re.compile(r"(?<![\d:])(?P<month>\d{1,2})[/-](?P<day>\d{1,2})(?:[/-](?P<year>\d{2,4}))?(?![\d:])")
WEEKDAY_RE = re.compile(
    r"\b(?P<weekday>Mon(?:day)?|Tue(?:s(?:day)?)?|Wed(?:nesday)?|Thu(?:r|rs(?:day)?)?|Fri(?:day)?|Sat(?:urday)?|Sun(?:day)?)['’]?s?\b",
    re.I,
)
RELATIVE_RE = re.compile(r"\b(today|tonight|tomorrow|this\s+(?:morning|afternoon|evening))\b", re.I)
IN_DAYS_RE = re.compile(
    r"\b(?:in\s+)?(?P<number>\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|twenty)\s+days?\b(?:\s+(?:from\s+now|away|to|until|before))?",
    re.I,
)

TIME_TOKEN = r"(?:noon|midnight|\d{1,2}(?:\s*:\s*\d{2})?\s*(?:a\.?m\.?|p\.?m\.?|[ap])?)"
RANGE_RE = re.compile(
    rf"(?P<start>{TIME_TOKEN})\s*(?P<connector>-|–|—|to|until|through|till)\s*(?P<end>{TIME_TOKEN})",
    re.I,
)
SINGLE_TIME_RE = re.compile(
    r"\b(?:noon|midnight)\b|"
    r"(?<!\d)(?:[01]?\d|2[0-3])\s*:\s*[0-5]\d\s*(?:a\.?m\.?|p\.?m\.?)?\b|"
    r"(?<!\d)\d{1,2}(?::[0-5]\d)?\s*(?:a\.?m\.?|p\.?m\.?|[ap])\b",
    re.I,
)

DUE_RE = re.compile(
    r"\b(?:due(?!\s+to)|deadline|submit|submission|complete(?:\s+\w+){0,2}\s+by|respond\s+by|register\s+by|"
    r"apply\s+by|closes?|until|extended?\b.{0,24}\bto|no\s+later\s+than|last\s+day|must\s+be\s+(?:submitted|completed|received))\b",
    re.I,
)
START_RE = re.compile(
    r"\b(?:opens?|available|released?|posted|published|begins?|start(?:s|ed|ing)?|goes?\s+live|launch(?:es|ed)?)\b",
    re.I,
)
EVENT_RE = re.compile(
    r"\b(?:session|lecture|tutorial|exam|test|midterm|quiz|office\s+hours?|workshop|"
    r"meeting|presentation|review|seminar|webinar|class|lab|showcase|check-?in|held|take\s+place|attend)\b",
    re.I,
)
CANCEL_RE = re.compile(r"\b(?:cancelled|canceled|postponed|rescheduled|no\s+class)\b", re.I)
END_OVERRIDE_RE = re.compile(r"\b(?:leave|finish|end|ends|ending)\b.{0,16}\b(?:at\s+)?(?P<time>\d{1,2}(?::\d{2})?\s*(?:a\.?m\.?|p\.?m\.?)?)", re.I)
MINUTES_TO_GO_RE = re.compile(r"\b(?:less\s+than\s+)?(?P<minutes>\d{1,3})\s+minutes?\s+to\s+go\b", re.I)
WEEKS_AFTER_RE = re.compile(r"\b(?P<number>\d+|one|two|three|four)\s+weeks?\s+after\b", re.I)


def clean(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"[ \t]+", " ", str(value)).strip()


def posted_at_toronto(value: Any) -> datetime:
    """Interpret corrected spreadsheet serials as Toronto-local wall time."""
    raw = clean(value)
    try:
        serial = float(raw)
        return datetime(1899, 12, 30) + timedelta(days=serial)
    except (TypeError, ValueError):
        pass
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(TORONTO).replace(tzinfo=None)
    except ValueError:
        return datetime(2025, 1, 1)


def infer_year(month: int, day: int, posted: datetime, explicit_year: int | None) -> datetime | None:
    year = explicit_year or posted.year
    try:
        candidate = datetime(year, month, day)
    except ValueError:
        return None
    if explicit_year is None:
        if (candidate - posted).days > 183:
            candidate = candidate.replace(year=year - 1)
        elif (posted - candidate).days > 183:
            candidate = candidate.replace(year=year + 1)
    return candidate


@dataclass(frozen=True)
class DateMention:
    value: datetime
    start: int
    end: int
    specificity: int
    text: str


@dataclass(frozen=True)
class TimeRange:
    start_minutes: int
    end_minutes: int
    start: int
    end: int
    text: str


@dataclass
class Candidate:
    segment: str
    segment_index: int
    source: str
    date: DateMention | None
    time_range: TimeRange | None
    single_minutes: list[int]
    due_cue: bool
    start_cue: bool
    event_cue: bool
    title_overlap: float
    explicit_time: bool
    inferred_date: bool = False


def month_number(value: str) -> int:
    return MONTHS[value.lower().rstrip(".")]


def date_mentions(text: str, posted: datetime) -> list[DateMention]:
    mentions: list[DateMention] = []
    occupied: list[tuple[int, int]] = []

    def add(match: re.Match[str], month: int, day: int, year: int | None, specificity: int) -> None:
        value = infer_year(month, day, posted, year)
        if value is None:
            return
        span = match.span()
        if any(span[0] < end and span[1] > start for start, end in occupied):
            return
        mentions.append(DateMention(value, span[0], span[1], specificity, match.group(0)))
        occupied.append(span)

    temporal_range_spans = [match.span() for match in RANGE_RE.finditer(text)]
    for pattern in (ISO_RE, MONTH_FIRST_RE, DAY_FIRST_RE, NUMERIC_RE):
        for match in pattern.finditer(text):
            if pattern is NUMERIC_RE and any(
                match.start() < range_end and match.end() > range_start
                for range_start, range_end in temporal_range_spans
            ):
                continue
            groups = match.groupdict()
            month_raw = groups["month"]
            month = month_number(month_raw) if not month_raw.isdigit() else int(month_raw)
            year_raw = groups.get("year")
            year = int(year_raw) if year_raw else None
            if year is not None and year < 100:
                year += 2000
            add(match, month, int(groups["day"]), year, 5 if year else 4)

    # Month propagation for lists/ranges such as "Apr 01 & 03" or "Oct 25 ... 30th".
    for base in list(mentions):
        tail = text[base.end : min(len(text), base.end + 45)]
        for follow in re.finditer(r"(?:&|and|through|until)\s*(\d{1,2})(?:st|nd|rd|th)?\b(?!\s*:)", tail, re.I):
            nearby = tail[max(0, follow.start() - 8) : min(len(tail), follow.end() + 8)]
            if follow.start() > 42 or (not follow.group(0).lower().lstrip().startswith("until") and re.search(r":|\b(?:a\.?m\.?|p\.?m\.?)\b", nearby, re.I)):
                continue
            value = infer_year(base.value.month, int(follow.group(1)), posted, base.value.year)
            if value:
                start = base.end + follow.start(1)
                mentions.append(DateMention(value, start, base.end + follow.end(1), 3, follow.group(0)))

    explicit_exists = bool(mentions)
    for match in RELATIVE_RE.finditer(text):
        token = match.group(1).lower()
        delta = 1 if token == "tomorrow" else 0
        value = (posted + timedelta(days=delta)).replace(hour=0, minute=0, second=0, microsecond=0)
        mentions.append(DateMention(value, match.start(), match.end(), 3, match.group(0)))

    for match in IN_DAYS_RE.finditer(text):
        raw = match.group("number").lower()
        days = int(raw) if raw.isdigit() else WORD_NUMBERS.get(raw, 0)
        if days:
            value = (posted + timedelta(days=days)).replace(hour=0, minute=0, second=0, microsecond=0)
            mentions.append(DateMention(value, match.start(), match.end(), 2, match.group(0)))

    # Standalone weekdays are weaker than explicit or relative dates.
    if not explicit_exists:
        for match in WEEKDAY_RE.finditer(text):
            token = re.sub(r"['’]s$", "", match.group("weekday").lower())
            weekday = WEEKDAYS.get(token[:3], WEEKDAYS.get(token))
            if weekday is None:
                continue
            delta = (weekday - posted.weekday()) % 7
            value = (posted + timedelta(days=delta)).replace(hour=0, minute=0, second=0, microsecond=0)
            mentions.append(DateMention(value, match.start(), match.end(), 1, match.group(0)))

    unique: dict[tuple[datetime, int, int], DateMention] = {}
    for mention in mentions:
        unique[(mention.value, mention.start, mention.end)] = mention
    return sorted(unique.values(), key=lambda item: (item.start, -item.specificity))


def _ampm(token: str) -> str:
    compact = token.lower().replace(".", "").strip()
    match = re.search(r"(am|pm|a|p)$", compact)
    if not match:
        return ""
    value = match.group(1)
    return "am" if value in {"am", "a"} else "pm"


def _clock(token: str) -> tuple[int, int]:
    compact = token.lower().strip()
    if "noon" in compact:
        return 12, 0
    if "midnight" in compact:
        return 0, 0
    match = re.search(r"(\d{1,2})(?:\s*:\s*(\d{2}))?", compact)
    if not match:
        raise ValueError(token)
    return int(match.group(1)), int(match.group(2) or 0)


def _to_minutes(hour: int, minute: int, marker: str) -> int:
    if marker == "pm" and hour < 12:
        hour += 12
    elif marker == "am" and hour == 12:
        hour = 0
    return hour * 60 + minute


def parse_range(start_token: str, end_token: str, context: str) -> tuple[int, int] | None:
    try:
        start_hour, start_minute = _clock(start_token)
        end_hour, end_minute = _clock(end_token)
    except ValueError:
        return None
    start_marker, end_marker = _ampm(start_token), _ampm(end_token)
    lower = context.lower()
    context_marker = "am" if "morning" in lower else "pm" if re.search(r"\b(?:afternoon|evening|tonight)\b", lower) else ""
    if not start_marker and end_marker:
        if end_marker == "pm" and start_hour == 12:
            start_marker = "pm"
        elif end_marker == "pm" and start_hour > end_hour:
            start_marker = "am"
        elif end_marker == "pm" and end_hour == 12 and start_hour < 12:
            start_marker = "am"
        else:
            start_marker = end_marker
    if start_marker and not end_marker:
        end_marker = start_marker
    if not start_marker and not end_marker:
        start_marker = end_marker = context_marker
        if not context_marker:
            start_marker = "pm" if start_hour <= 7 or start_hour == 12 else "am"
            end_marker = "pm" if end_hour <= 7 or end_hour == 12 else start_marker
    start = _to_minutes(start_hour, start_minute, start_marker)
    end = _to_minutes(end_hour, end_minute, end_marker)
    if end < start and end_marker == start_marker:
        if start_marker == "am" and end_hour <= 7:
            end = _to_minutes(end_hour, end_minute, "pm")
        elif start - end >= 8 * 60:
            end += 24 * 60
    return start, end


def time_ranges(text: str) -> list[TimeRange]:
    output: list[TimeRange] = []
    for match in RANGE_RE.finditer(text):
        raw = match.group(0)
        # Reject ordinary numeric ranges unless the context makes them time-like.
        time_like = bool(
            re.search(r":|\b(?:a\.?m\.?|p\.?m\.?|noon|midnight)\b|[ap]\s*$", raw, re.I)
            or re.search(r"\b(?:from|between|time|hours?|lecture|session|exam|test|office|tutorial|class|meeting)\b", text, re.I)
        )
        if not time_like:
            continue
        parsed = parse_range(match.group("start"), match.group("end"), text)
        if parsed is None:
            continue
        output.append(TimeRange(parsed[0], parsed[1], match.start(), match.end(), raw))
    return output


def single_times(text: str, excluded: Iterable[tuple[int, int]] = ()) -> list[tuple[int, int, int]]:
    excluded = list(excluded)
    output: list[tuple[int, int, int]] = []
    for match in SINGLE_TIME_RE.finditer(text):
        if any(match.start() < end and match.end() > start for start, end in excluded):
            continue
        token = match.group(0)
        try:
            hour, minute = _clock(token)
        except ValueError:
            continue
        if re.search(r"\bmidnight\b", token, re.I):
            output.append((0, match.start(), match.end()))
            continue
        if re.search(r"\bnoon\b", token, re.I):
            output.append((12 * 60, match.start(), match.end()))
            continue
        marker = _ampm(token)
        if not marker:
            lower = text.lower()
            marker = "am" if "morning" in lower else "pm" if re.search(r"\b(?:afternoon|evening|tonight)\b", lower) else ""
        if not marker and hour <= 7:
            marker = "pm"
        output.append((_to_minutes(hour, minute, marker), match.start(), match.end()))
    return output


def content_tokens(value: str) -> set[str]:
    stop = {"the", "a", "an", "and", "or", "to", "of", "for", "in", "on", "at", "is", "will", "with", "this", "that", "information", "reminder"}
    return {token for token in re.findall(r"[a-z0-9]+", value.lower()) if len(token) > 2 and token not in stop}


def title_overlap(title: str, segment: str) -> float:
    title_tokens = content_tokens(title)
    if not title_tokens:
        return 0.0
    return len(title_tokens & content_tokens(segment)) / len(title_tokens)


def split_segments(row: dict[str, Any]) -> list[tuple[str, str]]:
    title = clean(row.get("announcement_title"))
    body = str(row.get("body_text") or "").replace("\r", "\n")
    output: list[tuple[str, str]] = []
    if title:
        output.append(("title", title))
    for raw_line in re.split(r"\n+", body):
        line = clean(raw_line).strip("•*·-–— ")
        if not line:
            continue
        output.append(("body", line))
    return output


def nearest_date(mentions: list[DateMention], anchor_start: int, anchor_end: int) -> DateMention | None:
    if not mentions:
        return None
    def distance(item: DateMention) -> tuple[int, int]:
        if item.end < anchor_start:
            gap = anchor_start - item.end
        elif item.start > anchor_end:
            gap = item.start - anchor_end
        else:
            gap = 0
        return (gap, -item.specificity)
    return min(mentions, key=distance)


def build_candidates(row: dict[str, Any]) -> list[Candidate]:
    posted = posted_at_toronto(row.get("posted_at"))
    title = clean(row.get("announcement_title"))
    output: list[Candidate] = []
    for segment_index, (source, segment) in enumerate(split_segments(row)):
        dates = date_mentions(segment, posted)
        ranges = time_ranges(segment)
        if dates:
            ranges = [
                item
                for item in ranges
                if not any(item.start < date.end and item.end > date.start for date in dates)
            ]
        range_spans = [(item.start, item.end) for item in ranges]
        singles = single_times(segment, range_spans)
        common = {
            "segment": segment,
            "segment_index": segment_index,
            "source": source,
            "due_cue": bool(DUE_RE.search(segment)),
            "start_cue": bool(START_RE.search(segment)),
            "event_cue": bool(EVENT_RE.search(segment)),
            "title_overlap": title_overlap(title, segment),
        }
        if ranges:
            for item in ranges:
                output.append(
                    Candidate(
                        **common,
                        date=nearest_date(dates, item.start, item.end),
                        time_range=item,
                        single_minutes=[value for value, _, _ in singles],
                        explicit_time=True,
                    )
                )
        elif dates:
            for item in dates:
                nearby = sorted(singles, key=lambda value: min(abs(value[1] - item.end), abs(item.start - value[2])))
                output.append(
                    Candidate(
                        **common,
                        date=item,
                        time_range=None,
                        single_minutes=[value for value, _, _ in nearby],
                        explicit_time=bool(nearby),
                    )
                )
        elif singles:
            output.append(
                Candidate(
                    **common,
                    date=None,
                    time_range=None,
                    single_minutes=[value for value, _, _ in singles],
                    explicit_time=True,
                    inferred_date=True,
                )
            )
    return output


def candidate_score(candidate: Candidate, role: str) -> float:
    score = 4.0 if candidate.source == "body" else 0.0
    time_label = bool(re.match(r"\s*(?:date\s+and\s+)?time\s*:", candidate.segment, re.I))
    if time_label:
        score += 6.0
    score += candidate.title_overlap * 3.0
    if candidate.date:
        score += candidate.date.specificity * 0.6
    if candidate.explicit_time:
        score += 1.0
    if role == "event":
        score += 7.0 if candidate.time_range else -1.0
        score += 2.5 if candidate.event_cue else 0.0
        score -= 2.0 if candidate.due_cue and not candidate.event_cue else 0.0
        score -= 5.0 if re.search(r"\banother\s+(?:course\s+)?(?:midterm|exam|test)\b", candidate.segment, re.I) else 0.0
        score += 3.0 if re.search(r"\b(?:specifically|confirmed?|booked|official)\b", candidate.segment, re.I) else 0.0
    elif role == "deadline":
        score += 7.0 if candidate.due_cue else 0.0
        score -= 2.0 if candidate.start_cue and not candidate.due_cue else 0.0
        score += 0.5 if candidate.time_range else 0.0
        if candidate.date:
            score += candidate.date.start * 0.01
        if re.search(r"\bextend(?:ed|ing)?\b", candidate.segment, re.I):
            score += 4.0
    else:
        actionable_start = candidate.start_cue and not re.search(r"\bstart\s+(?:you|thinking|to\s+think)\b", candidate.segment, re.I)
        score += 6.0 if actionable_start else 0.0
        score -= 3.0 if candidate.due_cue and not candidate.start_cue else 0.0
        score += 1.0 if candidate.event_cue else 0.0
        score -= 4.0 if candidate.time_range and not candidate.start_cue else 0.0
        if candidate.date:
            score += candidate.date.start * 0.01
            if re.search(r"\bfollowing\s+(?:tutorial|lecture|session|class)\b", candidate.segment, re.I):
                score += 4.2 - candidate.date.start * 0.02
        if candidate.source == "title" and CANCEL_RE.search(candidate.segment):
            score += 6.0
    # Prefer later body clauses only weakly; cue/specificity remains dominant.
    score += min(candidate.segment_index, 20) * 0.01
    return score


def format_datetime(date: datetime, minutes: int, inferred_time: bool) -> str:
    if minutes >= 24 * 60:
        date += timedelta(days=minutes // (24 * 60))
        minutes %= 24 * 60
    value = date.replace(hour=minutes // 60, minute=minutes % 60, second=0, microsecond=0)
    return value.strftime("%Y-%m-%d %H:%M") + (" (time inferred)" if inferred_time else "")


def choose_candidate(row: dict[str, Any], role: str) -> Candidate | None:
    candidates = build_candidates(row)
    if not candidates:
        return None
    return max(candidates, key=lambda item: candidate_score(item, role))


def extract_endpoints(row: dict[str, Any], has_start: bool, has_end: bool) -> tuple[str, str]:
    if not has_start and not has_end:
        return "", ""
    role = "event" if has_start and has_end else "notice" if has_start else "deadline"
    posted = posted_at_toronto(row.get("posted_at"))
    # Notices are anchored to announcement publication, never to a future date
    # mentioned in the title/body. This is the reviewed dataset contract.
    if role == "notice":
        return posted.strftime("%Y-%m-%d %H:%M"), ""
    candidate = choose_candidate(row, role)
    if candidate is None:
        if role == "deadline":
            weeks = WEEKS_AFTER_RE.search(clean(row.get("body_text")))
            if weeks:
                raw = weeks.group("number").lower()
                count = int(raw) if raw.isdigit() else WORD_NUMBERS.get(raw, 0)
                value = posted + timedelta(days=count * 7 + (1 if posted.hour >= 20 else 0))
                return "", value.strftime("%Y-%m-%d 23:59") + " (time inferred)"
            duration = MINUTES_TO_GO_RE.search(clean(row.get("body_text")))
            if duration:
                value = posted + timedelta(minutes=int(duration.group("minutes")))
                rounded_minutes = int(round(value.minute / 15.0) * 15)
                value = value.replace(minute=0) + timedelta(minutes=rounded_minutes)
                return "", value.strftime("%Y-%m-%d %H:%M") + " (time inferred)"
        date = posted.replace(hour=0, minute=0, second=0, microsecond=0)
        if role == "notice":
            return format_datetime(date, 0, True), ""
        if role == "deadline":
            return "", format_datetime(date, 23 * 60 + 59, True)
        return format_datetime(date, 0, True), format_datetime(date, 23 * 60 + 59, True)

    if candidate.date:
        date = candidate.date.value
    else:
        global_text = f"{clean(row.get('announcement_title'))}\n{clean(row.get('body_text'))}"
        global_dates = date_mentions(global_text, posted)
        best_global = max(global_dates, key=lambda item: (item.specificity, item.start), default=None)
        date = best_global.value if best_global else posted.replace(hour=0, minute=0, second=0, microsecond=0)
    start = end = ""
    if candidate.time_range:
        start_minutes = candidate.time_range.start_minutes
        end_minutes = candidate.time_range.end_minutes
        if role == "notice" and candidate.due_cue:
            return format_datetime(date, 0, True), ""
        override = END_OVERRIDE_RE.search(candidate.segment)
        if override:
            try:
                override_hour, override_minute = _clock(override.group("time"))
                override_marker = _ampm(override.group("time"))
                if not override_marker:
                    override_marker = "pm" if end_minutes >= 12 * 60 else "am"
                end_minutes = _to_minutes(override_hour, override_minute, override_marker)
            except ValueError:
                pass
        if has_start:
            start = format_datetime(date, start_minutes, False)
        if has_end:
            end = format_datetime(date, end_minutes, False)
        return start, end

    values = candidate.single_minutes
    if role == "notice":
        # A notice topology attached to a due-date phrase means the dated item
        # starts on that date; the due clock belongs to an end role and must not
        # be copied into start_time.
        use_explicit = bool(values) and (not candidate.due_cue or candidate.start_cue)
        start = format_datetime(date, values[0] if use_explicit else 0, not use_explicit)
    elif role == "deadline":
        if values:
            if values[-1] == 0 and re.search(r"\bmidnight\b", candidate.segment, re.I):
                end = format_datetime(date, 23 * 60 + 59, True)
            else:
                end = format_datetime(date, values[0], False)
        else:
            duration = MINUTES_TO_GO_RE.search(candidate.segment)
            if duration:
                value = posted + timedelta(minutes=int(duration.group("minutes")))
                rounded_minutes = int(round(value.minute / 15.0) * 15)
                value = value.replace(minute=0) + timedelta(minutes=rounded_minutes)
                end = value.strftime("%Y-%m-%d %H:%M") + " (time inferred)"
            else:
                end = format_datetime(date, 23 * 60 + 59, True)
    else:
        if len(values) >= 2:
            start = format_datetime(date, values[0], False)
            end = format_datetime(date, values[1], False)
        elif values:
            start = format_datetime(date, values[0], False)
            end = format_datetime(date, values[0], False)
        else:
            start = format_datetime(date, 0, True)
            end = format_datetime(date, 23 * 60 + 59, True)
    return start, end


def derive_classification(start_time: str, end_time: str) -> str:
    if start_time and end_time:
        return "event"
    if start_time:
        return "notice"
    if end_time:
        return "deadline"
    return "not related"


def normalize(value: Any) -> str:
    return re.sub(r"\s+", " ", clean(value).lower()).strip()


def evaluate_pipeline(frame: pd.DataFrame, presence: pd.DataFrame, output_csv: Path) -> dict[str, Any]:
    merged = frame.merge(
        presence[["reviewed_row_id", "predicted_has_start", "predicted_has_end"]],
        on="reviewed_row_id",
        how="inner",
        validate="one_to_one",
    )
    records = []
    for _, row in merged.iterrows():
        has_start = str(row["predicted_has_start"]).lower() == "true"
        has_end = str(row["predicted_has_end"]).lower() == "true"
        start_time, end_time = extract_endpoints(row.to_dict(), has_start, has_end)
        prediction = derive_classification(start_time, end_time)
        records.append(
            {
                "reviewed_row_id": row["reviewed_row_id"],
                "source_row": row["source_row"],
                "source_group_id": row["source_group_id"],
                "true_classification": row["classification"],
                "predicted_classification": prediction,
                "true_start_time": row["start_time"],
                "predicted_start_time": start_time,
                "true_end_time": row["end_time"],
                "predicted_end_time": end_time,
            }
        )
    output = pd.DataFrame(records)
    for field in ("classification", "start_time", "end_time"):
        output[f"correct_{field}"] = output[f"true_{field}"].map(normalize).eq(output[f"predicted_{field}"].map(normalize))
    output.to_csv(output_csv, index=False)
    class_report = classification_report(
        output["true_classification"], output["predicted_classification"], labels=LABELS, output_dict=True, zero_division=0
    )
    metrics: dict[str, Any] = {
        "rows": len(output),
        "classification": {
            "accuracy": float(output["correct_classification"].mean()),
            "macro_f1": float(f1_score(output["true_classification"], output["predicted_classification"], labels=LABELS, average="macro", zero_division=0)),
            "per_class_f1": {label: float(class_report[label]["f1-score"]) for label in LABELS},
        },
    }
    for field in ("start_time", "end_time"):
        truth_nonblank = output[f"true_{field}"].map(bool)
        metrics[field] = {
            "exact_match_accuracy": float(output[f"correct_{field}"].mean()),
            "nonblank_exact_match_accuracy": float(output.loc[truth_nonblank, f"correct_{field}"].mean()),
            "presence_accuracy": float(
                np.mean(output[f"true_{field}"].map(bool).to_numpy() == output[f"predicted_{field}"].map(bool).to_numpy())
            ),
        }
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--presence", type=Path, default=DEFAULT_PRESENCE)
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--oracle-presence", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(args.data, dtype=str, keep_default_na=False)
    frame = data[
        data["dataset_split"].eq(args.split) & data["row_origin"].eq("reviewed_real")
    ].copy().reset_index(drop=True)
    if args.oracle_presence:
        presence = frame[["reviewed_row_id"]].copy()
        presence["predicted_has_start"] = frame["start_time"].ne("")
        presence["predicted_has_end"] = frame["end_time"].ne("")
        suffix = "oracle"
    else:
        presence = pd.read_csv(args.presence, dtype=str, keep_default_na=False)
        suffix = "predicted"
    output_csv = args.output_dir / f"{args.split}_{suffix}_predictions.csv"
    metrics = evaluate_pipeline(frame, presence, output_csv)
    metrics.update(
        {
            "split": args.split,
            "presence_source": "gold endpoint presence (diagnostic oracle)" if args.oracle_presence else str(args.presence),
            "classification_rule": "both=event; start-only=notice; end-only=deadline; neither=not related",
        }
    )
    metrics_path = args.output_dir / f"{args.split}_{suffix}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
