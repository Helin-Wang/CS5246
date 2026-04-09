"""
Disaster event time extractor.

Entry point:
    extract_event_time(text, title, article_timestamp, event_type) -> TimeResult

Strategy (5 steps, in priority order):
  1. Explicit absolute date in text (highest confidence)
     Scan trigger-sentence ±1, then full text for Month-Day-Year patterns.
  2. Trigger-word ±1 sentence relative date
     Relative expressions ("last Friday", "on Tuesday") in sentences near
     disaster trigger verbs, resolved against article_timestamp.
  3. DR-specific "since [month/year]" start-point extraction.
  4. Full-text relative time scan (degraded precision for vague terms).
  5. Fallback to article_timestamp.date() — only for EQ/TC/WF/FL.
     DR returns unknown when nothing extracted.

Output fields:
    event_date      YYYY-MM-DD str, or None
    granularity     "day" | "month" | "year" | "unknown"
    time_type       "event_date" | "date_range" | "duration_only" | "unknown"
    raw_expression  verbatim snippet from text, or None
    method          which step succeeded, for debugging
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional

import dateparser

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TRIGGER_VERBS = re.compile(
    r"\b(struck|hit|made landfall|broke out|broken out|swept|battered|"
    r"devastated|ravaged|killed|claimed|destroyed|flooded|inundated|"
    r"submerged|erupted|triggered|caused|started|ignited|forced|ripped|"
    r"slammed|pounded|lashed|tore|displaced|affected|braced|approaching|"
    r"intensif|strengthen|washed away|"
    r"jolted|shook|shaken|rattled|rocked|quake|temblor|"  # earthquake
    r"made its way|struck down|cut off|overwhelmed)\b",     # general
    re.IGNORECASE,
)

# Sentences that describe aftermath/recovery — dates here are NOT event dates
_AFTERMATH_PAT = re.compile(
    r"\b(clean.?up|cleanup|recovery|rebuild|aftermath|"
    r"restoration|damage assessment|reopen(?:ed)?|resum(?:ed)?|"
    r"month(?:s)? (?:have |has )?pass|weeks? (?:have |has )?pass|"
    r"year(?:s)? (?:have |has )?pass|months? later|years? later|"
    r"anniversary|commemor|in the wake of)\b",
    re.IGNORECASE,
)

# Comparison "since YEAR": "driest/worst/best/... since 2008" — NOT a drought start
_COMPARISON_SINCE = re.compile(
    r"\b(?:driest|wettest|hottest|coldest|worst|best|highest|lowest|"
    r"biggest|largest|deepest|rarest|most severe|record)\b.{0,40}\bsince\s+\d{4}",
    re.IGNORECASE,
)

# Months for pattern matching
_MONTHS = (
    r"January|February|March|April|May|June|July|August|September|"
    r"October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec"
)

# Absolute date with day: "August 14", "Aug. 14, 2025", "14 August 2025", "2025-08-14"
_ABS_DATE_PAT = re.compile(
    rf"(?:(?:{_MONTHS})\.?\s+\d{{1,2}}(?:st|nd|rd|th)?"
    rf"(?:\s*,\s*\d{{4}})?)"
    rf"|(?:\d{{1,2}}(?:st|nd|rd|th)?\s+(?:{_MONTHS})\.?(?:\s+\d{{4}})?)"
    rf"|\d{{4}}-\d{{2}}-\d{{2}}",
    re.IGNORECASE,
)

# Month-only mention: "in January", "in January 2025"
_MONTH_ONLY_PAT = re.compile(
    rf"\bin\s+(?:early\s+|late\s+|mid-?\s*)?({_MONTHS})(?:\s+(\d{{4}}))?",
    re.IGNORECASE,
)

# Since-start patterns for DR
_SINCE_PAT = re.compile(
    rf"since\s+(?:early\s+|late\s+|mid-?\s*)?(?:({_MONTHS})(?:\s+(\d{{4}}))?"
    rf"|(\d{{4}}))",
    re.IGNORECASE,
)

# Vague relative terms → degrade to month precision
_VAGUE_TERMS = re.compile(
    r"\b(this week|last week|recent days|recent weeks|in recent|over the past|"
    r"the past few|past several|past \d+ weeks|past \d+ days)\b",
    re.IGNORECASE,
)

# Relative terms that dateparser can resolve precisely
_RELATIVE_PAT = re.compile(
    r"\b(yesterday|last (?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)|"
    r"on (?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)|"
    r"(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)(?:\s+(?:morning|afternoon|evening|night))?|"
    r"\d+ days? ago|\d+ weeks? ago)\b",
    re.IGNORECASE,
)

DATEPARSER_SETTINGS = {
    "PREFER_DAY_OF_MONTH": "first",
    "RETURN_AS_TIMEZONE_AWARE": False,
    "PREFER_LOCALE_DATE_ORDER": False,
}

DR_TYPES = {"drought"}
NO_FALLBACK_TYPES = {"drought"}


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class TimeResult:
    event_date:     Optional[str] = None   # YYYY-MM-DD
    granularity:    str           = "unknown"
    time_type:      str           = "unknown"
    raw_expression: Optional[str] = None
    method:         str           = "none"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _split_sentences(text: str) -> list[str]:
    """Simple sentence splitter that keeps [SEP] titles separate."""
    text = re.sub(r"\s+", " ", text).strip()
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def _parse_date(expr: str, base: datetime) -> Optional[date]:
    """Run dateparser on expr with base as reference. Returns date or None."""
    settings = dict(DATEPARSER_SETTINGS)
    settings["RELATIVE_BASE"] = base
    try:
        result = dateparser.parse(expr, settings=settings)
        if result is None:
            return None
        # Sanity: must be within 5 years of base
        diff = abs((result.date() - base.date()).days)
        if diff > 5 * 365:
            return None
        return result.date()
    except Exception:
        return None


def _date_to_str(d: date) -> str:
    return d.strftime("%Y-%m-%d")


def _find_trigger_sentences(sentences: list[str]) -> list[int]:
    """Return indices of sentences containing a trigger verb."""
    return [i for i, s in enumerate(sentences) if TRIGGER_VERBS.search(s)]


def _sentences_near_triggers(sentences: list[str], window: int = 1) -> list[str]:
    """Return sentences within ±window of any trigger sentence (deduplicated)."""
    trigger_idx = set(_find_trigger_sentences(sentences))
    near = set()
    for i in trigger_idx:
        for j in range(max(0, i - window), min(len(sentences), i + window + 1)):
            near.add(j)
    return [sentences[i] for i in sorted(near)]


# ---------------------------------------------------------------------------
# Step implementations
# ---------------------------------------------------------------------------

def _step1_absolute(sentences: list[str], base: datetime) -> Optional[TimeResult]:
    """Step 1: explicit absolute date.
    - Trigger sentences ±1: accept any match.
    - Full text (non-trigger): skip aftermath/cleanup sentences.
    Also extracts month-only mentions ("in January 2025") at month precision.
    """
    trigger_sents = set(_sentences_near_triggers(sentences))
    non_trigger = [s for s in sentences if s not in trigger_sents
                   and not _AFTERMATH_PAT.search(s)]

    # Pass 1: trigger sentences (skip aftermath even within ±1 window)
    for sent in trigger_sents:
        if _AFTERMATH_PAT.search(sent):
            continue
        for m in _ABS_DATE_PAT.finditer(sent):
            expr = m.group(0)
            d = _parse_date(expr, base)
            if d:
                return TimeResult(
                    event_date=_date_to_str(d),
                    granularity="day",
                    time_type="event_date",
                    raw_expression=expr,
                    method="step1_absolute",
                )

    # Pass 2: non-aftermath sentences (day precision)
    for sent in non_trigger:
        for m in _ABS_DATE_PAT.finditer(sent):
            expr = m.group(0)
            d = _parse_date(expr, base)
            if d:
                return TimeResult(
                    event_date=_date_to_str(d),
                    granularity="day",
                    time_type="event_date",
                    raw_expression=expr,
                    method="step1_absolute",
                )

    # Pass 3: month-only mentions (trigger sentences first)
    for sent in list(trigger_sents) + non_trigger:
        for m in _MONTH_ONLY_PAT.finditer(sent):
            month_name = m.group(1)
            year_str   = m.group(2)
            year = int(year_str) if year_str else base.year
            d = _parse_date(f"{month_name} 1 {year}", base)
            if d:
                # Roll back year if date is in the future
                if d > base.date():
                    d = date(d.year - 1, d.month, 1)
                return TimeResult(
                    event_date=_date_to_str(d),
                    granularity="month",
                    time_type="event_date",
                    raw_expression=m.group(0),
                    method="step1_month_only",
                )
    return None


def _step2_trigger_relative(sentences: list[str], base: datetime) -> Optional[TimeResult]:
    """Step 2: relative expressions near trigger verbs."""
    near = _sentences_near_triggers(sentences)
    for sent in near:
        for m in _RELATIVE_PAT.finditer(sent):
            expr = m.group(0)
            d = _parse_date(expr, base)
            if d:
                return TimeResult(
                    event_date=_date_to_str(d),
                    granularity="day",
                    time_type="event_date",
                    raw_expression=expr,
                    method="step2_trigger_relative",
                )
    return None


def _step3_drought_since(text: str, base: datetime) -> Optional[TimeResult]:
    # Pre-filter: remove comparison phrases like "driest since 2008"
    text = _COMPARISON_SINCE.sub("", text)
    """Step 3 (DR only): since [month/year] start point."""
    for m in _SINCE_PAT.finditer(text):
        month_str = m.group(1)  # month name
        year_str = m.group(2)   # optional year after month
        year_only = m.group(3)  # year-only form

        if year_only:
            # "since 2022"
            try:
                yr = int(year_only)
                d = date(yr, 1, 1)
                return TimeResult(
                    event_date=_date_to_str(d),
                    granularity="year",
                    time_type="date_range",
                    raw_expression=m.group(0),
                    method="step3_since_year",
                )
            except ValueError:
                continue
        elif month_str:
            # "since May" or "since May 2023"
            year_to_use = year_str if year_str else str(base.year)
            expr = f"{month_str} 1 {year_to_use}"
            d = _parse_date(expr, base)
            if d:
                # If inferred year puts date in the future, roll back one year
                if d > base.date():
                    d = date(d.year - 1, d.month, d.day)
                return TimeResult(
                    event_date=_date_to_str(d),
                    granularity="month",
                    time_type="date_range",
                    raw_expression=m.group(0),
                    method="step3_since_month",
                )
    return None


def _step4_fulltext_relative(sentences: list[str], base: datetime) -> Optional[TimeResult]:
    """Step 4: full-text relative scan; vague terms → month precision."""
    # First try precise relative expressions in any sentence
    for sent in sentences:
        for m in _RELATIVE_PAT.finditer(sent):
            expr = m.group(0)
            d = _parse_date(expr, base)
            if d:
                return TimeResult(
                    event_date=_date_to_str(d),
                    granularity="day",
                    time_type="event_date",
                    raw_expression=expr,
                    method="step4_fulltext_relative",
                )

    # Then accept vague terms at week/month precision
    full_text = " ".join(sentences)
    m = _VAGUE_TERMS.search(full_text)
    if m:
        expr = m.group(0).lower()
        d = base.date()
        if "this week" in expr:
            # Monday of the current week
            monday = d - __import__("datetime").timedelta(days=d.weekday())
            event_d, gran = monday, "day"
        elif "last week" in expr:
            monday = d - __import__("datetime").timedelta(days=d.weekday() + 7)
            event_d, gran = monday, "day"
        else:
            # "recent days/weeks" → use start of current month as proxy
            event_d = date(d.year, d.month, 1)
            gran = "month"
        return TimeResult(
            event_date=_date_to_str(event_d),
            granularity=gran,
            time_type="event_date",
            raw_expression=m.group(0),
            method="step4_vague",
        )
    return None


def _step5_fallback(base: datetime, event_type: str, sentences: list[str]) -> Optional[TimeResult]:
    """Step 5: fallback to article timestamp date (not for DR).

    Skipped if article is clearly retrospective: contains aftermath language
    AND no trigger verb in the first 3 sentences (= not a breaking-news article).
    """
    if event_type in NO_FALLBACK_TYPES:
        return None

    first_3 = " ".join(sentences[:3])
    has_early_trigger = bool(TRIGGER_VERBS.search(first_3))

    # Retrospective check: aftermath keywords present + no trigger in first 3 sentences
    has_aftermath = bool(_AFTERMATH_PAT.search(" ".join(sentences)))
    if has_aftermath and not has_early_trigger:
        return None  # Likely a retrospective/analysis article; return unknown

    return TimeResult(
        event_date=_date_to_str(base.date()),
        granularity="day",
        time_type="event_date",
        raw_expression=None,
        method="step5_fallback",
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def extract_event_time(
    text: str,
    title: str = "",
    article_timestamp: Optional[datetime] = None,
    event_type: str = "",
) -> TimeResult:
    """
    Extract the primary disaster event time from article text.

    Args:
        text: article body (after [SEP] if present)
        title: article title (prepended to sentences for trigger detection)
        article_timestamp: GKG DATE as datetime (used as RELATIVE_BASE)
        event_type: disaster category string, e.g. "earthquake", "drought"

    Returns:
        TimeResult with event_date (YYYY-MM-DD or None), granularity, etc.
    """
    if article_timestamp is None:
        article_timestamp = datetime.now()

    event_type = event_type.lower().strip()

    # Normalise text; split title from [SEP] if not already provided
    full_text = str(text or "")
    if not title and " [SEP] " in full_text:
        parts = full_text.split(" [SEP] ", 1)
        title, full_text = parts[0], parts[1]

    combined = (title + ". " + full_text).strip() if title else full_text
    sentences = _split_sentences(combined)

    # Step 1 — explicit absolute date
    result = _step1_absolute(sentences, article_timestamp)
    if result:
        return result

    # Step 2 — trigger-bound relative date
    result = _step2_trigger_relative(sentences, article_timestamp)
    if result:
        return result

    # Step 3 — DR "since" start point
    if event_type in DR_TYPES:
        result = _step3_drought_since(combined, article_timestamp)
        if result:
            return result

    # Step 4 — full-text relative / vague
    result = _step4_fulltext_relative(sentences, article_timestamp)
    if result:
        return result

    # Step 5 — fallback
    result = _step5_fallback(article_timestamp, event_type, sentences)
    if result:
        return result

    return TimeResult()
