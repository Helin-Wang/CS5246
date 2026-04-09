"""
Unified event field extractor for disaster news.

This module keeps a single extractor API, but internally separates field logic
into hazard-aware rules so the output stays aligned with downstream severity
models:

- EQ: magnitude, depth, rapidpopdescription
- TC: maximum_wind_speed_kmh, maximum_storm_surge_m, exposed_population
- WF: duration_days, burned_area_ha, people_affected
- DR: duration_days, affected_area_km2, affected_country_count
- FL: dead, displaced

It also emits auxiliary metrics for stock-impact analysis, plus confidence and
evidence for each extracted field.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Pattern, Sequence, Tuple

import pandas as pd


JSON_COLUMN = "event_features"

# Maps DistilBERT label strings (and abbreviations) → internal two-letter code.
LABEL_TO_TYPE_CODE: Dict[str, str] = {
    "earthquake": "EQ", "eq": "EQ",
    "cyclone": "TC", "tc": "TC", "typhoon": "TC", "hurricane": "TC",
    "wildfire": "WF", "wf": "WF", "bushfire": "WF",
    "drought": "DR", "dr": "DR",
    "flood": "FL", "fl": "FL", "flooding": "FL",
}

POP_UNIT_MULTIPLIER = {
    "k": 1_000.0,
    "thousand": 1_000.0,
    "m": 1_000_000.0,
    "million": 1_000_000.0,
    "b": 1_000_000_000.0,
    "billion": 1_000_000_000.0,
}

_N = r"\d[\d,]*(?:\.\d+)?"
_NS = _N + r"(?:\s*(?:k|m|b|thousand|million|billion))?"


def normalize_text(text: str) -> str:
    text = str(text or "")
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_sentences(text: str) -> List[str]:
    text = normalize_text(text)
    if not text:
        return []
    parts = re.split(r"(?<=[\.\!\?])\s+", text)
    return [part.strip() for part in parts if part.strip()]


def sentence_for_span(sentences: List[str], full_text: str, start: int, end: int) -> str:
    if not sentences:
        return ""
    running = 0
    for sent in sentences:
        idx = full_text.find(sent, running)
        if idx < 0:
            continue
        s0, s1 = idx, idx + len(sent)
        running = s1
        if start >= s0 and end <= s1:
            return sent
    return ""


def parse_number(num_text: str) -> Optional[float]:
    text = normalize_text(num_text).lower().replace(",", "")
    if not text:
        return None

    m = re.fullmatch(r"(?P<num>\d+(?:\.\d+)?)\s*(?P<scale>k|m|b|thousand|million|billion)?", text)
    if not m:
        return None

    value = float(m.group("num"))
    scale = m.group("scale")
    if scale:
        value *= POP_UNIT_MULTIPLIER[scale]
    return value


def to_float(x) -> Optional[float]:
    try:
        value = float(x)
    except Exception:
        return None
    if math.isnan(value):
        return None
    return value


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def is_missing(value) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    try:
        return math.isnan(float(value))
    except Exception:
        return False


def jsonable_value(value):
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


@dataclass
class Candidate:
    field: str
    value: object
    unit: str
    confidence: float
    raw: str
    evidence: str
    start: int
    end: int


@dataclass(frozen=True)
class FieldRule:
    field: str
    patterns: Tuple[Pattern[str], ...]
    anchors: Tuple[str, ...]
    event_types: Tuple[str, ...] = ()
    value_kind: str = "numeric"
    base_confidence: float = 0.52
    min_binding_score: float = 0.18


class UnifiedEventExtractor:
    def __init__(self):
        self.sector_keywords = {
            "energy": ("power grid", "electricity", "oil", "gas", "refinery", "pipeline"),
            "transportation": ("airport", "flight", "rail", "port", "shipping", "highway"),
            "agriculture": ("crop", "harvest", "farmland", "livestock", "agriculture"),
            "insurance": ("insured losses", "claims", "insurer", "reinsurance"),
            "tourism": ("tourism", "hotel", "resort", "tourist"),
            "technology": ("semiconductor", "chip plant", "data center", "factory"),
            "mining": ("mine", "copper", "lithium", "iron ore"),
        }
        self.negation_cues = (
            "no",
            "not",
            "without",
            "zero",
            "none",
            "unlikely",
            "unconfirmed",
            "could",
            "may",
            "might",
        )
        self.low_confidence_key_fields = {
            "EQ": ("magnitude", "depth"),
            "TC": ("maximum_wind_speed_kmh",),
            "WF": ("burned_area_ha", "people_affected"),
            "DR": ("duration_days",),
            "FL": ("dead", "displaced"),
        }
        self.field_rules = self._build_field_rules()

    def _build_field_rules(self) -> List[FieldRule]:
        return [
            FieldRule(
                field="magnitude",
                patterns=(
                    # magnitude N / richter N
                    re.compile(rf"(?:magnitude|richter|mw|mb|ml|ms)\s*[:\-]?\s*(?P<num>{_N})", re.I),
                    # M6.2 / M 6.2
                    re.compile(rf"\bM\s*(?P<num>{_N})\b", re.I),
                    # N magnitude / N-magnitude / N richter
                    re.compile(rf"(?P<num>{_N})[- ]?(?:magnitude|richter|mw|mb|ml|ms)\b", re.I),
                ),
                anchors=("magnitude", "richter", "earthquake", "quake", "aftershock"),
                event_types=("EQ",),
                base_confidence=0.60,
            ),
            FieldRule(
                field="depth",
                patterns=(
                    # depth of N km / depth: N km
                    re.compile(
                        rf"(?:depth|deep)\s*(?:of\s*|:\s*)?(?P<num>{_N})\s*(?P<unit>km|kilometers?|mi|miles?)",
                        re.I,
                    ),
                    # N km depth / N km deep / N-km-deep
                    re.compile(
                        rf"(?P<num>{_N})\s*(?P<unit>km|kilometers?|mi|miles?)\s*(?:depth|deep)\b",
                        re.I,
                    ),
                ),
                anchors=("depth", "deep", "earthquake", "hypocenter", "epicenter"),
                event_types=("EQ",),
                base_confidence=0.58,
            ),
            FieldRule(
                field="rapidpopdescription",
                patterns=(
                    re.compile(
                        rf"(?P<raw>(?:few|{_N})\s*(?:thousand|million|billion)?\s*people(?:\s+in\s+MMI\s+[IVX]+)?)",
                        re.I,
                    ),
                    re.compile(
                        rf"(?P<raw>MMI\s+[IVX]+\s*(?:felt by|affecting)\s*(?:few|{_N})\s*(?:thousand|million|billion)?\s*people)",
                        re.I,
                    ),
                ),
                anchors=("mmi", "intensity", "earthquake", "quake", "aftershock", "shaking"),
                event_types=("EQ",),
                value_kind="text",
                base_confidence=0.56,
                min_binding_score=0.22,
            ),
            FieldRule(
                field="maximum_wind_speed_kmh",
                patterns=(
                    re.compile(
                        rf"(?P<num>{_N})\s*(?P<unit>km/h|kph|mph|knots?|kt|m/s)\s*(?:sustained\s*)?(?:winds?|gusts?)?",
                        re.I,
                    ),
                ),
                anchors=("wind", "gust", "cyclone", "typhoon", "hurricane", "storm", "landfall"),
                event_types=("TC",),
                base_confidence=0.54,
            ),
            FieldRule(
                field="maximum_storm_surge_m",
                patterns=(
                    re.compile(
                        rf"storm surge(?:\s*of)?\s*(?P<num>{_N})\s*(?P<unit>m|meters?|ft|feet)",
                        re.I,
                    ),
                ),
                anchors=("storm surge", "surge", "coastal flooding", "inundation", "cyclone", "typhoon", "hurricane"),
                event_types=("TC",),
                base_confidence=0.62,
            ),
            FieldRule(
                field="exposed_population",
                patterns=(
                    re.compile(
                        rf"(?P<num>{_NS})\s*(?:people|residents|persons?)\s*(?:were\s*)?(?:exposed|at risk|in the path|under threat|vulnerable)",
                        re.I,
                    ),
                    re.compile(
                        rf"(?:exposed|at risk|in the path)\s*(?:population\s*)?(?:of\s*)?(?P<num>{_NS})",
                        re.I,
                    ),
                ),
                anchors=("exposed", "at risk", "in the path", "track", "cyclone", "typhoon", "hurricane"),
                event_types=("TC",),
                base_confidence=0.58,
            ),
            FieldRule(
                field="rainfall_mm",
                patterns=(
                    re.compile(
                        rf"(?P<num>{_N})\s*(?P<unit>mm|millimeters?|cm|inches?|in)\s*(?:of\s*)?(?:rain|rainfall|precipitation)",
                        re.I,
                    ),
                ),
                anchors=("rain", "rainfall", "precipitation", "downpour", "flood", "storm"),
                base_confidence=0.50,
            ),
            FieldRule(
                field="burned_area_ha",
                patterns=(
                    # N ha/hectares/acres burned  (unit before optional verb)
                    re.compile(
                        rf"(?P<num>{_N})\s*(?P<unit>ha|hectares?|acres?|km2|km²|square kilometers?|sq\.?\s*km)\s*(?:of\s*(?:land|forest|bush|vegetation|terrain))?\s*(?:burned|scorched|charred|destroyed|ablaze|on fire)?",
                        re.I,
                    ),
                    # burned/destroyed N ha/km2  (verb before number)
                    re.compile(
                        rf"(?:burned|scorched|charred|destroyed|consumed|razed)\s*(?:(?:over|about|more than|nearly|around|some)\s*)?(?P<num>{_N})\s*(?P<unit>ha|hectares?|acres?|km2|km²|square kilometers?|sq\.?\s*km)",
                        re.I,
                    ),
                    # N ha/km2 of land burned  (unit then "of land/forest" then verb)
                    re.compile(
                        rf"(?P<num>{_N})\s*(?P<unit>ha|hectares?|acres?|km2|km²|square kilometers?|sq\.?\s*km)\s*(?:of\s*(?:land|forest|bush|vegetation|terrain))\s*(?:burned|scorched|charred|destroyed|consumed|were burned|have burned)?",
                        re.I,
                    ),
                    # fire has burned/spread across N km2/ha
                    re.compile(
                        rf"(?:fire|blaze|wildfire|bushfire)\s*(?:has\s*)?(?:burned|covered|spread over|scorched)\s*(?:(?:over|about|more than|nearly|around)\s*)?(?P<num>{_N})\s*(?P<unit>ha|hectares?|acres?|km2|km²|square kilometers?|sq\.?\s*km)",
                        re.I,
                    ),
                ),
                anchors=("burned", "scorched", "charred", "wildfire", "bushfire", "blaze", "fire", "hectare", "acre"),
                event_types=("WF",),
                base_confidence=0.58,
            ),
            FieldRule(
                field="affected_area_km2",
                patterns=(
                    re.compile(
                        rf"(?P<num>{_N})\s*(?P<unit>km2|km²|square kilometers?|sq\.?\s*km|sq\.?\s*mi|square miles?)\s*(?:affected|under water|impacted|parched|burned)?",
                        re.I,
                    ),
                    re.compile(
                        rf"(?P<num>{_N})\s*(?P<unit>km2|km²|square kilometers?|sq\.?\s*km|sq\.?\s*mi|square miles?)\s*(?:of\s*)?(?:farmland|land|terrain|area)",
                        re.I,
                    ),
                ),
                anchors=("affected area", "under water", "impacted area", "drought", "flooded area", "parched", "farmland", "land"),
                event_types=("DR", "FL"),
                base_confidence=0.54,
            ),
            FieldRule(
                field="duration_days",
                patterns=(
                    # for/lasting/burning/raging N days/weeks/months
                    re.compile(
                        rf"(?:for|lasting|lasted|persisting for|persisted for|burning for|raging for|continuing for|going on for)\s*(?P<num>{_N})\s*(?P<unit>days?|weeks?|months?)",
                        re.I,
                    ),
                    # N days/weeks of drought/wildfire
                    re.compile(
                        rf"(?P<num>{_N})\s*(?P<unit>days?|weeks?|months?)\s*(?:of\s*)?(?:drought|dry spell|wildfire|bushfire|fires?)",
                        re.I,
                    ),
                    # N-day/N-week wildfire / drought
                    re.compile(
                        rf"(?P<num>{_N})-(?P<unit>day|week|month)\s*(?:wildfire|bushfire|blaze|drought|dry spell|fire)",
                        re.I,
                    ),
                    # drought that has lasted N months / fire that burned for N days
                    re.compile(
                        rf"(?:drought|fire|blaze|wildfire)\s*(?:that\s*)?(?:has\s*)?(?:lasted|burned|continued|persisted|gone on)\s*(?:for\s*)?(?P<num>{_N})\s*(?P<unit>days?|weeks?|months?)",
                        re.I,
                    ),
                    # since N months/weeks ago
                    re.compile(
                        rf"(?P<num>{_N})\s*(?P<unit>days?|weeks?|months?)\s*ago\b",
                        re.I,
                    ),
                ),
                anchors=("lasting", "lasted", "duration", "days", "weeks", "months", "drought", "wildfire", "blaze", "burning", "persisted"),
                event_types=("WF", "DR"),
                base_confidence=0.52,
            ),
            FieldRule(
                field="economic_loss_usd",
                patterns=(
                    re.compile(
                        rf"(?P<currency>\$|usd)\s*(?P<num>{_N})\s*(?P<scale>k|m|b|thousand|million|billion)?\s*(?:in\s*)?(?:losses?|damage)",
                        re.I,
                    ),
                    re.compile(
                        rf"(?:losses?|damage)\s*(?:estimated at|of|around|near)?\s*(?P<currency>\$|usd)\s*(?P<num>{_N})\s*(?P<scale>k|m|b|thousand|million|billion)?",
                        re.I,
                    ),
                ),
                anchors=("loss", "losses", "damage", "insured", "economic"),
                base_confidence=0.55,
            ),
            FieldRule(
                field="affected_country_count",
                patterns=(
                    re.compile(r"(?:across|affecting|spanning)\s*(?P<num>\d+)\s*(?:countries|nations)", re.I),
                    re.compile(r"(?P<num>\d+)\s*(?:countries|nations)\s*(?:affected|impacted|hit|experiencing)", re.I),
                ),
                anchors=("countries", "nations", "across", "drought", "affecting"),
                event_types=("DR",),
                base_confidence=0.62,
            ),
            FieldRule(
                field="dead",
                patterns=(
                    # N people killed/dead/fatalities
                    re.compile(rf"(?P<num>{_NS})\s*(?:people\s*|persons?\s*|civilians?\s*)?(?:were\s*|have\s*been\s*)?(?:killed|dead|died|fatalities?)", re.I),
                    # killed/claimed at least/over N people/lives
                    re.compile(rf"(?:killed|claimed|left|took)\s*(?:at least\s*)?(?:more than\s*)?(?:over\s*)?(?:about\s*)?(?P<num>{_NS})\s*(?:people|lives|persons?|dead|fatalities?)?", re.I),
                    # death toll / deaths reached N
                    re.compile(rf"(?:death toll|deaths?)\s*(?:rose to|climbed to|reached|surpassed|exceeded|at|of|:|stands at|now at)\s*(?:at least\s*)?(?:over\s*)?(?P<num>{_NS})", re.I),
                    # N killed in / N dead in
                    re.compile(rf"(?P<num>{_NS})\s*(?:people\s*)?(?:killed|dead)\s*(?:in|after|following|from|by)", re.I),
                    # claiming/costing N lives
                    re.compile(rf"(?:claiming|costing|cost)\s*(?:at least\s*)?(?:over\s*)?(?:more than\s*)?(?P<num>{_NS})\s*(?:lives?|deaths?|fatalities?)", re.I),
                    # at least N people lost their lives / perished
                    re.compile(rf"(?:at least\s*|over\s*|more than\s*)?(?P<num>{_NS})\s*(?:people|persons?)\s*(?:lost their lives|perished|were killed|have died)", re.I),
                ),
                anchors=("dead", "killed", "fatality", "fatalities", "death toll", "deaths", "died", "lives", "perished"),
                base_confidence=0.60,
            ),
            FieldRule(
                field="injured",
                patterns=(
                    re.compile(rf"(?P<num>{_NS})\s*(?:people\s*)?(?:were\s*)?injured", re.I),
                    re.compile(rf"injur(?:ed|ing)\s*(?:at least\s*)?(?:more than\s*)?(?P<num>{_NS})\s*(?:people|persons?)?", re.I),
                ),
                anchors=("injured", "wounded", "hurt"),
                base_confidence=0.58,
            ),
            FieldRule(
                field="missing",
                patterns=(
                    re.compile(rf"(?P<num>{_NS})\s*(?:people\s*)?(?:are|were|remain)?\s*missing", re.I),
                    re.compile(rf"missing\s*(?:persons?\s*)?(?:toll\s*)?(?:rose to|reached|:)?\s*(?P<num>{_NS})", re.I),
                ),
                anchors=("missing", "unaccounted"),
                base_confidence=0.58,
            ),
            FieldRule(
                field="displaced",
                patterns=(
                    # N people displaced/evacuated/forced to flee
                    re.compile(rf"(?P<num>{_NS})\s*(?:people\s*|residents\s*|persons?\s*|families\s*)?(?:were\s*|have\s*been\s*)?(?:displaced|evacuated|homeless|relocated|forced to flee|fled their homes?)", re.I),
                    # displaced/evacuated N people
                    re.compile(rf"(?:displaced|evacuated|forced to flee|forced from their homes?)\s*(?:at least\s*)?(?:more than\s*)?(?:over\s*)?(?:about\s*)?(?:some\s*)?(?P<num>{_NS})\s*(?:people|residents|families|persons?)?", re.I),
                    # N people fled / N refugees
                    re.compile(rf"(?P<num>{_NS})\s*(?:people|residents|persons?|families)\s*(?:fled|were forced to leave|were uprooted)", re.I),
                    # N people seeking shelter / in evacuation centers
                    re.compile(rf"(?P<num>{_NS})\s*(?:people|residents|persons?|families)\s*(?:are\s*)?(?:seeking shelter|in evacuation cent(?:er|re)|in emergency shelter|in relief camp)", re.I),
                ),
                anchors=("displaced", "evacuated", "homeless", "relocated", "fled", "shelter", "evacuation"),
                base_confidence=0.60,
            ),
            FieldRule(
                field="people_affected",
                patterns=(
                    # N people (were/are/have been) affected/impacted/stranded
                    re.compile(rf"(?P<num>{_NS})\s*(?:people|residents|persons?|families|households?)\s*(?:were\s*|are\s*|have been\s*)?(?:affected|impacted|stranded|displaced by|hit by)", re.I),
                    # affecting/affected N people
                    re.compile(rf"affect(?:ing|ed)\s*(?:at least\s*)?(?:more than\s*)?(?:about\s*)?(?:over\s*)?(?:around\s*)?(?:some\s*)?(?P<num>{_NS})\s*(?:people|residents|persons?|families|households?)", re.I),
                    # evacuation orders/advisories for N residents
                    re.compile(rf"evacuation\s*(?:orders?|advisories?|warnings?|notices?)\s*(?:for|issued\s*(?:for|to)|affecting|covering)\s*(?:(?:more than|over|about|nearly|around|some)\s*)?(?P<num>{_NS})\s*(?:people|residents|persons?|homes?|households?)?", re.I),
                    # N people/residents evacuated / under evacuation orders
                    re.compile(rf"(?P<num>{_NS})\s*(?:people|residents|persons?|homes?|households?)\s*(?:were\s*)?(?:under\s*)?(?:evacuation|evacuat(?:ed|ion))\s*(?:orders?|advisories?|warnings?)?", re.I),
                    # N residents ordered/told/forced to evacuate
                    re.compile(rf"(?P<num>{_NS})\s*(?:people|residents|persons?)\s*(?:were\s*)?(?:ordered|told|forced|asked)\s*(?:to\s*)?evacuate", re.I),
                    # N people at risk / under threat / in harm's way
                    re.compile(rf"(?P<num>{_NS})\s*(?:people|residents|persons?)\s*(?:are\s*|were\s*)?(?:at risk|under threat|in harm|in the path|threatened)\b", re.I),
                ),
                anchors=("affected", "affecting", "impacted", "stranded", "people", "residents",
                         "evacuation", "evacuated", "at risk", "under threat"),
                base_confidence=0.55,
            ),
        ]

    @staticmethod
    def _phrase_regex(phrase: str) -> str:
        """Strict word-boundary regex — used for negation cue detection."""
        return r"\b" + re.escape(phrase).replace(r"\ ", r"\s+") + r"\b"

    def _unit_convert(self, field: str, value: float, unit: str) -> Tuple[float, str]:
        normalized_unit = re.sub(r"\s+", " ", (unit or "").strip().lower())

        if field == "depth":
            if normalized_unit in {"mi", "mile", "miles"}:
                return value * 1.60934, "km"
            return value, "km"

        if field == "maximum_wind_speed_kmh":
            if normalized_unit == "mph":
                return value * 1.60934, "km/h"
            if normalized_unit in {"knot", "knots", "kt"}:
                return value * 1.852, "km/h"
            if normalized_unit == "m/s":
                return value * 3.6, "km/h"
            return value, "km/h"

        if field == "rainfall_mm":
            if normalized_unit == "cm":
                return value * 10.0, "mm"
            if normalized_unit in {"inch", "inches", "in"}:
                return value * 25.4, "mm"
            return value, "mm"

        if field == "burned_area_ha":
            if normalized_unit in {"acre", "acres"}:
                return value * 0.404686, "ha"
            if normalized_unit in {"km2", "km²", "square kilometer", "square kilometers",
                                   "sq km", "sq. km", "sq.km"}:
                return value * 100.0, "ha"
            return value, "ha"

        if field == "affected_area_km2":
            if normalized_unit in {"sq mi", "sq. mi", "square mile", "square miles"}:
                return value * 2.58999, "km2"
            return value, "km2"

        if field == "maximum_storm_surge_m":
            if normalized_unit in {"ft", "feet"}:
                return value * 0.3048, "m"
            return value, "m"

        if field == "duration_days":
            if normalized_unit.startswith("week"):
                return value * 7.0, "days"
            if normalized_unit.startswith("month"):
                return value * 30.0, "days"
            return value, "days"

        if field == "economic_loss_usd":
            return value, "usd"

        if field in {"dead", "injured", "missing", "displaced", "people_affected", "exposed_population", "affected_country_count"}:
            unit_out = "countries" if field == "affected_country_count" else "people"
            return value, unit_out

        return value, normalized_unit

    @staticmethod
    def _token_spans(text: str) -> List[Tuple[int, int, str]]:
        spans: List[Tuple[int, int, str]] = []
        for m in re.finditer(r"\b[\w/]+\b", text.lower()):
            spans.append((m.start(), m.end(), m.group(0)))
        return spans

    @staticmethod
    def _token_index_for_char(spans: List[Tuple[int, int, str]], char_pos: int) -> int:
        for idx, (start, end, _) in enumerate(spans):
            if start <= char_pos < end:
                return idx
        if not spans:
            return -1
        if char_pos < spans[0][0]:
            return 0
        return len(spans) - 1

    def _window_binding_score(self, text: str, anchors: Sequence[str], start: int, end: int) -> float:
        if not anchors:
            return 0.5

        spans = self._token_spans(text)
        if not spans:
            return 0.5

        idx_s = self._token_index_for_char(spans, start)
        idx_e = self._token_index_for_char(spans, max(start, end - 1))
        if idx_s < 0 or idx_e < 0:
            return 0.5

        lo = max(0, idx_s - 8)
        hi = min(len(spans) - 1, idx_e + 8)
        window_tokens = [tok for _, _, tok in spans[lo : hi + 1]]
        window_text = " ".join(window_tokens)

        hits = 0
        min_dist = 999
        for anchor in anchors:
            anchor_lower = anchor.lower()
            if " " in anchor_lower:
                if anchor_lower in window_text:
                    hits += 1
                    min_dist = min(min_dist, 1)
                continue
            for tok_idx in range(lo, hi + 1):
                token = spans[tok_idx][2]
                if token == anchor_lower:
                    hits += 1
                    min_dist = min(min_dist, min(abs(tok_idx - idx_s), abs(tok_idx - idx_e)))

        if hits == 0:
            return 0.08
        score = 0.42 + 0.12 * min(hits, 3) + max(0.0, 0.26 - 0.03 * min_dist)
        return clamp(score, 0.0, 1.0)

    def _has_negation_near_span(self, text: str, start: int, end: int) -> bool:
        spans = self._token_spans(text)
        if not spans:
            return False
        idx_s = self._token_index_for_char(spans, start)
        idx_e = self._token_index_for_char(spans, max(start, end - 1))
        lo = max(0, idx_s - 5)
        hi = min(len(spans) - 1, idx_e + 5)
        context = " ".join(tok for _, _, tok in spans[lo : hi + 1])
        return any(re.search(self._phrase_regex(cue), context) for cue in self.negation_cues)

    def _extract_with_rule(
        self,
        text: str,
        sentences: List[str],
        rule: FieldRule,
    ) -> List[Candidate]:
        candidates: List[Candidate] = []
        for pattern in rule.patterns:
            for match in pattern.finditer(text):
                evidence = sentence_for_span(sentences, text, match.start(), match.end()) or match.group(0)

                if rule.value_kind == "text":
                    value = normalize_text(match.groupdict().get("raw", "") or match.group(0))
                    if not value:
                        continue
                    unit = ""
                else:
                    value = parse_number(match.groupdict().get("num", ""))
                    if value is None:
                        continue
                    scale = match.groupdict().get("scale", "")
                    if scale:
                        value = parse_number(f"{value} {scale}") or value
                    unit = match.groupdict().get("unit", "")
                    value, unit = self._unit_convert(rule.field, value, unit)

                binding_score = self._window_binding_score(text, rule.anchors, match.start(), match.end())
                if binding_score < rule.min_binding_score:
                    continue

                confidence = rule.base_confidence + 0.24 * binding_score
                if self._has_negation_near_span(text, match.start(), match.end()):
                    confidence -= 0.20
                if rule.value_kind != "text" and "official" in evidence.lower():
                    confidence += 0.04
                candidates.append(
                    Candidate(
                        field=rule.field,
                        value=value,
                        unit=unit,
                        confidence=clamp(confidence, 0.05, 0.98),
                        raw=match.group(0),
                        evidence=evidence,
                        start=match.start(),
                        end=match.end(),
                    )
                )
        return candidates

    @staticmethod
    def _candidate_numeric_priority(value: object) -> float:
        if isinstance(value, (int, float)) and not is_missing(value):
            return float(value)
        return 0.0

    def _best_candidate(self, candidates: List[Candidate]) -> Optional[Candidate]:
        if not candidates:
            return None
        return sorted(
            candidates,
            key=lambda cand: (cand.confidence, self._candidate_numeric_priority(cand.value)),
            reverse=True,
        )[0]

    def _collect_sector_hints(self, text: str) -> Dict[str, int]:
        lowered = text.lower()
        scores: Dict[str, int] = {}
        for sector, keywords in self.sector_keywords.items():
            count = sum(1 for keyword in keywords if keyword in lowered)
            if count > 0:
                scores[sector] = count
        return scores

    def _parse_rapidpopdescription(self, text: str) -> Dict[str, float]:
        raw = normalize_text(text)
        lower = raw.lower()
        result = {
            "rapid_pop_people": math.nan,
            "rapid_pop_log": math.nan,
            "rapid_missing": 0.0,
            "rapid_few_people": 0.0,
            "rapid_unparsed": 0.0,
        }

        if not raw:
            result["rapid_missing"] = 1.0
            return result

        if "few people" in lower:
            result["rapid_few_people"] = 1.0
            result["rapid_pop_people"] = 100.0
            result["rapid_pop_log"] = math.log1p(100.0)

        pop_match = re.search(r"(\d+(?:\.\d+)?)\s*(thousand|million|billion)?", lower)
        if pop_match:
            people = float(pop_match.group(1)) * POP_UNIT_MULTIPLIER.get(pop_match.group(2) or "", 1.0)
            result["rapid_pop_people"] = people
            result["rapid_pop_log"] = math.log1p(people)
        elif result["rapid_few_people"] == 0.0:
            result["rapid_unparsed"] = 1.0

        return result

    def _add_derived_metrics(self, metrics: Dict[str, Dict]) -> None:
        source = metrics.get("rapidpopdescription")
        if not source:
            return

        parsed = self._parse_rapidpopdescription(str(source.get("value", "")))
        derived_confidence = max(0.0, float(source.get("confidence", 0.5)) - 0.05)
        derived_units = {
            "rapid_pop_people": "people",
            "rapid_pop_log": "log_people",
            "rapid_missing": "flag",
            "rapid_few_people": "flag",
            "rapid_unparsed": "flag",
        }
        for field, value in parsed.items():
            metrics[field] = {
                "value": None if is_missing(value) else round(float(value), 6),
                "unit": derived_units[field],
                "confidence": round(derived_confidence, 3),
                "evidence": source["evidence"],
                "raw": source["raw"],
            }

    @staticmethod
    def _normalize_event_type(event_type: str) -> str:
        """Convert DistilBERT label or abbreviation to internal two-letter code."""
        raw = (event_type or "").strip()
        if raw.upper() in {"EQ", "TC", "WF", "DR", "FL"}:
            return raw.upper()
        return LABEL_TO_TYPE_CODE.get(raw.lower(), "")

    def _low_confidence(self, metrics: Dict[str, Dict], type_code: str) -> bool:
        """True when every key field for the given event type is missing."""
        fields = self.low_confidence_key_fields.get(type_code, [])
        if not fields:
            return False
        return all(is_missing(metrics.get(f, {}).get("value")) for f in fields)

    def _build_stock_impact_summary(self, metrics: Dict[str, Dict], sector_hints: Dict[str, int]) -> Dict:
        dead = to_float(metrics.get("dead", {}).get("value")) or 0.0
        displaced = to_float(metrics.get("displaced", {}).get("value")) or 0.0
        affected = to_float(metrics.get("people_affected", {}).get("value")) or 0.0
        econ_loss = to_float(metrics.get("economic_loss_usd", {}).get("value")) or 0.0
        wind = to_float(metrics.get("maximum_wind_speed_kmh", {}).get("value")) or 0.0
        magnitude = to_float(metrics.get("magnitude", {}).get("value")) or 0.0
        burned = to_float(metrics.get("burned_area_ha", {}).get("value")) or 0.0

        score = 0.0
        score += 18.0 * math.log1p(dead)
        score += 8.0 * math.log1p(displaced)
        score += 4.0 * math.log1p(affected)
        score += 12.0 * math.log1p(econ_loss / 1_000_000.0)
        score += 0.08 * wind
        score += 2.5 * magnitude
        score += 3.0 * math.log1p(burned / 1_000.0)
        score = clamp(score, 0.0, 100.0)

        if score >= 65:
            tier = "high"
        elif score >= 35:
            tier = "medium"
        else:
            tier = "low"

        ranked_sectors = sorted(sector_hints.keys(), key=lambda sector: sector_hints[sector], reverse=True)
        return {
            "impact_score_0_100": round(score, 3),
            "impact_tier": tier,
            "likely_affected_sectors": ranked_sectors,
        }

    def extract(
        self,
        text: str,
        event_type: str,
        timestamp: str = "",
        location_text: str = "",
    ) -> Dict:
        """Extract structured event fields from a news article.

        Parameters
        ----------
        text          : article text (title [SEP] body)
        event_type    : DistilBERT predicted label, e.g. "earthquake" / "EQ".
                        Only rules relevant to that type (plus universal rules) are run.
        timestamp     : article publication timestamp (passed through to meta)
        location_text : article location hint (passed through to meta)
        """
        type_code = self._normalize_event_type(event_type)
        clean_text = normalize_text(text)

        if not clean_text:
            return {
                "event_type": type_code or None,
                "metrics": {},
                "sector_hints": {},
                "affected_sectors": [],
                "low_confidence": self._low_confidence({}, type_code) if type_code else None,
                "stock_impact_summary": {
                    "impact_score_0_100": 0.0,
                    "impact_tier": "low",
                    "likely_affected_sectors": [],
                },
                "meta": {
                    "timestamp": normalize_text(timestamp),
                    "location_text": normalize_text(location_text),
                },
            }

        sentences = split_sentences(clean_text)
        active_rules = [
            r for r in self.field_rules
            if not r.event_types or type_code in r.event_types
        ]

        all_candidates: List[Candidate] = []
        for rule in active_rules:
            all_candidates.extend(self._extract_with_rule(clean_text, sentences, rule))

        grouped: Dict[str, List[Candidate]] = {}
        for candidate in all_candidates:
            grouped.setdefault(candidate.field, []).append(candidate)

        metrics: Dict[str, Dict] = {}
        for field, candidates in grouped.items():
            best = self._best_candidate(candidates)
            if best is None:
                continue
            value = best.value
            if isinstance(value, (int, float)) and not is_missing(value):
                value = round(float(value), 6)
            metrics[field] = {
                "value": jsonable_value(value),
                "unit": best.unit,
                "confidence": round(best.confidence, 3),
                "evidence": best.evidence,
                "raw": best.raw,
            }

        self._add_derived_metrics(metrics)

        # For EQ: if rapidpopdescription was not found in text, set rapid_missing=1.0
        # so Module C receives an explicit flag rather than NaN (which would be median-imputed)
        if type_code == "EQ" and "rapidpopdescription" not in metrics:
            for field, unit, val in [
                ("rapid_missing",    "flag",        1.0),
                ("rapid_few_people", "flag",        0.0),
                ("rapid_unparsed",   "flag",        0.0),
            ]:
                metrics[field] = {"value": val, "unit": unit, "confidence": 1.0,
                                  "evidence": "", "raw": ""}

        sector_hints = self._collect_sector_hints(clean_text)
        affected_sectors = sorted(sector_hints.keys(), key=lambda s: sector_hints[s], reverse=True)
        stock_summary = self._build_stock_impact_summary(metrics, sector_hints)

        return {
            "event_type": type_code or None,
            "metrics": metrics,
            "sector_hints": sector_hints,
            "affected_sectors": affected_sectors,
            "low_confidence": self._low_confidence(metrics, type_code) if type_code else None,
            "stock_impact_summary": stock_summary,
            "meta": {
                "timestamp": normalize_text(timestamp),
                "location_text": normalize_text(location_text),
            },
        }


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix in {".csv", ".tsv"}:
        sep = "\t" if suffix == ".tsv" else ","
        return pd.read_csv(path, sep=sep, encoding_errors="replace")
    raise ValueError(f"Unsupported input format: {path.suffix}")


def process_file(
    input_path: Path,
    output_path: Path,
    text_column: str,
    timestamp_column: str,
    location_column: str,
    event_type_column: str = "",
) -> None:
    df = read_table(input_path)
    if text_column not in df.columns:
        raise ValueError(f"text column '{text_column}' not found. columns={list(df.columns)}")

    extractor = UnifiedEventExtractor()
    features_json: List[str] = []

    ts_series  = df[timestamp_column]   if timestamp_column   in df.columns else pd.Series([""] * len(df))
    loc_series = df[location_column]    if location_column    in df.columns else pd.Series([""] * len(df))
    et_series  = df[event_type_column]  if event_type_column  in df.columns else pd.Series([""] * len(df))

    for text, ts, loc, et in zip(
        df[text_column].fillna(""),
        ts_series.fillna(""),
        loc_series.fillna(""),
        et_series.fillna(""),
    ):
        feat = extractor.extract(str(text), event_type=str(et), timestamp=str(ts), location_text=str(loc))
        features_json.append(json.dumps(feat, ensure_ascii=False))

    df_out = df.copy()
    df_out[JSON_COLUMN] = features_json

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() in {".xlsx", ".xls"}:
        df_out.to_excel(output_path, index=False)
    else:
        df_out.to_csv(output_path, index=False, encoding="utf-8-sig")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract unified event fields from news for stock-impact analysis."
    )
    parser.add_argument("--input", required=True, help="Input CSV/XLSX path.")
    parser.add_argument("--output", required=True, help="Output CSV/XLSX path.")
    parser.add_argument("--text-column", default="text_cleaned", help="News text column name.")
    parser.add_argument("--timestamp-column", default="timestamp", help="Timestamp column name.")
    parser.add_argument("--location-column", default="location", help="Location text column name.")
    parser.add_argument(
        "--event-type-column", default="",
        help="Column containing DistilBERT-predicted event type (e.g. 'label' or 'pred_label'). "
             "When provided, extraction is scoped to that event type per row."
    )
    return parser


def main():
    parser = build_argparser()
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()

    process_file(
        input_path=input_path,
        output_path=output_path,
        text_column=args.text_column,
        timestamp_column=args.timestamp_column,
        location_column=args.location_column,
        event_type_column=args.event_type_column,
    )
    print(f"Saved extraction output to {output_path}")


if __name__ == "__main__":
    main()
