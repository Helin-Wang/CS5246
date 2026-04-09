"""
Module D — Event Clustering and Deduplication

Groups per-article extraction records into unique disaster events using a
two-layer rule-based approach. Articles reporting on the same disaster are
merged so that Module C receives one feature-complete record per event.

Design rationale
----------------
DBSCAN and other metric-space clustering methods are unsuitable here because
lat/lon coordinate coverage is ~0% in this dataset (test: 0.1%, train: 0.0%).
Instead we use two layers of rule-based grouping on the available features:

  Layer 1 (hard partition):
      Group by (event_type, country_iso2, 7-day time bucket).
      Fast: eliminates cross-country and cross-type collisions.

  Layer 2 (location text refinement within each group):
      Two articles in the same country/type/time bucket are merged only if
      their location_text strings share a common token (substring overlap).
      This prevents a California wildfire and a Texas wildfire from being
      collapsed into one event just because both have country_iso2="US".

This is semantically equivalent to complete-linkage clustering — every pair
within a cluster satisfies the merge condition — avoiding the chaining
problem of single-linkage (Union-Find).

Input format (list of dicts, one per article):
  {
      "idx":            int
      "event_type":     str           e.g. "earthquake" / "EQ"
      "event_date":     str | None    "YYYY-MM-DD"
      "location_text":  str | None
      "country_iso2":   str | None    ISO-3166-1 alpha-2
      "lat":            float | None  (rarely available; kept for future use)
      "lon":            float | None
      "low_confidence": bool
      # numeric fields (all may be None)
      "magnitude", "depth", "rapidpopdescription",
      "maximum_wind_speed_kmh", "maximum_storm_surge_m", "exposed_population",
      "burned_area_ha", "people_affected", "duration_days",
      "affected_area_km2", "affected_country_count", "dead", "displaced",
      "rapid_missing", "rapid_few_people", "rapid_unparsed"
  }

Output format (list of dicts, one per cluster):
  Same keys as input, plus:
      "article_count":   int
      "article_indices": list[int]
  "idx" is removed (replaced by "article_indices").
"""

from __future__ import annotations

import math
import re
from collections import Counter
from datetime import date
from typing import Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TIME_BUCKET_DAYS = 7     # two articles more than this apart → different events
FALLBACK_BUCKET  = "UNKNOWN_DATE"

NUMERIC_FIELDS = [
    "magnitude", "depth",
    "maximum_wind_speed_kmh", "maximum_storm_surge_m", "exposed_population",
    "burned_area_ha", "people_affected", "duration_days",
    "affected_area_km2", "affected_country_count",
    "dead", "displaced",
    "rapid_missing", "rapid_few_people", "rapid_unparsed",
]
TEXT_FIELD = "rapidpopdescription"

_LABEL_TO_CODE: Dict[str, str] = {
    "earthquake": "EQ", "eq": "EQ",
    "cyclone": "TC", "tc": "TC", "typhoon": "TC", "hurricane": "TC",
    "wildfire": "WF", "wf": "WF", "bushfire": "WF",
    "drought": "DR", "dr": "DR",
    "flood": "FL", "fl": "FL", "flooding": "FL",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _type_code(label: str) -> str:
    return _LABEL_TO_CODE.get(str(label).lower().strip(), str(label).upper()[:2])


def _parse_date(s) -> Optional[date]:
    if not s or str(s).strip().lower() in ("none", "nan", ""):
        return None
    s = str(s).strip()
    if len(s) == 7:          # YYYY-MM → pad to first of month
        s = s + "-01"
    try:
        return date.fromisoformat(s[:10])
    except ValueError:
        return None


def _time_bucket(event_date: Optional[date]) -> str:
    """Map a date to a 7-day bucket label 'YYYY-WNN' (ISO week)."""
    if event_date is None:
        return FALLBACK_BUCKET
    # Use ISO week: (year, week_number)
    iso = event_date.isocalendar()
    return f"{iso[0]}-W{iso[1]:02d}"


def _location_tokens(text) -> Set[str]:
    """Normalise location_text to a set of lowercase word tokens (≥3 chars)."""
    if not text or str(text).strip().lower() in ("none", "nan", ""):
        return set()
    tokens = re.findall(r"[a-zA-Z]{3,}", str(text).lower())
    # Remove very generic stop words that would cause false matches
    stop = {"the", "and", "for", "area", "region", "province", "district",
            "county", "city", "state", "states", "united", "kingdom",
            "republic", "democratic", "north", "south", "east", "west",
            "central", "upper", "lower", "new", "greater", "metropolitan"}
    return set(tokens) - stop


def _location_overlap(a: dict, b: dict) -> bool:
    """
    Return True if the two articles share at least one meaningful location token,
    OR if either has no parseable location (benefit of the doubt).
    """
    tok_a = _location_tokens(a.get("location_text"))
    tok_b = _location_tokens(b.get("location_text"))
    if not tok_a or not tok_b:
        return True    # can't tell apart → allow merge
    return bool(tok_a & tok_b)


def _mode(values: list):
    filtered = [v for v in values if v is not None and str(v).lower() not in ("none", "nan", "")]
    return Counter(filtered).most_common(1)[0][0] if filtered else None


def _max_numeric(values: list) -> Optional[float]:
    nums = []
    for v in values:
        if v is None:
            continue
        try:
            f = float(v)
            if not math.isnan(f):
                nums.append(f)
        except (ValueError, TypeError):
            pass
    return max(nums) if nums else None


def _merge_articles(articles: List[dict]) -> dict:
    """Aggregate a list of article dicts into one cluster record."""
    type_codes = [_type_code(a.get("event_type", "")) for a in articles]
    event_type = type_codes[0]

    dates = [_parse_date(a.get("event_date")) for a in articles]
    valid_dates = [d for d in dates if d is not None]
    event_date = min(valid_dates).isoformat() if valid_dates else None

    primary_country = _mode([a.get("country_iso2") for a in articles])
    location_text   = _mode([a.get("location_text") for a in articles])

    # Representative coords: first article with both lat and lon
    lat, lon = None, None
    for a in articles:
        if a.get("lat") is not None and a.get("lon") is not None:
            try:
                lat, lon = float(a["lat"]), float(a["lon"])
                break
            except (ValueError, TypeError):
                pass

    numerics = {f: _max_numeric([a.get(f) for a in articles]) for f in NUMERIC_FIELDS}

    rapid_texts = [a.get(TEXT_FIELD) for a in articles if a.get(TEXT_FIELD)]
    rapid_text  = max(rapid_texts, key=len) if rapid_texts else None

    low_confidence = all(a.get("low_confidence", True) for a in articles)

    cluster = {
        "event_type":       event_type,
        "event_date":       event_date,
        "primary_country":  primary_country,
        "location_text":    location_text,
        "lat":              lat,
        "lon":              lon,
        "low_confidence":   low_confidence,
        "article_count":    len(articles),
        "article_indices":  [a["idx"] for a in articles if "idx" in a],
        TEXT_FIELD:         rapid_text,
    }
    cluster.update(numerics)
    return cluster


# ---------------------------------------------------------------------------
# Two-layer clustering
# ---------------------------------------------------------------------------

def _layer1_key(article: dict) -> Tuple[str, str]:
    """
    Layer 1 partition key: (event_type_code, country_iso2).
    Time is intentionally excluded — fixed week buckets cause boundary
    artifacts (12% of within-7-day pairs fall in different buckets).
    Time proximity is enforced in Layer 2 instead.
    """
    etype   = _type_code(article.get("event_type", ""))
    country = str(article.get("country_iso2") or "UNKNOWN_COUNTRY").upper()
    return (etype, country)


def _date_ok(article: dict, cluster: List[dict]) -> bool:
    """
    Complete-linkage date check: new article must be within TIME_BUCKET_DAYS
    of EVERY existing cluster member.
    Equivalent to: cluster date span stays ≤ TIME_BUCKET_DAYS after adding
    the new article.
    Articles with missing dates are given benefit of the doubt (allowed).
    """
    new_date = _parse_date(article.get("event_date"))
    if new_date is None:
        return True
    for member in cluster:
        member_date = _parse_date(member.get("event_date"))
        if member_date is None:
            continue
        if abs((new_date - member_date).days) > TIME_BUCKET_DAYS:
            return False
    return True


def _layer2_clusters(articles: List[dict]) -> List[List[dict]]:
    """
    Within a Layer-1 group, cluster using complete-linkage on two conditions:
      1. Date span of cluster stays ≤ TIME_BUCKET_DAYS (sliding window, no
         fixed boundary — fixes the ISO-week boundary artifact)
      2. Location-text token overlap with every existing cluster member

    Articles are processed in event_date order so earlier dates anchor clusters.
    """
    # Sort by event_date (None dates go last)
    def sort_key(a):
        d = _parse_date(a.get("event_date"))
        return d.isoformat() if d else "9999-99-99"

    sorted_articles = sorted(articles, key=sort_key)
    clusters: List[List[dict]] = []

    for article in sorted_articles:
        placed = False
        for cluster in clusters:
            if _date_ok(article, cluster) and all(
                _location_overlap(article, m) for m in cluster
            ):
                cluster.append(article)
                placed = True
                break
        if not placed:
            clusters.append([article])

    return clusters


class EventClusterer:
    """
    Two-layer rule-based event clustering.

    Layer 1: hard partition by (event_type, country_iso2, 7-day time bucket)
    Layer 2: complete-linkage location-text token overlap within each group

    Usage:
        clusterer = EventClusterer()
        clusters  = clusterer.cluster(article_records)
    """

    def __init__(self, time_bucket_days: int = TIME_BUCKET_DAYS):
        self.time_bucket_days = time_bucket_days

    def cluster(self, articles: List[dict]) -> List[dict]:
        """
        Parameters
        ----------
        articles : list of article dicts (see module docstring for schema)

        Returns
        -------
        list of merged cluster dicts, one per unique event
        """
        if not articles:
            return []

        # Layer 1: group by (event_type, country_iso2, time_bucket)
        groups: Dict[Tuple, List[dict]] = {}
        for article in articles:
            key = _layer1_key(article)
            groups.setdefault(key, []).append(article)

        # Layer 2: within each group, split by location overlap
        result = []
        for group in groups.values():
            for sub_cluster in _layer2_clusters(group):
                result.append(_merge_articles(sub_cluster))

        return result
