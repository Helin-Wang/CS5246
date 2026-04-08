"""
Location extractor for disaster news articles.

Extracts four fields from article text (title + body):
    location_text  : most specific place name where the disaster occurred
    country_iso2   : ISO 3166-1 alpha-2 country code
    lat            : decimal latitude  (None if unavailable)
    lon            : decimal longitude (None if unavailable)

Resolution priority for lat/lon:
    1. Explicit coordinates in text  (e.g. "23.97°N, 121.6°E")
    2. geonamescache city centroid   (city name → coords)
    3. Country centroid (static dict) (country code → approx centre)
    4. None

Dependencies:
    pip install spacy geonamescache pycountry
    python -m spacy download en_core_web_sm
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import geonamescache
import pycountry
import spacy

# ---------------------------------------------------------------------------
# Lazy-load spaCy model once
# ---------------------------------------------------------------------------
_nlp = None

def _get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
    return _nlp


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
@dataclass
class LocationResult:
    location_text: Optional[str] = None
    country_iso2:  Optional[str] = None
    lat:           Optional[float] = None
    lon:           Optional[float] = None
    confidence:    str = "none"   # "coords_text" | "city" | "country" | "none"


# Country name alias patches (common variants not in pycountry)
_COUNTRY_ALIASES: dict[str, str] = {
    "america": "US", "the us": "US", "the united states": "US",
    "usa": "US", "u.s.": "US", "u.s.a.": "US",
    "south korea": "KR", "north korea": "KP",
    "britain": "GB", "great britain": "GB", "england": "GB",
    "uk": "GB", "u.k.": "GB",
    "russia": "RU", "uae": "AE",
    "tanzania": "TZ", "congo": "CD",
    "syria": "SY", "iran": "IR", "taiwan": "TW",
    "vietnam": "VN", "laos": "LA", "myanmar": "MM",
    "ivory coast": "CI", "cape verde": "CV",
}

# Country centroids for ~60 high-frequency disaster countries
_COUNTRY_CENTROIDS: dict[str, tuple[float, float]] = {
    "US": (38.0, -97.0), "JP": (36.2, 138.3), "CN": (35.0, 105.0),
    "IN": (20.6, 79.0),  "AU": (-25.3, 133.8),"ID": (-2.5, 118.0),
    "PH": (12.9, 121.8), "MX": (23.6, -102.6),"BR": (-14.2, -51.9),
    "CL": (-35.7, -71.5),"PE": (-9.2, -75.0), "EC": (-1.8, -78.2),
    "CO": (4.6, -74.1),  "GT": (15.8, -90.2), "HT": (19.0, -72.3),
    "NP": (28.4, 84.1),  "PK": (30.4, 69.3),  "AF": (33.9, 67.7),
    "IR": (32.4, 53.7),  "TR": (38.9, 35.2),  "IT": (41.9, 12.6),
    "GR": (39.1, 21.8),  "FR": (46.2, 2.2),   "DE": (51.2, 10.5),
    "ES": (40.5, -3.7),  "PT": (39.4, -8.2),  "RO": (45.9, 24.9),
    "RU": (61.5, 105.3), "UA": (48.4, 31.2),  "TR": (38.9, 35.2),
    "EG": (26.8, 30.8),  "MA": (31.8, -7.1),  "DZ": (28.0, 1.7),
    "NG": (9.1, 8.7),    "KE": (-0.2, 37.9),  "ET": (9.1, 40.5),
    "MZ": (-18.7, 35.5), "MW": (-13.3, 34.3), "ZW": (-20.0, 30.0),
    "ZA": (-30.6, 22.9), "MG": (-18.8, 46.9), "TZ": (-6.4, 34.9),
    "SD": (12.9, 30.2),  "SO": (5.2, 46.2),   "CD": (-4.0, 21.8),
    "TH": (15.9, 100.9), "VN": (14.1, 108.3), "MM": (21.9, 95.9),
    "BD": (23.7, 90.4),  "LK": (7.9, 80.8),   "TW": (23.6, 121.0),
    "KR": (35.9, 127.8), "KP": (40.3, 127.5), "MN": (46.9, 103.8),
    "NZ": (-40.9, 174.9),"FJ": (-17.7, 178.1),"PG": (-6.3, 143.9),
    "TO": (-21.2, -175.2),"WS": (-13.8, -172.1),"VU": (-15.4, 166.9),
    "GB": (55.4, -3.4),  "IE": (53.4, -8.2),  "NO": (60.5, 8.5),
    "CA": (56.1, -106.3),"AR": (-38.4, -63.6),"BO": (-16.3, -63.6),
    "VE": (6.4, -66.6),  "PA": (8.5, -80.8),  "SV": (13.8, -88.9),
    "HN": (15.2, -86.2), "NI": (12.9, -85.2), "CR": (9.7, -83.8),
    "DO": (18.7, -70.2), "CU": (21.5, -79.5),
}

# Disaster trigger words for location relevance scoring
_TRIGGER_VERBS = re.compile(
    r"\b(struck|hit|devastated|slammed|battered|flooded|inundated|"
    r"burned|ravaged|swept|affected|impacted|killed|displaced|"
    r"made landfall|struck|triggered|erupted|broke out)\b",
    re.I,
)

# Coordinate patterns in text
_COORD_PATTERNS = [
    # 23.97°N, 121.6°E  or  23.97N 121.6E
    re.compile(
        r"(?P<lat>\d{1,3}(?:\.\d+)?)\s*°?\s*(?P<lat_h>[NS])[,\s]+?"
        r"(?P<lon>\d{1,3}(?:\.\d+)?)\s*°?\s*(?P<lon_h>[EW])",
        re.I,
    ),
    # latitude 38.1, longitude 142.4
    re.compile(
        r"lat(?:itude)?\s*[:\s]\s*(?P<lat>-?\d{1,3}(?:\.\d+)?).*?"
        r"lon(?:gitude)?\s*[:\s]\s*(?P<lon>-?\d{1,3}(?:\.\d+)?)",
        re.I | re.S,
    ),
    # (38.1, 142.4)  — heuristic: first is lat, second lon
    re.compile(
        r"\(\s*(?P<lat>-?\d{1,2}(?:\.\d+)?)\s*,\s*(?P<lon>-?\d{2,3}(?:\.\d+)?)\s*\)"
    ),
]

# Dateline pattern: "CITY (Reuters) —" or "CITY, COUNTRY (AP)"
_DATELINE = re.compile(r"^([A-Z][A-Z\s]{1,30}?)\s*[,(]")

# ---------------------------------------------------------------------------
# geonamescache helpers
# ---------------------------------------------------------------------------
_gc = geonamescache.GeonamesCache()

def _build_city_index():
    """Build a lowercase city-name → (country_iso2, lat, lon) index."""
    idx: dict[str, tuple[str, float, float]] = {}
    for city in _gc.get_cities().values():
        name = city["name"].lower()
        idx[name] = (city["countrycode"], float(city["latitude"]), float(city["longitude"]))
        # also index ASCII name if different
        alt = city.get("alternatenames", "")
        if alt:
            for a in alt.split(","):
                a = a.strip().lower()
                if a and a not in idx:
                    idx[a] = (city["countrycode"], float(city["latitude"]), float(city["longitude"]))
    return idx

_CITY_INDEX: dict[str, tuple[str, float, float]] | None = None

def _get_city_index():
    global _CITY_INDEX
    if _CITY_INDEX is None:
        _CITY_INDEX = _build_city_index()
    return _CITY_INDEX

def _build_country_name_index():
    """lowercase country name / alpha2 / alpha3 → alpha2."""
    idx: dict[str, str] = {}
    for c in _gc.get_countries().values():
        idx[c["name"].lower()] = c["iso"]
        idx[c["iso"].lower()]  = c["iso"]
    for country in pycountry.countries:
        key = country.name.lower()
        idx[key] = country.alpha_2
        if hasattr(country, "common_name"):
            idx[country.common_name.lower()] = country.alpha_2
        if hasattr(country, "official_name"):
            idx[country.official_name.lower()] = country.alpha_2
    for alias, code in _COUNTRY_ALIASES.items():
        idx[alias] = code
    return idx

_COUNTRY_INDEX: dict[str, str] | None = None

def _get_country_index():
    global _COUNTRY_INDEX
    if _COUNTRY_INDEX is None:
        _COUNTRY_INDEX = _build_country_name_index()
    return _COUNTRY_INDEX


# ---------------------------------------------------------------------------
# Sub-extractors
# ---------------------------------------------------------------------------

def _extract_coords_from_text(text: str) -> tuple[Optional[float], Optional[float]]:
    """Try to find explicit coordinate expressions in text."""
    for pat in _COORD_PATTERNS:
        m = pat.search(text)
        if m:
            gd = m.groupdict()
            try:
                lat = float(gd["lat"])
                lon = float(gd["lon"])
                if "lat_h" in gd and gd["lat_h"].upper() == "S":
                    lat = -lat
                if "lon_h" in gd and gd["lon_h"].upper() == "W":
                    lon = -lon
                if -90 <= lat <= 90 and -180 <= lon <= 180:
                    return round(lat, 4), round(lon, 4)
            except (ValueError, TypeError):
                continue
    return None, None


def _name_to_country(name: str) -> Optional[str]:
    """Map a place name to ISO-2 country code."""
    ci = _get_country_index()
    n = name.lower().strip()

    # Direct country name match
    if n in ci:
        return ci[n]

    # Try as city → country via geonamescache
    city_idx = _get_city_index()
    if n in city_idx:
        return city_idx[n][0]

    # Partial match: check if name ends with a known country name
    for cname, code in ci.items():
        if n.endswith(f", {cname}") or n.endswith(f" {cname}"):
            return code

    return None


def _city_to_coords(name: str) -> tuple[Optional[str], Optional[float], Optional[float]]:
    """Return (country_iso2, lat, lon) if name matches a city, else (None, None, None)."""
    city_idx = _get_city_index()
    n = name.lower().strip()
    if n in city_idx:
        code, lat, lon = city_idx[n]
        return code, lat, lon
    return None, None, None


def _extract_gpe_entities(text: str, title: str) -> list[str]:
    """
    Return GPE/LOC entities from title + first 3 sentences of body,
    with dateline entity stripped and trigger-sentence entities first.
    """
    nlp = _get_nlp()
    entities: list[tuple[str, int]] = []  # (text, priority)

    # --- title ---
    title_doc = nlp(title)
    title_gpes = [ent.text for ent in title_doc.ents if ent.label_ in ("GPE", "LOC")]

    # --- body: first 3 sentences ---
    # Split body (after [SEP] if present)
    body = text.split("[SEP]", 1)[-1].strip() if "[SEP]" in text else text

    # Strip dateline: "CITY (Agency) —"
    dateline_name: Optional[str] = None
    dm = _DATELINE.match(body)
    if dm:
        dateline_name = dm.group(1).strip()

    sentences = re.split(r"(?<=[.!?])\s+", body)[:3]
    body_snippet = " ".join(sentences)
    body_doc = nlp(body_snippet)

    body_gpes: list[tuple[str, int]] = []
    for ent in body_doc.ents:
        if ent.label_ not in ("GPE", "LOC"):
            continue
        if dateline_name and ent.text.strip().lower() == dateline_name.lower():
            continue  # skip dateline city
        # Check if in a trigger sentence
        sent_text = ent.sent.text if ent.sent else ""
        priority = 0 if _TRIGGER_VERBS.search(sent_text) else 1
        body_gpes.append((ent.text, priority))

    # Sort body GPEs: trigger sentences first
    body_gpes.sort(key=lambda x: x[1])

    # Combine: title GPEs first (high relevance), then sorted body GPEs
    seen = set()
    result = []
    for name in title_gpes:
        n = name.strip()
        if n and n.lower() not in seen:
            seen.add(n.lower())
            result.append(n)
    for name, _ in body_gpes:
        n = name.strip()
        if n and n.lower() not in seen:
            seen.add(n.lower())
            result.append(n)

    return result


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def extract_location(text: str, title: str = "") -> LocationResult:
    """
    Extract location_text, country_iso2, lat, lon from article text.

    Args:
        text:  full article text (may contain "[SEP]" separator)
        title: article title (used preferentially for location extraction)
    """
    result = LocationResult()

    # 1. Try explicit coordinates in text
    lat, lon = _extract_coords_from_text(text + " " + title)
    if lat is not None:
        result.lat = lat
        result.lon = lon
        result.confidence = "coords_text"

    # 2. Extract GPE entities
    gpes = _extract_gpe_entities(text, title)

    if not gpes:
        return result

    # 3. Find the best entity: first one that resolves to a city or country
    for name in gpes:
        # Try city first (more specific)
        code, clat, clon = _city_to_coords(name)
        if code:
            result.location_text = name
            result.country_iso2  = code
            if result.lat is None:
                result.lat = clat
                result.lon = clon
                result.confidence = "city"
            return result

        # Try as country name
        code = _name_to_country(name)
        if code:
            result.location_text = name
            result.country_iso2  = code
            if result.lat is None and code in _COUNTRY_CENTROIDS:
                result.lat, result.lon = _COUNTRY_CENTROIDS[code]
                result.confidence = "country"
            return result

    # 4. Fallback: use first GPE as location_text even if we can't resolve it
    result.location_text = gpes[0]
    # Try country resolution one more time with partial matching
    for name in gpes:
        code = _name_to_country(name)
        if code:
            result.country_iso2 = code
            if result.lat is None and code in _COUNTRY_CENTROIDS:
                result.lat, result.lon = _COUNTRY_CENTROIDS[code]
                result.confidence = "country"
            break

    return result


# ---------------------------------------------------------------------------
# Batch extraction helper
# ---------------------------------------------------------------------------

def extract_location_from_row(row) -> dict:
    """Convenience wrapper for DataFrame.apply usage."""
    text  = str(row.get("text", "") or "")
    title = str(row.get("title", "") or "")
    if not title and "[SEP]" in text:
        title = text.split("[SEP]", 1)[0].strip()
    res = extract_location(text, title)
    return {
        "location_text": res.location_text,
        "country_iso2":  res.country_iso2,
        "lat":           res.lat,
        "lon":           res.lon,
        "loc_confidence": res.confidence,
    }


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        ("Cyclone Belal hits France's Reunion Island, Mauritius on high alert",
         "Cyclone Belal made landfall on Reunion Island on Monday. Mauritius was also on alert."),
        ("M7.4 earthquake strikes Hualien, Taiwan",
         "A 7.4-magnitude earthquake struck Hualien County in eastern Taiwan at 7:58 a.m. "
         "The epicenter was located at 23.97°N, 121.59°E, at a depth of 15 km."),
        ("Wildfires force 10,000 evacuations in Los Angeles",
         "LOS ANGELES (AP) — A fast-moving wildfire broke out Tuesday in the hills north of LA, "
         "forcing 10,000 residents to flee. The blaze has burned 5,000 acres."),
        ("Severe drought hits the Horn of Africa",
         "Somalia, Ethiopia and Kenya are experiencing their worst drought in 40 years. "
         "Over 20 million people are at risk of starvation."),
        ("Floods kill dozens in central Vietnam",
         "HANOI (Reuters) — Flash floods and landslides have killed at least 35 people "
         "in Quang Nam and Quang Tri provinces in central Vietnam."),
    ]

    for title, text in tests:
        r = extract_location(text, title)
        print(f"Title : {title[:60]}")
        print(f"  location : {r.location_text}")
        print(f"  country  : {r.country_iso2}")
        print(f"  lat/lon  : {r.lat}, {r.lon}  [{r.confidence}]")
        print()
