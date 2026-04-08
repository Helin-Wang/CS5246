"""
Location extractor for disaster news articles.

Extracts four fields from article text (title + body):
    location_text  : most specific place name where the disaster occurred
    country_iso2   : ISO 3166-1 alpha-2 country code
    lat            : decimal latitude  (None if unavailable)
    lon            : decimal longitude (None if unavailable)

lat/lon:
    Only extracted from explicit coordinate expressions in the article text
    (e.g. "23.97°N, 121.6°E", "latitude 38.1 longitude 142.4").
    No centroid inference — if the text doesn't mention coordinates, lat/lon = None.

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
        _nlp = spacy.load("en_core_web_sm", disable=["lemmatizer"])
        if "sentencizer" not in _nlp.pipe_names and "senter" not in _nlp.pipe_names:
            _nlp.add_pipe("sentencizer")
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
    confidence:    str = "none"   # "coords_text" | "location" | "none"
    # coords_text: lat/lon extracted from explicit text coordinates (e.g. "38.1°N 142.4°E")
    # location:    location_text/country resolved from NER; lat/lon remain None
    # none:        no location resolved


# Country name alias patches (common variants not in pycountry)
_COUNTRY_ALIASES: dict[str, str] = {
    "america": "US", "the us": "US", "the united states": "US",
    "usa": "US", "u.s.": "US", "u.s.a.": "US",
    "south korea": "KR", "north korea": "KP",
    "britain": "GB", "great britain": "GB", "england": "GB",
    "scotland": "GB", "wales": "GB",
    "uk": "GB", "u.k.": "GB",
    "russia": "RU", "uae": "AE",
    "tanzania": "TZ", "congo": "CD",
    "syria": "SY", "iran": "IR", "taiwan": "TW",
    "vietnam": "VN", "laos": "LA", "myanmar": "MM",
    "ivory coast": "CI", "cape verde": "CV",
    # US states → US
    "alabama": "US", "alaska": "US", "arizona": "US", "arkansas": "US",
    "california": "US", "colorado": "US", "connecticut": "US", "delaware": "US",
    "florida": "US", "georgia": "US", "hawaii": "US", "idaho": "US",
    "illinois": "US", "indiana": "US", "iowa": "US", "kansas": "US",
    "kentucky": "US", "louisiana": "US", "maine": "US", "maryland": "US",
    "massachusetts": "US", "michigan": "US", "minnesota": "US", "mississippi": "US",
    "missouri": "US", "montana": "US", "nebraska": "US", "nevada": "US",
    "new hampshire": "US", "new jersey": "US", "new mexico": "US", "new york": "US",
    "north carolina": "US", "north dakota": "US", "ohio": "US", "oklahoma": "US",
    "oregon": "US", "pennsylvania": "US", "rhode island": "US", "south carolina": "US",
    "south dakota": "US", "tennessee": "US", "texas": "US", "utah": "US",
    "vermont": "US", "virginia": "US", "washington": "US", "west virginia": "US",
    "wisconsin": "US", "wyoming": "US",
    # US territories / common references
    "puerto rico": "US", "us virgin islands": "US", "guam": "US",
    # Province/state abbreviations
    "b.c.": "CA", "bc": "CA", "ont.": "CA", "que.": "CA", "sask.": "CA",
    "n.s.": "CA", "n.b.": "CA", "p.e.i.": "CA", "nfld.": "CA",
    "n.s.w.": "AU", "qld.": "AU", "w.a.": "AU", "s.a.": "AU",
    # Canadian provinces → CA
    "british columbia": "CA", "alberta": "CA", "saskatchewan": "CA", "manitoba": "CA",
    "ontario": "CA", "quebec": "CA", "nova scotia": "CA", "new brunswick": "CA",
    "newfoundland": "CA", "prince edward island": "CA", "yukon": "CA",
    "northwest territories": "CA", "nunavut": "CA",
    # Australian states → AU
    "new south wales": "AU", "victoria": "AU", "queensland": "AU",
    "western australia": "AU", "south australia": "AU", "tasmania": "AU",
    "northern territory": "AU", "australian capital territory": "AU",
    # Indian states → IN
    "rajasthan": "IN", "gujarat": "IN", "maharashtra": "IN", "uttar pradesh": "IN",
    "madhya pradesh": "IN", "karnataka": "IN", "tamil nadu": "IN", "andhra pradesh": "IN",
    "telangana": "IN", "odisha": "IN", "west bengal": "IN", "assam": "IN",
    "bihar": "IN", "jharkhand": "IN", "kerala": "IN", "punjab": "IN",
    "haryana": "IN", "himachal pradesh": "IN", "uttarakhand": "IN",
    # Chinese provinces → CN
    "sichuan": "CN", "yunnan": "CN", "guangdong": "CN", "fujian": "CN",
    "zhejiang": "CN", "jiangsu": "CN", "xinjiang": "CN", "tibet": "CN",
    "gansu": "CN", "qinghai": "CN", "shanxi": "CN", "shandong": "CN",
    "henan": "CN", "hubei": "CN", "hunan": "CN", "guizhou": "CN",
    "guangxi": "CN", "inner mongolia": "CN",
    # Japanese prefectures → JP
    "hokkaido": "JP", "miyagi": "JP", "iwate": "JP", "fukushima": "JP",
    "tokyo": "JP", "osaka": "JP", "kyoto": "JP", "hiroshima": "JP",
    "kumamoto": "JP", "okinawa": "JP",
    # Indonesian provinces → ID
    "aceh": "ID", "java": "ID", "sumatra": "ID", "sulawesi": "ID",
    "kalimantan": "ID", "papua": "ID", "lombok": "ID", "bali": "ID",
    # Philippines regions → PH
    "luzon": "PH", "mindanao": "PH", "visayas": "PH", "cebu": "PH",
    # Other common sub-national to country
    "kashmir": "IN", "crimea": "UA", "catalonia": "ES",
    "sicily": "IT", "sardinia": "IT",
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
    # Europe
    "RS": (44.0, 21.0),  "BA": (44.2, 17.9),  "HR": (45.1, 15.2),
    "SI": (46.1, 14.8),  "MK": (41.6, 21.7),  "AL": (41.2, 20.2),
    "ME": (42.7, 19.4),  "BG": (42.7, 25.5),  "HU": (47.2, 19.5),
    "SK": (48.7, 19.7),  "CZ": (49.8, 15.5),  "PL": (51.9, 19.1),
    "LT": (55.2, 23.9),  "LV": (56.9, 24.6),  "EE": (58.6, 25.0),
    "FI": (61.9, 25.7),  "SE": (60.1, 18.6),  "DK": (56.3, 9.5),
    "NL": (52.1, 5.3),   "BE": (50.5, 4.5),   "CH": (46.8, 8.2),
    "AT": (47.5, 14.6),  "LU": (49.8, 6.1),
    # Middle East
    "SA": (23.9, 45.1),  "IQ": (33.2, 43.7),  "YE": (15.6, 48.5),
    "JO": (30.6, 36.2),  "LB": (33.9, 35.9),  "IL": (31.1, 35.0),
    "SY": (34.8, 38.9),  "KW": (29.3, 47.5),  "QA": (25.4, 51.2),
    "OM": (21.5, 55.9),  "AE": (23.4, 53.8),  "BH": (26.0, 50.6),
    # Africa
    "TN": (33.9, 9.5),   "LY": (26.3, 17.2),  "GH": (7.9, -1.0),
    "SN": (14.5, -14.5), "CM": (3.9, 11.5),   "CI": (5.4, -4.0),
    "ML": (17.6, -2.0),  "BF": (12.4, -1.6),  "NE": (17.6, 8.1),
    "TD": (15.5, 18.7),  "SS": (7.9, 30.2),   "UG": (1.4, 32.3),
    "RW": (-1.9, 29.9),  "BI": (-3.4, 29.9),  "AO": (-11.2, 17.9),
    "ZM": (-13.1, 27.8), "BW": (-22.3, 24.7), "NA": (-22.9, 18.5),
    # South / SE Asia
    "KH": (12.6, 104.9), "LA": (17.9, 102.6), "SG": (1.4, 103.8),
    "MY": (4.2, 108.0),  "BN": (4.5, 114.7),  "TL": (-8.9, 125.7),
    "NP": (28.4, 84.1),  "BT": (27.5, 90.4),  "MV": (0.0, 73.5),
    # Central Asia
    "KZ": (48.0, 66.9),  "UZ": (41.4, 64.6),  "TM": (38.9, 59.6),
    "TJ": (38.9, 71.3),  "KG": (41.2, 74.8),
    # Pacific / others
    "CK": (-21.2, -159.8),"KI": (1.9, -157.4), "FM": (6.9, 158.2),
    "MH": (7.1, 171.2),  "PW": (7.5, 134.6),  "SB": (-9.6, 160.2),
    "NC": (-20.9, 165.6), "PF": (-17.7, -149.4),
    "CY": (35.1, 33.4),  "AM": (40.1, 45.0),  "GE": (42.3, 43.4),
    "AZ": (40.1, 47.6),
    # Asian territories
    "HK": (22.3, 114.2), "MO": (22.2, 113.5), "TW": (23.6, 121.0),
    "SG": (1.35, 103.8),
}

# Subregion centroids: location_text (lowercase) → (lat, lon)
# Used when the resolved country is correct but coords would default to
# the national centroid (e.g. US state, Canadian province, AU state).
_SUBREGION_CENTROIDS: dict[str, tuple[float, float]] = {
    # US states
    "alabama": (32.8, -86.8), "alaska": (64.2, -153.4), "arizona": (34.3, -111.1),
    "arkansas": (34.8, -92.2), "california": (36.8, -119.4), "colorado": (39.0, -105.5),
    "connecticut": (41.6, -72.7), "delaware": (38.9, -75.5), "florida": (27.8, -81.5),
    "georgia": (32.7, -83.2), "hawaii": (20.8, -156.3), "idaho": (44.1, -114.5),
    "illinois": (40.3, -89.0), "indiana": (40.3, -86.1), "iowa": (42.0, -93.5),
    "kansas": (38.5, -98.4), "kentucky": (37.5, -85.3), "louisiana": (31.2, -91.8),
    "maine": (45.3, -69.0), "maryland": (39.0, -76.8), "massachusetts": (42.2, -71.5),
    "michigan": (44.3, -85.4), "minnesota": (46.4, -93.1), "mississippi": (32.7, -89.6),
    "missouri": (38.3, -92.5), "montana": (47.0, -109.6), "nebraska": (41.5, -99.9),
    "nevada": (38.5, -117.1), "new hampshire": (43.9, -71.6), "new jersey": (40.2, -74.7),
    "new mexico": (34.5, -106.2), "new york": (42.9, -75.5), "north carolina": (35.5, -79.4),
    "north dakota": (47.5, -100.5), "ohio": (40.4, -82.8), "oklahoma": (35.6, -96.9),
    "oregon": (43.8, -120.6), "pennsylvania": (41.2, -77.2), "rhode island": (41.7, -71.5),
    "south carolina": (33.8, -80.9), "south dakota": (44.4, -100.4), "tennessee": (35.9, -86.7),
    "texas": (31.1, -97.6), "utah": (39.3, -111.1), "vermont": (44.1, -72.7),
    "virginia": (37.7, -78.7), "washington": (47.4, -120.4), "west virginia": (38.5, -80.4),
    "wisconsin": (44.5, -89.5), "wyoming": (43.1, -107.6),
    "hawaii": (20.8, -156.3), "puerto rico": (18.2, -66.6),
    # Canadian provinces
    "british columbia": (53.7, -127.6), "b.c.": (53.7, -127.6), "bc": (53.7, -127.6),
    "alberta": (53.9, -116.6), "saskatchewan": (52.9, -106.4), "manitoba": (53.8, -98.8),
    "ontario": (51.3, -85.3), "quebec": (52.9, -73.5), "nova scotia": (45.0, -63.2),
    "new brunswick": (46.6, -66.5), "newfoundland": (53.1, -57.7),
    "prince edward island": (46.4, -63.1), "yukon": (64.3, -135.0),
    # Australian states
    "new south wales": (-32.0, 147.0), "n.s.w.": (-32.0, 147.0),
    "victoria": (-36.9, 144.0), "queensland": (-22.6, 144.1),
    "western australia": (-25.3, 122.0), "south australia": (-30.0, 135.8),
    "tasmania": (-41.5, 145.9), "northern territory": (-19.5, 133.9),
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
            # alternatenames may be a list or a comma-separated string
            names = alt if isinstance(alt, list) else alt.split(",")
            for a in names:
                a = str(a).strip().lower()
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

    # Partial match: "City, Country" format — only match after ", " to avoid
    # false positives like "new jersey" matching "jersey" (JE).
    for cname, code in ci.items():
        if n.endswith(f", {cname}"):
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

    # 3. Find the best entity: first one that resolves to a city or country.
    # Check country-level first (alias + pycountry index), then city lookup for
    # country code. lat/lon is NEVER inferred from centroids — only from text (step 1).
    # This prevents city homonyms (e.g. a town called "India" in Serbia)
    # from shadowing the actual country.
    ci = _get_country_index()
    for name in gpes:
        n = name.lower().strip()
        # Country alias / direct country name → prefer over city lookup
        if n in ci:
            code = ci[n]
            result.location_text = name
            result.country_iso2  = code
            if result.confidence == "none":
                result.confidence = "location"
            return result

        # City lookup — gives us country code; lat/lon only if text coords not found
        code, clat, clon = _city_to_coords(name)
        if code:
            result.location_text = name
            result.country_iso2  = code
            if result.confidence == "none":
                result.confidence = "location"
            return result

    # 4. Fallback: use first GPE as location_text even if we can't resolve country
    result.location_text = gpes[0]
    # Try country resolution one more time with partial matching
    for name in gpes:
        code = _name_to_country(name)
        if code:
            result.country_iso2 = code
            if result.confidence == "none":
                result.confidence = "location"
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
