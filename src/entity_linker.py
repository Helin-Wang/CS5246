"""
Module B — Entity Linking

Maps disaster event location strings to:
  1. ISO-3166-1 alpha-2 country code
  2. Stock market index ticker for that country
  3. Key industries likely affected by the disaster

Resolution chain (in order):
  a. Already have country_iso2 (from location_extractor) → use it directly
  b. Parse location_text with pycountry (name / alpha-2 / alpha-3)
  c. Alias dictionary (handles common short forms not in pycountry)
  d. geonamescache country name search

Falls back to None if resolution fails.

Usage:
    from src.entity_linker import EntityLinker
    linker = EntityLinker()
    result = linker.link({
        "country_iso2": "NP",
        "location_text": "Kathmandu, Nepal",
        "event_type": "EQ",
    })
    # {"country_iso2": "NP", "country_name": "Nepal",
    #  "index_ticker": "EWJ", "key_industries": [...], "linked": True}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import pycountry

try:
    import geonamescache as _gnc
    _GC = _gnc.GeonamesCache()
    _GC_COUNTRIES: Dict[str, dict] = _GC.get_countries()  # iso2 → info
    _GC_NAME_TO_ISO2: Dict[str, str] = {
        v["name"].lower(): k for k, v in _GC_COUNTRIES.items()
    }
except Exception:
    _GC_NAME_TO_ISO2 = {}


BASE_DIR = Path(__file__).resolve().parents[1]
KB_FILE  = BASE_DIR / "data" / "country_knowledge_base.json"

# --------------------------------------------------------------------------
# Alias dictionary: lower-cased → ISO-2 code
# Covers common abbreviations, demonyms, and regional names not in pycountry
# --------------------------------------------------------------------------
COUNTRY_ALIASES: Dict[str, str] = {
    # Southeast Asia
    "philippines": "PH", "luzon": "PH", "mindanao": "PH", "visayas": "PH",
    "vietnam": "VN", "viet nam": "VN",
    "myanmar": "MM", "burma": "MM",
    "cambodia": "KH",
    "thailand": "TH",
    "laos": "LA",
    "timor-leste": "TL", "east timor": "TL",
    # South Asia
    "india": "IN", "indian": "IN",
    "bangladesh": "BD",
    "pakistan": "PK",
    "nepal": "NP", "nepali": "NP",
    "sri lanka": "LK",
    # East Asia
    "china": "CN", "chinese": "CN",
    "japan": "JP", "japanese": "JP",
    "south korea": "KR", "korea": "KR",
    "north korea": "KP",
    "taiwan": "TW",
    # Pacific
    "papua new guinea": "PG", "png": "PG",
    "fiji": "FJ",
    "vanuatu": "VU",
    "solomon islands": "SB",
    "tonga": "TO",
    "samoa": "WS", "american samoa": "AS",
    "new caledonia": "NC",
    "new zealand": "NZ",
    "australia": "AU", "australian": "AU",
    # Americas
    "usa": "US", "united states": "US", "u.s.": "US", "u.s.a.": "US",
    "mexico": "MX", "mexican": "MX",
    "brazil": "BR", "brazilian": "BR",
    "chile": "CL", "chilean": "CL",
    "colombia": "CO", "colombian": "CO",
    "peru": "PE", "peruvian": "PE",
    "ecuador": "EC", "ecuadorian": "EC",
    "bolivia": "BO",
    "venezuela": "VE", "venezuelan": "VE",
    "argentina": "AR", "argentine": "AR", "argentinian": "AR",
    "haiti": "HT", "haitian": "HT",
    "cuba": "CU", "cuban": "CU",
    "dominican republic": "DO",
    "puerto rico": "PR",
    "canada": "CA", "canadian": "CA",
    "guatemala": "GT",
    "el salvador": "SV",
    "honduras": "HN",
    "nicaragua": "NI",
    "costa rica": "CR",
    "panama": "PA",
    # Europe
    "turkey": "TR", "türkiye": "TR", "turkiye": "TR",
    "greece": "GR", "greek": "GR",
    "italy": "IT", "italian": "IT",
    "france": "FR", "french": "FR",
    "germany": "DE", "german": "DE",
    "spain": "ES", "spanish": "ES",
    "portugal": "PT", "portuguese": "PT",
    "united kingdom": "GB", "uk": "GB", "britain": "GB", "england": "GB",
    "ukraine": "UA", "ukrainian": "UA",
    "russia": "RU", "russian": "RU",
    # Middle East
    "iran": "IR", "iranian": "IR",
    "iraq": "IQ", "iraqi": "IQ",
    "afghanistan": "AF", "afghan": "AF",
    "yemen": "YE", "yemeni": "YE",
    "syria": "SY", "syrian": "SY",
    # Africa
    "nigeria": "NG", "nigerian": "NG",
    "kenya": "KE", "kenyan": "KE",
    "ethiopia": "ET", "ethiopian": "ET",
    "south africa": "ZA", "s. africa": "ZA",
    "mozambique": "MZ",
    "madagascar": "MG",
    "malawi": "MW",
    "zimbabwe": "ZW",
    "somalia": "SO", "somali": "SO",
    "sudan": "SD", "sudanese": "SD",
    "south sudan": "SS",
    "congo": "CD", "democratic republic of the congo": "CD", "dr congo": "CD", "drc": "CD",
    "cameroon": "CM", "cameroonian": "CM",
    "ghana": "GH", "ghanaian": "GH",
    "tanzania": "TZ", "tanzanian": "TZ",
    "uganda": "UG", "ugandan": "UG",
    "indonesia": "ID", "indonesian": "ID",
    "malaysia": "MY", "malaysian": "MY",
}


def _load_kb() -> Dict[str, dict]:
    if KB_FILE.exists():
        with open(KB_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Remove comment key if present
        data.pop("_comment", None)
        return data
    return {}


_KB: Dict[str, dict] = _load_kb()

# ---------------------------------------------------------------------------
# Sector ETF map  (sector label → ETF ticker)
# ---------------------------------------------------------------------------
_ETF_MAP_FILE = BASE_DIR / "data" / "sector_etf_map.json"

def _load_etf_map() -> Dict[str, str]:
    if _ETF_MAP_FILE.exists():
        data = json.loads(_ETF_MAP_FILE.read_text())
        data.pop("_comment", None)
        return data
    return {}

_ETF_MAP: Dict[str, str] = _load_etf_map()

# Static prior: event_type → sectors validated on training split (2024-01 ~ 2025-07).
# Refined from domain-knowledge candidates via training-set CAR analysis (full-text pipeline):
#   kept:    EQ→construction (T+3 -0.78% p=0.018), DR→insurance (T+5 -0.44% p=0.014)
#   removed: TC→energy (positive direction, inconsistent with disruption), TC→tourism (n=16 only),
#            FL→utilities (n.s.), WF→timber (n.s.)
#   kept on theory: EQ→insurance, TC→insurance, WF→timber
_EVENT_TYPE_SECTORS: Dict[str, List[str]] = {
    "EQ": ["insurance", "construction"],
    "TC": ["insurance", "agriculture"],
    "WF": ["insurance", "utilities", "timber"],
    "DR": ["agriculture", "insurance"],
    "FL": ["insurance", "agriculture", "construction"],
}


def _resolve_iso2(country_iso2: Optional[str], location_text: Optional[str]) -> Optional[str]:
    """Try to resolve an ISO-2 code from existing code or location text."""
    # Direct ISO-2 already provided
    if country_iso2 and len(str(country_iso2).strip()) == 2:
        code = str(country_iso2).strip().upper()
        if pycountry.countries.get(alpha_2=code):
            return code

    if not location_text or str(location_text).strip().lower() in ("none", "nan", ""):
        return country_iso2.strip().upper() if country_iso2 else None

    text = str(location_text).strip()

    # Try tokens longest-first (handles "United States", "New Zealand", etc.)
    words = text.replace(",", " ").split()
    for length in range(min(len(words), 4), 0, -1):
        for start in range(len(words) - length + 1):
            phrase = " ".join(words[start: start + length]).lower().strip(".")
            # Alias dict
            if phrase in COUNTRY_ALIASES:
                return COUNTRY_ALIASES[phrase]
            # pycountry name
            try:
                country = pycountry.countries.lookup(phrase)
                return country.alpha_2
            except LookupError:
                pass
            # geonamescache
            if phrase in _GC_NAME_TO_ISO2:
                return _GC_NAME_TO_ISO2[phrase]

    return None


class EntityLinker:
    """
    Resolve disaster event locations and texts to affected industries and
    corresponding sector ETF tickers for Module E stock analysis.

    Two complementary signals are merged:
      Branch A (static / KB): event_type → prior sector list from domain knowledge
      Branch B (dynamic / text): news text → explicit industry mention extraction

    The union of both branches determines which sector ETFs to analyse.

    Parameters
    ----------
    kb_path    : optional override for country knowledge base JSON path.
    use_spacy  : pass True to enable spaCy ORG extraction in IndustryExtractor.
    """

    def __init__(self, kb_path: Optional[Path] = None, use_spacy: bool = False):
        self.kb: Dict[str, dict] = _load_kb() if kb_path is None else json.loads(kb_path.read_text())
        from industry_extractor import IndustryExtractor
        self._ix = IndustryExtractor(use_spacy=use_spacy)

    def link(self, event: dict) -> dict:
        """
        Resolve a single event cluster dict.

        Input keys used:
            country_iso2   : from location extractor
            location_text  : fallback for country resolution
            event_type     : "EQ" / "TC" etc. — drives Branch A sector prior
            article_texts  : optional list of raw article strings for Branch B
                             (if absent, only Branch A is used)

        Returns
        -------
        dict with keys:
            country_iso2      : resolved ISO-2 (or None)
            country_name      : full country name (or None)
            key_industries    : merged sector list (Branch A ∪ Branch B)
            sector_etfs       : list of ETF tickers for key_industries
            sectors_from_kb   : Branch A sectors (event_type prior)
            sectors_from_text : Branch B sectors (text extraction)
            linked            : True if country_iso2 resolved
        """
        iso2 = _resolve_iso2(event.get("country_iso2"), event.get("location_text"))

        country_name = None
        if iso2:
            pc = pycountry.countries.get(alpha_2=iso2)
            country_name = pc.name if pc else iso2

        # Branch A: static prior by event_type
        from event_clusterer import _type_code
        etype = _type_code(event.get("event_type", ""))
        kb_sectors = list(_EVENT_TYPE_SECTORS.get(etype, []))

        # Also add country-KB industries if available (supplementary)
        kb_entry = self.kb.get(iso2, {}) if iso2 else {}
        for s in kb_entry.get("key_industries", []):
            if s not in kb_sectors:
                kb_sectors.append(s)

        # Branch B: text-based extraction
        text_sectors: List[str] = []
        article_texts = event.get("article_texts") or []
        if isinstance(article_texts, str):
            article_texts = [article_texts]
        for text in article_texts[:5]:   # cap at 5 articles per event for speed
            for s in self._ix.extract(text):
                if s not in text_sectors:
                    text_sectors.append(s)

        # Merge
        all_sectors = list(dict.fromkeys(kb_sectors + text_sectors))   # preserves order, dedupes

        # Map sectors → ETF tickers
        etfs = list(dict.fromkeys(
            _ETF_MAP[s] for s in all_sectors if s in _ETF_MAP
        ))

        return {
            "country_iso2":       iso2,
            "country_name":       country_name,
            "key_industries":     all_sectors,
            "sector_etfs":        etfs,
            "sectors_from_kb":    kb_sectors,
            "sectors_from_text":  text_sectors,
            "linked":             iso2 is not None or len(etfs) > 0,
        }

    def link_batch(self, events: List[dict]) -> List[dict]:
        return [self.link(ev) for ev in events]
