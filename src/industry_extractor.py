"""
Text-based industry impact extractor.

Scans news article text for explicit mentions of affected industries/sectors
using two signals:
  1. Industry keyword matching — curated keyword → sector mapping
  2. spaCy ORG entity → sector lookup (company name → sector)

Output: list of sector strings, e.g. ["insurance", "agriculture", "utilities"]

These are merged with the static knowledge-base sectors in entity_linker.py
to produce a combined affected-sector list used for ETF selection in Module E.

Usage:
    from src.industry_extractor import IndustryExtractor
    ix = IndustryExtractor()
    sectors = ix.extract("Flooding destroyed rice crops; insurers face claims")
    # → ["agriculture", "insurance"]
"""

from __future__ import annotations

import re
from typing import List

# ---------------------------------------------------------------------------
# Keyword → sector mapping
# Each entry: (compiled_pattern, sector_label)
# Patterns are matched case-insensitively against the full article text.
# ---------------------------------------------------------------------------
_KEYWORD_RULES: List[tuple] = []

def _add(pattern: str, sector: str):
    _KEYWORD_RULES.append((re.compile(pattern, re.I), sector))

# Insurance / reinsurance (universally affected by natural disasters)
_add(r"\b(insur(?:ance|er|ed|ers)|reinsur\w+|catastrophe\s+bond|cat\s+bond|claims?\s+payout|underwriter)\b", "insurance")

# Agriculture / food
_add(r"\b(crop(?:s|land)?|harvest|farmland|agricult\w+|livestock|grain|wheat|rice|corn|soybean|food\s+supply|drought.{0,30}farm|flood.{0,30}farm|farm.{0,20}flood|farm.{0,20}drought)\b", "agriculture")

# Energy (oil, gas, power)
_add(r"\b(oil\s+(?:field|platform|refiner|spill|price|export)|natural\s+gas|pipeline|petroleum|fuel\s+supply|energy\s+(?:sector|infrastructure|supply)|blackout|power\s+(?:plant|grid|outage|cut|station))\b", "energy")

# Utilities (electricity, water)
_add(r"\b(utilit(?:y|ies)|electricity\s+(?:supply|grid|outage)|water\s+(?:supply|utility|treatment|shortage)|grid\s+(?:failure|collapse|damage)|power\s+line)\b", "utilities")

# Construction / real estate
_add(r"\b(construction|rebuild(?:ing)?|reconstruct\w+|infrastructure\s+damage|bridge\s+(?:collapse|damage)|road\s+(?:damage|destroyed)|building\s+(?:collapse|damage|destroyed)|real\s+estate|property\s+damage|housing\s+damage)\b", "construction")

# Tourism / hospitality
_add(r"\b(tourism|tourist|hotel|resort|airline|flight\s+cancel|airport\s+(?:close|shut|damage)|travel\s+(?:warning|ban|disruption)|evacuat\w+.{0,30}tourist)\b", "tourism")

# Timber / forestry
_add(r"\b(timber|forest(?:ry)?|lumber|logging|wood\s+(?:supply|industry)|wildfire.{0,30}forest|forest.{0,30}burned)\b", "timber")

# Mining
_add(r"\b(mining|mine\s+(?:flood|collapse|shut)|mineral\s+(?:export|production)|coal\s+(?:mine|production)|copper\s+(?:mine|output))\b", "mining")

# Textiles / manufacturing
_add(r"\b(garment|textile|manufactur\w+\s+(?:disruption|halt|shutdown)|factory\s+(?:flood|damage|shutdown)|supply\s+chain\s+(?:disruption|breakdown|interrupt))\b", "manufacturing")

# Shipping / trade
_add(r"\b(port\s+(?:damage|closed|operations)|shipping\s+(?:route|disruption)|trade\s+(?:disruption|halted)|export\s+(?:disruption|halted|ban)|import\s+(?:disruption|shortage))\b", "shipping")

# Financial / banking
_add(r"\b(bank\w*\s+(?:loss|exposure|write.?off)|stock\s+(?:market|exchange)\s+(?:fell|drop|plunge|crash)|financial\s+(?:loss|impact|market))\b", "financial")


# ---------------------------------------------------------------------------
# Company → sector lookup (common multinationals that appear in disaster news)
# ---------------------------------------------------------------------------
_ORG_SECTOR: dict = {
    # Insurance / reinsurance
    "aig": "insurance", "allianz": "insurance", "axa": "insurance",
    "munich re": "insurance", "swiss re": "insurance", "lloyd's": "insurance",
    "zurich": "insurance", "berkshire": "insurance", "allstate": "insurance",
    "state farm": "insurance", "tokio marine": "insurance",
    # Energy
    "chevron": "energy", "exxon": "energy", "shell": "energy", "bp": "energy",
    "total": "energy", "conocophillips": "energy", "halliburton": "energy",
    "schlumberger": "energy", "petronas": "energy", "sinopec": "energy",
    "pg&e": "utilities", "duke energy": "utilities", "nextera": "utilities",
    # Agriculture / food
    "cargill": "agriculture", "adm": "agriculture", "bunge": "agriculture",
    "tyson": "agriculture", "nestle": "agriculture", "unilever": "agriculture",
    # Construction
    "caterpillar": "construction", "lafarge": "construction",
    "bechtel": "construction", "fluor": "construction",
    # Mining
    "bhp": "mining", "rio tinto": "mining", "vale": "mining",
    "barrick": "mining", "freeport": "mining", "glencore": "mining",
    # Airlines / tourism
    "united airlines": "tourism", "delta": "tourism", "american airlines": "tourism",
    "southwest": "tourism", "lufthansa": "tourism", "air france": "tourism",
    "marriott": "tourism", "hilton": "tourism",
}


class IndustryExtractor:
    """
    Extract affected industry sectors from article text.

    Two signals:
      - Keyword patterns (reliable, domain-specific)
      - spaCy ORG entities mapped to sectors (optional, loads spaCy model)

    Parameters
    ----------
    use_spacy : bool
        Whether to run spaCy ORG extraction (adds ~100ms per article).
        Disable for speed; keyword matching alone is usually sufficient.
    """

    def __init__(self, use_spacy: bool = False):
        self._nlp = None
        if use_spacy:
            try:
                import spacy
                self._nlp = spacy.load("en_core_web_sm")
            except Exception:
                pass

    def extract(self, text: str) -> List[str]:
        """
        Return a deduplicated list of affected sector strings.
        """
        text = str(text or "")
        sectors = set()

        # Signal 1: keyword patterns
        for pattern, sector in _KEYWORD_RULES:
            if pattern.search(text):
                sectors.add(sector)

        # Signal 2: spaCy ORG → sector
        if self._nlp:
            doc = self._nlp(text[:1000])   # limit for speed
            for ent in doc.ents:
                if ent.label_ == "ORG":
                    name_lower = ent.text.lower().strip()
                    for org_key, sector in _ORG_SECTOR.items():
                        if org_key in name_lower:
                            sectors.add(sector)
                            break

        return sorted(sectors)
