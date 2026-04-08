"""
Unified, type-agnostic event field extractor for disaster news.

Why this file exists:
- Extract event-related fields from each news article without requiring
  per-disaster-type routing first.
- Produce structured outputs that are immediately useful for downstream
  stock-impact analysis (sector hints + impact score + normalized metrics).

Input:
- CSV or XLSX with a text column (default: text_cleaned)
- Optional timestamp/location columns

Output:
- Original rows + one JSON column ("event_features")

This module intentionally uses a practical hybrid:
- Regex + normalization for quantitative fields
- Keyword-based context checks
- Confidence scoring + evidence span tracking
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


JSON_COLUMN = "event_features"


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
    return [p.strip() for p in parts if p.strip()]


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
    t = str(num_text or "").strip().lower().replace(",", "")
    if not t:
        return None
    multiplier = 1.0
    if t.endswith(("k", "thousand")):
        t = re.sub(r"(k|thousand)$", "", t).strip()
        multiplier = 1_000.0
    elif t.endswith(("m", "million")):
        t = re.sub(r"(m|million)$", "", t).strip()
        multiplier = 1_000_000.0
    elif t.endswith(("b", "billion")):
        t = re.sub(r"(b|billion)$", "", t).strip()
        multiplier = 1_000_000_000.0
    try:
        return float(t) * multiplier
    except ValueError:
        return None


def to_float(x) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


@dataclass
class Candidate:
    field: str
    value: float
    unit: str
    confidence: float
    raw: str
    evidence: str
    start: int
    end: int


class UnifiedEventExtractor:
    def __init__(self):
        self.hazard_keywords = [
            "earthquake",
            "aftershock",
            "flood",
            "flooding",
            "storm",
            "cyclone",
            "typhoon",
            "hurricane",
            "wildfire",
            "bushfire",
            "drought",
            "landslide",
            "tsunami",
            "eruption",
            "volcano",
            "heatwave",
        ]

        self.impact_keywords = {
            "dead": ["dead", "killed", "fatalities", "death toll", "deaths"],
            "injured": ["injured", "wounded", "hurt"],
            "missing": ["missing", "unaccounted"],
            "displaced": ["displaced", "evacuated", "homeless", "relocated"],
            "affected_population": ["affected", "at risk", "impacted", "stranded"],
        }

        self.sector_keywords = {
            "energy": ["power grid", "electricity", "oil", "gas", "refinery", "pipeline"],
            "transportation": ["airport", "flight", "rail", "port", "shipping", "highway"],
            "agriculture": ["crop", "harvest", "farmland", "livestock", "agriculture"],
            "insurance": ["insured losses", "claims", "insurer", "reinsurance"],
            "tourism": ["tourism", "hotel", "resort", "tourist"],
            "technology": ["semiconductor", "chip plant", "data center", "factory"],
            "mining": ["mine", "copper", "lithium", "iron ore"],
        }
        self.negation_cues = [
            "no ",
            "not ",
            "without ",
            "zero ",
            "none ",
            "unlikely ",
            "unconfirmed ",
            "could ",
            "may ",
            "might ",
        ]
        # Field-specific anchor terms used for local syntactic-window binding.
        self.field_anchors = {
            "magnitude": ["magnitude", "richter", "earthquake", "quake", "aftershock"],
            "depth_km": ["depth", "deep", "earthquake", "hypocenter", "epicenter"],
            "wind_speed_kmh": ["wind", "gust", "cyclone", "typhoon", "hurricane", "storm"],
            "rainfall_mm": ["rain", "rainfall", "precipitation", "downpour"],
            "burned_area_ha": ["burned", "scorched", "wildfire", "fire", "forest"],
            "affected_area_km2": ["affected area", "under water", "flooded area", "impacted area"],
            "storm_surge_m": ["storm surge", "surge", "coastal flooding", "inundation"],
            "duration_days": ["for", "lasting", "lasted", "duration", "days", "weeks", "months"],
            "economic_loss_usd": ["loss", "losses", "damage", "insured", "economic"],
            "dead": ["dead", "killed", "fatality", "fatalities", "death toll", "deaths"],
            "injured": ["injured", "wounded", "hurt"],
            "missing": ["missing", "unaccounted"],
            "displaced": ["displaced", "evacuated", "homeless", "relocated"],
            "affected_population": ["affected", "at risk", "impacted", "stranded"],
        }

        # Precompile patterns for practical extraction.
        self.patterns = {
            "magnitude": [
                re.compile(
                    r"(?:magnitude|m|richter)\s*[:\-]?\s*(?P<num>\d+(?:\.\d+)?)",
                    re.I,
                ),
                re.compile(
                    r"\bM(?P<num>\d+(?:\.\d+)?)\b",
                    re.I,
                ),
            ],
            "depth_km": [
                re.compile(
                    r"(?:depth|deep)\s*(?:of\s*)?(?P<num>\d+(?:\.\d+)?)\s*(?P<unit>km|kilometers?|mi|miles?)",
                    re.I,
                )
            ],
            "wind_speed_kmh": [
                re.compile(
                    r"(?P<num>\d+(?:\.\d+)?)\s*(?P<unit>km/h|kph|mph|knots?|kt|m/s)\s*(?:winds?|gusts?)?",
                    re.I,
                )
            ],
            "rainfall_mm": [
                re.compile(
                    r"(?P<num>\d+(?:\.\d+)?)\s*(?P<unit>mm|millimeters?|cm|inches?|in)\s*(?:of\s*)?(?:rain|rainfall|precipitation)",
                    re.I,
                )
            ],
            "burned_area_ha": [
                re.compile(
                    r"(?P<num>\d+(?:\.\d+)?)\s*(?P<unit>ha|hectares?|acres?)\s*(?:burned|scorched|destroyed)?",
                    re.I,
                )
            ],
            "affected_area_km2": [
                re.compile(
                    r"(?P<num>\d+(?:\.\d+)?)\s*(?P<unit>km2|km²|square kilometers?|sq\.?\s*km|sq\.?\s*mi|square miles?)\s*(?:affected|under water|impacted)?",
                    re.I,
                )
            ],
            "storm_surge_m": [
                re.compile(
                    r"storm surge(?:\s*of)?\s*(?P<num>\d+(?:\.\d+)?)\s*(?P<unit>m|meters?|ft|feet)",
                    re.I,
                )
            ],
            "duration_days": [
                re.compile(
                    r"(?:for|lasting|lasted)\s*(?P<num>\d+(?:\.\d+)?)\s*(?P<unit>days?|weeks?|months?)",
                    re.I,
                )
            ],
            "economic_loss_usd": [
                re.compile(
                    r"(?P<currency>\$|usd)\s*(?P<num>\d+(?:\.\d+)?)\s*(?P<scale>k|m|b|thousand|million|billion)?\s*(?:in\s*)?(?:losses?|damage)",
                    re.I,
                ),
                re.compile(
                    r"(?:losses?|damage)\s*(?:estimated at|of|around|near)?\s*(?P<currency>\$|usd)\s*(?P<num>\d+(?:\.\d+)?)\s*(?P<scale>k|m|b|thousand|million|billion)?",
                    re.I,
                ),
            ],
        }

        self.human_patterns = {
            "dead": [
                re.compile(
                    r"(?P<num>\d+(?:\.\d+)?(?:\s*(?:k|m|b|thousand|million|billion))?)\s*(?:people\s*)?(?:were\s*)?(?:killed|dead|fatalities?)",
                    re.I,
                ),
                re.compile(
                    r"(?:death toll|deaths?)\s*(?:rose to|reached|at|of)?\s*(?P<num>\d+(?:\.\d+)?(?:\s*(?:k|m|b|thousand|million|billion))?)",
                    re.I,
                ),
            ],
            "injured": [
                re.compile(
                    r"(?P<num>\d+(?:\.\d+)?(?:\s*(?:k|m|thousand|million))?)\s*(?:people\s*)?(?:were\s*)?injured",
                    re.I,
                )
            ],
            "missing": [
                re.compile(
                    r"(?P<num>\d+(?:\.\d+)?(?:\s*(?:k|m|thousand|million))?)\s*(?:people\s*)?(?:are|were)?\s*missing",
                    re.I,
                )
            ],
            "displaced": [
                re.compile(
                    r"(?P<num>\d+(?:\.\d+)?(?:\s*(?:k|m|thousand|million))?)\s*(?:people\s*)?(?:displaced|evacuated|homeless|relocated)",
                    re.I,
                )
            ],
            "affected_population": [
                re.compile(
                    r"(?P<num>\d+(?:\.\d+)?(?:\s*(?:k|m|b|thousand|million|billion))?)\s*(?:people\s*)?(?:affected|at risk|impacted|stranded)",
                    re.I,
                )
            ],
        }

    def _unit_convert(self, field: str, value: float, unit: str) -> Tuple[float, str]:
        u = (unit or "").lower()

        if field == "depth_km":
            if u in {"mi", "mile", "miles"}:
                return value * 1.60934, "km"
            return value, "km"

        if field == "wind_speed_kmh":
            if u in {"mph"}:
                return value * 1.60934, "km/h"
            if u in {"knot", "knots", "kt"}:
                return value * 1.852, "km/h"
            if u in {"m/s"}:
                return value * 3.6, "km/h"
            return value, "km/h"

        if field == "rainfall_mm":
            if u in {"cm"}:
                return value * 10.0, "mm"
            if u in {"inch", "inches", "in"}:
                return value * 25.4, "mm"
            return value, "mm"

        if field == "burned_area_ha":
            if u in {"acre", "acres"}:
                return value * 0.404686, "ha"
            return value, "ha"

        if field == "affected_area_km2":
            if u in {"sq mi", "sq. mi", "square mile", "square miles"}:
                return value * 2.58999, "km2"
            return value, "km2"

        if field == "storm_surge_m":
            if u in {"ft", "feet"}:
                return value * 0.3048, "m"
            return value, "m"

        if field == "duration_days":
            if u.startswith("week"):
                return value * 7.0, "days"
            if u.startswith("month"):
                return value * 30.0, "days"
            return value, "days"

        if field == "economic_loss_usd":
            return value, "usd"

        return value, unit

    @staticmethod
    def _token_spans(text: str) -> List[Tuple[int, int, str]]:
        spans: List[Tuple[int, int, str]] = []
        for m in re.finditer(r"\b\w+\b", text.lower()):
            spans.append((m.start(), m.end(), m.group(0)))
        return spans

    @staticmethod
    def _token_index_for_char(spans: List[Tuple[int, int, str]], char_pos: int) -> int:
        for i, (s, e, _) in enumerate(spans):
            if s <= char_pos < e:
                return i
        if not spans:
            return -1
        if char_pos < spans[0][0]:
            return 0
        return len(spans) - 1

    def _window_binding_score(self, text: str, field: str, start: int, end: int) -> float:
        """
        Score candidate-field binding by checking anchor keywords in a local token window.
        Returns score in [0, 1].
        """
        anchors = self.field_anchors.get(field, [])
        if not anchors:
            return 0.5

        spans = self._token_spans(text)
        if not spans:
            return 0.5

        idx_s = self._token_index_for_char(spans, start)
        idx_e = self._token_index_for_char(spans, max(start, end - 1))
        if idx_s < 0 or idx_e < 0:
            return 0.5

        win = 8
        lo = max(0, idx_s - win)
        hi = min(len(spans) - 1, idx_e + win)
        window_tokens = [tok for _, _, tok in spans[lo : hi + 1]]
        window_text = " ".join(window_tokens)

        hits = 0
        min_dist = 999
        for a in anchors:
            a_low = a.lower()
            if " " in a_low:
                if a_low in window_text:
                    hits += 1
                    min_dist = min(min_dist, 1)
                continue
            for j in range(lo, hi + 1):
                tok = spans[j][2]
                if tok == a_low:
                    hits += 1
                    anchor_dist = min(abs(j - idx_s), abs(j - idx_e))
                    min_dist = min(min_dist, anchor_dist)

        if hits == 0:
            return 0.1
        # More anchor hits and shorter distance -> stronger binding.
        score = 0.45 + 0.12 * min(hits, 3) + max(0.0, 0.25 - 0.03 * min_dist)
        return clamp(score, 0.0, 1.0)

    def _has_negation_near_span(self, text: str, start: int, end: int) -> bool:
        spans = self._token_spans(text)
        if not spans:
            return False
        idx_s = self._token_index_for_char(spans, start)
        idx_e = self._token_index_for_char(spans, max(start, end - 1))
        lo = max(0, idx_s - 5)
        hi = min(len(spans) - 1, idx_e + 5)
        context = " ".join(tok for _, _, tok in spans[lo : hi + 1]) + " "
        return any(cue in context for cue in self.negation_cues)

    def _extract_general_fields(self, text: str, sentences: List[str]) -> List[Candidate]:
        cands: List[Candidate] = []
        for field, field_patterns in self.patterns.items():
            for pat in field_patterns:
                for m in pat.finditer(text):
                    num = parse_number(m.groupdict().get("num", ""))
                    if num is None:
                        continue
                    unit = m.groupdict().get("unit", "")
                    scale = m.groupdict().get("scale", "")
                    if field == "economic_loss_usd":
                        if scale:
                            num = parse_number(f"{num} {scale}") or num
                    norm_val, norm_unit = self._unit_convert(field, num, unit)
                    evidence = sentence_for_span(sentences, text, m.start(), m.end()) or m.group(0)
                    # Syntactic-window binding around candidate.
                    bind_score = self._window_binding_score(text, field, m.start(), m.end())
                    confidence = 0.50 + 0.40 * bind_score
                    if self._has_negation_near_span(text, m.start(), m.end()):
                        confidence -= 0.18
                    # Drop obvious mismatches if window binding is too weak.
                    if bind_score < 0.15:
                        continue
                    cands.append(
                        Candidate(
                            field=field,
                            value=norm_val,
                            unit=norm_unit,
                            confidence=clamp(confidence, 0.05, 0.95),
                            raw=m.group(0),
                            evidence=evidence,
                            start=m.start(),
                            end=m.end(),
                        )
                    )
        return cands

    def _extract_human_impact(self, text: str, sentences: List[str]) -> List[Candidate]:
        cands: List[Candidate] = []
        for field, field_patterns in self.human_patterns.items():
            for pat in field_patterns:
                for m in pat.finditer(text):
                    num = parse_number(m.groupdict().get("num", ""))
                    if num is None:
                        continue
                    evidence = sentence_for_span(sentences, text, m.start(), m.end()) or m.group(0)
                    bind_score = self._window_binding_score(text, field, m.start(), m.end())
                    conf = 0.58 + 0.35 * bind_score
                    if "report" in evidence.lower() or "official" in evidence.lower():
                        conf += 0.06
                    if self._has_negation_near_span(text, m.start(), m.end()):
                        conf -= 0.22
                    if bind_score < 0.18:
                        continue
                    cands.append(
                        Candidate(
                            field=field,
                            value=num,
                            unit="people",
                            confidence=clamp(conf, 0.05, 0.98),
                            raw=m.group(0),
                            evidence=evidence,
                            start=m.start(),
                            end=m.end(),
                        )
                    )
        return cands

    @staticmethod
    def _best_candidate(cands: List[Candidate]) -> Optional[Candidate]:
        if not cands:
            return None
        # Prefer higher confidence, then larger absolute value for impact fields.
        return sorted(cands, key=lambda x: (x.confidence, x.value), reverse=True)[0]

    def _collect_hazard_terms(self, text: str) -> List[str]:
        lowered = text.lower()
        found = [k for k in self.hazard_keywords if k in lowered]
        # preserve order and dedupe
        seen = set()
        out = []
        for x in found:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    def _collect_sector_hints(self, text: str) -> Dict[str, int]:
        lowered = text.lower()
        sector_scores: Dict[str, int] = {}
        for sector, kws in self.sector_keywords.items():
            cnt = sum(1 for kw in kws if kw in lowered)
            if cnt > 0:
                sector_scores[sector] = cnt
        return sector_scores

    def _build_stock_impact_summary(self, metrics: Dict[str, Dict], sector_hints: Dict[str, int]) -> Dict:
        dead = to_float(metrics.get("dead", {}).get("value")) or 0.0
        displaced = to_float(metrics.get("displaced", {}).get("value")) or 0.0
        affected = to_float(metrics.get("affected_population", {}).get("value")) or 0.0
        econ_loss = to_float(metrics.get("economic_loss_usd", {}).get("value")) or 0.0
        wind = to_float(metrics.get("wind_speed_kmh", {}).get("value")) or 0.0
        magnitude = to_float(metrics.get("magnitude", {}).get("value")) or 0.0
        burned = to_float(metrics.get("burned_area_ha", {}).get("value")) or 0.0

        # A simple bounded score in [0, 100] for downstream event-study stratification.
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

        return {
            "impact_score_0_100": round(score, 3),
            "impact_tier": tier,
            "likely_affected_sectors": sorted(
                sector_hints.keys(),
                key=lambda s: sector_hints[s],
                reverse=True,
            ),
        }

    def extract(self, text: str, timestamp: str = "", location_text: str = "") -> Dict:
        clean_text = normalize_text(text)
        sentences = split_sentences(clean_text)
        if not clean_text:
            return {
                "hazard_terms": [],
                "metrics": {},
                "sector_hints": {},
                "stock_impact_summary": {
                    "impact_score_0_100": 0.0,
                    "impact_tier": "low",
                    "likely_affected_sectors": [],
                },
                "meta": {"timestamp": normalize_text(timestamp), "location_text": normalize_text(location_text)},
            }

        all_cands = self._extract_general_fields(clean_text, sentences)
        all_cands.extend(self._extract_human_impact(clean_text, sentences))

        grouped: Dict[str, List[Candidate]] = {}
        for c in all_cands:
            grouped.setdefault(c.field, []).append(c)

        metrics: Dict[str, Dict] = {}
        for field, cands in grouped.items():
            best = self._best_candidate(cands)
            if best is None:
                continue
            metrics[field] = {
                "value": round(best.value, 6),
                "unit": best.unit,
                "confidence": round(best.confidence, 3),
                "evidence": best.evidence,
                "raw": best.raw,
            }

        sector_hints = self._collect_sector_hints(clean_text)
        hazard_terms = self._collect_hazard_terms(clean_text)
        stock_summary = self._build_stock_impact_summary(metrics, sector_hints)

        return {
            "hazard_terms": hazard_terms,
            "metrics": metrics,
            "sector_hints": sector_hints,
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
) -> None:
    df = read_table(input_path)
    if text_column not in df.columns:
        raise ValueError(f"text column '{text_column}' not found. columns={list(df.columns)}")

    extractor = UnifiedEventExtractor()
    features_json: List[str] = []

    ts_series = df[timestamp_column] if timestamp_column in df.columns else pd.Series([""] * len(df))
    loc_series = df[location_column] if location_column in df.columns else pd.Series([""] * len(df))

    for text, ts, loc in zip(df[text_column].fillna(""), ts_series.fillna(""), loc_series.fillna("")):
        feat = extractor.extract(str(text), str(ts), str(loc))
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
    )
    print(f"Saved extraction output to {output_path}")


if __name__ == "__main__":
    main()
