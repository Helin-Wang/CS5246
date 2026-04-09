"""
GDACS Event Matcher

After clustering, tries to match each news-derived event cluster to a GDACS
structured event record.  When a match is found the cluster can use GDACS
severity (alertlevel) and numeric parameters directly instead of relying on
noisy news-text NER.

Matching criteria (all three must hold):
  1. eventtype matches (two-letter code: EQ / TC / WF / DR / FL)
  2. country_iso2 matches (entity_linker resolution applied to GDACS country name)
  3. |event_date − gdacs_date| ≤ DATE_TOLERANCE_DAYS  (default 7)

If multiple GDACS rows qualify, the one with the smallest date gap is chosen.

Usage:
    from src.gdacs_matcher import GDACSmatcher
    matcher = GDACSmatcher()          # loads data/gdacs_all_fields_v2.csv
    result  = matcher.match(cluster)  # cluster dict from EventClusterer
    # result is None  →  no GDACS match
    # result is dict  →  {alertlevel, magnitude, wind_speed_kmh, …, gdacs_matched: True}
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_GDACS_CSV = BASE_DIR / "data" / "gdacs_all_fields_v2.csv"

DATE_TOLERANCE_DAYS = 7

# GDACS alertlevel string → canonical label used by SeverityPredictor
ALERTLEVEL_MAP = {
    "green":  "green",
    "orange": "orange_or_red",
    "red":    "orange_or_red",
}

# GDACS country full name → ISO-2  (resolved lazily via entity_linker)
_COUNTRY_CACHE: Dict[str, Optional[str]] = {}


def _gdacs_country_to_iso2(country_name: str) -> Optional[str]:
    if country_name in _COUNTRY_CACHE:
        return _COUNTRY_CACHE[country_name]
    try:
        import pycountry
        hit = pycountry.countries.lookup(country_name)
        iso2 = hit.alpha_2
    except Exception:
        from entity_linker import _resolve_iso2
        iso2 = _resolve_iso2(None, country_name)
    _COUNTRY_CACHE[country_name] = iso2
    return iso2


def _parse_date(val) -> Optional[date]:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    try:
        return pd.Timestamp(str(val)).date()
    except Exception:
        return None


class GDACSmatcher:
    """
    Load a GDACS CSV once and expose a match() method for single-cluster lookup.

    Parameters
    ----------
    gdacs_csv : path to GDACS CSV (default: data/gdacs_all_fields_v2.csv)
    date_tolerance : max days between cluster event_date and GDACS fromdate
    """

    def __init__(
        self,
        gdacs_csv: str | Path = DEFAULT_GDACS_CSV,
        date_tolerance: int = DATE_TOLERANCE_DAYS,
    ):
        self.date_tolerance = date_tolerance
        path = Path(gdacs_csv)
        if not path.exists():
            self._df = pd.DataFrame()
            return

        df = pd.read_csv(path, low_memory=False)
        df["_event_date"] = pd.to_datetime(df["fromdate"], errors="coerce").dt.date
        # Resolve country → ISO-2 once at load time
        df["_iso2"] = df["country"].apply(
            lambda c: _gdacs_country_to_iso2(str(c)) if pd.notna(c) else None
        )
        df["_etype"] = df["eventtype"].str.upper().str.strip()
        self._df = df

    # ------------------------------------------------------------------
    def match(self, cluster: dict) -> Optional[dict]:
        """
        Find the best GDACS record for a cluster.

        Parameters
        ----------
        cluster : dict with at minimum:
            event_type  (two-letter code)
            country_iso2  (may be None)
            event_date  (ISO date string or None)

        Returns
        -------
        None if no match; otherwise a dict with GDACS fields merged in and
        gdacs_matched=True, gdacs_alertlevel, plus numeric fields.
        """
        if self._df.empty:
            return None

        etype   = str(cluster.get("event_type", "")).upper().strip()
        iso2    = cluster.get("country_iso2")
        c_date  = _parse_date(cluster.get("event_date"))

        if not etype or not iso2 or c_date is None:
            return None

        df = self._df
        mask = (df["_etype"] == etype) & (df["_iso2"] == iso2)
        candidates = df[mask].copy()
        if candidates.empty:
            return None

        # Filter by date tolerance
        candidates["_days"] = candidates["_event_date"].apply(
            lambda d: abs((d - c_date).days) if d is not None else 9999
        )
        candidates = candidates[candidates["_days"] <= self.date_tolerance]
        if candidates.empty:
            return None

        best = candidates.loc[candidates["_days"].idxmin()]

        alertlevel_raw = str(best.get("alertlevel", "")).lower().strip()
        alertlevel     = ALERTLEVEL_MAP.get(alertlevel_raw)

        # Build result dict with GDACS numeric fields
        def _g(col):
            v = best.get(col)
            try:
                return float(v) if pd.notna(v) else None
            except (TypeError, ValueError):
                return None

        return {
            "gdacs_matched":            True,
            "gdacs_event_id":           best.get("eventid"),
            "gdacs_alertlevel":         alertlevel_raw,
            "predicted_alert":          alertlevel,
            "prob_orange_or_red":       1.0 if alertlevel == "orange_or_red" else 0.0,
            "low_confidence":           alertlevel is None,
            # Numeric fields (override NER if present)
            "magnitude":                _g("magnitude"),
            "depth":                    _g("depth"),
            "rapidpopdescription":      best.get("rapidpopdescription"),
            "maximum_wind_speed_kmh":   _g("maximum_wind_speed_kmh"),
            "maximum_storm_surge_m":    _g("maximum_storm_surge_m"),
            "exposed_population":       _g("exposed_population"),
            "duration_days":            _g("duration_days"),
            "burned_area_ha":           _g("burned_area_ha"),
            "people_affected":          _g("people_affected"),
            "affected_area_km2":        _g("affected_area_km2"),
            "affected_country_count":   _g("affected_country_count"),
        }
