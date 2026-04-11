"""
Clustering Evaluation via GDACS (GDACS-centric direction)

Direction: GDACS event → news clusters  (reverse of eval_gdacs_matching.py)

Metrics:
  1. GDACS Recall: % of GDACS events in the test window that have ≥1 matching cluster
  2. Over-splitting rate: % of "found" GDACS events that are matched by ≥2 clusters
     (one real-world event split into multiple clusters = deduplication failure)

Matching criteria (same as pipeline Module D linking):
  event_type match  +  country_iso2 match  +  |date_diff| ≤ DATE_TOLERANCE days

Country caveat: clusters without a resolved country_iso2 cannot be matched.
The recall figure therefore represents a lower bound; the "matchable" subset
(clusters WITH country_iso2) is reported separately.

Usage:
    python scripts/eval_clustering_gdacs.py
    python scripts/eval_clustering_gdacs.py --gdacs-csv data/gdacs_all_fields.csv
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DATE_TOLERANCE = 7   # days
TEST_START     = "2025-08-01"
TEST_END       = "2025-12-31"

ETYPE_MAP = {          # GDACS eventtype → pipeline label
    "EQ": "EQ", "TC": "TC", "WF": "WF", "DR": "DR", "FL": "FL",
}


def iso2_from_name(name: str, _cache: dict = {}):
    """Convert country name to ISO-3166-1 alpha-2 via pycountry."""
    if name in _cache:
        return _cache[name]
    try:
        import pycountry
        result = pycountry.countries.lookup(str(name)).alpha_2
    except Exception:
        result = None
    _cache[name] = result
    return result


def load_gdacs(path: Path, start: str, end: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df["_date"] = pd.to_datetime(df["fromdate"], errors="coerce")
    df["_etype"] = df["eventtype"].str.upper().str.strip()
    df["_iso2"]  = df["country"].apply(iso2_from_name)
    mask = (df["_date"] >= start) & (df["_date"] <= end)
    return df[mask].copy().reset_index(drop=True)


def find_matching_clusters(gdacs_row: pd.Series,
                           clusters: pd.DataFrame) -> pd.DataFrame:
    """Return all clusters that match a GDACS event."""
    etype    = gdacs_row["_etype"]
    iso2     = gdacs_row["_iso2"]
    gdate    = gdacs_row["_date"]

    if pd.isna(iso2) or pd.isna(gdate):
        return pd.DataFrame()

    mask = (
        (clusters["event_type"] == etype) &
        (clusters["country_iso2"] == iso2) &
        (clusters["_event_date"].notna()) &
        ((clusters["_event_date"] - gdate).abs().dt.days <= DATE_TOLERANCE)
    )
    return clusters[mask]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--events",    default=str(ROOT / "data" / "results" / "events.csv"))
    parser.add_argument("--gdacs-csv", default=str(ROOT / "data" / "gdacs_all_fields.csv"))
    parser.add_argument("--out",       default=str(ROOT / "data" / "results" / "clustering_eval_gdacs.csv"))
    parser.add_argument("--start",     default=TEST_START)
    parser.add_argument("--end",       default=TEST_END)
    args = parser.parse_args()

    # ── Load ──────────────────────────────────────────────────────────────────
    clusters = pd.read_csv(args.events)
    clusters["_event_date"] = pd.to_datetime(clusters["event_date"], errors="coerce")

    gdacs = load_gdacs(Path(args.gdacs_csv), args.start, args.end)

    n_gdacs    = len(gdacs)
    n_clusters = len(clusters)
    n_with_country = clusters["country_iso2"].notna().sum()

    print(f"GDACS events in test window ({args.start} ~ {args.end}): {n_gdacs}")
    print(f"  Type dist: {gdacs['_etype'].value_counts().to_dict()}")
    print(f"  ISO2 resolved: {gdacs['_iso2'].notna().sum()}/{n_gdacs}")
    print()
    print(f"News clusters: {n_clusters}")
    print(f"  With country_iso2: {n_with_country} ({100*n_with_country/n_clusters:.1f}%)")
    print()

    # ── Per-GDACS-event matching ───────────────────────────────────────────────
    rows = []
    for _, gev in gdacs.iterrows():
        matched = find_matching_clusters(gev, clusters)
        n_matched = len(matched)
        rows.append({
            "gdacs_eventid":   gev.get("eventid"),
            "event_type":      gev["_etype"],
            "alertlevel":      gev.get("alertlevel"),
            "country":         gev.get("country"),
            "country_iso2":    gev["_iso2"],
            "gdacs_date":      gev["_date"].date() if pd.notna(gev["_date"]) else None,
            "n_matching_clusters": n_matched,
            "found":           n_matched >= 1,
            "over_split":      n_matched >= 2,
            "matching_cluster_ids": ";".join(matched["event_id"].astype(str).tolist()) if n_matched > 0 else "",
        })

    result = pd.DataFrame(rows)

    # ── Summary metrics ────────────────────────────────────────────────────────
    # Only GDACS events with a resolved ISO2 can potentially be matched
    matchable   = result[result["country_iso2"].notna()]
    n_matchable = len(matchable)
    found       = matchable["found"].sum()
    over_split  = matchable["over_split"].sum()

    recall      = found / n_matchable if n_matchable > 0 else 0.0
    os_rate     = over_split / found  if found > 0       else 0.0

    print("=" * 60)
    print("CLUSTERING EVALUATION  (GDACS-centric)")
    print("=" * 60)
    print(f"GDACS events in window:          {n_gdacs}")
    print(f"  with resolvable country ISO2:  {n_matchable} ({100*n_matchable/n_gdacs:.1f}%)")
    print()
    print(f"[Metric 1] GDACS Recall")
    print(f"  GDACS events found in clusters: {found}/{n_matchable} "
          f"({100*recall:.1f}%)")
    print()
    print(f"[Metric 2] Over-splitting")
    print(f"  Found events matched by ≥2 clusters: {over_split}/{found} "
          f"({'N/A' if found==0 else f'{100*os_rate:.1f}%'})")
    if over_split > 0:
        print("  Over-split cases:")
        for _, row in matchable[matchable["over_split"]].iterrows():
            print(f"    GDACS {row['gdacs_eventid']} ({row['event_type']}, "
                  f"{row['country']}, {row['gdacs_date']}) → "
                  f"{row['n_matching_clusters']} clusters: {row['matching_cluster_ids']}")
    print()

    # Per event-type breakdown
    print("Per event-type breakdown (matchable subset):")
    for etype, grp in matchable.groupby("event_type"):
        n_g  = len(grp)
        f_g  = grp["found"].sum()
        os_g = grp["over_split"].sum()
        rec_g = f_g / n_g if n_g > 0 else float("nan")
        print(f"  {etype}: n={n_g:3d}  found={f_g:3d} ({100*rec_g:.1f}%)  "
              f"over-split={os_g}")

    # ── Save ──────────────────────────────────────────────────────────────────
    result.to_csv(args.out, index=False)
    print(f"\nDetailed results saved → {args.out}")


if __name__ == "__main__":
    main()
