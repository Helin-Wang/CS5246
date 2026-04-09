"""
GDACS Matching Evaluation

For each pipeline event cluster, independently looks up GDACS using gold labels
(event_type + country_iso2 + article timestamps), then compares with what the
pipeline actually matched.

Metrics:
  - GDACS coverage:    % of clusters where a matching GDACS record EXISTS
  - Pipeline match rate: % of clusters where pipeline DID match GDACS
  - Recall:            pipeline matches / clusters with coverage
  - False-match rate:  pipeline matched but gold lookup disagrees

Usage:
    python scripts/eval_gdacs_matching.py
    python scripts/eval_gdacs_matching.py --events data/results/events.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DATE_TOLERANCE = 7  # days

# Map pipeline label → GDACS eventtype
LABEL_TO_ETYPE = {
    "earthquake": "EQ",
    "cyclone":    "TC",
    "wildfire":   "WF",
    "drought":    "DR",
    "flood":      "FL",
    "EQ": "EQ", "TC": "TC", "WF": "WF", "DR": "DR", "FL": "FL",
}


def load_gdacs(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df["_from"] = pd.to_datetime(df["fromdate"], errors="coerce")
    df["_etype"] = df["eventtype"].str.upper().str.strip()
    # Resolve country name → ISO2
    try:
        import pycountry
        def _iso2(name):
            try:
                return pycountry.countries.lookup(str(name)).alpha_2
            except Exception:
                return None
        df["_iso2"] = df["country"].apply(_iso2)
    except ImportError:
        df["_iso2"] = None
    return df


def gold_lookup(gdacs: pd.DataFrame, etype: str, iso2: str, ref_date: pd.Timestamp) -> bool:
    """Return True if GDACS has a record matching (etype, iso2, date ±7d)."""
    if pd.isna(iso2) or pd.isna(etype) or pd.isna(ref_date):
        return False
    mask = gdacs["_etype"] == etype
    if iso2:
        mask &= gdacs["_iso2"] == iso2
    sub = gdacs[mask]
    if sub.empty:
        return False
    diffs = (sub["_from"] - ref_date).abs().dt.days
    return bool((diffs <= DATE_TOLERANCE).any())


def best_gdacs_record(gdacs: pd.DataFrame, etype: str, iso2: str, ref_date: pd.Timestamp):
    """Return the closest matching GDACS row, or None."""
    if pd.isna(iso2) or pd.isna(etype) or pd.isna(ref_date):
        return None
    mask = (gdacs["_etype"] == etype)
    if iso2:
        mask &= gdacs["_iso2"] == iso2
    sub = gdacs[mask]
    if sub.empty:
        return None
    sub = sub.copy()
    sub["_diff"] = (sub["_from"] - ref_date).abs().dt.days
    sub = sub[sub["_diff"] <= DATE_TOLERANCE]
    if sub.empty:
        return None
    return sub.loc[sub["_diff"].idxmin()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--events",   default=str(ROOT / "data" / "results" / "events.csv"))
    parser.add_argument("--test-csv", default=str(ROOT / "data" / "splits" / "test.csv"))
    parser.add_argument("--loc-labels", default=str(ROOT / "data" / "llm_labels" / "location_labels_test.csv"))
    parser.add_argument("--gdacs-csv",  default=str(ROOT / "data" / "gdacs_all_fields_v2.csv"))
    parser.add_argument("--out",        default=str(ROOT / "data" / "results" / "gdacs_match_eval.csv"))
    args = parser.parse_args()

    # ── Load data ──────────────────────────────────────────────────────────────
    events   = pd.read_csv(args.events)
    test_df  = pd.read_csv(args.test_csv)
    loc_gold = pd.read_csv(args.loc_labels)
    gdacs    = load_gdacs(Path(args.gdacs_csv))

    # Build lookup: idx → gold country_iso2
    gold_iso2_map = dict(zip(loc_gold["idx"], loc_gold["country_iso2"]))
    # Build lookup: idx → timestamp
    ts_map = dict(zip(test_df["idx"], pd.to_datetime(test_df["timestamp"])))

    print(f"Events loaded: {len(events)}")
    print(f"GDACS records: {len(gdacs)}")
    print()

    # ── Per-cluster evaluation ─────────────────────────────────────────────────
    rows = []
    for _, ev in events.iterrows():
        etype      = LABEL_TO_ETYPE.get(str(ev.get("event_type", "")), "")
        pipeline_matched = str(ev.get("severity_source", "")).lower() == "gdacs"

        # Parse article_indices
        raw_idx = ev.get("article_indices", "[]")
        try:
            import ast
            article_idxs = ast.literal_eval(str(raw_idx)) if raw_idx else []
        except Exception:
            article_idxs = []

        # Gold country: majority vote from gold labels
        gold_countries = [gold_iso2_map.get(i) for i in article_idxs
                          if gold_iso2_map.get(i) is not None]
        gold_iso2 = max(set(gold_countries), key=gold_countries.count) if gold_countries else None

        # Reference date: earliest article timestamp in cluster
        ts_list = [ts_map.get(i) for i in article_idxs if ts_map.get(i) is not None]
        ref_date = min(ts_list) if ts_list else None

        # Gold GDACS lookup
        has_gdacs = gold_lookup(gdacs, etype, gold_iso2, ref_date)
        best_row  = best_gdacs_record(gdacs, etype, gold_iso2, ref_date) if has_gdacs else None
        gold_alertlevel = str(best_row["alertlevel"]).lower() if best_row is not None else None

        # Pipeline country (for comparison)
        pipe_iso2 = ev.get("country_iso2")

        rows.append({
            "event_id":          ev.get("event_id"),
            "event_type":        etype,
            "article_count":     len(article_idxs),
            "gold_country_iso2": gold_iso2,
            "pipe_country_iso2": pipe_iso2,
            "country_match":     gold_iso2 == pipe_iso2 if gold_iso2 and pipe_iso2 else None,
            "ref_date":          ref_date.date() if ref_date else None,
            "pipe_event_date":   ev.get("event_date"),
            "gdacs_coverage":    has_gdacs,       # should pipeline have matched?
            "pipeline_matched":  pipeline_matched, # did it?
            "gold_alertlevel":   gold_alertlevel,
            "pipe_alertlevel":   ev.get("gdacs_alertlevel"),
        })

    result = pd.DataFrame(rows)

    # ── Metrics ────────────────────────────────────────────────────────────────
    n          = len(result)
    covered    = result["gdacs_coverage"].sum()
    matched    = result["pipeline_matched"].sum()
    # True hits: pipeline matched AND gold says a record exists
    true_hits  = (result["gdacs_coverage"] & result["pipeline_matched"]).sum()
    # False misses: gold says record exists but pipeline didn't match
    false_miss = (result["gdacs_coverage"] & ~result["pipeline_matched"]).sum()
    # False hits: pipeline matched but gold says no record (possible incorrect match)
    false_hit  = (~result["gdacs_coverage"] & result["pipeline_matched"]).sum()

    recall    = true_hits / covered if covered > 0 else 0.0
    precision = true_hits / matched if matched > 0 else 0.0

    print("=" * 55)
    print("GDACS MATCHING EVALUATION")
    print("=" * 55)
    print(f"Total clusters evaluated:       {n}")
    print(f"Clusters with GDACS coverage:   {covered} ({100*covered/n:.1f}%)")
    print(f"Clusters pipeline matched:       {matched} ({100*matched/n:.1f}%)")
    print(f"  True hits  (covered & matched):{true_hits}")
    print(f"  False miss (covered, missed):  {false_miss}")
    print(f"  False hit  (uncovered, matched):{false_hit}")
    print(f"Recall    (hits/covered):        {recall:.3f}")
    print(f"Precision (hits/matched):        {precision:.3f}" if matched else "Precision: N/A (no matches)")
    print()

    # Per event-type breakdown
    print("Per event-type breakdown:")
    for etype, grp in result.groupby("event_type"):
        n_g   = len(grp)
        cov_g = grp["gdacs_coverage"].sum()
        mat_g = grp["pipeline_matched"].sum()
        th_g  = (grp["gdacs_coverage"] & grp["pipeline_matched"]).sum()
        rec_g = th_g / cov_g if cov_g > 0 else float("nan")
        print(f"  {etype}: n={n_g:3d}  coverage={cov_g:3d} ({100*cov_g/n_g:5.1f}%)  "
              f"matched={mat_g:3d}  recall={rec_g:.2f}")

    print()
    print("Country resolution accuracy:")
    sub_country = result[result["gold_country_iso2"].notna() & result["pipe_country_iso2"].notna()]
    if len(sub_country) > 0:
        acc = sub_country["country_match"].mean()
        print(f"  {sub_country['country_match'].sum()}/{len(sub_country)} clusters "
              f"correct country ({100*acc:.1f}%)")
    else:
        print("  (no overlapping country annotations)")

    # Save
    result.to_csv(args.out, index=False)
    print(f"\nDetailed results saved → {args.out}")


if __name__ == "__main__":
    main()
