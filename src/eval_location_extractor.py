"""
Evaluate the rule-based location extractor against LLM ground-truth labels.

Reads:
  data/llm_labels/location_labels_test.csv   (GT from label_locations.py)
  data/splits/test.csv                        (article text)

Runs extract_location() on each article and compares to GT.

Metrics:
  - country_acc:   pred_country == gt_country  (on rows where gt_country is valid)
  - location_coverage: pred_location_text is non-null
  - lat_error_km:  haversine distance (only when pred confidence=="coords_text",
                   i.e. extractor found explicit coords in text; GT lat/lon from LLM knowledge)
  - confidence breakdown: coords_text / location / none

Note on lat/lon: the rule-based extractor only produces lat/lon when the article contains
an explicit coordinate expression (e.g. "38.1°N 142.4°E"). The LLM GT uses its own
knowledge for lat/lon (approximate city centroids). Comparison is only meaningful when
both sides have coordinates, which mostly occurs for earthquake articles.

Usage:
    python src/eval_location_extractor.py
    python src/eval_location_extractor.py --verbose
    python src/eval_location_extractor.py --per-class
"""

import argparse
import math
from pathlib import Path

import pandas as pd

ROOT       = Path(__file__).parent.parent
GT_FILE    = ROOT / "data" / "llm_labels" / "location_labels_test.csv"
SPLITS_DIR = ROOT / "data" / "splits"
RESULTS_DIR = ROOT / "data" / "results"


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def run_eval(verbose=False, per_class=False):
    # Import here so spaCy only loads when needed
    from location_extractor import extract_location

    gt = pd.read_csv(GT_FILE)
    articles = pd.read_csv(SPLITS_DIR / "test.csv")

    # Drop rows where LLM returned error
    gt_ok = gt[~gt["location_text"].isin(["error", "Error"]) & gt["location_text"].notna()].copy()
    print(f"GT rows: {len(gt)} total, {len(gt_ok)} valid (non-error)")

    # Merge with article text
    merged = gt_ok.merge(articles[["idx", "text", "label"]], on="idx", how="inner", suffixes=("_gt", "_article"))
    print(f"Merged: {len(merged)} rows")

    records = []
    for i, row in merged.iterrows():
        title = ""
        text = str(row["text"])
        # Try to split title from "[SEP]" format used in classifier splits
        if " [SEP] " in text:
            parts = text.split(" [SEP] ", 1)
            title, text = parts[0], parts[1]

        pred = extract_location(text, title)

        # Normalise country codes to uppercase
        gt_country = str(row["country_iso2"]).strip().upper() if pd.notna(row["country_iso2"]) else None
        if gt_country in ("ERROR", "N/A", "NA", "NONE", ""):
            gt_country = None

        pred_country = pred.country_iso2.strip().upper() if pred.country_iso2 else None

        country_match = (pred_country == gt_country) if (gt_country and pred_country) else None
        gt_lat  = row["lat"]  if pd.notna(row.get("lat"))  else None
        gt_lon  = row["lon"]  if pd.notna(row.get("lon"))  else None

        dist_km = None
        # Only compare coords when extractor found explicit text coordinates
        if pred.confidence == "coords_text" and gt_lat and gt_lon and pred.lat and pred.lon:
            dist_km = haversine_km(gt_lat, gt_lon, pred.lat, pred.lon)

        records.append({
            "idx":              row["idx"],
            "label":            row["label_article"],
            "gt_location":      row["location_text"],
            "gt_country":       gt_country,
            "gt_lat":           gt_lat,
            "gt_lon":           gt_lon,
            "pred_location":    pred.location_text,
            "pred_country":     pred_country,
            "pred_lat":         pred.lat,
            "pred_lon":         pred.lon,
            "confidence":       pred.confidence,
            "country_match":    country_match,
            "dist_km":          dist_km,
        })

        if verbose and i % 50 == 0:
            print(f"  [{i}] GT={row['location_text']} | pred={pred.location_text} | conf={pred.confidence} | match={country_match}")

    df = pd.DataFrame(records)

    # Save per-sample results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "location_extractor_eval.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} rows → {out_path}")

    _print_metrics(df, "OVERALL")

    if per_class:
        for lbl in sorted(df["label"].unique()):
            _print_metrics(df[df["label"] == lbl], lbl.upper())


def _print_metrics(df: pd.DataFrame, title: str):
    n = len(df)
    if n == 0:
        return

    # Country accuracy (only on rows where both gt and pred country are available)
    country_rows = df["country_match"].notna()
    country_acc  = df.loc[country_rows, "country_match"].mean() if country_rows.any() else float("nan")

    # Coverage: fraction of rows where we got a non-null location_text
    covered = df["pred_location"].notna().sum()
    coverage = covered / n

    # Coordinate coverage and median error
    coord_rows = df["dist_km"].notna()
    median_dist = df.loc[coord_rows, "dist_km"].median() if coord_rows.any() else float("nan")
    lt_100km = (df.loc[coord_rows, "dist_km"] < 100).mean() if coord_rows.any() else float("nan")
    lt_500km = (df.loc[coord_rows, "dist_km"] < 500).mean() if coord_rows.any() else float("nan")

    # Confidence breakdown
    conf_counts = df["confidence"].value_counts()

    print(f"\n{'='*55}")
    print(f"  {title}  (n={n})")
    print(f"{'='*55}")
    print(f"  Country accuracy:   {country_acc:.3f}  ({country_rows.sum()} comparable rows)")
    print(f"  Location coverage:  {coverage:.3f}  ({covered}/{n} non-null)")
    print(f"  Coord pairs:        {coord_rows.sum()} rows have both gt+pred coords")
    print(f"  Median dist (km):   {median_dist:.1f}")
    print(f"  Within 100km:       {lt_100km:.3f}")
    print(f"  Within 500km:       {lt_500km:.3f}")
    print(f"\n  Confidence breakdown:")
    for conf_val in ["coords_text", "location", "none"]:
        cnt = conf_counts.get(conf_val, 0)
        print(f"    {conf_val:14s}: {cnt:4d}  ({cnt/n:.1%})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose",   action="store_true")
    parser.add_argument("--per-class", action="store_true")
    args = parser.parse_args()
    run_eval(verbose=args.verbose, per_class=args.per_class)
