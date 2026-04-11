"""
Evaluate the rule-based NER extractor against LLM ground-truth labels.

Reads:
  data/llm_labels/ner_labels_test.csv   (GT from label_ner_fields.py)
  data/splits/test.csv                   (article text + timestamps)

Runs UnifiedEventExtractor.extract() on each article and compares extracted
numeric fields to LLM-labeled ground truth.

Metrics per field:
  - coverage:    fraction of rows where extractor produced a non-null value
                 (only on rows where GT has a value)
  - exact_frac:  fraction within 5% relative error  (tight accuracy)
  - within_20pct: fraction within 20% relative error
  - within_50pct: fraction within 50% relative error
  - median_relerr: median relative error on comparable rows

Only rows where GT has a valid (non-null) value for the field are included
in accuracy metrics for that field.

Field → extractor key mapping:
  GT field           →  metrics key in extractor output
  magnitude          →  magnitude
  depth_km           →  depth
  wind_speed_kmh     →  maximum_wind_speed_kmh
  storm_surge_m      →  maximum_storm_surge_m
  exposed_population →  exposed_population
  burned_area_ha     →  burned_area_ha
  people_affected    →  people_affected
  duration_days      →  duration_days
  affected_area_km2  →  affected_area_km2
  affected_country_count → affected_country_count
  dead               →  dead
  displaced          →  displaced

Usage:
    python src/eval_ner_extractor.py
    python src/eval_ner_extractor.py --verbose
    python src/eval_ner_extractor.py --per-class
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT        = Path(__file__).parent.parent
GT_FILE     = ROOT / "data" / "llm_labels" / "ner_labels_test.csv"
SPLITS_DIR  = ROOT / "data" / "splits"
RESULTS_DIR = ROOT / "data" / "results"

# Maps GT column name → extractor metrics key
FIELD_MAP = {
    "magnitude":             "magnitude",
    "depth_km":              "depth",
    "wind_speed_kmh":        "maximum_wind_speed_kmh",
    "storm_surge_m":         "maximum_storm_surge_m",
    "exposed_population":    "exposed_population",
    "burned_area_ha":        "burned_area_ha",
    "people_affected":       "people_affected",
    "duration_days":         "duration_days",
    "affected_area_km2":     "affected_area_km2",
    "affected_country_count":"affected_country_count",
    "dead":                  "dead",
    "displaced":             "displaced",
}

# NER extraction uses full article text (no truncation).
# TEXT_LIMIT applies only to Module A (time + location extraction).


def run_eval(verbose: bool = False, per_class: bool = False):
    sys.path.insert(0, str(ROOT / "src"))
    from unified_event_extractor import UnifiedEventExtractor
    extractor = UnifiedEventExtractor()

    gt = pd.read_csv(GT_FILE)
    articles = pd.read_csv(SPLITS_DIR / "test.csv")

    # Drop api_error rows
    gt_ok = gt[gt["source_note"] != "api_error"].copy()
    print(f"GT rows: {len(gt)} total, {len(gt_ok)} valid (non-error)")

    # Merge: GT has idx + label + field columns; articles has idx + text + timestamp
    merged = gt_ok.merge(
        articles[["idx", "text", "timestamp", "label"]],
        on="idx", how="inner",
        suffixes=("_gt", "_article")
    )
    print(f"Merged: {len(merged)} rows")

    # Use GT label for event_type (authoritative over article label)
    label_col = "label_gt" if "label_gt" in merged.columns else "label"

    records = []
    for _, row in merged.iterrows():
        raw_text = str(row["text"])
        title = ""
        if " [SEP] " in raw_text:
            parts = raw_text.split(" [SEP] ", 1)
            title, body = parts[0], parts[1]
        else:
            body = raw_text

        # Use full body text for NER (no truncation)
        text_for_ner = f"{title} [SEP] {body}" if title else body

        event_type = str(row[label_col])
        result = extractor.extract(text_for_ner, event_type)
        metrics = result.get("metrics", {})

        rec: dict = {
            "idx":   row["idx"],
            "label": event_type,
        }

        for gt_field, extractor_key in FIELD_MAP.items():
            gt_val = row.get(gt_field, None)
            gt_val = float(gt_val) if pd.notna(gt_val) else None

            ext_entry = metrics.get(extractor_key)
            pred_val = float(ext_entry["value"]) if ext_entry and ext_entry.get("value") is not None else None

            # Relative error (only when both available and GT > 0)
            rel_err = None
            if gt_val is not None and pred_val is not None and gt_val > 0:
                rel_err = abs(pred_val - gt_val) / gt_val

            rec[f"gt_{gt_field}"]   = gt_val
            rec[f"pred_{gt_field}"] = pred_val
            rec[f"relerr_{gt_field}"] = rel_err

        records.append(rec)

        if verbose:
            print(f"  [{row['idx']}] {event_type}", end="  ")
            for gt_field in FIELD_MAP:
                gt_v = rec[f"gt_{gt_field}"]
                pr_v = rec[f"pred_{gt_field}"]
                if gt_v is not None or pr_v is not None:
                    print(f"{gt_field}=GT:{gt_v}/PRED:{pr_v}", end=" ")
            print()

    df = pd.DataFrame(records)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "ner_extractor_eval.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} rows → {out_path}")

    _print_metrics(df, "OVERALL")

    if per_class:
        label_col_df = "label"
        for lbl in sorted(df[label_col_df].unique()):
            _print_metrics(df[df[label_col_df] == lbl], lbl.upper())


def _print_metrics(df: pd.DataFrame, title: str):
    n = len(df)
    if n == 0:
        return

    print(f"\n{'='*65}")
    print(f"  {title}  (n={n})")
    print(f"{'='*65}")
    print(f"  {'Field':<28} {'Cover%':>7} {'Cov/GT':>8} "
          f"{'W5%':>6} {'W20%':>6} {'W50%':>7} {'MedRE':>7} {'Cmp':>5}")
    print(f"  {'-'*28} {'-'*7} {'-'*8} {'-'*6} {'-'*6} {'-'*7} {'-'*7} {'-'*5}")

    for gt_field in FIELD_MAP:
        gt_col   = f"gt_{gt_field}"
        pred_col = f"pred_{gt_field}"
        re_col   = f"relerr_{gt_field}"

        if gt_col not in df.columns:
            continue

        gt_has    = df[gt_col].notna()
        pred_has  = df[pred_col].notna()
        n_gt      = gt_has.sum()

        if n_gt == 0:
            continue

        # Coverage: among rows where GT has value, how many did extractor produce?
        coverage = pred_has[gt_has].mean()
        cov_str  = f"{pred_has[gt_has].sum()}/{n_gt}"

        # Accuracy on comparable rows (both have value)
        comparable = df[gt_has & pred_has]
        nc = len(comparable)

        if nc > 0:
            w5   = (comparable[re_col] <= 0.05).mean()
            w20  = (comparable[re_col] <= 0.20).mean()
            w50  = (comparable[re_col] <= 0.50).mean()
            mre  = comparable[re_col].median()
        else:
            w5 = w20 = w50 = mre = float("nan")

        print(f"  {gt_field:<28} {coverage:>6.1%} {cov_str:>8} "
              f"{w5:>6.1%} {w20:>6.1%} {w50:>7.1%} {mre:>7.2f} {nc:>5}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose",   action="store_true")
    parser.add_argument("--per-class", action="store_true")
    args = parser.parse_args()
    run_eval(verbose=args.verbose, per_class=args.per_class)
