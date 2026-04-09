"""
Evaluate the rule-based time extractor against LLM ground-truth labels.

Reads:
  data/llm_labels/time_labels_test.csv   (GT from label_times.py)
  data/splits/test.csv                    (article text + timestamps)

Runs extract_event_time() on each article and compares to GT.

Metrics:
  - exact_match:    pred_date == gt_date (YYYY-MM-DD)
  - within_1d:      |pred - gt| <= 1 day
  - within_7d:      |pred - gt| <= 7 days
  - coverage:       pred_date is non-null (on rows where GT has a date)
  - median_error:   median |pred - gt| in days (on comparable rows)

Only rows where GT has a valid event_date_iso are used for date accuracy.
Rows where both pred and GT are null count as correct (both unknown).

Usage:
    python src/eval_time_extractor.py
    python src/eval_time_extractor.py --verbose
    python src/eval_time_extractor.py --per-class
"""

from __future__ import annotations

import argparse
from datetime import date, datetime
from pathlib import Path

import pandas as pd

ROOT        = Path(__file__).parent.parent
GT_FILE     = ROOT / "data" / "llm_labels" / "time_labels_test.csv"
SPLITS_DIR  = ROOT / "data" / "splits"
RESULTS_DIR = ROOT / "data" / "results"


def _parse_ts(ts_str: str) -> datetime:
    s = str(ts_str).strip()
    for fmt in ("%Y%m%d%H%M%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return datetime.now()


def run_eval(verbose: bool = False, per_class: bool = False):
    from time_extractor import extract_event_time

    gt = pd.read_csv(GT_FILE)
    articles = pd.read_csv(SPLITS_DIR / "test.csv")

    # Drop api_error rows
    gt_ok = gt[gt["source_note"] != "api_error"].copy()
    print(f"GT rows: {len(gt)} total, {len(gt_ok)} valid (non-error)")

    merged = gt_ok.merge(articles[["idx", "text", "timestamp"]], on="idx", how="inner")
    print(f"Merged: {len(merged)} rows")

    records = []
    for _, row in merged.iterrows():
        text = str(row["text"])
        title = ""
        if " [SEP] " in text:
            parts = text.split(" [SEP] ", 1)
            title, text = parts[0], parts[1]

        ts = _parse_ts(row["timestamp"])
        pred = extract_event_time(text, title, ts, str(row["label"]))

        gt_iso = str(row["event_date_iso"]).strip() if pd.notna(row["event_date_iso"]) else None
        # Normalise: accept YYYY-MM (month-only) by padding to YYYY-MM-01
        if gt_iso and len(gt_iso) == 7:
            gt_iso = gt_iso + "-01"

        pred_date = pred.event_date  # YYYY-MM-DD or None

        # Compute diff in days (only when both have dates)
        diff_days = None
        if gt_iso and pred_date:
            try:
                d1 = date.fromisoformat(gt_iso)
                d2 = date.fromisoformat(pred_date)
                diff_days = abs((d1 - d2).days)
            except ValueError:
                pass

        exact   = (gt_iso == pred_date) if (gt_iso or pred_date) else True
        w1      = diff_days is not None and diff_days <= 1
        w7      = diff_days is not None and diff_days <= 7

        records.append({
            "idx":        row["idx"],
            "label":      row["label"],
            "gt_date":    gt_iso,
            "gt_gran":    row["granularity"],
            "gt_type":    row["time_type"],
            "pred_date":  pred_date,
            "pred_gran":  pred.granularity,
            "pred_method":pred.method,
            "diff_days":  diff_days,
            "exact":      exact,
            "within_1d":  w1,
            "within_7d":  w7,
        })

        if verbose:
            print(f"  [{row['idx']}] GT={gt_iso} PRED={pred_date} diff={diff_days} "
                  f"exact={exact} method={pred.method}")

    df = pd.DataFrame(records)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "time_extractor_eval.csv"
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

    # Rows where GT has a date
    gt_has_date = df["gt_date"].notna()
    pred_has_date = df["pred_date"].notna()

    # Coverage: among rows where GT has a date, how many did we predict?
    gt_dated = df[gt_has_date]
    coverage = pred_has_date[gt_has_date].mean() if gt_has_date.any() else float("nan")

    # Accuracy metrics: only on rows where both have a date
    comparable = df[gt_has_date & pred_has_date]
    nc = len(comparable)

    exact_acc = comparable["exact"].mean()     if nc else float("nan")
    w1_acc    = comparable["within_1d"].mean() if nc else float("nan")
    w7_acc    = comparable["within_7d"].mean() if nc else float("nan")
    med_err   = comparable["diff_days"].median() if nc else float("nan")

    # Both-null agreement (GT unknown, pred unknown) — correct abstentions
    both_null = ((~gt_has_date) & (~pred_has_date)).sum()
    gt_null   = (~gt_has_date).sum()

    # Method breakdown
    method_counts = df["pred_method"].value_counts()

    print(f"\n{'='*55}")
    print(f"  {title}  (n={n})")
    print(f"{'='*55}")
    print(f"  GT has date:        {gt_has_date.sum()} rows")
    print(f"  Coverage (pred≠null when GT≠null): {coverage:.3f}  ({pred_has_date[gt_has_date].sum()}/{gt_has_date.sum()})")
    print(f"  Comparable (both have date): {nc} rows")
    print(f"  Exact match:        {exact_acc:.3f}")
    print(f"  Within 1 day:       {w1_acc:.3f}")
    print(f"  Within 7 days:      {w7_acc:.3f}")
    print(f"  Median error (days):{med_err:.1f}")
    print(f"  Both-null agreement:{both_null}/{gt_null} GT-null rows")
    print(f"\n  Method breakdown:")
    for m in ["step1_absolute", "step1_month_only", "step2_trigger_relative",
              "step3_since_month", "step4_fulltext_relative", "step4_vague",
              "step5_fallback", "none"]:
        cnt = method_counts.get(m, 0)
        print(f"    {m:28s}: {cnt:4d}  ({cnt/n:.1%})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose",   action="store_true")
    parser.add_argument("--per-class", action="store_true")
    args = parser.parse_args()
    run_eval(verbose=args.verbose, per_class=args.per_class)
