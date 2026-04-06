"""
Prepare train/val/test splits for the Stage 2 event type classifier.

Split strategy: time-based to prevent data leakage.
  train : 2024-01 – 2025-04  (~70%)
  val   : 2025-05 – 2025-07  (~15%)
  test  : 2025-08 – 2025-12  (~15%)

Input:
  data/llm_labels/llm_labels_0_26326_s2000.csv  (LLM labels)
  data/training_events_gdelt.xlsx               (original text + timestamps)

Output:
  data/splits/train.csv
  data/splits/val.csv
  data/splits/test.csv
  Each file has columns: idx, timestamp, label, text

Usage:
  conda run -n gdelt python scripts/prepare_classifier_data.py
"""

from pathlib import Path
import pandas as pd

ROOT       = Path(__file__).parent.parent
LABELS     = ROOT / "data" / "llm_labels" / "llm_labels_0_26326_s2000.csv"
ORIG       = ROOT / "data" / "training_events_gdelt.xlsx"
SPLITS_DIR = ROOT / "data" / "splits"

TRAIN_END = "2025-04-30"
VAL_END   = "2025-07-31"

TEXT_MAX_CHARS = 512  # title + first N chars of text_cleaned fed to classifier


def build_text(title: str, text: str) -> str:
    title   = str(title or "").strip()
    snippet = str(text  or "").strip()[:TEXT_MAX_CHARS]
    return f"{title} [SEP] {snippet}" if title else snippet


def main():
    print("Loading data...")
    labels = pd.read_csv(LABELS)
    orig   = pd.read_excel(ORIG)

    # Drop error rows
    labels = labels[labels["llm_event_type"] != "error"].copy()
    print(f"  Valid labels: {len(labels)}")

    # Merge to get timestamp + text
    orig_indexed = (orig[["timestamp", "title", "text_cleaned"]]
                    .reset_index()
                    .rename(columns={"index": "idx"}))
    orig_indexed["timestamp"] = pd.to_datetime(orig_indexed["timestamp"], format="mixed")

    df = labels.merge(orig_indexed, on="idx")
    df["text"] = df.apply(lambda r: build_text(r["title"], r["text_cleaned"]), axis=1)
    df = df[["idx", "timestamp", "llm_event_type", "text"]].rename(
        columns={"llm_event_type": "label"})

    # Time-based split
    train = df[df["timestamp"] <= TRAIN_END]
    val   = df[(df["timestamp"] > TRAIN_END) & (df["timestamp"] <= VAL_END)]
    test  = df[df["timestamp"] > VAL_END]

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(SPLITS_DIR / "train.csv", index=False)
    val.to_csv(SPLITS_DIR / "val.csv",   index=False)
    test.to_csv(SPLITS_DIR / "test.csv", index=False)

    # Report
    total = len(df)
    print(f"\nSplit results (total {total}):")
    for name, split, end in [("train", train, f"≤ {TRAIN_END}"),
                              ("val",   val,   f"{TRAIN_END} ~ {VAL_END}"),
                              ("test",  test,  f"> {VAL_END}")]:
        print(f"  {name:5s} [{end}]: {len(split):5d} rows ({100*len(split)/total:.1f}%)")

    print("\nLabel distribution per split:")
    for name, split in [("train", train), ("val", val), ("test", test)]:
        dist = split["label"].value_counts()
        dist_str = "  ".join(f"{k}={v}" for k, v in dist.items())
        print(f"  {name:5s}: {dist_str}")

    print(f"\nSaved → {SPLITS_DIR}")


if __name__ == "__main__":
    main()
