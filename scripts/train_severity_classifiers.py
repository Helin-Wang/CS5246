"""
Train binary severity classifiers (green vs orange_or_red) for EQ/TC/WF/DR.

Input:  data/gdacs_all_fields.csv  (produced by fetch_gdacs_all_fields.py)
Output: models/{eq,tc,wf,dr}_alertlevel_binary_classifier.pkl

Split protocol (§4.5.1 of project_plan.md):
  T_val  = 2025-04-30  →  train: fromdate < T_val
  T_test = 2025-07-31  →  val:   T_val ≤ fromdate < T_test
                           test:  fromdate ≥ T_test

Anti-leakage goal: GDACS training events must not overlap with GDELT inference
articles (enforced externally by keeping GDELT articles from a later window
than the GDACS training data). GDACS-internal temporal ordering provides
additional evaluation rigour but is not the primary leakage boundary.

Small-test augmentation (§4.5.1 回退方案):
  If a type's native test set has < MIN_TEST_PER_CLASS rows for either class,
  that class is topped up by stratified sampling from the train pool. The
  augmented samples remain in the train+val pool used for final model fitting,
  so augmented-test scores are slightly optimistic — native-test-only metrics
  are reported separately for reference.
"""

import argparse
import pickle
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = BASE_DIR / "data" / "gdacs_all_fields.csv"
DEFAULT_MODEL_DIR = BASE_DIR / "models"

POSITIVE_LABEL = "orange_or_red"
GREEN_LABEL = "green"

T_VAL = pd.Timestamp("2025-04-30")
T_TEST = pd.Timestamp("2025-07-31")
MIN_TEST_PER_CLASS = 30
MIN_TRAIN_PER_CLASS = 20  # if time split yields fewer, fall back to random split

POP_UNIT_MULTIPLIER = {
    "thousand": 1_000,
    "million": 1_000_000,
    "billion": 1_000_000_000,
}

# Feature columns per event type — must match what Module A extracts at inference time
TYPE_FEATURES: Dict[str, List[str]] = {
    "EQ": [
        "magnitude", "depth",
        "rapid_pop_people", "rapid_pop_log",
        "rapid_missing", "rapid_few_people", "rapid_unparsed",
    ],
    "TC": ["maximum_wind_speed_kmh", "maximum_storm_surge_m", "exposed_population"],
    "WF": ["duration_days", "burned_area_ha", "people_affected"],
    "DR": ["duration_days", "affected_area_km2", "affected_country_count"],
}

TYPE_MODEL_NAMES = {
    "EQ": "eq_alertlevel_binary_classifier.pkl",
    "TC": "tc_alertlevel_binary_classifier.pkl",
    "WF": "wf_alertlevel_binary_classifier.pkl",
    "DR": "dr_alertlevel_binary_classifier.pkl",
}

# RandomForest hyperparameters per type
TYPE_RF_PARAMS: Dict[str, Dict] = {
    "EQ": {"n_estimators": 300, "max_depth": 10, "min_samples_leaf": 2},
    "TC": {"n_estimators": 300, "max_depth": 10, "min_samples_leaf": 2},
    "WF": {"n_estimators": 300, "max_depth": 8,  "min_samples_leaf": 2},
    "DR": {"n_estimators": 300, "max_depth": 8,  "min_samples_leaf": 2},
}


# ── EQ-specific feature engineering ──────────────────────────────────────────

def parse_rapidpopdescription(text) -> Dict[str, float]:
    """
    Parse GDACS rapidpopdescription text into numeric features.

    The field typically contains phrases like:
      "500 thousand people", "1.2 million people", "few people", ""

    Note: if detail enrichment failed during data fetch, this field may be
    empty — handled via rapid_missing flag.
    """
    raw = "" if pd.isna(text) else str(text).strip()
    lower = raw.lower()

    result: Dict[str, float] = {
        "rapid_pop_people": np.nan,
        "rapid_pop_log": np.nan,
        "rapid_missing": 0.0,
        "rapid_few_people": 0.0,
        "rapid_unparsed": 0.0,
    }

    if not raw:
        result["rapid_missing"] = 1.0
        return result

    if "few people" in lower:
        result["rapid_few_people"] = 1.0
        result["rapid_pop_people"] = 100.0
        result["rapid_pop_log"] = np.log1p(100.0)

    pop_match = re.search(
        r"(\d+(?:\.\d+)?)\s*(thousand|million|billion)?",
        lower,
    )
    if pop_match:
        value = float(pop_match.group(1))
        unit = pop_match.group(2)
        multiplier = POP_UNIT_MULTIPLIER.get(unit, 1)
        people = value * multiplier
        result["rapid_pop_people"] = people
        result["rapid_pop_log"] = np.log1p(people)
    elif result["rapid_few_people"] == 0.0:
        result["rapid_unparsed"] = 1.0

    return result


# ── Feature builders (per event type) ────────────────────────────────────────

def build_features(df: pd.DataFrame, event_type: str) -> Tuple[pd.DataFrame, pd.Series]:
    work = df.copy()
    work["alertlevel"] = work["alertlevel"].astype(str).str.strip().str.lower()
    work = work[work["alertlevel"].isin({"green", "orange", "red"})].copy()
    work["target"] = np.where(work["alertlevel"] == "green", GREEN_LABEL, POSITIVE_LABEL)

    feature_cols = TYPE_FEATURES[event_type]

    if event_type == "EQ":
        work["magnitude"] = pd.to_numeric(work["magnitude"], errors="coerce")
        work["depth"] = pd.to_numeric(work["depth"], errors="coerce")
        parsed = (
            work["rapidpopdescription"]
            .apply(parse_rapidpopdescription)
            .apply(pd.Series)
        )
        work = pd.concat([work, parsed], axis=1)
    else:
        for col in feature_cols:
            if col in work.columns:
                work[col] = pd.to_numeric(work[col], errors="coerce")

    x = work[feature_cols].copy()
    y = work["target"].copy()
    return x, y


# ── Time-based split ──────────────────────────────────────────────────────────

def time_split(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    work = df.copy()
    ts = pd.to_datetime(work["fromdate"], errors="coerce", utc=True).dt.tz_convert(None)
    work["_ts"] = ts

    train = work[work["_ts"] < T_VAL].drop(columns=["_ts"])
    val   = work[(work["_ts"] >= T_VAL) & (work["_ts"] < T_TEST)].drop(columns=["_ts"])
    test  = work[work["_ts"] >= T_TEST].drop(columns=["_ts"])
    return train, val, test


def _split_ok(train: pd.DataFrame) -> bool:
    """Return True if the time-based train split is usable for training."""
    al = train["alertlevel"].str.strip().str.lower()
    n_green = int((al == "green").sum())
    n_oor   = int(al.isin(["orange", "red"]).sum())
    return n_green >= MIN_TRAIN_PER_CLASS and n_oor >= MIN_TRAIN_PER_CLASS


def random_split(
    df: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.1,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified random split as fallback when time split is not viable."""
    from sklearn.model_selection import train_test_split as _tts
    al = df["alertlevel"].str.strip().str.lower()
    strat = al.apply(lambda x: "green" if x == "green" else "oor")
    if strat.nunique() < 2:
        # Single class — can't stratify; return all as train, empty val/test
        return df.copy(), df.iloc[:0].copy(), df.iloc[:0].copy()

    train_val, test = _tts(df, test_size=test_size, random_state=random_state, stratify=strat)
    strat_tv = train_val["alertlevel"].str.strip().str.lower().apply(
        lambda x: "green" if x == "green" else "oor"
    )
    relative_val = val_size / (1 - test_size)
    train, val = _tts(
        train_val, test_size=relative_val, random_state=random_state, stratify=strat_tv
    )
    return train, val, test


def check_no_overlap(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame
) -> None:
    def ids(df):
        return set(zip(df["eventtype"].astype(str), df["eventid"].astype(str)))

    tv = ids(train) & ids(val)
    tt = ids(train) & ids(test)
    vt = ids(val)   & ids(test)
    if tv or tt or vt:
        raise RuntimeError(
            f"Anti-leakage check failed — eventid overlap: "
            f"train∩val={len(tv)}, train∩test={len(tt)}, val∩test={len(vt)}"
        )


# ── Small-test augmentation ───────────────────────────────────────────────────

def augment_test(
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
    target_per_class: int,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, bool]:
    """
    Top up test_df with samples from train_df until each alertlevel class
    has at least target_per_class rows.

    Augmented rows are flagged with _augmented=True so callers can compute
    native-test-only metrics for reference.
    """
    al = test_df["alertlevel"].str.strip().str.lower()
    pieces = [test_df.assign(_augmented=False)]
    was_augmented = False

    class_pool_filter = {
        GREEN_LABEL:    lambda d: d["alertlevel"].str.strip().str.lower() == "green",
        POSITIVE_LABEL: lambda d: d["alertlevel"].str.strip().str.lower().isin(["orange", "red"]),
    }
    class_test_count = {
        GREEN_LABEL:    int((al == "green").sum()),
        POSITIVE_LABEL: int(al.isin(["orange", "red"]).sum()),
    }

    for label, count in class_test_count.items():
        n_need = max(0, target_per_class - count)
        if n_need == 0:
            continue
        pool = train_df[class_pool_filter[label](train_df)]
        n_sample = min(n_need, len(pool))
        if n_sample == 0:
            print(f"    [augment] class={label}: need {n_need} but train pool empty")
            continue
        sampled = pool.sample(n=n_sample, random_state=random_state, replace=False)
        pieces.append(sampled.assign(_augmented=True))
        was_augmented = True
        print(
            f"    [augment] class={label}: have {count}, "
            f"need {n_need}, sampled {n_sample} from train pool"
        )

    result = pd.concat(pieces, ignore_index=True)
    return result, was_augmented


# ── Model pipeline ────────────────────────────────────────────────────────────

def make_pipeline(event_type: str, random_state: int = 42) -> Pipeline:
    feature_cols = TYPE_FEATURES[event_type]
    rf_params = TYPE_RF_PARAMS[event_type]
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                feature_cols,
            )
        ]
    )
    model = RandomForestClassifier(
        class_weight="balanced",
        random_state=random_state,
        **rf_params,
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("clf", model)])


# ── Evaluation ────────────────────────────────────────────────────────────────

def compute_metrics(pipeline: Pipeline, x: pd.DataFrame, y: pd.Series) -> Dict:
    pred = pipeline.predict(x)
    probs = pipeline.predict_proba(x)
    pos_idx = list(pipeline.named_steps["clf"].classes_).index(POSITIVE_LABEL)
    pos_probs = probs[:, pos_idx]
    y_bin = (y == POSITIVE_LABEL).astype(int)

    has_both_classes = len(y.unique()) > 1

    return {
        "n": len(y),
        "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
        "accuracy": accuracy_score(y, pred),
        "roc_auc": roc_auc_score(y_bin, pos_probs) if has_both_classes else float("nan"),
        "pr_auc": average_precision_score(y_bin, pos_probs) if has_both_classes else float("nan"),
        "minority_recall": f1_score(
            y, pred, labels=[POSITIVE_LABEL], average="macro", zero_division=0
        ),
        "report": classification_report(y, pred, digits=4, zero_division=0),
        "cm": confusion_matrix(y, pred, labels=[GREEN_LABEL, POSITIVE_LABEL]),
    }


def print_metrics(tag: str, m: Dict) -> None:
    print(
        f"  [{tag}] n={m['n']}  "
        f"Macro-F1={m['macro_f1']:.4f}  "
        f"ROC-AUC={m['roc_auc']:.4f}  "
        f"PR-AUC={m['pr_auc']:.4f}  "
        f"minority-recall={m['minority_recall']:.4f}"
    )


# ── Data diagnostics ──────────────────────────────────────────────────────────

def label_dist(df: pd.DataFrame) -> Dict[str, int]:
    al = df["alertlevel"].str.strip().str.lower()
    return {
        "green": int((al == "green").sum()),
        "orange_or_red": int(al.isin(["orange", "red"]).sum()),
    }


def print_data_stats(
    event_type: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    print(f"\n  --- {event_type} data statistics ---")
    print(f"  n_train={len(train_df)}  n_val={len(val_df)}  n_test(native)={len(test_df)}")
    print(f"  train  labels: {label_dist(train_df)}")
    print(f"  val    labels: {label_dist(val_df)}")
    print(f"  test   labels: {label_dist(test_df)}")

    # Key field null/empty rates in training data
    combined = pd.concat([train_df, val_df], ignore_index=True)
    raw_key_cols = {
        "EQ": ["magnitude", "depth", "rapidpopdescription"],
        "TC": ["maximum_wind_speed_kmh", "maximum_storm_surge_m", "exposed_population"],
        "WF": ["duration_days", "burned_area_ha", "people_affected"],
        "DR": ["duration_days", "affected_area_km2", "affected_country_count"],
    }[event_type]

    for col in raw_key_cols:
        if col not in combined.columns:
            continue
        series = combined[col]
        if pd.api.types.is_numeric_dtype(series):
            null_rate = series.isna().mean()
        else:
            # Text column: count empty strings and NaN as missing
            null_rate = (series.isna() | (series.astype(str).str.strip() == "")).mean()
        print(f"  {col} null rate (train+val): {null_rate:.1%}")


# ── Per-type training ─────────────────────────────────────────────────────────

def train_one_type(
    event_type: str,
    df_type: pd.DataFrame,
    model_dir: Path,
    random_state: int = 42,
) -> None:
    feature_cols = TYPE_FEATURES[event_type]

    print(f"\n{'='*60}")
    print(f"  {event_type} classifier")
    print(f"{'='*60}")

    # 1. Check overall class balance — skip if too few samples to train
    al_all = df_type["alertlevel"].str.strip().str.lower()
    n_green_total = int((al_all == "green").sum())
    n_oor_total   = int(al_all.isin(["orange", "red"]).sum())
    min_viable = max(2, MIN_TRAIN_PER_CLASS // 2)
    if n_green_total < min_viable or n_oor_total < min_viable:
        print(
            f"  [SKIP] {event_type}: insufficient class samples "
            f"(green={n_green_total}, orange_or_red={n_oor_total}, "
            f"min_viable_per_class={min_viable}). "
            f"Cannot train binary classifier — check data fetch."
        )
        return

    # 2. Time split; fall back to stratified random split if train is too small
    train_df, val_df, test_df = time_split(df_type)
    split_method = "time"
    if not _split_ok(train_df):
        print(
            f"  [warn] Time split yields insufficient train data "
            f"(green={int((train_df['alertlevel'].str.strip().str.lower()=='green').sum())}, "
            f"orange_or_red={int(train_df['alertlevel'].str.strip().str.lower().isin(['orange','red']).sum())}). "
            f"Falling back to stratified random split (train/val/test = 70/10/20%)."
        )
        train_df, val_df, test_df = random_split(df_type, random_state=random_state)
        split_method = "random"

    if split_method == "time":
        check_no_overlap(train_df, val_df, test_df)

    # 3. Data diagnostics (before augmentation)
    print_data_stats(event_type, train_df, val_df, test_df)
    print(f"  split_method: {split_method}")

    # 4. Augment test from train pool if either class is underrepresented
    native_test_n = len(test_df)
    test_df, was_augmented = augment_test(
        test_df, train_df,
        target_per_class=MIN_TEST_PER_CLASS,
        random_state=random_state,
    )
    if was_augmented:
        print(
            f"  [note] test set: {native_test_n} native → {len(test_df)} total "
            f"({len(test_df) - native_test_n} augmented from train pool)"
        )
        print(f"  test   labels (augmented): {label_dist(test_df)}")

    # 5. Feature engineering
    x_train, y_train = build_features(train_df, event_type)
    x_val,   y_val   = build_features(val_df,   event_type)
    x_test,  y_test  = build_features(test_df,  event_type)

    if len(y_train) == 0:
        print(f"  [skip] no training data after alertlevel filtering")
        return
    if len(y_train.unique()) < 2:
        print(f"  [skip] only one class present in training data: {y_train.unique()}")
        return

    # 6. Train on train split
    pipeline = make_pipeline(event_type, random_state)
    pipeline.fit(x_train, y_train)

    # 6. Evaluate on val
    if len(y_val) > 0 and len(y_val.unique()) > 1:
        m_val = compute_metrics(pipeline, x_val, y_val)
        print_metrics("val", m_val)
    else:
        print(f"  [val] n={len(y_val)} (skipping metrics — single class or empty)")

    # 7. Evaluate on test (augmented)
    if len(y_test) > 0:
        m_test = compute_metrics(pipeline, x_test, y_test)
        tag = "test (augmented)" if was_augmented else "test"
        print_metrics(tag, m_test)
        print(m_test["report"])
        print(f"  Confusion matrix [green, {POSITIVE_LABEL}]:")
        print(m_test["cm"])

        # Native-test-only metrics for reference (if augmented)
        if was_augmented and native_test_n > 0:
            native_mask = ~test_df["_augmented"].values if "_augmented" in test_df.columns \
                else np.ones(len(test_df), dtype=bool)
            x_native = x_test.iloc[:native_test_n]
            y_native = y_test.iloc[:native_test_n]
            if len(y_native.unique()) > 1:
                m_native = compute_metrics(pipeline, x_native, y_native)
                print_metrics("test (native only, reference)", m_native)
            else:
                print(f"  [test native] n={native_test_n} — single class, skipping metrics")

    # 8. 5-fold CV on train split (stability estimate)
    if len(y_train) >= 10:
        n_splits = min(5, int(y_train.value_counts().min()))
        n_splits = max(2, n_splits)
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        cv_f1 = cross_val_score(pipeline, x_train, y_train, cv=cv, scoring="f1_macro")
        print(f"  {n_splits}-fold CV on train: {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")

    # 9. Final model: retrain on train + val
    x_trainval = pd.concat([x_train, x_val], ignore_index=True)
    y_trainval  = pd.concat([y_train, y_val],  ignore_index=True)
    if len(y_trainval.unique()) < 2:
        print("  [warn] train+val has single class — saving train-only model")
        final_pipeline = pipeline
        y_trainval = y_train
        x_trainval = x_train
    else:
        final_pipeline = make_pipeline(event_type, random_state)
        final_pipeline.fit(x_trainval, y_trainval)

    # 10. Save
    model_dir.mkdir(parents=True, exist_ok=True)
    out_path = model_dir / TYPE_MODEL_NAMES[event_type]
    payload = {
        "model": final_pipeline,
        "feature_columns": feature_cols,
        "labels": sorted(y_trainval.unique().tolist()),
        "event_type": event_type,
        "train_val_size": int(len(y_trainval)),
    }
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)
    print(f"\n  Saved → {out_path}  (train+val n={len(y_trainval)})")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    global MIN_TEST_PER_CLASS

    parser = argparse.ArgumentParser(
        description="Train EQ/TC/WF/DR severity classifiers from gdacs_all_fields.csv."
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help="Path to gdacs_all_fields.csv.",
    )
    parser.add_argument(
        "--model-dir",
        default=str(DEFAULT_MODEL_DIR),
        help="Directory to write .pkl files.",
    )
    parser.add_argument(
        "--event-types",
        default="EQ,TC,WF,DR",
        help="Comma-separated event types to train (subset of EQ,TC,WF,DR).",
    )
    parser.add_argument(
        "--min-test-per-class",
        type=int,
        default=MIN_TEST_PER_CLASS,
        help=(
            "Min rows per class in native test set before augmenting "
            "from train pool (default: 30)."
        ),
    )
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    MIN_TEST_PER_CLASS = args.min_test_per_class

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(input_path, encoding_errors="replace")
    df["eventtype"]  = df["eventtype"].astype(str).str.strip().str.upper()
    df["eventid"]    = df["eventid"].astype(str).str.strip()
    df["alertlevel"] = df["alertlevel"].astype(str).str.strip().str.lower()
    df = df[df["alertlevel"].isin({"green", "orange", "red"})].copy()

    model_dir = Path(args.model_dir)
    requested = [t.strip().upper() for t in args.event_types.split(",")]
    event_types = [t for t in requested if t in TYPE_FEATURES]

    if not event_types:
        print(f"Error: no valid event types in --event-types '{args.event_types}'", file=sys.stderr)
        sys.exit(1)

    print(f"Input: {input_path}  ({len(df)} rows after alertlevel filter)")
    print(f"Training: {event_types}")
    print(f"T_val={T_VAL.date()}  T_test={T_TEST.date()}")
    print(f"Min test per class (before augmentation): {MIN_TEST_PER_CLASS}")

    for event_type in event_types:
        df_type = df[df["eventtype"] == event_type].copy()
        if df_type.empty:
            print(f"\n[skip] {event_type}: no rows in dataset")
            continue
        train_one_type(event_type, df_type, model_dir, args.random_state)

    print(f"\n{'='*60}")
    print("All done.")


if __name__ == "__main__":
    main()
