"""
Stage 2 Event Type Classifier — TF-IDF + Logistic Regression baseline.

Input : data/splits/{train,val,test}.csv
Output: data/models/event_classifier_tfidf_lr.pkl

Usage:
    conda run -n gdelt python scripts/train_event_classifier.py
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight

ROOT       = Path(__file__).parent.parent
SPLITS_DIR = ROOT / "data" / "splits"
MODELS_DIR = ROOT / "models"

LABELS_ORDER = ["earthquake", "flood", "cyclone", "wildfire", "drought", "not_related"]


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
def load_split(name: str) -> tuple[list[str], list[str]]:
    df = pd.read_csv(SPLITS_DIR / f"{name}.csv")
    return df["text"].fillna("").tolist(), df["label"].tolist()


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------
def evaluate(model, X: list[str], y: list[str], split_name: str) -> dict:
    y_pred = model.predict(X)
    report = classification_report(y, y_pred, labels=LABELS_ORDER,
                                   zero_division=0, output_dict=True)
    print(f"\n{'='*55}")
    print(f"  {split_name.upper()} SET")
    print(f"{'='*55}")
    print(classification_report(y, y_pred, labels=LABELS_ORDER, zero_division=0))

    print("Confusion matrix (rows=true, cols=pred):")
    cm = confusion_matrix(y, y_pred, labels=LABELS_ORDER)
    cm_df = pd.DataFrame(cm, index=LABELS_ORDER, columns=LABELS_ORDER)
    print(cm_df.to_string())
    return report


# ---------------------------------------------------------------------------
# Baseline: predict old_event_type (V1THEMES rule mapping, no not_related)
# ---------------------------------------------------------------------------
def baseline_evaluate(split_name: str) -> None:
    df = pd.read_csv(SPLITS_DIR / f"{split_name}.csv")
    # old_event_type acts as the "before LLM" prediction; not_related rows
    # are always wrong since V1THEMES has no such class.
    labels_df = pd.read_csv(
        ROOT / "data" / "llm_labels" / "llm_labels_0_26326_s1000.csv")
    df2 = df.merge(
        labels_df[["idx", "old_event_type"]],
        left_on=df.index, right_on=labels_df.index, how="left")
    y_true = df["label"].tolist()
    y_pred = df2["old_event_type"].fillna("not_related").tolist()
    print(f"\n{'='*55}")
    print(f"  BASELINE (V1THEMES) — {split_name.upper()}")
    print(f"{'='*55}")
    print(classification_report(y_true, y_pred, labels=LABELS_ORDER,
                                zero_division=0))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Loading splits...")
    X_train, y_train = load_split("train")
    X_val,   y_val   = load_split("val")
    X_test,  y_test  = load_split("test")
    print(f"  train={len(X_train)}  val={len(X_val)}  test={len(X_test)}")

    # Class weights for imbalanced cyclone
    classes   = np.array(LABELS_ORDER)
    weights   = compute_class_weight("balanced", classes=classes, y=y_train)
    cw = dict(zip(classes, weights))
    print(f"\nClass weights: { {k: round(v,2) for k,v in cw.items()} }")

    # --- Model ---
    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            max_features=100_000,
            sublinear_tf=True,
            min_df=2,
        )),
        ("lr", LogisticRegression(
            C=5.0,
            max_iter=1000,
            class_weight=cw,
            solver="lbfgs",
            multi_class="multinomial",
            n_jobs=-1,
        )),
    ])

    print("\nTraining TF-IDF + LR...")
    model.fit(X_train, y_train)

    # --- Baseline comparison on val ---
    try:
        baseline_evaluate("val")
    except Exception as e:
        print(f"[WARN] Baseline eval skipped: {e}")

    # --- Model evaluation ---
    val_report  = evaluate(model, X_val,  y_val,  "val")
    test_report = evaluate(model, X_test, y_test, "test")

    # --- Summary ---
    print(f"\n{'='*55}")
    print("  SUMMARY")
    print(f"{'='*55}")
    print(f"  Val  Macro-F1 : {val_report['macro avg']['f1-score']:.4f}")
    print(f"  Test Macro-F1 : {test_report['macro avg']['f1-score']:.4f}")
    for label in LABELS_ORDER:
        vf = val_report.get(label, {}).get("f1-score", 0)
        tf = test_report.get(label, {}).get("f1-score", 0)
        print(f"    {label:12s}  val-F1={vf:.3f}  test-F1={tf:.3f}")

    # --- Save ---
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out = MODELS_DIR / "event_classifier_tfidf_lr.pkl"
    with open(out, "wb") as f:
        pickle.dump({"model": model, "labels": LABELS_ORDER}, f)
    print(f"\nModel saved → {out}")


if __name__ == "__main__":
    main()
