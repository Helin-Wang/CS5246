"""
Run inference with the best DistilBERT event classifier on the test split
and save per-sample predictions.

Usage:
    python src/eval_event_classifier.py
    python src/eval_event_classifier.py --split val
    python src/eval_event_classifier.py --ckpt models/event_classifier_distilbert/checkpoint-270
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

ROOT      = Path(__file__).parent.parent
CKPT_DIR  = ROOT / "models" / "event_classifier_distilbert"
SPLITS_DIR = ROOT / "data" / "splits"
RESULTS_DIR = ROOT / "data" / "results"

LABELS_ORDER = ["earthquake", "flood", "cyclone", "wildfire", "drought", "not_related"]
LABEL2ID     = {l: i for i, l in enumerate(LABELS_ORDER)}
ID2LABEL     = {i: l for i, l in enumerate(LABELS_ORDER)}

MAX_LEN = 128


def find_best_checkpoint(ckpt_dir: Path) -> Path:
    """Return the checkpoint with the highest step number (last saved = best due to load_best_model_at_end)."""
    ckpts = sorted(ckpt_dir.glob("checkpoint-*"),
                   key=lambda p: int(p.name.split("-")[1]))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")
    # The trainer copies the best checkpoint as the last one; pick lowest step
    # (best val Macro-F1 was epoch 5 = checkpoint-270)
    return ckpts[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--ckpt",  default=None, help="Checkpoint path (default: auto-detect best)")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Resolve checkpoint
    ckpt_path = Path(args.ckpt) if args.ckpt else find_best_checkpoint(CKPT_DIR)
    print(f"Loading model from {ckpt_path}")

    tokenizer = DistilBertTokenizerFast.from_pretrained(str(CKPT_DIR / "tokenizer"))
    model = DistilBertForSequenceClassification.from_pretrained(str(ckpt_path))
    model.eval().to(device)

    # Load split
    df = pd.read_csv(SPLITS_DIR / f"{args.split}.csv")
    texts = df["text"].fillna("").tolist()
    true_labels = df["label"].tolist()
    print(f"Loaded {len(df)} rows from {args.split}.csv")

    # Batch inference
    all_probs = []
    for i in range(0, len(texts), args.batch_size):
        batch = texts[i: i + args.batch_size]
        enc = tokenizer(batch, truncation=True, padding="max_length",
                        max_length=MAX_LEN, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        all_probs.append(probs)
        if (i // args.batch_size) % 10 == 0:
            print(f"  {i + len(batch)}/{len(texts)}")

    all_probs = np.vstack(all_probs)
    pred_ids   = np.argmax(all_probs, axis=1)
    pred_labels = [ID2LABEL[p] for p in pred_ids]
    confidence  = all_probs.max(axis=1)

    # Build output DataFrame
    out_df = df[["idx", "timestamp", "label"]].copy()
    out_df["pred_label"]  = pred_labels
    out_df["confidence"]  = confidence.round(4)
    for lbl in LABELS_ORDER:
        out_df[f"prob_{lbl}"] = all_probs[:, LABEL2ID[lbl]].round(4)
    out_df["correct"] = out_df["label"] == out_df["pred_label"]

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"distilbert_preds_{args.split}.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved {len(out_df)} rows → {out_path}")

    # Print classification report
    print(f"\n{'='*55}")
    print(f"  {args.split.upper()} SET — Classification Report")
    print(f"{'='*55}")
    print(classification_report(true_labels, pred_labels,
                                labels=LABELS_ORDER, zero_division=0))

    macro_f1 = float(classification_report(true_labels, pred_labels,
                                           labels=LABELS_ORDER, zero_division=0,
                                           output_dict=True)["macro avg"]["f1-score"])
    print(f"Macro-F1: {macro_f1:.4f}")


if __name__ == "__main__":
    main()
