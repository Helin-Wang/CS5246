"""
Stage 2 Event Type Classifier — DistilBERT fine-tuning.

Input : data/splits/{train,val,test}.csv
Output: data/models/event_classifier_distilbert/  (best checkpoint)

Usage:
    conda run -n gdelt python scripts/train_event_classifier_distilbert.py
    conda run -n gdelt python scripts/train_event_classifier_distilbert.py --epochs 3 --batch-size 32
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.utils.data import Dataset
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

ROOT       = Path(__file__).parent.parent
SPLITS_DIR = ROOT / "data" / "splits"
MODELS_DIR = ROOT / "models"
CKPT_DIR   = ROOT / "models" / "event_classifier_distilbert"

LABELS_ORDER = ["earthquake", "flood", "cyclone", "wildfire", "drought", "not_related"]
LABEL2ID     = {l: i for i, l in enumerate(LABELS_ORDER)}
ID2LABEL     = {i: l for i, l in enumerate(LABELS_ORDER)}

MODEL_NAME = "distilbert-base-uncased"
MAX_LEN    = 128


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class EventDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], tokenizer):
        enc = tokenizer(texts, truncation=True, padding="max_length",
                        max_length=MAX_LEN, return_tensors="pt")
        self.input_ids      = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]
        self.labels         = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels":         self.labels[idx],
        }


# ---------------------------------------------------------------------------
# Weighted-loss Trainer (handles class imbalance)
# ---------------------------------------------------------------------------
class WeightedTrainer(Trainer):
    def __init__(self, class_weights: torch.Tensor, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels  = inputs.pop("labels")
        outputs = model(**inputs)
        logits  = outputs.logits
        loss_fn = nn.CrossEntropyLoss(
            weight=self.class_weights.to(logits.device))
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


# ---------------------------------------------------------------------------
# Metrics for Trainer
# ---------------------------------------------------------------------------
def make_compute_metrics(label_list):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
        return {"macro_f1": macro_f1}
    return compute_metrics


# ---------------------------------------------------------------------------
# Load split
# ---------------------------------------------------------------------------
def load_split(name: str) -> tuple[list[str], list[int]]:
    df = pd.read_csv(SPLITS_DIR / f"{name}.csv")
    texts  = df["text"].fillna("").tolist()
    labels = [LABEL2ID[l] for l in df["label"]]
    return texts, labels


# ---------------------------------------------------------------------------
# Evaluate with full classification report
# ---------------------------------------------------------------------------
def full_eval(trainer, dataset, texts, true_labels, split_name):
    result = trainer.predict(dataset)
    preds  = np.argmax(result.predictions, axis=-1)
    pred_labels = [ID2LABEL[p] for p in preds]
    true_str    = [ID2LABEL[l] for l in true_labels]
    print(f"\n{'='*55}")
    print(f"  {split_name.upper()} SET")
    print(f"{'='*55}")
    print(classification_report(true_str, pred_labels,
                                labels=LABELS_ORDER, zero_division=0))
    report = classification_report(true_str, pred_labels,
                                   labels=LABELS_ORDER, zero_division=0,
                                   output_dict=True)
    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=5)
    parser.add_argument("--batch-size", type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=2e-5)
    args = parser.parse_args()

    # Device
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Device: {device}  ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")

    # Load data
    print("Loading splits...")
    X_train, y_train = load_split("train")
    X_val,   y_val   = load_split("val")
    X_test,  y_test  = load_split("test")
    print(f"  train={len(X_train)}  val={len(X_val)}  test={len(X_test)}")

    # Tokenizer + datasets
    print(f"Loading tokenizer ({MODEL_NAME})...")
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    train_ds = EventDataset(X_train, y_train, tokenizer)
    val_ds   = EventDataset(X_val,   y_val,   tokenizer)
    test_ds  = EventDataset(X_test,  y_test,  tokenizer)

    # Class weights
    classes = np.arange(len(LABELS_ORDER))
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    cw = torch.tensor(weights, dtype=torch.float32)
    print(f"Class weights: { {LABELS_ORDER[i]: round(w,2) for i,w in enumerate(weights)} }")

    # Model
    print("Loading model...")
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS_ORDER),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    # Training args
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(CKPT_DIR),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=64,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=50,
        fp16=torch.cuda.is_available(),   # enable on GPU, off on CPU
        report_to="none",
        save_total_limit=2,
    )

    trainer = WeightedTrainer(
        class_weights=cw,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=make_compute_metrics(LABELS_ORDER),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print(f"\nTraining ({args.epochs} epochs, batch={args.batch_size}, lr={args.lr})...")
    trainer.train()

    # Full evaluation
    val_report  = full_eval(trainer, val_ds,  X_val,  y_val,  "val")
    test_report = full_eval(trainer, test_ds, X_test, y_test, "test")

    # Summary
    print(f"\n{'='*55}")
    print("  SUMMARY")
    print(f"{'='*55}")
    print(f"  Val  Macro-F1 : {val_report['macro avg']['f1-score']:.4f}")
    print(f"  Test Macro-F1 : {test_report['macro avg']['f1-score']:.4f}")
    for label in LABELS_ORDER:
        vf = val_report.get(label,  {}).get("f1-score", 0)
        tf = test_report.get(label, {}).get("f1-score", 0)
        print(f"    {label:12s}  val-F1={vf:.3f}  test-F1={tf:.3f}")

    # Save tokenizer alongside model for inference
    tokenizer.save_pretrained(str(CKPT_DIR / "tokenizer"))
    print(f"\nBest model + tokenizer saved → {CKPT_DIR}")


if __name__ == "__main__":
    main()
