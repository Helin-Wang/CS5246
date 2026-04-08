# CS5246 — Storms 'n' Stocks

End-to-end NLP pipeline: disaster event detection from news → severity prediction → stock market impact analysis.

**Supported disaster types**: EQ (earthquake), TC (tropical cyclone), WF (wildfire), DR (drought), FL (flood). Volcano excluded.

---

## Environment

```bash
conda activate gdelt   # Python 3.10+
```

---

## Data

| File | Description |
|------|-------------|
| `data/gdacs_all_fields_v2.csv` | GDACS severity training data — 2910 rows (EQ=670, TC=1438, WF=544, DR=258) |
| `data/training_events_gdelt.xlsx` | GDELT news articles for event-type classifier training |
| `data/llm_labels/` | LLM-labeled event-type annotations |
| `data/splits/` | Time-based train/val/test splits for event-type classifier |

---

## 1. Fetch GDACS severity training data

GDACS API returns events in reverse-chronological order. Use `--balanced-per-level` to
ensure orange/red events are represented (without it, the row cap is exhausted on green
events before reaching orange/red pages).

**EQ / WF / DR** — balanced fetch from 2018:

```bash
python scripts/fetch_gdacs_all_fields.py \
  --event-types EQ,WF,DR \
  --fromdate 2018-01-01 \
  --balanced-per-level 500 \
  --page-cap 5000 \
  --workers 6 \
  --output data/gdacs_eq_wf_dr_balanced.csv
```

**TC** — fetches historical seasons back to 2008; TC endpoint may return HTTP 403,
use conservative throttling:

```bash
python scripts/fetch_gdacs_all_fields.py \
  --event-types TC \
  --fromdate 2008-01-01 \
  --balanced-per-level 500 \
  --page-cap 5000 \
  --tc-workers 2 \
  --request-sleep-sec 0.3 \
  --retry-attempts 5 \
  --output data/gdacs_tc_balanced.csv
```

After fetching, merge into the final training file:

```python
import pandas as pd
eq_wf_dr = pd.read_csv("data/gdacs_eq_wf_dr_balanced.csv")
tc       = pd.read_csv("data/gdacs_tc_balanced.csv")
pd.concat([eq_wf_dr, tc], ignore_index=True).to_csv("data/gdacs_all_fields_v2.csv", index=False)
```

---

## 2. Train severity classifiers

Trains binary classifiers (green vs orange_or_red) for EQ / TC / WF / DR.
FL uses a rule (no model needed).

Orange and red are merged into one class because: (1) red events are too rare to train
a reliable 3-class model (as few as 5 globally for WF); (2) both orange and red trigger
significant stock market impact in the same direction, matching the granularity needed
for Module E; (3) the semantic gap between green and orange is far larger than between
orange and red in GDACS's own severity scale.

```bash
python scripts/train_severity_classifiers.py \
  --input data/gdacs_all_fields_v2.csv \
  --model-dir models/
```

**Key options**:

| Flag | Default | Description |
|------|---------|-------------|
| `--event-types` | `EQ,TC,WF,DR` | Subset of types to train |
| `--min-test-per-class` | `30` | Augment test set from train pool if below this |
| `--random-state` | `42` | Reproducibility seed |

**Split logic**: time-based split per type (train < 2025-04-30, val < 2025-07-31, test ≥ 2025-07-31).
Falls back to stratified random split (70/10/20) if the train set has < 10 samples per class after
the time split. Test sets with < 30 samples per class are augmented from the train pool.

**Trained models** saved to `models/`:

| Model file | Type | Macro-F1 (test) | Features |
|------------|------|----------------|---------|
| `eq_alertlevel_binary_classifier.pkl` | EQ | 0.878 | magnitude, depth, rapid_pop_people, rapid_pop_log, rapid_missing, rapid_few_people, rapid_unparsed |
| `tc_alertlevel_binary_classifier.pkl` | TC | 0.815 | maximum_wind_speed_kmh, maximum_storm_surge_m, exposed_population |
| `wf_alertlevel_binary_classifier.pkl` | WF | 0.798 | duration_days, burned_area_ha, people_affected |
| `dr_alertlevel_binary_classifier.pkl` | DR | ~0.51 (CV) | duration_days, affected_area_km2, affected_country_count |

---

## 3. Train event-type classifier (Stage 2)

Classifies news articles into: earthquake / flood / cyclone / wildfire / drought / not_related.

**Baseline — TF-IDF + Logistic Regression**:

```bash
python src/train_event_classifier.py
# Output: models/event_classifier_tfidf_lr.pkl
# Test Macro-F1: 0.661
```

**Main model — DistilBERT fine-tuned**:

```bash
python src/train_event_classifier_distilbert.py
# Output: models/event_classifier_distilbert/
```

---

## 4. Severity inference (Module C)

```python
from src.severity_predictor import SeverityPredictor

predictor = SeverityPredictor()   # loads models lazily on first use

# Single event (dict of extracted fields from Module A after Module D aggregation)
result = predictor.predict({
    "event_type": "EQ",
    "magnitude": 6.8,
    "depth": 35.0,
    "rapidpopdescription": "500 thousand people in MMI VI",
})
# {"predicted_alert": "orange_or_red", "prob_orange_or_red": 0.87, "low_confidence": False}

# DataFrame of events
results_df = predictor.predict_df(events_df)
# Appends columns: predicted_alert, prob_orange_or_red, low_confidence
```

**low_confidence = True** when all key fields for the event type are missing after
Module D aggregation (e.g. every article in a cluster failed feature extraction).

**FL** is always rule-based: `dead > 100 or displaced > 80_000 → orange_or_red`.

---

## Pipeline overview

```
GDELT GKG  →  Stage 1 (V1THEMES filter)
           →  full-text crawl
           →  Stage 2 (event-type classifier)       [src/train_event_classifier*.py]
           →  Module A (per-article NER + regex)    [src/unified_event_extractor.py]
           →  Module D (cluster & aggregate)
           →  Module C (severity prediction)        [src/severity_predictor.py]
           →  Module B (entity linking → country)
           →  Module E (event study, yfinance CAR)
```
