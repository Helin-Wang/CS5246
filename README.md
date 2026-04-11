# Storms 'n' Stocks

End-to-end NLP pipeline: disaster event detection from news → severity prediction → stock market impact analysis (CAR).

**Supported types**: EQ (earthquake), TC (tropical cyclone), WF (wildfire), DR (drought), FL (flood).

---

## Environment

```bash
conda env create -f environment.yml
conda activate gdelt
python -m spacy download en_core_web_sm
```

---

## Repository Layout

```
scripts/      Data collection, LLM labeling, training helpers, HPC SLURM scripts
src/          Pipeline modules and evaluation scripts
models/       Trained models (pre-built; see §Training)
data/
  gdacs_all_fields.csv          GDACS severity training data (2910 events)
  training_events_gdelt.xlsx       GDELT news articles (source)
  splits/                          Time-based train/val/test splits
    train.csv / val.csv / test.csv  Labeled articles for classifier training & eval
    trainval.csv                    Train+val merged (used for full-run sector validation)
  llm_labels/                      LLM ground-truth annotations for eval
    llm_labels_0_26326_s1000.csv   Event-type labels (4998 articles)
    ner_labels_test.csv            Numeric field labels (test set)
    location_labels_test.csv       Location labels (test set)
    time_labels_test.csv           Time labels (test set)
  results/                         Full-pipeline outputs (train+val+test input)
  results_train/                Training-split pipeline run (sector mapping validation)
  results_test/                 Held-out test-split pipeline run (final evaluation)
  country_knowledge_base.json      Country → key industries KB
  sector_etf_map.json              Sector label → ETF ticker map
```

---

## Pipeline Architecture

```
GDELT articles (pre-labeled)
      │
      ▼
[Module A] Event-type classifier (DistilBERT) → filter not_related
[Module B] Per-article NER: time, location, type-specific numeric fields
[Module C] Event clustering: same type + geo ≤500km + Δt ≤7d → unique events
[Module D] GDACS matching: hit → use GDACS severity; miss → Module E
[Module E] Severity classifier (RF) per event type (EQ/TC/WF/DR) or rule (FL)
[Module F] Entity linking: country → sector ETFs (static prior + text extraction)
[Module G] Event study (OLS, ACWI benchmark, 35-day window) → CAR T+1/T+3/T+5
```

---

## 1. Data Collection

### 1a. GDACS severity training data

Fetches historical GDACS records with balanced alert levels. **Required only to retrain severity classifiers** — pre-trained models are already in `models/`.

```bash
# EQ / WF / DR  (2018–2026, balanced green/orange/red)
python scripts/fetch_gdacs_all_fields.py \
  --event-types EQ,WF,DR \
  --fromdate 2018-01-01 --todate 2026-04-07 \
  --balanced-per-level 500 --page-cap 5000 \
  --workers 6 --request-sleep-sec 0.15 \
  --output data/gdacs_eq_wf_dr_balanced.csv

# TC  (2008–2026; use fewer workers to avoid 403 rate-limit)
python scripts/fetch_gdacs_all_fields.py \
  --event-types TC \
  --fromdate 2008-01-01 --todate 2026-04-07 \
  --balanced-per-level 500 --page-cap 5000 \
  --workers 2 --request-sleep-sec 0.30 \
  --output data/gdacs_tc_balanced.csv

# Merge
python -c "
import pandas as pd
pd.concat([
    pd.read_csv('data/gdacs_eq_wf_dr_balanced.csv'),
    pd.read_csv('data/gdacs_tc_balanced.csv'),
], ignore_index=True).to_csv('data/gdacs_all_fields.csv', index=False)
"
```

### 1b. LLM labeling (DeepSeek-V3 via SiliconFlow)

These labels are **pre-computed** in `data/llm_labels/`. Re-run only if re-labeling from scratch.

```bash
# Event-type labels for classifier training (requires SILICONFLOW_API_KEY)
python scripts/label_event_types.py

# Ground-truth labels for extraction evaluation (test set only)
python scripts/label_ner_fields.py      # numeric severity fields
python scripts/label_locations.py       # country / lat-lon
python scripts/label_times.py           # event date
```

HPC SLURM equivalents: `scripts/label_ner_fields_slurm.sh`, `scripts/label_times_slurm.sh`.

---

## 2. Training

### 2a. Event-type classifier (DistilBERT)

**Pre-trained model**: `models/event_classifier_distilbert/checkpoint-890` (best checkpoint, Macro-F1 0.901 on test).

To retrain from scratch (GPU required, ~25s on H100):

```bash
# Local
python src/train_event_classifier_distilbert.py

# NUS HPC (SLURM)
sbatch scripts/train_distilbert_slurm.sh
```

First, prepare the train/val/test splits:

```bash
python scripts/prepare_classifier_data.py
# Output: data/splits/train.csv, val.csv, test.csv
```

### 2b. Severity classifiers (Random Forest, per disaster type)

**Pre-trained models**: `models/{eq,tc,wf,dr}_alertlevel_binary_classifier.pkl`.

To retrain:

```bash
python scripts/train_severity_classifiers.py \
  --input data/gdacs_all_fields.csv \
  --model-dir models/
```

| Model | Type | Test Macro-F1 | ROC-AUC | Features |
|-------|------|--------------|---------|---------|
| `eq_alertlevel_binary_classifier.pkl` | EQ | 0.878 | 0.935 | magnitude, depth, rapid_pop fields |
| `tc_alertlevel_binary_classifier.pkl` | TC | 0.815 | 0.926 | wind speed, storm surge, exposed pop |
| `wf_alertlevel_binary_classifier.pkl` | WF | 0.798 | 0.960 | duration, burned area, people affected |
| `dr_alertlevel_binary_classifier.pkl` | DR | ~0.51 (CV) | — | duration, affected area, country count |

FL uses a deterministic rule (no model): `dead > 100 or displaced > 80,000 → orange_or_red`.

---

## 3. Inference Pipeline

Runs the full pipeline (Modules A–G) on a classified article CSV.

```bash
python src/pipeline.py --input data/splits/test.csv

# Key options
python src/pipeline.py \
  --input   data/splits/test.csv \   # input CSV (must have: idx, label, text, timestamp)
  --output-dir data/results_test/ \  # where to write results
  --max-rows 100 \                   # process first N rows only (0 = all)
  --skip-stock \                     # skip Module G (CAR analysis)
  --verbose
```

**Input CSV columns**: `idx`, `label` (event type), `text` (article text), `timestamp`.  
Articles with `label=not_related` are automatically filtered out.

**Output files**:

| File | Description |
|------|-------------|
| `events.csv` | One row per unique event cluster with all extracted fields + severity + sector ETFs |
| `car_results.csv` | CAR(T+1/T+3/T+5) per event × ETF ticker pair |
| `group_analysis.csv` | Group-level CAR statistics (mean, t-stat, p-value) by event_type × severity × sector |

**NUS HPC** (SLURM):

```bash
# Edit pipeline_slurm.sh to set INPUT and OUTPUT_DIR, then:
sbatch scripts/pipeline_slurm.sh
```

---

## 4. Evaluation

All eval scripts read from `data/splits/test.csv` and `data/llm_labels/` by default.

### Event-type classifier

```bash
python src/eval_event_classifier.py
# --split val              evaluate on val set
# --ckpt models/event_classifier_distilbert/checkpoint-890
```

Output: `data/results/distilbert_preds_test.csv` (per-sample predictions + confidence).

### Location extractor

```bash
python src/eval_location_extractor.py
# Output: data/results/location_extractor_eval.csv
```

### Time extractor

```bash
python src/eval_time_extractor.py
# Output: data/results/time_extractor_eval.csv
```

### NER / numeric field extractor

```bash
python src/eval_ner_extractor.py
python src/eval_ner_extractor.py --per-class   # breakdown by disaster type
# Output: data/results/ner_extractor_eval.csv
```

### Clustering (GDACS-centric recall)

```bash
python scripts/eval_clustering_gdacs.py
# Output: data/results/clustering_eval_gdacs.csv
```

---

## 5. Pre-computed Results

The `data/results/` directory contains outputs from the **full pipeline run** (all 4,998 labeled articles). Key files for paper reproduction:

| File | Description |
|------|-------------|
| `distilbert_preds_test.csv` | Test-set classifier predictions (Macro-F1 0.901) |
| `location_extractor_eval.csv` | Location accuracy (91.1% overall) |
| `time_extractor_eval.csv` | Time accuracy (58.2% exact, 86.1% within 7 days) |
| `ner_extractor_eval.csv` | NER field extraction quality |
| `clustering_eval_gdacs.csv` | GDACS-centric clustering recall (32.4%) |
| `events.csv` | 1,664 unique events |
| `car_results.csv` | 5,161 event × ticker CAR pairs |
| `group_analysis.csv` | Group-level CAR statistics |
| `car_random_baseline.csv` | Random-date baseline CAR (for comparison) |

**Training-split validation** (`data/results_train/`): pipeline run on `trainval.csv` used to empirically validate the candidate sector mapping. Results feed into `tab:car_train` in the paper.

**Held-out test evaluation** (`data/results_test/`): pipeline run on `test.csv` only. 6,335 event × ticker pairs, 61 significant group-level CAR findings at p < 0.05.

---

## 6. Figures

```bash
python scripts/generate_report_figures.py
# Generates figures to a local output directory (default: figures/)
```

---

## Module API (quick reference)

```python
from src.location_extractor   import extract_location
from src.time_extractor        import extract_event_time
from src.unified_event_extractor import UnifiedEventExtractor
from src.severity_predictor    import SeverityPredictor
from src.entity_linker         import EntityLinker
from src.stock_analyser        import StockAnalyser

extractor = UnifiedEventExtractor()
result    = extractor.extract("M6.8 earthquake struck Nepal ...", "earthquake")
# result["metrics"]["magnitude"]["value"] → 6.8

predictor = SeverityPredictor()
pred      = predictor.predict({"event_type": "EQ", "magnitude": 6.8, "depth": 35.0})
# pred["predicted_alert"] → "orange_or_red"

linker = EntityLinker()
link   = linker.link({"event_type": "WF", "country_iso2": "US", "location_text": "California"})
# link["sector_etfs"] → ["IAK", "XLU", "WOOD", "XHB"]
```
