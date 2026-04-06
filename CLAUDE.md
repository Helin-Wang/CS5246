# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Storms 'n' Stocks** — CS5246 Text Mining course project. An end-to-end NLP pipeline that detects disaster events from news, extracts structured information (time, location, type, severity), and assesses potential stock market impact.

Key documents:
- `task.md` — original assignment specification
- `project_plan.md` — detailed project plan including data sources, pipeline design, and evaluation protocol

---

## Pipeline Architecture

Two paths: offline training (GDACS → models) and online inference (GDELT news → prediction).

```
【Offline Training】
GDACS API → fetch_gdacs_*.py → data/gdacs_*_fields.csv
          → train_*.py → models/*.pkl (RandomForest binary classifier per disaster type)
          Label: alertlevel green=0 / orange_or_red=1

【Online Inference — 5 Modules, in order】
GDELT GKG (V1THEMES filter) → crawl news text
  ↓
Module A  per-article extraction
          event_type: from V1THEMES mapping
          location:   spaCy GPE/LOC (text) + GKG V1LOCATIONS (lat/lon)
          time:       GKG DATE field (col 1)
          per-type disaster params (regex, unit-normalised):
            EQ: magnitude, depth, rapidpopdescription
            TC: wind_speed (→km/h), storm_surge (→m), exposed_population
            WF: duration_days, burned_area (→ha), people_affected
            DR: duration_days, affected_area (→km2), country_count
            FL: dead, displaced
  ↓
Module D  Cluster by (event_type, geo-proximity ≤500km or location-text-match, Δtime ≤7d)
          Aggregate per cluster: numeric fields → max; location → most frequent; date → earliest
          No Entity Linking needed here — uses raw location text/coords from Module A
  ↓
Module C  Severity prediction on aggregated event record (one prediction per unique event)
          load models/*.pkl → predict(aggregated_features)
          FL: rule dead>100 or displaced>80000 on aggregated max values
          low_confidence=True if all key fields still NaN after aggregation
  ↓
Module B  Entity Linking (per unique event, just before stock analysis)
          aggregated location text → pycountry + alias dict → ISO country code
          ISO → knowledge base JSON → index_ticker + key_industries
  ↓
Module E  Event study: yfinance + OLS → CAR(T+1/T+3/T+5), group by event_type×severity
```

Final output per unique event: `event_id`, `event_type`, `event_date`, `primary_country`, `predicted_alert`, `article_count`, `car_t1`, `car_t3`, `car_t5`.

---

## Data Sources (all verified accessible)

| Source | Role | Key fields |
|--------|------|-----------|
| GDELT v2 Export CSV | News URL entry | col60=`SOURCEURL`, col26=`EventCode`, col56/57=`Lat/Long` |
| GDELT v2 GKG CSV | Disaster filtering + event type | col4=`URL`, col7=`V1THEMES`, col9=`V1LOCATIONS` |
| GDACS API | **Severity training labels** + structured features | `alertlevel`, `eventtype`, `magnitude`, `rapidpopdescription`, `maximum_wind_speed_kmh` |
| yfinance | Stock/ETF prices | Daily OHLCV |
| NewsAPI | Fallback for failed GDELT URLs | Requires free API key |

**Do not use**: ReliefWeb API (410 Gone), USGS (replaced by GDACS for severity labels), Twitter/X API (paid).

**Verified GKG disaster theme keywords** (confirmed against 2024-04-03 Taiwan M7.4):
`NATURAL_DISASTER`, `NATURAL_DISASTER_EARTHQUAKE`, `NATURAL_DISASTER_TSUNAMI`, `NATURAL_DISASTER_FLOOD`, `NATURAL_DISASTER_TREMOR`, `MANMADE_DISASTER_IMPLIED`, `DISASTER_FIRE`, `TERROR`, `CRISISLEX_CRISISLEXREC`

**GDELT column layout** (verified live): Export has 61 cols (older docs say 58). GKG has 27 cols. GKG files occasionally contain non-UTF-8 bytes; use `encoding_errors="replace"`.

**GDACS API**: `https://www.gdacs.org/gdacsapi/api/events/geteventlist/SEARCH`. Requires `ssl._create_unverified_context()`. EQ `magnitude`/`depth` fields need `--enrich-details` (detail endpoint per event); basic fetch already returns `rapidpopdescription` with magnitude as text ("Magnitude 4.7M, Depth:131km").

---

## Tech Stack

- **NLP**: spaCy `en_core_web_sm` (GPE/LOC NER only; no fine-tuning needed)
- **Feature extraction**: Python `re` regex, per-type with unit normalisation
- **Severity models**: scikit-learn `Pipeline` (median imputation + `RandomForestClassifier`, `class_weight="balanced"`)
- **Entity linking**: `pycountry` + `geonamescache` (offline, no API calls)
- **Keyword extraction**: KeyBERT (for display/analysis only, not model input)
- **Stock data**: yfinance
- **Environment**: Python 3.10+, conda env `gdelt`

## Reference Implementation

`/Users/wanghelin/Documents/course/CS5246/pj/` contains a working version of the severity classification pipeline. Key scripts to adapt:
- `scripts/fetch_gdacs_eq_fields.py` — GDACS data fetching with `--enrich-details` flag for full structured features
- `scripts/train_eq_alert_classifier.py` — RandomForest training with `rapidpopdescription` parsing
- `scripts/rule_based_alert_classifier.py` — rule-based classifier for EQ + FL
- `docs/NER.md` — per-type regex feature extraction specification
- `scripts/pipeline_step1_predict.py` — full inference pipeline (Module A+C)

---

## Evaluation Protocol

- **Feature extraction (Module A)**: field parse rate per type (EQ magnitude ≥60%, FL dead ≥55%); numeric accuracy on 200 human-verified samples
- **Severity (Module C)**: test Macro-F1 + ROC-AUC per disaster type on GDACS hold-out (25%); expected: EQ ~0.91, TC ~0.78, WF ~0.77, DR ~0.58 (known weak); baseline = rule classifier
- **Deduplication (Module D)**: merge rate + false-merge rate on 100 manually grouped articles
- **Entity Linking (Module B)**: country resolution rate ≥70%, accuracy ≥85% on 100-sample check
- **End-to-end (Module E)**: CAR(T+1/T+3/T+5) group means; t-test H₀: CAR=0; expect orange_or_red group < green group

Volcano is explicitly excluded (no wind-effect feature → no model). DR classifier performance is a known limitation (~F1 0.58); flag in report.
