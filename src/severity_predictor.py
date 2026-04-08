"""
Module C — Severity Predictor

Predicts binary alert level (green / orange_or_red) for a single aggregated
disaster event. Called after Module D (clustering) has merged per-article
feature fields into one record per unique event.

Supported event types:
  EQ / TC / WF / DR  →  trained RandomForest models (models/*.pkl)
  FL                 →  rule-based (dead > 100 or displaced > 80000)

Usage (single event):
    from src.severity_predictor import SeverityPredictor
    predictor = SeverityPredictor()
    result = predictor.predict({
        "event_type": "EQ",
        "magnitude": 6.8,
        "depth": 35.0,
        "rapidpopdescription": "500 thousand people in MMI VI",
    })
    # result = {"predicted_alert": "orange_or_red",
    #           "prob_orange_or_red": 0.87,
    #           "low_confidence": False}

Usage (DataFrame of aggregated events):
    results_df = predictor.predict_df(events_df)
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = BASE_DIR / "models"

POSITIVE_LABEL = "orange_or_red"
GREEN_LABEL = "green"

MODEL_FILES = {
    "EQ": "eq_alertlevel_binary_classifier.pkl",
    "TC": "tc_alertlevel_binary_classifier.pkl",
    "WF": "wf_alertlevel_binary_classifier.pkl",
    "DR": "dr_alertlevel_binary_classifier.pkl",
}

# Fields that must not all be NaN for a prediction to be considered reliable.
# If every key field is NaN, low_confidence is set to True.
KEY_FIELDS: Dict[str, list] = {
    "EQ": ["magnitude", "depth", "rapidpopdescription"],
    "TC": ["maximum_wind_speed_kmh", "maximum_storm_surge_m", "exposed_population"],
    "WF": ["duration_days", "burned_area_ha", "people_affected"],
    "DR": ["duration_days", "affected_area_km2", "affected_country_count"],
    "FL": ["dead", "displaced"],
}

POP_UNIT_MULTIPLIER = {
    "thousand": 1_000,
    "million":  1_000_000,
    "billion":  1_000_000_000,
}


# ── EQ feature engineering (mirrors train_severity_classifiers.py) ────────────

def parse_rapidpopdescription(text) -> Dict[str, float]:
    """
    Convert rapidpopdescription text to numeric features.

    At inference time this field comes from Module A regex extraction on news
    text, so it may differ in format from GDACS training data. The parser is
    intentionally lenient: any number followed by an optional scale word is
    accepted as the population figure.
    """
    raw = "" if pd.isna(text) else str(text).strip()
    lower = raw.lower()

    result: Dict[str, float] = {
        "rapid_pop_people": np.nan,
        "rapid_pop_log":    np.nan,
        "rapid_missing":    0.0,
        "rapid_few_people": 0.0,
        "rapid_unparsed":   0.0,
    }

    if not raw:
        result["rapid_missing"] = 1.0
        return result

    if "few people" in lower:
        result["rapid_few_people"] = 1.0
        result["rapid_pop_people"] = 100.0
        result["rapid_pop_log"]    = np.log1p(100.0)

    m = re.search(r"(\d+(?:\.\d+)?)\s*(thousand|million|billion)?", lower)
    if m:
        value      = float(m.group(1))
        multiplier = POP_UNIT_MULTIPLIER.get(m.group(2), 1)
        people     = value * multiplier
        result["rapid_pop_people"] = people
        result["rapid_pop_log"]    = np.log1p(people)
    elif result["rapid_few_people"] == 0.0:
        result["rapid_unparsed"] = 1.0

    return result


def _build_eq_features(event: Dict) -> pd.DataFrame:
    rapid = parse_rapidpopdescription(event.get("rapidpopdescription"))
    row = {
        "magnitude":        _to_float(event.get("magnitude")),
        "depth":            _to_float(event.get("depth")),
        "rapid_pop_people": rapid["rapid_pop_people"],
        "rapid_pop_log":    rapid["rapid_pop_log"],
        "rapid_missing":    rapid["rapid_missing"],
        "rapid_few_people": rapid["rapid_few_people"],
        "rapid_unparsed":   rapid["rapid_unparsed"],
    }
    return pd.DataFrame([row])


def _build_generic_features(event: Dict, feature_cols: list) -> pd.DataFrame:
    row = {col: _to_float(event.get(col)) for col in feature_cols}
    return pd.DataFrame([row])


def _to_float(value) -> Optional[float]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    try:
        return float(value)
    except (ValueError, TypeError):
        return np.nan


# ── FL rule ───────────────────────────────────────────────────────────────────

def _classify_fl(event: Dict) -> str:
    dead       = _to_float(event.get("dead"))       or 0.0
    displaced  = _to_float(event.get("displaced"))  or 0.0
    if dead > 100 or displaced > 80_000:
        return POSITIVE_LABEL
    return GREEN_LABEL


# ── Predictor ─────────────────────────────────────────────────────────────────

class SeverityPredictor:
    """
    Load severity models once and predict for any number of events.

    Parameters
    ----------
    model_dir : path to directory containing *.pkl files.
                Defaults to <project_root>/models/.
    """

    def __init__(self, model_dir: Optional[Path] = None):
        self._model_dir = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
        self._models: Dict[str, Dict] = {}

    def _load(self, event_type: str) -> Optional[Dict]:
        if event_type in self._models:
            return self._models[event_type]
        fname = MODEL_FILES.get(event_type)
        if not fname:
            return None
        path = self._model_dir / fname
        if not path.exists():
            raise FileNotFoundError(
                f"Model not found for {event_type}: {path}. "
                f"Run scripts/train_severity_classifiers.py first."
            )
        with open(path, "rb") as f:
            payload = pickle.load(f)
        self._models[event_type] = payload
        return payload

    def _low_confidence(self, event: Dict, event_type: str) -> bool:
        """True if every key field for this event type is missing."""
        key_fields = KEY_FIELDS.get(event_type, [])
        if not key_fields:
            return False
        for field in key_fields:
            val = event.get(field)
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                s = str(val).strip()
                if s and s.lower() not in ("nan", "none", ""):
                    return False
        return True

    def predict(self, event: Dict) -> Dict:
        """
        Predict severity for a single aggregated event record.

        Parameters
        ----------
        event : dict with keys matching the event's extracted fields.
                Required key: "event_type" (case-insensitive).

        Returns
        -------
        dict with:
          predicted_alert      : "green" or "orange_or_red"
          prob_orange_or_red   : float in [0, 1], or None for FL
          low_confidence       : bool — True if all key fields were missing
        """
        event_type = str(event.get("event_type", "")).strip().upper()
        low_conf   = self._low_confidence(event, event_type)

        # FL: rule-based, no model
        if event_type == "FL":
            label = _classify_fl(event)
            return {
                "predicted_alert":    label,
                "prob_orange_or_red": None,
                "low_confidence":     low_conf,
            }

        payload = self._load(event_type)
        if payload is None:
            raise ValueError(
                f"Unsupported event_type '{event_type}'. "
                f"Supported: {list(MODEL_FILES)} + FL."
            )

        pipeline     = payload["model"]
        feature_cols = payload["feature_columns"]

        if event_type == "EQ":
            x = _build_eq_features(event)
        else:
            x = _build_generic_features(event, feature_cols)

        # Ensure column order matches training
        x = x[feature_cols]

        label      = pipeline.predict(x)[0]
        probs      = pipeline.predict_proba(x)[0]
        classes    = list(pipeline.named_steps["clf"].classes_)
        pos_idx    = classes.index(POSITIVE_LABEL)
        prob_pos   = float(probs[pos_idx])

        return {
            "predicted_alert":    label,
            "prob_orange_or_red": prob_pos,
            "low_confidence":     low_conf,
        }

    def predict_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict severity for a DataFrame of aggregated events.

        The DataFrame must contain an "event_type" column plus the relevant
        feature columns. Returns a copy of df with three new columns appended:
          predicted_alert, prob_orange_or_red, low_confidence.
        """
        results = df.apply(lambda row: self.predict(row.to_dict()), axis=1)
        out = df.copy()
        out["predicted_alert"]    = results.map(lambda r: r["predicted_alert"])
        out["prob_orange_or_red"] = results.map(lambda r: r["prob_orange_or_red"])
        out["low_confidence"]     = results.map(lambda r: r["low_confidence"])
        return out
