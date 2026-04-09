"""
LLM-based numeric field labeling for Module A ground-truth evaluation.

For each disaster article in data/splits/test.csv (non-not_related rows), asks
DeepSeek-V3 to extract the per-type key fields that feed the severity models.
The prompt and expected fields differ by event type:

  earthquake : magnitude, depth_km, rapidpopdescription, dead, displaced
  cyclone    : maximum_wind_speed_kmh, maximum_storm_surge_m, exposed_population,
               dead, displaced
  wildfire   : burned_area_ha, duration_days, people_affected, dead
  drought    : duration_days, affected_area_km2, affected_country_count
  flood      : dead, displaced, people_affected

Output: data/llm_labels/field_labels_test.csv
Columns:
  idx, label,
  magnitude, depth_km, rapidpopdescription,
  maximum_wind_speed_kmh, maximum_storm_surge_m, exposed_population,
  burned_area_ha, duration_days, people_affected,
  affected_area_km2, affected_country_count,
  dead, displaced,
  source_note

Irrelevant fields for a given type are left blank (NaN).

Usage:
  python scripts/label_fields.py
  python scripts/label_fields.py --workers 2
  python scripts/label_fields.py --max-rows 50
  python scripts/label_fields.py --force-relabel
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import re
import threading
import time
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import requests
from tqdm import tqdm

API_KEY = os.getenv("SILICONFLOW_API_KEY", "sk-jbarivjnimgypzwvaxnzpgbccuuhsxeddhwgwliwoewhgast")
API_URL = "https://api.siliconflow.cn/v1/chat/completions"
MODEL   = "deepseek-ai/DeepSeek-V3"

SPLITS_DIR = Path(__file__).parent.parent / "data" / "splits"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "llm_labels"
INPUT_FILE = SPLITS_DIR / "test.csv"
OUT_FILE   = OUTPUT_DIR / "field_labels_test.csv"

CHECKPOINT_EVERY = 50

# All output columns (besides idx and label); absent fields get NaN.
ALL_FIELD_COLS = [
    "magnitude",
    "depth_km",
    "rapidpopdescription",
    "maximum_wind_speed_kmh",
    "maximum_storm_surge_m",
    "exposed_population",
    "burned_area_ha",
    "duration_days",
    "people_affected",
    "affected_area_km2",
    "affected_country_count",
    "dead",
    "displaced",
]

SYSTEM_PROMPT = (
    "You are a precise numeric field extractor for disaster news articles. "
    "Extract only values explicitly stated in the article. "
    "Always respond in the exact format requested. "
    "Output N/A for any field not mentioned in the article."
)

# ── Per-type prompt templates ──────────────────────────────────────────────────

_PROMPTS: Dict[str, str] = {

"earthquake": """\
Extract numeric fields from this earthquake article. Convert units where needed.

Rules:
- MAGNITUDE: Richter / Mw value (number only, e.g. 6.8). N/A if not stated.
- DEPTH_KM: earthquake depth in km. If given in miles, multiply by 1.609. N/A if not stated.
- RAPID_POP_DESCRIPTION: verbatim phrase about how many people felt or were exposed to shaking \
(e.g. "1.2 million people felt shaking", "few people in MMI VII"). N/A if not stated.
- DEAD: total confirmed death count (number only). N/A if not stated.
- DISPLACED: total displaced / evacuated count (number only). N/A if not stated.
- NOTE: one sentence summarising what was found.

Article timestamp: {timestamp}
Article (title + body):
{text}

Respond ONLY in this exact format:
MAGNITUDE: <number or N/A>
DEPTH_KM: <number or N/A>
RAPID_POP_DESCRIPTION: <verbatim phrase or N/A>
DEAD: <number or N/A>
DISPLACED: <number or N/A>
NOTE: <one sentence>""",

"cyclone": """\
Extract numeric fields from this tropical cyclone / typhoon / hurricane article.

Rules:
- WIND_SPEED_KMH: maximum sustained wind speed in km/h. \
  Convert: mph × 1.609, knots × 1.852, m/s × 3.6. N/A if not stated.
- STORM_SURGE_M: storm surge height in metres. Convert: feet × 0.305. N/A if not stated.
- EXPOSED_POPULATION: number of people exposed / at risk / in the storm's path (number only). \
  N/A if not stated.
- DEAD: total confirmed death count (number only). N/A if not stated.
- DISPLACED: total displaced / evacuated count (number only). N/A if not stated.
- NOTE: one sentence summarising what was found.

Article timestamp: {timestamp}
Article (title + body):
{text}

Respond ONLY in this exact format:
WIND_SPEED_KMH: <number or N/A>
STORM_SURGE_M: <number or N/A>
EXPOSED_POPULATION: <number or N/A>
DEAD: <number or N/A>
DISPLACED: <number or N/A>
NOTE: <one sentence>""",

"wildfire": """\
Extract numeric fields from this wildfire / bushfire article.

Rules:
- BURNED_AREA_HA: total burned area in hectares. \
  Convert: acres × 0.405, km² × 100, sq mi × 259. N/A if not stated.
- DURATION_DAYS: how many days the fire has been burning / lasted. \
  Convert: weeks × 7, months × 30. N/A if not stated.
- PEOPLE_AFFECTED: total people affected, evacuated, or under evacuation orders (number only). \
  N/A if not stated.
- DEAD: total confirmed death count (number only). N/A if not stated.
- NOTE: one sentence summarising what was found.

Article timestamp: {timestamp}
Article (title + body):
{text}

Respond ONLY in this exact format:
BURNED_AREA_HA: <number or N/A>
DURATION_DAYS: <number or N/A>
PEOPLE_AFFECTED: <number or N/A>
DEAD: <number or N/A>
NOTE: <one sentence>""",

"drought": """\
Extract numeric fields from this drought article.

Rules:
- DURATION_DAYS: how long the drought has lasted. \
  Convert: weeks × 7, months × 30, years × 365. N/A if not stated.
- AFFECTED_AREA_KM2: geographic area affected by drought in km². \
  Convert: sq mi × 2.590, hectares × 0.01, acres × 0.004047. N/A if not stated.
- COUNTRY_COUNT: number of countries affected (integer). N/A if not stated.
- NOTE: one sentence summarising what was found.

Article timestamp: {timestamp}
Article (title + body):
{text}

Respond ONLY in this exact format:
DURATION_DAYS: <number or N/A>
AFFECTED_AREA_KM2: <number or N/A>
COUNTRY_COUNT: <integer or N/A>
NOTE: <one sentence>""",

"flood": """\
Extract numeric fields from this flood article.

Rules:
- DEAD: total confirmed death count (number only). N/A if not stated.
- DISPLACED: total displaced / evacuated / fled count (number only). N/A if not stated.
- PEOPLE_AFFECTED: total people otherwise affected / stranded / at risk (number only, \
  do not double-count with DISPLACED if the same figure). N/A if not stated.
- NOTE: one sentence summarising what was found.

Article timestamp: {timestamp}
Article (title + body):
{text}

Respond ONLY in this exact format:
DEAD: <number or N/A>
DISPLACED: <number or N/A>
PEOPLE_AFFECTED: <number or N/A>
NOTE: <one sentence>""",
}


# ── Parsing helpers ────────────────────────────────────────────────────────────

def _get(pattern: str, text: str) -> Optional[str]:
    m = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
    return m.group(1).strip() if m else None


def _parse_number(s: Optional[str]) -> Optional[float]:
    """Return float from string like '6.8', '140', '1200000'. N/A → None."""
    if not s:
        return None
    s = s.strip()
    if s.upper() in {"N/A", "NA", "NONE", ""}:
        return None
    # Strip trailing non-numeric noise (e.g. "140 km/h" → "140")
    m = re.match(r"^([\d,]+(?:\.\d+)?)", s.replace(",", ""))
    return float(m.group(1)) if m else None


def _parse_text(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    s = s.strip()
    if s.upper() in {"N/A", "NA", "NONE", ""}:
        return None
    return re.sub(r"\s+", " ", s)[:300]


def _parse_response(label: str, raw: str) -> dict:
    """Parse LLM response into a dict with ALL_FIELD_COLS keys."""
    result: dict = {col: None for col in ALL_FIELD_COLS}
    result["source_note"] = (_get(r"^NOTE:\s*(.+)$", raw) or "").strip()[:220]

    if label == "earthquake":
        result["magnitude"]            = _parse_number(_get(r"^MAGNITUDE:\s*(.+)$", raw))
        result["depth_km"]             = _parse_number(_get(r"^DEPTH_KM:\s*(.+)$", raw))
        result["rapidpopdescription"]  = _parse_text(_get(r"^RAPID_POP_DESCRIPTION:\s*(.+)$", raw))
        result["dead"]                 = _parse_number(_get(r"^DEAD:\s*(.+)$", raw))
        result["displaced"]            = _parse_number(_get(r"^DISPLACED:\s*(.+)$", raw))

    elif label == "cyclone":
        result["maximum_wind_speed_kmh"] = _parse_number(_get(r"^WIND_SPEED_KMH:\s*(.+)$", raw))
        result["maximum_storm_surge_m"]  = _parse_number(_get(r"^STORM_SURGE_M:\s*(.+)$", raw))
        result["exposed_population"]     = _parse_number(_get(r"^EXPOSED_POPULATION:\s*(.+)$", raw))
        result["dead"]                   = _parse_number(_get(r"^DEAD:\s*(.+)$", raw))
        result["displaced"]              = _parse_number(_get(r"^DISPLACED:\s*(.+)$", raw))

    elif label == "wildfire":
        result["burned_area_ha"]    = _parse_number(_get(r"^BURNED_AREA_HA:\s*(.+)$", raw))
        result["duration_days"]     = _parse_number(_get(r"^DURATION_DAYS:\s*(.+)$", raw))
        result["people_affected"]   = _parse_number(_get(r"^PEOPLE_AFFECTED:\s*(.+)$", raw))
        result["dead"]              = _parse_number(_get(r"^DEAD:\s*(.+)$", raw))

    elif label == "drought":
        result["duration_days"]          = _parse_number(_get(r"^DURATION_DAYS:\s*(.+)$", raw))
        result["affected_area_km2"]      = _parse_number(_get(r"^AFFECTED_AREA_KM2:\s*(.+)$", raw))
        result["affected_country_count"] = _parse_number(_get(r"^COUNTRY_COUNT:\s*(.+)$", raw))

    elif label == "flood":
        result["dead"]            = _parse_number(_get(r"^DEAD:\s*(.+)$", raw))
        result["displaced"]       = _parse_number(_get(r"^DISPLACED:\s*(.+)$", raw))
        result["people_affected"] = _parse_number(_get(r"^PEOPLE_AFFECTED:\s*(.+)$", raw))

    return result


# ── API call ───────────────────────────────────────────────────────────────────

def call_api(label: str, text: str, timestamp: str, max_retries: int = 5) -> dict:
    prompt_template = _PROMPTS.get(label)
    if prompt_template is None:
        return {col: None for col in ALL_FIELD_COLS} | {"source_note": "unsupported_label"}

    user_content = prompt_template.format(
        timestamp=timestamp,
        text=text[:1400],
    )
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ],
        "max_tokens": 220,
        "temperature": 0.0,
    }
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(API_URL, json=payload, headers=headers, timeout=60)
            if resp.status_code in (429, 403):
                time.sleep(30 * (attempt + 1))
                continue
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"].strip()
            return _parse_response(label, raw)
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(5 * (attempt + 1))

    return {col: None for col in ALL_FIELD_COLS} | {"source_note": "api_error"}


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="LLM GT labeling for Module A numeric field extraction."
    )
    parser.add_argument("--workers",      type=int, default=2)
    parser.add_argument("--max-rows",     type=int, default=0,
                        help="Cap on disaster rows to process (0 = all).")
    parser.add_argument("--force-relabel", action="store_true",
                        help="Ignore existing output and relabel from scratch.")
    args = parser.parse_args()

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing input: {INPUT_FILE}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df_all = pd.read_csv(INPUT_FILE)
    df = df_all[df_all["label"] != "not_related"].copy().reset_index(drop=True)
    if args.max_rows > 0:
        df = df.head(args.max_rows).copy()

    print(f"Disaster rows: {len(df)}")
    print(df["label"].value_counts().to_string())

    # Resume logic: skip successfully labelled rows; always retry api_error.
    done: dict[int, dict] = {}
    if OUT_FILE.exists() and not args.force_relabel:
        prev = pd.read_csv(OUT_FILE)
        prev_ok = prev[prev["source_note"] != "api_error"]
        for _, row in prev_ok.iterrows():
            done[int(row["idx"])] = row.to_dict()
        print(f"Resuming: {len(done)} done, {len(prev) - len(done)} api_error will retry.")
    elif args.force_relabel and OUT_FILE.exists():
        print("Force relabel: existing output ignored for this run.")

    rows_to_do = [
        (int(row["idx"]), row)
        for _, row in df.iterrows()
        if int(row["idx"]) not in done
    ]
    print(f"Remaining: {len(rows_to_do)} rows.")
    if not rows_to_do:
        print("Nothing to do.")
        return

    results: list[dict] = list(done.values())
    lock = threading.Lock()

    def process(item):
        idx, row = item
        result = call_api(
            label=str(row["label"]),
            text=str(row["text"]),
            timestamp=str(row["timestamp"]),
        )
        return {"idx": idx, "label": row["label"], **result}

    with tqdm(total=len(rows_to_do), desc="Labeling fields", unit="row") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(process, item): item for item in rows_to_do}
            for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
                res = future.result()
                with lock:
                    results.append(res)
                pbar.update(1)

                if i % CHECKPOINT_EVERY == 0:
                    with lock:
                        _save(results)

    _save(results)
    _print_summary()


def _save(results: list[dict]) -> None:
    cols = ["idx", "label"] + ALL_FIELD_COLS + ["source_note"]
    pd.DataFrame(results, columns=cols).sort_values("idx").to_csv(OUT_FILE, index=False)


def _print_summary() -> None:
    df = pd.read_csv(OUT_FILE)
    print(f"\nSaved {len(df)} rows → {OUT_FILE}")
    print(f"  api_error rows : {(df['source_note'] == 'api_error').sum()}")
    numeric_cols = [c for c in ALL_FIELD_COLS if c != "rapidpopdescription"]
    for col in numeric_cols:
        if col in df.columns:
            n = df[col].notna().sum()
            print(f"  {col:<30} non-null: {n} ({100 * n / len(df):.1f}%)")


if __name__ == "__main__":
    main()
