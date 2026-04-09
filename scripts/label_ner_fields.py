"""
LLM-based NER field labeling for disaster news (ground-truth for numeric parameters).

Reads:
  data/splits/test.csv                   (idx, label, timestamp)
  data/training_events_gdelt.xlsx        (full article text, no truncation)

Asks DeepSeek-V3 to extract type-specific numeric fields from each article.

Output: data/llm_labels/ner_labels_test.csv
Columns: idx, label, <field_1>, <field_2>, ..., source_note

Fields extracted per event type:
  EQ : magnitude, depth_km, rapid_pop_description
  TC : wind_speed_kmh, storm_surge_m, exposed_population
  WF : burned_area_ha, people_affected, duration_days
  DR : affected_area_km2, duration_days, affected_country_count
  FL : dead, displaced

Usage:
  python scripts/label_ner_fields.py
  python scripts/label_ner_fields.py --workers 2 --max-rows 20
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import re
import threading
import time
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

API_KEY = os.getenv("SILICONFLOW_API_KEY", "sk-jbarivjnimgypzwvaxnzpgbccuuhsxeddhwgwliwoewhgast")
API_URL = "https://api.siliconflow.cn/v1/chat/completions"
MODEL   = "deepseek-ai/DeepSeek-V3"

ROOT       = Path(__file__).parent.parent
SPLITS_DIR = ROOT / "data" / "splits"
ORIG_FILE  = ROOT / "data" / "training_events_gdelt.xlsx"
OUTPUT_DIR = ROOT / "data" / "llm_labels"
OUT_FILE   = OUTPUT_DIR / "ner_labels_test.csv"

CHECKPOINT_EVERY = 50

# ---------------------------------------------------------------------------
# Per-type field specs: (output_key, prompt_label, unit_hint)
# ---------------------------------------------------------------------------
TYPE_FIELDS = {
    "earthquake": [
        ("magnitude",             "MAGNITUDE",             "Richter scale number, e.g. 6.2"),
        ("depth_km",              "DEPTH_KM",              "kilometers (convert from miles if needed)"),
        ("rapid_pop_description", "RAPID_POP_DESCRIPTION", "verbatim text describing population affected by shaking, e.g. '1.2 million people', 'few people', or N/A if absent"),
    ],
    "cyclone": [
        ("wind_speed_kmh",    "WIND_SPEED_KMH",    "km/h (convert from mph ×1.609, knots ×1.852, m/s ×3.6)"),
        ("storm_surge_m",     "STORM_SURGE_M",     "meters (convert from feet ×0.3048)"),
        ("exposed_population","EXPOSED_POPULATION","number of people exposed/at risk (integer or N/A)"),
    ],
    "wildfire": [
        ("burned_area_ha",  "BURNED_AREA_HA",  "hectares (convert from acres ×0.4047, km² ×100)"),
        ("people_affected", "PEOPLE_AFFECTED", "number of people affected/evacuated (integer or N/A)"),
        ("duration_days",   "DURATION_DAYS",   "days (convert from weeks ×7, months ×30)"),
    ],
    "drought": [
        ("affected_area_km2",      "AFFECTED_AREA_KM2",      "km² (convert from sq mi ×2.59)"),
        ("duration_days",          "DURATION_DAYS",          "days (convert from weeks ×7, months ×30)"),
        ("affected_country_count", "AFFECTED_COUNTRY_COUNT", "integer number of countries"),
    ],
    "flood": [
        ("dead",       "DEAD",       "total death toll (integer or N/A)"),
        ("displaced",  "DISPLACED",  "total displaced/evacuated people (integer or N/A)"),
    ],
}

ALL_FIELDS = sorted({f for fields in TYPE_FIELDS.values() for f, _, _ in fields})

SYSTEM_PROMPT = (
    "You are a precise disaster news parameter extractor. "
    "Extract only explicitly stated numeric values. "
    "Never guess or infer values not directly mentioned in the text. "
    "Always respond in the exact format requested."
)

USER_TEMPLATE = """\
Extract disaster parameters from the article below.

Rules:
- Output the numeric value in the specified unit. Perform unit conversions as instructed.
- Output N/A if the value is not explicitly stated in the article.
- Do NOT round values; preserve the precision given in the article.
- If multiple values appear (e.g. updated figures), use the highest/most recent one.

Disaster type: {label}
Article:
{text}

Respond ONLY in this exact format (no extra text or explanation):
{format_lines}"""


def _build_format_lines(label: str) -> str:
    fields = TYPE_FIELDS.get(label, [])
    return "\n".join(f"{prompt_label}: <{unit_hint}>" for _, prompt_label, unit_hint in fields)


def _parse_number(s: str) -> float | None:
    if not s or s.strip().upper() in ("N/A", "NA", "NONE", ""):
        return None
    s = s.strip().replace(",", "")
    # Strip trailing units like "km/h", "km²" etc. (keep leading number)
    s = re.sub(r"[^\d.\-].*$", "", s)
    try:
        return float(s)
    except ValueError:
        return None


def _parse_text_field(s: str) -> str | None:
    if not s or s.strip().upper() in ("N/A", "NA", "NONE", ""):
        return None
    return re.sub(r"\s+", " ", s.strip())[:300]


def _parse_response(raw: str, label: str) -> dict:
    fields = TYPE_FIELDS.get(label, [])
    result: dict = {f: None for f, _, _ in fields}
    result["source_note"] = ""

    for output_key, prompt_label, _ in fields:
        pattern = rf"^{re.escape(prompt_label)}:\s*(.+)$"
        m = re.search(pattern, raw, re.IGNORECASE | re.MULTILINE)
        value_str = m.group(1).strip() if m else None

        if output_key == "rapid_pop_description":
            result[output_key] = _parse_text_field(value_str)
        else:
            result[output_key] = _parse_number(value_str)

    return result


def call_api(text: str, label: str, max_retries: int = 5) -> dict:
    format_lines = _build_format_lines(label)
    # Limit text to ~3000 chars to stay within token budget while using full text
    text_trimmed = text[:3000]

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": USER_TEMPLATE.format(
                label=label, text=text_trimmed, format_lines=format_lines)},
        ],
        "max_tokens": 200,
        "temperature": 0.0,
    }
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type":  "application/json",
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(API_URL, json=payload, headers=headers, timeout=90)
            if resp.status_code in (429, 403):
                time.sleep(30 * (attempt + 1))
                continue
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"].strip()
            result = _parse_response(raw, label)
            result["source_note"] = "ok"
            return result
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(5 * (attempt + 1))
    # All retries failed
    result = {f: None for fields in TYPE_FIELDS.values() for f, _, _ in fields}
    result["source_note"] = "api_error"
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers",      type=int, default=2)
    parser.add_argument("--max-rows",     type=int, default=0)
    parser.add_argument("--force-relabel",action="store_true")
    args = parser.parse_args()

    # Load test split (idx + label + timestamp)
    test_df = pd.read_csv(SPLITS_DIR / "test.csv")
    test_df = test_df[test_df["label"] != "not_related"].copy()

    # Load full articles from xlsx (idx = row index)
    print("Loading full article text from xlsx...")
    orig = pd.read_excel(ORIG_FILE)
    orig = orig.reset_index().rename(columns={"index": "idx"})
    text_col  = "text_cleaned" if "text_cleaned" in orig.columns else "text"
    title_col = "title"        if "title"        in orig.columns else None

    orig_lookup = orig.set_index("idx")[[title_col, text_col] if title_col else [text_col]]

    def get_full_text(idx: int) -> str:
        row   = orig_lookup.loc[idx]
        title = str(row[title_col] or "") if title_col else ""
        body  = str(row[text_col]  or "")
        return f"{title} [SEP] {body}" if title else body

    if args.max_rows and args.max_rows > 0:
        test_df = test_df.head(args.max_rows).copy()

    print(f"Rows to process: {len(test_df)}")

    # Resume logic
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    done: dict[int, dict] = {}
    if OUT_FILE.exists() and not args.force_relabel:
        prev    = pd.read_csv(OUT_FILE)
        prev_ok = prev[prev["source_note"] != "api_error"]
        for _, row in prev_ok.iterrows():
            done[int(row["idx"])] = row.to_dict()
        print(f"Resuming: {len(done)} done, {len(prev)-len(done)} api_error rows will retry.")
    elif args.force_relabel and OUT_FILE.exists():
        print("Force relabel: ignoring existing output.")

    rows_to_do = [(int(row["idx"]), row) for _, row in test_df.iterrows()
                  if int(row["idx"]) not in done]
    print(f"Remaining: {len(rows_to_do)} rows.")
    if not rows_to_do:
        print("Nothing to do.")
        return

    results: list[dict] = list(done.values())
    lock = threading.Lock()

    def process(item):
        idx, row = item
        text  = get_full_text(idx)
        label = str(row["label"])
        result = call_api(text, label)
        return {"idx": idx, "label": label, **result}

    with tqdm(total=len(rows_to_do), desc="Labeling NER fields", unit="row") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(process, item): item for item in rows_to_do}
            for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
                res = future.result()
                with lock:
                    results.append(res)
                pbar.update(1)

                if i % CHECKPOINT_EVERY == 0:
                    with lock:
                        ck = pd.DataFrame(results).sort_values("idx")
                        for col in ["idx","label"] + ALL_FIELDS + ["source_note"]:
                            if col not in ck.columns: ck[col] = None
                        ck[["idx","label"] + ALL_FIELDS + ["source_note"]].to_csv(OUT_FILE, index=False)

    # Ensure all field columns are present even if some event types weren't processed
    all_cols = ["idx", "label"] + ALL_FIELDS + ["source_note"]
    final_df = pd.DataFrame(results).sort_values("idx")
    for col in all_cols:
        if col not in final_df.columns:
            final_df[col] = None
    final_df[all_cols].to_csv(OUT_FILE, index=False)
    out_df = pd.read_csv(OUT_FILE)
    print(f"\nDone. Saved {len(out_df)} rows → {OUT_FILE}")
    errors = (out_df["source_note"] == "api_error").sum()
    print(f"  api_error rows: {errors}")
    for f in ALL_FIELDS:
        if f in out_df.columns:
            has = out_df[f].notna().sum()
            print(f"  {f:30s}: {has}/{len(out_df)} = {100*has/len(out_df):.1f}% non-null")


if __name__ == "__main__":
    main()
