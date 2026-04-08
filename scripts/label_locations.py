"""
LLM-based location labeling for GT evaluation data.

Reads data/splits/test.csv and asks DeepSeek-V3 to extract the main
disaster location from each article. Output is used as ground truth
for evaluating the rule-based location extractor.

Output: data/llm_labels/location_labels_test.csv
Columns: idx, label, location_text, country_iso2, lat, lon, source_note

Usage:
    python scripts/label_locations.py
    python scripts/label_locations.py --workers 2
"""

import argparse
import concurrent.futures
import re
import threading
import time
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

API_KEY = "sk-jbarivjnimgypzwvaxnzpgbccuuhsxeddhwgwliwoewhgast"
API_URL = "https://api.siliconflow.cn/v1/chat/completions"
MODEL   = "deepseek-ai/DeepSeek-V3"

SPLITS_DIR = Path(__file__).parent.parent / "data" / "splits"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "llm_labels"
OUT_FILE   = OUTPUT_DIR / "location_labels_test.csv"

CHECKPOINT_EVERY = 50

SYSTEM_PROMPT = (
    "You are a geographic information extractor for disaster news articles. "
    "Your task is to identify where the disaster physically occurred. "
    "Always respond in the exact structured format requested."
)

USER_TEMPLATE = """\
Extract the primary location where the disaster event physically occurred.

Rules:
- location_text: the most specific place name mentioned (city, region, or country). \
Use the place where the disaster happened, NOT where the news agency is located. \
If a dateline like "TOKYO (Reuters)" appears but the event happened elsewhere, ignore Tokyo.
- country_iso2: the ISO 3166-1 alpha-2 country code (e.g. US, JP, AU). \
If the event spans multiple countries, give the most affected one.
- lat/lon: decimal degrees if you can determine them confidently from the article text \
(e.g. coordinates explicitly mentioned, or a well-known city). \
Use positive for N/E, negative for S/W. Output N/A if uncertain.

Disaster type: {label}
Article (title + snippet):
{text}

Respond ONLY in this exact format (no extra text):
LOCATION: <place name or N/A>
COUNTRY: <ISO-2 code or N/A>
LAT: <decimal or N/A>
LON: <decimal or N/A>
NOTE: <one sentence explaining your reasoning>"""


def call_api(text: str, label: str, max_retries: int = 5):
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": USER_TEMPLATE.format(
                label=label, text=text[:600])},
        ],
        "max_tokens": 120,
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
            return _parse(raw)
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(5 * (attempt + 1))
    return {"location_text": "error", "country_iso2": "error",
            "lat": None, "lon": None, "source_note": "api_error"}


def _parse(raw: str) -> dict:
    def get(pattern, text):
        m = re.search(pattern, text, re.IGNORECASE)
        return m.group(1).strip() if m else None

    location = get(r"^LOCATION:\s*(.+)$", raw, ) or get(r"LOCATION:\s*(.+)", raw)
    country  = get(r"^COUNTRY:\s*(.+)$",  raw) or get(r"COUNTRY:\s*(.+)",  raw)
    lat_raw  = get(r"^LAT:\s*(.+)$",      raw) or get(r"LAT:\s*(.+)",      raw)
    lon_raw  = get(r"^LON:\s*(.+)$",      raw) or get(r"LON:\s*(.+)",      raw)
    note     = get(r"^NOTE:\s*(.+)$",     raw) or get(r"NOTE:\s*(.+)",     raw)

    def to_float(s):
        if not s or s.strip().upper() in ("N/A", "NA", "NONE", ""):
            return None
        try:
            return round(float(re.sub(r"[^\d.\-]", "", s)), 4)
        except Exception:
            return None

    def clean(s):
        if not s or s.strip().upper() in ("N/A", "NA", "NONE"):
            return None
        # ISO-2: uppercase, strip noise
        return s.strip()

    loc = clean(location)
    cty = clean(country)
    if cty:
        cty = cty.upper()[:2]  # ensure ISO-2

    return {
        "location_text": loc or "error",
        "country_iso2":  cty or "error",
        "lat":           to_float(lat_raw),
        "lon":           to_float(lon_raw),
        "source_note":   (note or "").strip()[:200],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(SPLITS_DIR / "test.csv")
    print(f"Loaded {len(df)} rows from test.csv")

    # Resume: skip already-labeled rows (non-error only)
    done: dict[int, dict] = {}
    if OUT_FILE.exists():
        prev = pd.read_csv(OUT_FILE)
        prev_ok = prev[prev["location_text"] != "error"]
        for _, row in prev_ok.iterrows():
            done[row["idx"]] = row.to_dict()
        print(f"Resuming: {len(done)} done, {len(prev)-len(done)} errors will retry.")

    rows_to_do = [(row["idx"], row) for _, row in df.iterrows() if row["idx"] not in done]
    print(f"Remaining: {len(rows_to_do)} rows.")

    if not rows_to_do:
        print("Nothing to do.")
        return

    results: list[dict] = list(done.values())
    lock = threading.Lock()

    def process(item):
        idx, row = item
        result = call_api(str(row["text"]), str(row["label"]))
        return {"idx": idx, "label": row["label"], **result}

    with tqdm(total=len(rows_to_do), desc="Labeling locations", unit="row") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(process, item): item for item in rows_to_do}
            for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
                res = future.result()
                with lock:
                    results.append(res)
                pbar.update(1)
                pbar.set_postfix({"loc": res["location_text"][:20]})

                if i % CHECKPOINT_EVERY == 0:
                    with lock:
                        pd.DataFrame(results).sort_values("idx").to_csv(OUT_FILE, index=False)

    pd.DataFrame(results).sort_values("idx").to_csv(OUT_FILE, index=False)
    df_out = pd.read_csv(OUT_FILE)
    print(f"\nDone. Saved {len(df_out)} rows → {OUT_FILE}")

    errors = (df_out["location_text"] == "error").sum()
    na_loc = df_out["location_text"].isna().sum()
    has_lat = df_out["lat"].notna().sum()
    has_country = (df_out["country_iso2"] != "error") & df_out["country_iso2"].notna()
    print(f"  errors:       {errors}")
    print(f"  no location:  {na_loc}")
    print(f"  has lat/lon:  {has_lat} ({100*has_lat/len(df_out):.1f}%)")
    print(f"  has country:  {has_country.sum()} ({100*has_country.sum()/len(df_out):.1f}%)")


if __name__ == "__main__":
    main()
