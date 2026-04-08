"""
LLM-based time labeling for non-not_related test news.

Reads data/llm_labels/time_labels_test_input.csv and asks DeepSeek-V3 to extract
disaster event time information.

Output: data/llm_labels/time_labels_test.csv
Columns:
  idx, label, event_date_iso, event_date_raw, granularity,
  time_type, source_note

Note:
- `event_date_iso`: LLM-inferred YYYY-MM-DD (uses article timestamp as reference
  for weekday/relative expressions). N/A when genuinely undeterminable.
- `event_date_raw`: verbatim time expression copied from article text.
- No week-level granularity; types simplified to event_date/date_range/duration_only/unknown.

Usage:
  python scripts/label_times.py
  python scripts/label_times.py --workers 2
  python scripts/label_times.py --max-rows 50
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
MODEL = "deepseek-ai/DeepSeek-V3"

INPUT_FILE = Path(__file__).parent.parent / "data" / "llm_labels" / "time_labels_test_input.csv"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "llm_labels"
OUT_FILE = OUTPUT_DIR / "time_labels_test.csv"

CHECKPOINT_EVERY = 50

VALID_GRANULARITY = {"day", "month", "year", "unknown", "n/a"}
VALID_TIME_TYPE = {"event_date", "date_range", "duration_only", "unknown", "n/a"}

SYSTEM_PROMPT = (
    "You are a temporal information extractor for disaster news. "
    "Extract the primary disaster event time, not the publication time. "
    "Always respond in the exact format requested."
)

USER_TEMPLATE = """\
Extract the time when the PRIMARY disaster event occurred or started.

Rules:
- event_date_raw: copy the verbatim time expression from the article (e.g. "Thursday", "last month", "July 30", "since May"). N/A if no time expression exists.
- event_date_iso: infer the calendar date in YYYY-MM-DD format using the article timestamp as reference.
  - Weekday expressions ("Thursday"): resolve relative to the article timestamp.
  - Relative expressions ("yesterday", "two days ago"): subtract from the article timestamp.
  - Month-only expressions ("since March"): use YYYY-MM-01; infer year from timestamp context.
  - Only output N/A if the date is genuinely impossible to infer (e.g., "years ago" with no further clues).
- granularity: precision of the inferred date — day / month / year / unknown.
- time_type:
  - event_date    : single event date or start of event is identifiable
  - date_range    : article indicates a start-to-end period
  - duration_only : only a duration is given, no start date inferable
  - unknown       : no usable time information for the event
- Ignore datelines and publication timestamps.
- Do NOT output week as granularity.

Disaster type: {label}
Article timestamp (use as reference for relative time resolution): {timestamp}
Article (title + snippet):
{text}

Respond ONLY in this exact format (no extra text):
EVENT_DATE_RAW: <verbatim expression or N/A>
EVENT_DATE_ISO: <YYYY-MM-DD or N/A>
GRANULARITY: <day|month|year|unknown>
TIME_TYPE: <event_date|date_range|duration_only|unknown>
NOTE: <one sentence reasoning>"""


def _normalize_raw(value: str | None) -> str | None:
    if not value:
        return None
    v = value.strip()
    if v.upper() in {"N/A", "NA", "NONE", ""}:
        return None
    return re.sub(r"\s+", " ", v)


def _normalize_iso(value: str | None) -> str | None:
    if not value:
        return None
    v = value.strip()
    if v.upper() in {"N/A", "NA", "NONE", ""}:
        return None
    # Accept YYYY-MM-DD or YYYY-MM
    if re.match(r"^\d{4}-\d{2}(-\d{2})?$", v):
        return v
    return None


def _normalize_granularity(value: str | None) -> str:
    if not value:
        return "unknown"
    v = value.strip().lower()
    if v in VALID_GRANULARITY:
        return "unknown" if v == "n/a" else v
    return "unknown"


def _normalize_time_type(value: str | None) -> str:
    if not value:
        return "unknown"
    v = value.strip().lower()
    if v in VALID_TIME_TYPE:
        return "unknown" if v == "n/a" else v
    return "unknown"


def _parse(raw: str) -> dict:
    def get(pattern: str, text: str):
        m = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        return m.group(1).strip() if m else None

    event_date_raw = get(r"^EVENT_DATE_RAW:\s*(.+)$", raw) or get(r"EVENT_DATE_RAW:\s*(.+)", raw)
    event_date_iso = get(r"^EVENT_DATE_ISO:\s*(.+)$", raw) or get(r"EVENT_DATE_ISO:\s*(.+)", raw)
    granularity    = get(r"^GRANULARITY:\s*(.+)$", raw) or get(r"GRANULARITY:\s*(.+)", raw)
    time_type      = get(r"^TIME_TYPE:\s*(.+)$", raw) or get(r"TIME_TYPE:\s*(.+)", raw)
    note           = get(r"^NOTE:\s*(.+)$", raw) or get(r"NOTE:\s*(.+)", raw)

    return {
        "event_date_raw": _normalize_raw(event_date_raw),
        "event_date_iso": _normalize_iso(event_date_iso),
        "granularity":    _normalize_granularity(granularity),
        "time_type":      _normalize_time_type(time_type),
        "source_note":    (note or "").strip()[:220],
    }


def call_api(text: str, label: str, timestamp: str, max_retries: int = 5) -> dict:
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_TEMPLATE.format(
                    label=label,
                    timestamp=timestamp,
                    text=text[:1200],
                ),
            },
        ],
        "max_tokens": 180,
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
    return {
        "event_date_raw": None,
        "event_date_iso": None,
        "granularity":    "unknown",
        "time_type":      "unknown",
        "source_note":    "api_error",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-rows", type=int, default=0, help="Optional cap for quick trial; 0 means all rows.")
    parser.add_argument(
        "--force-relabel",
        action="store_true",
        help="Ignore existing output and relabel selected rows from scratch.",
    )
    args = parser.parse_args()

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_FILE}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INPUT_FILE)

    if args.max_rows and args.max_rows > 0:
        df = df.head(args.max_rows).copy()

    print(f"Loaded {len(df)} rows from {INPUT_FILE.name}")

    done: dict[int, dict] = {}
    if OUT_FILE.exists() and not args.force_relabel:
        prev = pd.read_csv(OUT_FILE)
        prev_ok = prev[prev["source_note"] != "api_error"]
        for _, row in prev_ok.iterrows():
            done[int(row["idx"])] = row.to_dict()
        print(f"Resuming: {len(done)} done, {len(prev) - len(done)} api_error rows will retry.")
    elif args.force_relabel and OUT_FILE.exists():
        print("Force relabel enabled: existing output will be ignored for this run.")

    rows_to_do = [(int(row["idx"]), row) for _, row in df.iterrows() if int(row["idx"]) not in done]
    print(f"Remaining: {len(rows_to_do)} rows.")
    if not rows_to_do:
        print("Nothing to do.")
        return

    results: list[dict] = list(done.values())
    lock = threading.Lock()

    def process(item):
        idx, row = item
        result = call_api(
            text=str(row["text"]),
            label=str(row["label"]),
            timestamp=str(row["timestamp"]),
        )
        return {"idx": idx, "label": row["label"], **result}

    with tqdm(total=len(rows_to_do), desc="Labeling times", unit="row") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(process, item): item for item in rows_to_do}
            for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
                res = future.result()
                with lock:
                    results.append(res)
                pbar.update(1)
                pbar.set_postfix(
                    {
                        "iso":  (res["event_date_iso"] or "N/A"),
                        "type": res["time_type"],
                    }
                )

                if i % CHECKPOINT_EVERY == 0:
                    with lock:
                        pd.DataFrame(results).sort_values("idx").to_csv(OUT_FILE, index=False)

    pd.DataFrame(results).sort_values("idx").to_csv(OUT_FILE, index=False)
    out_df = pd.read_csv(OUT_FILE)

    print(f"\nDone. Saved {len(out_df)} rows -> {OUT_FILE}")
    has_raw  = out_df["event_date_raw"].notna().sum()
    has_iso  = out_df["event_date_iso"].notna().sum()
    api_errors = (out_df["source_note"] == "api_error").sum()
    print(f"  has raw expression: {has_raw} ({100 * has_raw / len(out_df):.1f}%)")
    print(f"  has ISO date:       {has_iso} ({100 * has_iso / len(out_df):.1f}%)")
    print(f"  api_error rows:     {api_errors}")


if __name__ == "__main__":
    main()
