"""
LLM-based time labeling for non-not_related test news.

Reads data/llm_labels/time_labels_test_input.csv and asks DeepSeek-V3 to extract
disaster event time information.

Output: data/llm_labels/time_labels_test.csv
Columns:
  idx, label, event_occurrence_date, granularity, date_range_end,
  duration_days, time_type, source_note

Note:
- `event_occurrence_date` stores raw time expression from article
  (e.g., "Thursday", "last month", "July 30"), not normalized date.
- No date inference should be performed by LLM.

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

VALID_GRANULARITY = {"day", "week", "month", "year", "unknown", "n/a"}
VALID_TIME_TYPE = {"event_date", "date_range", "duration_only", "non_event_time", "unknown", "n/a"}

SYSTEM_PROMPT = (
    "You are a temporal information extractor for disaster news. "
    "Extract the primary disaster event time, not publication time. "
    "Always respond in the exact format requested."
)

USER_TEMPLATE = """\
Extract time information for the PRIMARY disaster event in this article.

Rules:
- event_occurrence_date: copy the original time expression from the article for when the disaster occurred or started.
  Examples: "Thursday", "last month", "July 30", "Aug. 1, 2025", "since May".
- Do NOT infer or convert to a precise calendar date if the article is relative/ambiguous.
- granularity: one of day/week/month/year/unknown.
- date_range_end: raw end-time expression if a range is explicitly stated, else N/A.
- duration_days: numeric days if the article states duration (e.g., "for two weeks" -> 14). Else N/A.
- time_type:
  - event_date      : a single event date is clear
  - date_range      : article indicates start/end period
  - duration_only   : only duration is given, start date unclear
  - non_event_time  : time mentions are forecast/background/policy and not main event time
  - unknown         : insufficient evidence
- Ignore publication time/dateline.
- Do NOT use article timestamp to infer missing year/day.

Disaster type: {label}
Article timestamp (metadata only, do not infer from it): {timestamp}
Article (title + snippet):
{text}

Respond ONLY in this exact format (no extra text):
EVENT_DATE: <raw expression or N/A>
GRANULARITY: <day|week|month|year|unknown>
DATE_RANGE_END: <raw expression or N/A>
DURATION_DAYS: <number or N/A>
TIME_TYPE: <event_date|date_range|duration_only|non_event_time|unknown>
NOTE: <one sentence reasoning>"""


def _normalize_time_text(value: str | None) -> str | None:
    if not value:
        return None
    v = value.strip()
    if v.upper() in {"N/A", "NA", "NONE", ""}:
        return None
    v = re.sub(r"\s+", " ", v)
    # Guardrail: if model appends inferred calendar date in parentheses after weekday,
    # keep only the original weekday expression.
    weekday_prefix = re.match(
        r"^(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b",
        v,
        flags=re.IGNORECASE,
    )
    if weekday_prefix and "(" in v and ")" in v:
        v = weekday_prefix.group(1)
    return v


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


def _normalize_duration(value: str | None) -> float | None:
    if not value:
        return None
    v = value.strip()
    if v.upper() in {"N/A", "NA", "NONE", ""}:
        return None
    try:
        cleaned = re.sub(r"[^0-9.\-]", "", v)
        if not cleaned:
            return None
        return round(float(cleaned), 2)
    except Exception:
        return None


def _parse(raw: str) -> dict:
    def get(pattern: str, text: str):
        m = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        return m.group(1).strip() if m else None

    event_date = get(r"^EVENT_DATE:\s*(.+)$", raw) or get(r"EVENT_DATE:\s*(.+)", raw)
    granularity = get(r"^GRANULARITY:\s*(.+)$", raw) or get(r"GRANULARITY:\s*(.+)", raw)
    date_range_end = get(r"^DATE_RANGE_END:\s*(.+)$", raw) or get(r"DATE_RANGE_END:\s*(.+)", raw)
    duration = get(r"^DURATION_DAYS:\s*(.+)$", raw) or get(r"DURATION_DAYS:\s*(.+)", raw)
    time_type = get(r"^TIME_TYPE:\s*(.+)$", raw) or get(r"TIME_TYPE:\s*(.+)", raw)
    note = get(r"^NOTE:\s*(.+)$", raw) or get(r"NOTE:\s*(.+)", raw)

    return {
        "event_occurrence_date": _normalize_time_text(event_date),
        "granularity": _normalize_granularity(granularity),
        "date_range_end": _normalize_time_text(date_range_end),
        "duration_days": _normalize_duration(duration),
        "time_type": _normalize_time_type(time_type),
        "source_note": (note or "").strip()[:220],
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
                    text=text[:800],
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
        "event_occurrence_date": None,
        "granularity": "unknown",
        "date_range_end": None,
        "duration_days": None,
        "time_type": "unknown",
        "source_note": "api_error",
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
                        "date": (res["event_occurrence_date"] or "N/A"),
                        "type": res["time_type"],
                    }
                )

                if i % CHECKPOINT_EVERY == 0:
                    with lock:
                        pd.DataFrame(results).sort_values("idx").to_csv(OUT_FILE, index=False)

    pd.DataFrame(results).sort_values("idx").to_csv(OUT_FILE, index=False)
    out_df = pd.read_csv(OUT_FILE)

    print(f"\nDone. Saved {len(out_df)} rows -> {OUT_FILE}")
    has_date = out_df["event_occurrence_date"].notna().sum()
    has_range_end = out_df["date_range_end"].notna().sum()
    has_duration = out_df["duration_days"].notna().sum()
    api_errors = (out_df["source_note"] == "api_error").sum()
    print(f"  has event date: {has_date} ({100 * has_date / len(out_df):.1f}%)")
    print(f"  has range end:  {has_range_end} ({100 * has_range_end / len(out_df):.1f}%)")
    print(f"  has duration:   {has_duration} ({100 * has_duration / len(out_df):.1f}%)")
    print(f"  api_error rows: {api_errors}")


if __name__ == "__main__":
    main()
