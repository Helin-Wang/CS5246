"""
LLM-based event type labeling for Stage 2 classifier training data.

Reads training_events_gdelt.xlsx, labels each article with one of:
  earthquake / flood / cyclone / wildfire / drought / not_related

Uses SiliconFlow DeepSeek-R1-Distill-Qwen-7B (OpenAI-compatible API).

Usage:
    # Label rows 0–999
    conda run -n gdelt python label_event_types.py --start 0 --end 1000

    # Label rows 1000–1999 in parallel in another terminal
    conda run -n gdelt python label_event_types.py --start 1000 --end 2000

    # Merge all output files afterwards
    conda run -n gdelt python label_event_types.py --merge

Output: data/llm_labels_<start>_<end>.csv
Columns: idx, url, old_event_type, llm_event_type
"""

import argparse
import concurrent.futures
import os
import re
import time
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
API_KEY = "sk-pnfnnlrdscgcemyjjhqtseekdcozuqylngsjcpenhpiqiujl"
API_URL = "https://api.siliconflow.cn/v1/chat/completions"
MODEL   = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

DATA_FILE  = Path(__file__).parent.parent / "data" / "training_events_gdelt.xlsx"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "llm_labels"

VALID_LABELS = {"earthquake", "flood", "cyclone", "wildfire", "drought", "not_related"}

SYSTEM_PROMPT = (
    "You are a news article classifier. "
    "Classify the article into exactly one category. "
    "Respond with ONLY the single category label — no explanation, no punctuation."
)

USER_TEMPLATE = """\
Categories:
- earthquake : earthquakes, tremors, seismic activity, aftershocks
- flood       : floods, flooding, flash floods, inundation
- cyclone     : tropical cyclones, hurricanes, typhoons, tropical storms
- wildfire    : wildfires, forest fires, bushfires
- drought     : droughts, water shortages, dry spells
- not_related : anything else — including volcano/eruption events, political news,
                human-interest stories, financial articles, or articles that only
                mention disasters briefly without being primarily about them

Title: {title}
Text (first 400 chars): {snippet}

Category:"""


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------
def call_api(title: str, snippet: str, max_retries: int = 3) -> str:
    """Call DeepSeek via SiliconFlow. Returns one of VALID_LABELS or 'error'."""
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": USER_TEMPLATE.format(title=title, snippet=snippet)},
        ],
        "max_tokens": 128,
        "temperature": 0.0,
    }
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(API_URL, json=payload, headers=headers, timeout=60)
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"].strip()
            return _parse_label(raw)
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return "error"
    return "error"


def _parse_label(raw: str) -> str:
    """Extract the final answer label from model output.

    DeepSeek-R1-Distill outputs a reasoning paragraph followed by the answer.
    Strategy: find the LAST occurrence of any valid label (the conclusion, not
    the mid-reasoning mentions), with a fixed priority order to break ties.
    """
    # Strip explicit <think>…</think> blocks if present
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip().lower()

    # Exact match (ideal case — model followed instructions perfectly)
    if cleaned in VALID_LABELS:
        return cleaned

    # Fixed-order label list (most specific first to avoid partial matches)
    ordered = ["not_related", "earthquake", "wildfire", "cyclone", "flood", "drought"]

    # Find the LAST position of each label in the text; pick the one that appears latest
    last_pos: dict[str, int] = {}
    for label in ordered:
        pos = cleaned.rfind(label)
        if pos != -1:
            last_pos[label] = pos

    if last_pos:
        return max(last_pos, key=last_pos.__getitem__)

    return "error"


# ---------------------------------------------------------------------------
# Main labeling logic
# ---------------------------------------------------------------------------
def label_range(start: int, end: int, workers: int, sample: int | None = None) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data slice
    df_full = pd.read_excel(DATA_FILE)
    end = min(end, len(df_full))
    df = df_full.iloc[start:end].copy()

    # Drop volcano — not a supported disaster type in our pipeline
    before = len(df)
    df = df[df["event_type"] != "volcano"].copy()
    if len(df) < before:
        print(f"Dropped {before - len(df)} volcano rows. Remaining: {len(df)}")

    # Uniform per-class sampling: each event_type gets at most `sample` rows.
    if sample is not None:
        parts = []
        for etype, group in df.groupby("event_type"):
            parts.append(group.sample(min(len(group), sample), random_state=42))
        df = pd.concat(parts).sample(frac=1, random_state=42)  # shuffle, preserve original index
        counts = df["event_type"].value_counts().to_dict()
        print(f"Sampled {len(df)} rows (uniform per-class, cap={sample}): {counts}")

    tag = f"{start}_{end}" + (f"_s{sample}" if sample else "")
    out_file = OUTPUT_DIR / f"llm_labels_{tag}.csv"
    print(f"Rows {start}–{end-1} ({len(df)} rows). Output → {out_file}")

    # Resume: skip already-labeled rows
    done: dict[int, str] = {}
    if out_file.exists():
        prev = pd.read_csv(out_file)
        done = dict(zip(prev["idx"], prev["llm_event_type"]))
        print(f"Resuming: {len(done)} rows already labeled.")

    rows_to_do = [(idx, row) for idx, row in df.iterrows() if idx not in done]
    print(f"Remaining: {len(rows_to_do)} rows to label.")

    if not rows_to_do:
        print("Nothing to do.")
        return

    results: list[dict] = [{"idx": k, "url": "", "old_event_type": "", "llm_event_type": v}
                           for k, v in done.items()]
    lock = __import__("threading").Lock()

    def process(item):
        idx, row = item
        title   = str(row.get("title", "") or "")
        text    = str(row.get("text_cleaned", "") or "")
        snippet = text[:400]
        label   = call_api(title, snippet)
        return {"idx": idx, "url": str(row.get("url", "")),
                "old_event_type": str(row.get("event_type", "")),
                "llm_event_type": label}

    checkpoint_every = 50
    with tqdm(total=len(rows_to_do), desc=f"Labeling [{start}:{end}]", unit="row") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(process, item): item for item in rows_to_do}
            for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
                res = future.result()
                with lock:
                    results.append(res)
                pbar.update(1)
                pbar.set_postfix({"last": res["llm_event_type"]})

                # Checkpoint
                if i % checkpoint_every == 0:
                    with lock:
                        pd.DataFrame(results).sort_values("idx").to_csv(out_file, index=False)

    # Final save
    pd.DataFrame(results).sort_values("idx").to_csv(out_file, index=False)
    print(f"Done. Saved {len(results)} rows → {out_file}")

    # Stats
    labels_series = pd.Series([r["llm_event_type"] for r in results])
    print("\nLabel distribution:")
    print(labels_series.value_counts().to_string())
    error_n = (labels_series == "error").sum()
    if error_n:
        print(f"\nWARN: {error_n} rows labeled 'error' — consider re-running this range.")


# ---------------------------------------------------------------------------
# Merge all output files
# ---------------------------------------------------------------------------
def merge_outputs() -> None:
    files = sorted(OUTPUT_DIR.glob("llm_labels_*.csv"))
    if not files:
        print("No label files found in", OUTPUT_DIR)
        return
    dfs = [pd.read_csv(f) for f in files]
    merged = pd.concat(dfs, ignore_index=True).drop_duplicates("idx").sort_values("idx")
    out = OUTPUT_DIR / "llm_labels_all.csv"
    merged.to_csv(out, index=False)
    print(f"Merged {len(files)} files → {out}  ({len(merged)} rows)")
    print("\nLabel distribution:")
    print(merged["llm_event_type"].value_counts().to_string())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="LLM event-type labeling for GDELT articles")
    sub = parser.add_subparsers(dest="cmd")

    label_p = sub.add_parser("label", help="Label a row range (default command)")
    label_p.add_argument("--start",   type=int, required=True, help="Start row index (inclusive)")
    label_p.add_argument("--end",     type=int, required=True, help="End row index (exclusive)")
    label_p.add_argument("--workers", type=int, default=4,     help="Parallel API workers (default 4)")
    label_p.add_argument("--sample",  type=int, default=None,  help="Per-class row cap: each event_type gets at most this many rows (volcano kept in full)")

    sub.add_parser("merge", help="Merge all partial output files into llm_labels_all.csv")

    # Also support flat args without subcommand for convenience:
    #   python label_event_types.py --start 0 --end 26326
    parser.add_argument("--start",   type=int)
    parser.add_argument("--end",     type=int)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--sample",  type=int, default=None)
    parser.add_argument("--merge",   action="store_true")

    args = parser.parse_args()

    if args.merge or args.cmd == "merge":
        merge_outputs()
    elif args.start is not None and args.end is not None:
        label_range(args.start, args.end, args.workers, args.sample)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
