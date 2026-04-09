"""
End-to-end Storms 'n' Stocks inference pipeline.

Revised architecture (2026-04-09):
  1. Module A (light)  — per-article: time + location only (no NER)
  2. Module D          — cluster articles into unique events
  3. GDACS match       — look up each cluster in GDACS structured database
       hit  → use GDACS severity (alertlevel) + numeric fields directly
       miss → run NER on cluster's articles → ML severity prediction
  4. Module B          — entity linking → affected industries + sector ETFs
  5. Module E          — event study (CAR) per sector ETF

This design prioritises authoritative GDACS data over noisy news-text NER,
while keeping NER as a graceful fallback for events not in GDACS.

Input CSV (minimum required columns):
    idx, label, text, timestamp
    (optional: location_text, country_iso2, lat, lon, event_date)

Output files:
    data/results/events.csv          — per-event cluster records
    data/results/car_results.csv     — CAR estimates
    data/results/group_analysis.csv  — grouped CAR statistics

Usage:
    python src/pipeline.py --input data/splits/test.csv
    python src/pipeline.py --input data/splits/test.csv --max-rows 50
    python src/pipeline.py --input data/splits/test.csv --skip-stock
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ROOT        = Path(__file__).parent.parent
SPLITS_DIR  = ROOT / "data" / "splits"
RESULTS_DIR = ROOT / "data" / "results"
sys.path.insert(0, str(ROOT / "src"))


def _parse_ts(ts_str: str) -> datetime:
    s = str(ts_str).strip()
    for fmt in ("%Y%m%d%H%M%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return datetime.now()


def _extract_ner(text_trunc: str, label: str, extractor) -> dict:
    """Run NER extractor and return flat metrics dict."""
    ner_result = extractor.extract(text_trunc, label)
    metrics    = ner_result.get("metrics", {})

    def _v(field: str):
        entry = metrics.get(field)
        return entry["value"] if entry else None

    return {
        "low_confidence":           ner_result.get("low_confidence", True),
        "magnitude":                _v("magnitude"),
        "depth":                    _v("depth"),
        "rapidpopdescription":      _v("rapidpopdescription"),
        "rapid_missing":            _v("rapid_missing"),
        "rapid_few_people":         _v("rapid_few_people"),
        "rapid_unparsed":           _v("rapid_unparsed"),
        "maximum_wind_speed_kmh":   _v("maximum_wind_speed_kmh"),
        "maximum_storm_surge_m":    _v("maximum_storm_surge_m"),
        "exposed_population":       _v("exposed_population"),
        "burned_area_ha":           _v("burned_area_ha"),
        "people_affected":          _v("people_affected"),
        "duration_days":            _v("duration_days"),
        "affected_area_km2":        _v("affected_area_km2"),
        "affected_country_count":   _v("affected_country_count"),
        "dead":                     _v("dead"),
        "displaced":                _v("displaced"),
    }


def run_pipeline(
    input_csv: str,
    max_rows: int = 0,
    skip_stock: bool = False,
    verbose: bool = False,
) -> None:
    from location_extractor import extract_location
    from time_extractor import extract_event_time
    from event_clusterer import EventClusterer
    from gdacs_matcher import GDACSmatcher
    from unified_event_extractor import UnifiedEventExtractor
    from severity_predictor import SeverityPredictor
    from entity_linker import EntityLinker

    clusterer  = EventClusterer()
    matcher    = GDACSmatcher()
    extractor  = UnifiedEventExtractor()
    predictor  = SeverityPredictor()
    linker     = EntityLinker()

    gdacs_available = not matcher._df.empty
    print(f"GDACS records loaded: {len(matcher._df)}" if gdacs_available
          else "GDACS data not available — NER fallback will be used for all events")

    # ── Load articles ──────────────────────────────────────────────────────────
    df = pd.read_csv(input_csv)
    if "label" in df.columns:
        df = df[df["label"] != "not_related"].copy()
    if max_rows and max_rows > 0:
        df = df.head(max_rows).copy()
    print(f"Articles to process: {len(df)}")

    # Build article text lookup for NER fallback and Module B
    idx_to_row = {int(row["idx"]): row for _, row in df.iterrows()}

    # ── Module A (light): per-article time + location only ────────────────────
    TEXT_LIMIT = 512
    article_records = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Module A — time+location"):
        raw_text = str(row.get("text", ""))
        if " [SEP] " in raw_text:
            title, body = raw_text.split(" [SEP] ", 1)
        else:
            title, body = "", raw_text

        body_trunc = body[:TEXT_LIMIT]
        label = str(row.get("label", ""))
        ts    = _parse_ts(str(row.get("timestamp", "")))

        loc         = extract_location(body_trunc, title)
        time_result = extract_event_time(body_trunc, title, ts, label)

        article_records.append({
            "idx":           int(row.get("idx", -1)),
            "event_type":    label,
            "event_date":    time_result.event_date,
            "location_text": loc.location_text,
            "country_iso2":  loc.country_iso2,
            "lat":           loc.lat,
            "lon":           loc.lon,
            # NER fields left as None — filled in later if needed
            "low_confidence":           True,
            "magnitude":                None,
            "depth":                    None,
            "rapidpopdescription":      None,
            "rapid_missing":            None,
            "rapid_few_people":         None,
            "rapid_unparsed":           None,
            "maximum_wind_speed_kmh":   None,
            "maximum_storm_surge_m":    None,
            "exposed_population":       None,
            "burned_area_ha":           None,
            "people_affected":          None,
            "duration_days":            None,
            "affected_area_km2":        None,
            "affected_country_count":   None,
            "dead":                     None,
            "displaced":                None,
        })

        if verbose:
            print(f"  [{int(row.get('idx',-1))}] {label} | "
                  f"date={time_result.event_date} | loc={loc.location_text}")

    print(f"Module A complete: {len(article_records)} article records (time+location)")

    # ── Module D: clustering ───────────────────────────────────────────────────
    clusters = clusterer.cluster(article_records)
    print(f"Module D complete: {len(clusters)} unique events (from {len(article_records)} articles)")

    # ── GDACS match + conditional NER ─────────────────────────────────────────
    gdacs_hits = 0
    ner_runs   = 0
    events     = []

    for i, cluster in enumerate(tqdm(clusters, desc="GDACS match / NER fallback")):
        event_id = f"{cluster['event_type'].upper()}_{i:04d}"
        gdacs    = matcher.match(cluster)

        if gdacs:
            # ── GDACS hit: use structured data directly ────────────────────
            gdacs_hits += 1
            ev = {
                "event_id": event_id,
                **cluster,
                **{k: v for k, v in gdacs.items()
                   if k not in ("gdacs_matched", "gdacs_event_id",
                                "gdacs_alertlevel", "predicted_alert",
                                "prob_orange_or_red", "low_confidence")},
                "predicted_alert":    gdacs["predicted_alert"] or "green",
                "prob_orange_or_red": gdacs["prob_orange_or_red"],
                "low_confidence":     gdacs["low_confidence"],
                "severity_source":    "gdacs",
                "gdacs_alertlevel":   gdacs["gdacs_alertlevel"],
            }
        else:
            # ── GDACS miss: run NER on cluster articles, then ML severity ──
            ner_runs += 1
            article_indices = cluster.get("article_indices") or []

            # Aggregate NER over all articles in cluster
            agg: dict = {}
            NER_NUM_FIELDS = [
                "magnitude", "depth", "maximum_wind_speed_kmh",
                "maximum_storm_surge_m", "exposed_population",
                "burned_area_ha", "people_affected", "duration_days",
                "affected_area_km2", "affected_country_count",
                "dead", "displaced",
            ]
            NER_TEXT_FIELDS = ["rapidpopdescription"]
            all_low_conf = []

            for idx in article_indices:
                src_row = idx_to_row.get(idx)
                if src_row is None:
                    continue
                raw_text = str(src_row.get("text", ""))
                if " [SEP] " in raw_text:
                    title, body = raw_text.split(" [SEP] ", 1)
                else:
                    title, body = "", raw_text
                body_trunc = body[:TEXT_LIMIT]
                text_trunc = f"{title} [SEP] {body_trunc}" if title else body_trunc

                ner = _extract_ner(text_trunc, cluster["event_type"], extractor)
                all_low_conf.append(ner.get("low_confidence", True))

                for f in NER_NUM_FIELDS:
                    v = ner.get(f)
                    if v is not None:
                        agg[f] = max(agg.get(f) or float("-inf"), float(v))
                for f in NER_TEXT_FIELDS:
                    if ner.get(f) is not None and f not in agg:
                        agg[f] = ner[f]
                for f in ["rapid_missing", "rapid_few_people", "rapid_unparsed"]:
                    if ner.get(f) is not None:
                        agg[f] = ner[f]

            # Replace -inf sentinels with None
            agg = {k: (v if v != float("-inf") else None) for k, v in agg.items()}
            low_conf = all(all_low_conf) if all_low_conf else True

            merged = {**cluster, **agg, "low_confidence": low_conf}
            pred   = predictor.predict(merged)

            ev = {
                "event_id": event_id,
                **merged,
                "predicted_alert":    pred["predicted_alert"],
                "prob_orange_or_red": pred.get("prob_orange_or_red"),
                "low_confidence":     pred.get("low_confidence", low_conf),
                "severity_source":    "ml",
                "gdacs_alertlevel":   None,
            }

        events.append(ev)

    print(f"Severity resolved: {gdacs_hits} via GDACS, {ner_runs} via NER+ML "
          f"({100*gdacs_hits/len(events):.1f}% GDACS coverage)")

    # ── Module B: entity linking ───────────────────────────────────────────────
    idx_to_text = {idx: str(r.get("text", "")) for idx, r in idx_to_row.items()}
    for ev in events:
        ev["article_texts"] = [
            idx_to_text.get(i, "") for i in (ev.get("article_indices") or [])
        ]
        link = linker.link(ev)
        ev.update({
            "country_iso2":      link["country_iso2"],
            "country_name":      link["country_name"],
            "key_industries":    link["key_industries"],
            "sector_etfs":       link["sector_etfs"],
            "sectors_from_kb":   link["sectors_from_kb"],
            "sectors_from_text": link["sectors_from_text"],
        })
        ev.pop("article_texts", None)

    linked = sum(1 for ev in events if ev.get("sector_etfs"))
    print(f"Module B complete: {linked}/{len(events)} events have sector ETFs")

    # Save events
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    events_df  = pd.DataFrame(events)
    events_path = RESULTS_DIR / "events.csv"
    events_df.to_csv(events_path, index=False)
    print(f"Events saved → {events_path}")

    if skip_stock:
        print("Skipping Module E (--skip-stock)")
        return

    # ── Module E: stock impact analysis ───────────────────────────────────────
    from stock_analyser import StockAnalyser
    analyser = StockAnalyser()

    tradeable = [ev for ev in events if ev.get("sector_etfs") and ev.get("event_date")]
    print(f"Module E: {len(tradeable)} events with ticker + date for CAR computation")

    if tradeable:
        car_df   = analyser.compute_car_batch(tradeable)
        car_path = RESULTS_DIR / "car_results.csv"
        car_df.to_csv(car_path, index=False)
        print(f"CAR results saved → {car_path}  "
              f"(success: {car_df['error'].isna().sum()}/{len(car_df)})")

        group_df   = analyser.group_analysis(car_df, events_df)
        group_path = RESULTS_DIR / "group_analysis.csv"
        group_df.to_csv(group_path, index=False)
        print(f"Group analysis saved → {group_path}")
        print("\nGroup-level CAR summary:")
        print(group_df.to_string(index=False))
    else:
        print("No events with both ticker and event_date — skipping CAR computation.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Storms 'n' Stocks inference pipeline")
    parser.add_argument("--input",      default=str(SPLITS_DIR / "test.csv"),
                        help="Input CSV with classified articles")
    parser.add_argument("--max-rows",   type=int, default=0,
                        help="Process only first N rows (0 = all)")
    parser.add_argument("--skip-stock", action="store_true",
                        help="Skip Module E (stock analysis)")
    parser.add_argument("--verbose",    action="store_true")
    args = parser.parse_args()

    run_pipeline(
        input_csv  = args.input,
        max_rows   = args.max_rows,
        skip_stock = args.skip_stock,
        verbose    = args.verbose,
    )
