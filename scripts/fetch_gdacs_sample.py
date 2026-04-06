"""
Fetch sample data from the GDACS API to verify severity training pipeline feasibility.

GDACS provides:
  - alertlevel (green / orange / red)  ← binary classification label
  - eventtype (EQ / TC / WF / DR / FL)
  - structured features per event type (magnitude, wind speed, etc.)

This script fetches ~10 events per alert level for each supported disaster type
and saves samples to data/gdacs_*_sample.csv.

Usage:
    conda run -n gdelt python fetch_gdacs_sample.py

Reference: adapted from /Users/wanghelin/Documents/course/CS5246/pj/scripts/fetch_gdacs_eq_fields.py
"""

import csv
import json
import ssl
import sys
import urllib.request
from pathlib import Path

API_URL = "https://www.gdacs.org/gdacsapi/api/events/geteventlist/SEARCH"
DATA_DIR = Path(__file__).parent / "data"

REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
    )
}

# Fields saved per event type (aligns with training feature columns)
FIELDS = {
    "EQ": ["eventid", "eventtype", "alertlevel", "country", "fromdate",
           "magnitude", "depth", "rapidpopdescription"],
    "TC": ["eventid", "eventtype", "alertlevel", "country", "fromdate",
           "maximum_wind_speed_kmh", "maximum_storm_surge_m", "exposed_population"],
    "WF": ["eventid", "eventtype", "alertlevel", "country", "fromdate",
           "duration_days", "burned_area_ha", "people_affected"],
    "DR": ["eventid", "eventtype", "alertlevel", "country", "fromdate",
           "duration_days", "affected_area_km2", "affected_country_count"],
    "FL": ["eventid", "eventtype", "alertlevel", "country", "fromdate",
           "dead", "displaced"],
}


def _open_url(url: str):
    request = urllib.request.Request(url, headers=REQUEST_HEADERS)
    ctx = ssl._create_unverified_context()
    return urllib.request.urlopen(request, context=ctx, timeout=20)


def fetch_events(event_type: str, alert_level: str, limit: int = 10,
                 fromdate: str = "2023-01-01", todate: str = "2026-01-01") -> list[dict]:
    """Fetch up to `limit` events from GDACS API for a given type and alert level."""
    params = (
        f"eventlist={event_type}"
        f"&fromdate={fromdate}"
        f"&todate={todate}"
        f"&alertlevel={alert_level}"
        f"&pagenumber=0"
    )
    url = f"{API_URL}?{params}"
    try:
        with _open_url(url) as resp:
            doc = json.loads(resp.read().decode("utf-8", errors="replace"))
    except Exception as e:
        print(f"    [ERROR] {e}")
        return []

    rows = []
    for feat in doc.get("features", [])[:limit]:
        p = feat.get("properties", {})
        rows.append({
            "eventid":    str(p.get("eventid", "")),
            "eventtype":  str(p.get("eventtype", "")),
            "alertlevel": str(p.get("alertlevel", "")),
            "country":    str(p.get("country", "")),
            "fromdate":   str(p.get("fromdate", "")),
            # EQ fields
            "magnitude":            str(p.get("magnitude", "")),
            "depth":                str(p.get("depth", "")),
            "rapidpopdescription":  str(p.get("severitydata", {}).get("severitytext", "")
                                        if isinstance(p.get("severitydata"), dict) else ""),
            # TC fields (may be nested or in severitydata)
            "maximum_wind_speed_kmh":  str(p.get("severitydata", {}).get("severity", "")
                                           if p.get("eventtype") == "TC" else ""),
            "maximum_storm_surge_m":   "",
            "exposed_population":      str(p.get("population", {}).get("exposure", "")
                                           if isinstance(p.get("population"), dict) else ""),
            # WF / DR / FL fields — require detail API call; left blank at this stage
            "duration_days":        "",
            "burned_area_ha":       "",
            "people_affected":      "",
            "affected_area_km2":    "",
            "affected_country_count": "",
            "dead":                 "",
            "displaced":            "",
        })
    return rows


def save_sample(event_type: str, rows: list[dict]) -> Path:
    fields = FIELDS[event_type]
    out_path = DATA_DIR / f"gdacs_{event_type.lower()}_sample.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return out_path


def main():
    print("=" * 60)
    print("GDACS API feasibility check")
    print("=" * 60)

    all_pass = True
    results = {}

    for etype in ["EQ", "TC", "WF", "DR", "FL"]:
        print(f"\n[{etype}]")
        combined = []
        for level in ["green", "orange", "red"]:
            rows = fetch_events(etype, level, limit=5)
            print(f"  {level:6s}: {len(rows)} events")
            combined.extend(rows)

        results[etype] = len(combined)

        if combined:
            out = save_sample(etype, combined)
            print(f"  Saved {len(combined)} rows → {out.name}")

            # Quick content check
            levels_found = {r["alertlevel"].lower() for r in combined if r["alertlevel"]}
            print(f"  Alert levels present: {levels_found}")
        else:
            print(f"  [WARN] No data returned for {etype}")
            all_pass = False

    print("\n" + "=" * 60)
    print("Feasibility check results")
    checks = {
        "EQ data retrieved":  results.get("EQ", 0) > 0,
        "TC data retrieved":  results.get("TC", 0) > 0,
        "WF data retrieved":  results.get("WF", 0) > 0,
        "DR data retrieved":  results.get("DR", 0) > 0,
        "FL data retrieved":  results.get("FL", 0) > 0,
        "Multiple alert levels": any(
            len({r["alertlevel"].lower() for r in fetch_events("EQ", "green", 2)
                 + fetch_events("EQ", "orange", 2)}) > 1
            for _ in [1]  # evaluate once
        ),
    }
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        if not ok:
            all_pass = False

    print("\n  Overall:", "ALL PASS — GDACS severity training pipeline is feasible"
          if all_pass else "SOME CHECKS FAILED")


if __name__ == "__main__":
    main()
