"""
Unified GDACS fetcher for severity training data.

This script fetches EQ/TC/WF/DR/FL events from GDACS API in one run, enriches
type-specific fields (including detail endpoint extraction), and saves a
single merged CSV file.

Notes:
- FL (Flood) is now supported. Fields extracted: dead, displaced (from
  severitydata), plus alertlevel/country/dates for matching.
- Volcano (VO) is not supported.
"""

import argparse
import csv
import json
import random
import ssl
import sys
import time
import urllib.parse
import urllib.request
from urllib.error import HTTPError, URLError
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional

API_URL = "https://www.gdacs.org/gdacsapi/api/events/geteventlist/SEARCH"
BASE_DIR = Path(__file__).resolve().parents[1]
REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    )
}

SUPPORTED_TYPES = ("EQ", "TC", "WF", "DR", "FL")
DEFAULT_ALERT_LEVELS = "green;orange;red"

CSV_FIELDS = [
    "eventid",
    "eventtype",
    "alertlevel",
    "country",
    "fromdate",
    "todate",
    "magnitude",
    "depth",
    "rapidpopdescription",
    "maximum_wind_speed_kmh",
    "maximum_storm_surge_m",
    "exposed_population",
    "duration_days",
    "burned_area_ha",
    "people_affected",
    "affected_area_km2",
    "affected_country_count",
    "dead",
    "displaced",
    "severity_level",
    "severity_text",
]

SEVERITY_LEVEL_MAP = {
    "minor": 1,
    "medium": 2,
    "severe": 3,
    "extreme": 4,
}

REQUEST_SLEEP_SEC = 0.0
RETRY_ATTEMPTS = 5
RETRY_BASE_SEC = 1.0
RETRY_MAX_SEC = 20.0
MAX_PAGE_FAILURES = 6


def _open_url(url: str):
    if REQUEST_SLEEP_SEC > 0:
        time.sleep(REQUEST_SLEEP_SEC)
    req = urllib.request.Request(url, headers=REQUEST_HEADERS)
    ctx = ssl._create_unverified_context()
    return urllib.request.urlopen(req, context=ctx, timeout=30)


def fetch_json_url(url: str):
    with _open_url(url) as resp:
        payload = resp.read().decode("utf-8", errors="replace")
    return json.loads(payload)


def _compute_backoff(attempt_index: int) -> float:
    base = min(RETRY_BASE_SEC * (2 ** attempt_index), RETRY_MAX_SEC)
    jitter = random.uniform(0, 0.5)
    return base + jitter


def fetch_json_with_retry(url: str, attempts: Optional[int] = None):
    max_attempts = attempts if attempts is not None else RETRY_ATTEMPTS
    last_exc = None
    for i in range(max_attempts):
        try:
            return fetch_json_url(url)
        except HTTPError as exc:  # pragma: no cover
            last_exc = exc
            # 404 is usually permanent for this request, no need to retry.
            if exc.code == 404:
                break
            if i < max_attempts - 1:
                wait_sec = _compute_backoff(i)
                print(f"[warn] HTTP {exc.code}, retry in {wait_sec:.1f}s: {url}")
                time.sleep(wait_sec)
        except (URLError, TimeoutError, json.JSONDecodeError, ValueError) as exc:  # pragma: no cover
            last_exc = exc
            if i < max_attempts - 1:
                wait_sec = _compute_backoff(i)
                print(f"[warn] transient fetch error, retry in {wait_sec:.1f}s: {exc}")
                time.sleep(wait_sec)
        except Exception as exc:  # pragma: no cover
            last_exc = exc
            if i < max_attempts - 1:
                wait_sec = _compute_backoff(i)
                print(f"[warn] unexpected fetch error, retry in {wait_sec:.1f}s: {exc}")
                time.sleep(wait_sec)
    if last_exc:
        raise last_exc
    raise RuntimeError("Unknown fetch error")


def parse_levels(level_text: str) -> List[str]:
    levels = [x.strip().lower() for x in level_text.split(";") if x.strip()]
    return levels or ["green", "orange", "red"]


def parse_types(types_text: str) -> List[str]:
    parsed = [x.strip().upper() for x in types_text.split(",") if x.strip()]
    out = [x for x in parsed if x in SUPPORTED_TYPES]
    if not out:
        return list(SUPPORTED_TYPES)
    return out


def _safe_int(value) -> Optional[int]:
    try:
        return int(float(str(value).replace(",", "").strip()))
    except Exception:
        return None


def _safe_float(value) -> Optional[float]:
    try:
        return float(str(value).replace(",", "").strip())
    except Exception:
        return None


def parse_iso_datetime(value: str):
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def compute_duration_days(fromdate: str, todate: str):
    start = parse_iso_datetime(fromdate)
    end = parse_iso_datetime(todate)
    if not start or not end:
        return ""
    days = (end.date() - start.date()).days
    return max(days, 0)


def base_row_from_properties(p: Dict, event_type: str) -> Dict:
    return {
        "eventid": str(p.get("eventid", "")).strip(),
        "eventtype": str(p.get("eventtype", event_type)).strip(),
        "alertlevel": str(p.get("alertlevel", "")).strip().lower(),
        "country": str(p.get("country", "")).strip(),
        "fromdate": str(p.get("fromdate", "")).strip(),
        "todate": str(p.get("todate", "")).strip(),
        "magnitude": "",
        "depth": "",
        "rapidpopdescription": "",
        "maximum_wind_speed_kmh": "",
        "maximum_storm_surge_m": "",
        "exposed_population": "",
        "duration_days": "",
        "burned_area_ha": "",
        "people_affected": "",
        "affected_area_km2": "",
        "affected_country_count": "",
        "severity_level": "",
        "severity_text": "",
        "_details_url": "",
    }


def fetch_api_page(event_type: str, fromdate: str, todate: str, page: int, alertlevel: str):
    params = (
        f"eventlist={event_type}"
        f"&fromdate={fromdate}"
        f"&todate={todate}"
        f"&alertlevel={urllib.parse.quote(alertlevel, safe=';')}"
        f"&pagenumber={page}"
    )
    url = f"{API_URL}?{params}"
    return fetch_json_with_retry(url)


def parse_api_features(doc: Dict, event_type: str):
    rows = []
    for feat in doc.get("features", []):
        p = feat.get("properties", {})
        row = base_row_from_properties(p, event_type)
        url_obj = p.get("url", {})
        if isinstance(url_obj, dict):
            row["_details_url"] = str(url_obj.get("details", "")).strip()

        # DR fields available in list payload.
        if event_type == "DR":
            severity = p.get("severitydata", {}) if isinstance(p.get("severitydata"), dict) else {}
            severity_text = str(severity.get("severitytext", "")).strip()
            row["duration_days"] = compute_duration_days(row["fromdate"], row["todate"])
            area = _safe_float(severity.get("severity"))
            row["affected_area_km2"] = area if area is not None else ""
            row["affected_country_count"] = len(p.get("affectedcountries", []) or [])
            row["severity_text"] = severity_text
            row["severity_level"] = parse_severity_level(severity_text)

        # WF duration can be computed from list payload dates.
        if event_type == "WF":
            row["duration_days"] = compute_duration_days(row["fromdate"], row["todate"])

        # FL: extract dead/displaced from severitydata in list payload.
        if event_type == "FL":
            severity = p.get("severitydata", {}) if isinstance(p.get("severitydata"), dict) else {}
            severity_text = str(severity.get("severitytext", "")).strip()
            row["severity_text"] = severity_text
            row["severity_level"] = parse_severity_level(severity_text)
            # GDACS FL severitydata may contain humanimpact with dead/displaced
            human = p.get("humanimpact", {}) if isinstance(p.get("humanimpact"), dict) else {}
            dead = _safe_int(human.get("dead") or human.get("killed") or severity.get("dead"))
            displaced = _safe_int(
                human.get("displaced") or human.get("homeless") or severity.get("displaced")
            )
            if dead is not None:
                row["dead"] = dead
            if displaced is not None:
                row["displaced"] = displaced

        rows.append(row)
    return rows


def fetch_rows_for_type(
    event_type: str,
    fromdate: str,
    todate: str,
    level_filter: str,
    scan_limit: int,
):
    rows = []
    page = 0
    failed_pages = 0
    while len(rows) < scan_limit:
        try:
            doc = fetch_api_page(event_type, fromdate, todate, page, level_filter)
            failed_pages = 0
        except Exception as exc:
            failed_pages += 1
            print(
                f"[warn] {event_type} page={page} level={level_filter} "
                f"fetch failed ({failed_pages}/{MAX_PAGE_FAILURES}): {exc}"
            )
            if failed_pages >= MAX_PAGE_FAILURES:
                break
            page += 1
            continue
        page_rows = parse_api_features(doc, event_type)
        if not page_rows:
            break
        rows.extend(page_rows)
        if event_type == "TC":
            # TC endpoint is prone to temporary blocking; pace page requests.
            time.sleep(0.3)
        if len(page_rows) < 100:
            break
        page += 1
    return rows[:scan_limit]


def dedupe_rows(rows: List[Dict]) -> List[Dict]:
    seen = set()
    out = []
    for row in rows:
        key = (row.get("eventtype", ""), row.get("eventid", ""))
        if not key[1] or key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def parse_severity_level(severity_text: str):
    text = str(severity_text or "").strip().lower()
    for key, level in SEVERITY_LEVEL_MAP.items():
        if key in text:
            return level
    return ""


def _extract_sum_pop(impact_doc: Dict) -> Optional[int]:
    best = None
    for block in impact_doc.get("datums", []):
        for item in block.get("datum", []):
            scalars = (item.get("scalars", {}) or {}).get("scalar", [])
            for scalar in scalars:
                name = str(scalar.get("name", "")).upper()
                if not name.startswith("SUMPOP"):
                    continue
                v = _safe_int(scalar.get("value"))
                if v is None:
                    continue
                if best is None or v > best:
                    best = v
    return best


def _extract_tc_exposed_population(detail_props: Dict) -> str:
    impacts = detail_props.get("impacts", [])
    if not isinstance(impacts, list):
        return ""

    candidates = []
    for item in impacts:
        if not isinstance(item, dict):
            continue
        resource = item.get("resource", {})
        if not isinstance(resource, dict):
            continue
        for key in ("buffer74", "buffer39"):
            url = str(resource.get(key, "")).strip()
            if url:
                candidates.append(url)

    for url in candidates:
        try:
            impact_doc = fetch_json_with_retry(url)
            pop = _extract_sum_pop(impact_doc)
            if pop is not None:
                return str(pop)
        except Exception:
            continue
    return ""


def _extract_tc_max_storm_surge(detail_props: Dict) -> str:
    surge = detail_props.get("cyclonesurge", [])
    if not isinstance(surge, list):
        return ""

    detail_urls = []
    for source_item in surge:
        if not isinstance(source_item, dict):
            continue
        data_list = source_item.get("data", [])
        if not isinstance(data_list, list):
            continue
        last_items = [x for x in data_list if isinstance(x, dict) and bool(x.get("last"))]
        target_items = last_items if last_items else [x for x in data_list if isinstance(x, dict)]
        for entry in target_items:
            url = str(entry.get("url", "")).strip()
            if url:
                detail_urls.append(url)

    max_height = None
    for url in detail_urls:
        try:
            surge_doc = fetch_json_with_retry(url)
            props = surge_doc.get("properties", {}) if isinstance(surge_doc, dict) else {}
            h = _safe_float(props.get("maxheight"))
            if h is None:
                continue
            if max_height is None or h > max_height:
                max_height = h
        except Exception:
            continue

    if max_height is None:
        return ""
    return f"{max_height:.6f}".rstrip("0").rstrip(".")


def _extract_wf_impact_url(detail_doc: Dict) -> str:
    impacts = detail_doc.get("properties", {}).get("impacts", []) or []
    for item in impacts:
        if not isinstance(item, dict):
            continue
        resource = item.get("resource", {})
        if not isinstance(resource, dict):
            continue
        url = str(resource.get("impact", "")).strip()
        if url:
            return url
    return ""


def _extract_wf_people_affected(impact_doc: Dict) -> int:
    max_people = 0
    for datum_group in impact_doc.get("datums", []) or []:
        datum_list = datum_group.get("datum", []) or []
        for datum in datum_list:
            scalars = ((datum.get("scalars") or {}).get("scalar") or [])
            for scalar in scalars:
                name = str(scalar.get("name", "")).strip().upper()
                if name != "POPAFFECTED":
                    continue
                value = _safe_int(scalar.get("value")) or 0
                if value > max_people:
                    max_people = value
    return max_people


def _extract_wf_burned_area_ha(detail_doc: Dict):
    severity = detail_doc.get("properties", {}).get("severitydata", {}) or {}
    value = _safe_float(severity.get("severity"))
    if value is None:
        return ""
    return value


def enrich_row_with_details(row: Dict) -> Dict:
    details_url = row.get("_details_url", "")
    event_type = row.get("eventtype", "").upper()
    if details_url:
        try:
            detail_doc = fetch_json_with_retry(details_url)
            props = detail_doc.get("properties", {}) if isinstance(detail_doc, dict) else {}

            if event_type == "EQ":
                eq_detail = props.get("earthquakedetails", {}) if isinstance(props, dict) else {}
                row["magnitude"] = str(eq_detail.get("magnitude", "")).strip()
                row["depth"] = str(eq_detail.get("depth", "")).strip()
                row["rapidpopdescription"] = str(
                    eq_detail.get("rapidpopdescription", "")
                ).strip()

            elif event_type == "TC":
                severity = props.get("severitydata", {}) if isinstance(props, dict) else {}
                row["maximum_wind_speed_kmh"] = str(severity.get("severity", "")).strip()
                row["maximum_storm_surge_m"] = _extract_tc_max_storm_surge(props)
                row["exposed_population"] = _extract_tc_exposed_population(props)

            elif event_type == "WF":
                row["burned_area_ha"] = _extract_wf_burned_area_ha(detail_doc)
                impact_url = _extract_wf_impact_url(detail_doc)
                if impact_url:
                    impact_doc = fetch_json_with_retry(impact_url)
                    people = _extract_wf_people_affected(impact_doc)
                    row["people_affected"] = people if people > 0 else ""
        except Exception:
            pass

    row.pop("_details_url", None)
    return row


def enrich_rows_with_details(rows: List[Dict], workers: int) -> List[Dict]:
    if not rows:
        return rows

    enriched = [None] * len(rows)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(enrich_row_with_details, row): i for i, row in enumerate(rows)}
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                enriched[idx] = fut.result()
            except Exception:
                row = rows[idx]
                row.pop("_details_url", None)
                enriched[idx] = row
    return enriched


def to_csv_rows(rows: List[Dict]) -> List[Dict]:
    return [{k: row.get(k, "") for k in CSV_FIELDS} for row in rows]


def fetch_for_type_with_mode(
    event_type: str,
    fromdate: str,
    todate: str,
    alert_levels: List[str],
    limit_per_type: int,
    balanced_per_level: int,
    page_cap: int,
) -> List[Dict]:
    if balanced_per_level > 0:
        type_rows = []
        for level in alert_levels:
            try:
                level_rows = fetch_rows_for_type(
                    event_type, fromdate, todate, level, page_cap
                )
                level_rows = dedupe_rows(level_rows)
                selected = level_rows[:balanced_per_level]
                print(
                    f"[info] {event_type} {level}: got {len(selected)} "
                    f"unique events (target {balanced_per_level})"
                )
                type_rows.extend(selected)
            except Exception as exc:
                print(f"[warn] skip {event_type} {level} due to error: {exc}")
        return dedupe_rows(type_rows)

    try:
        rows = fetch_rows_for_type(
            event_type, fromdate, todate, ";".join(alert_levels), limit_per_type
        )
        rows = dedupe_rows(rows)
        return rows[:limit_per_type]
    except Exception as exc:
        print(f"[warn] skip {event_type} due to error: {exc}")
        return []


def main():
    global REQUEST_SLEEP_SEC
    global RETRY_ATTEMPTS
    global RETRY_BASE_SEC
    global RETRY_MAX_SEC
    global MAX_PAGE_FAILURES

    parser = argparse.ArgumentParser(
        description=(
            "Fetch GDACS EQ/TC/WF/DR fields in one script and save one merged CSV."
        )
    )
    parser.add_argument(
        "--event-types",
        type=str,
        default="EQ,TC,WF,DR",
        help="Comma-separated event types. Supported: EQ,TC,WF,DR,FL.",
    )
    parser.add_argument(
        "--fromdate",
        type=str,
        default="2024-01-01",
        help="API start date, format YYYY-MM-DD.",
    )
    parser.add_argument(
        "--todate",
        type=str,
        default=str(date.today()),
        help="API end date, format YYYY-MM-DD.",
    )
    parser.add_argument(
        "--alertlevel",
        type=str,
        default=DEFAULT_ALERT_LEVELS,
        help="Semicolon-separated alert levels, e.g. green;orange;red.",
    )
    parser.add_argument(
        "--limit-per-type",
        type=int,
        default=1000,
        help="Rows per event type in non-balanced mode.",
    )
    parser.add_argument(
        "--balanced-per-level",
        type=int,
        default=0,
        help="If >0, rows per alert level per type (overrides --limit-per-type behavior).",
    )
    parser.add_argument(
        "--page-cap",
        type=int,
        default=5000,
        help="Max rows scanned per type-level before dedupe/capping.",
    )
    parser.add_argument(
        "--skip-enrich",
        action="store_true",
        help="Skip per-event detail enrichment (geteventdata endpoint). "
             "Use when doing a full/bulk pull — the detail endpoint is fragile "
             "under high request volume. List fields (alertlevel, country, date) "
             "are sufficient for GDACS matching.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Thread workers for detail enrichment.",
    )
    parser.add_argument(
        "--tc-workers",
        type=int,
        default=2,
        help="Thread workers for TC detail enrichment.",
    )
    parser.add_argument(
        "--request-sleep-sec",
        type=float,
        default=0.15,
        help="Sleep seconds before each HTTP request.",
    )
    parser.add_argument(
        "--retry-attempts",
        type=int,
        default=5,
        help="Max retry attempts per request.",
    )
    parser.add_argument(
        "--retry-base-sec",
        type=float,
        default=1.0,
        help="Exponential backoff base seconds.",
    )
    parser.add_argument(
        "--retry-max-sec",
        type=float,
        default=20.0,
        help="Exponential backoff max seconds.",
    )
    parser.add_argument(
        "--max-page-failures",
        type=int,
        default=6,
        help="Abort one event-type fetch after this many consecutive page failures.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(BASE_DIR / "data" / "gdacs_all_fields.csv"),
        help="Output merged CSV path.",
    )
    args = parser.parse_args()
    REQUEST_SLEEP_SEC = max(0.0, args.request_sleep_sec)
    RETRY_ATTEMPTS = max(1, args.retry_attempts)
    RETRY_BASE_SEC = max(0.1, args.retry_base_sec)
    RETRY_MAX_SEC = max(RETRY_BASE_SEC, args.retry_max_sec)
    MAX_PAGE_FAILURES = max(1, args.max_page_failures)

    event_types = parse_types(args.event_types)
    alert_levels = parse_levels(args.alertlevel)

    try:
        merged_rows = []
        for event_type in event_types:
            rows = fetch_for_type_with_mode(
                event_type=event_type,
                fromdate=args.fromdate,
                todate=args.todate,
                alert_levels=alert_levels,
                limit_per_type=args.limit_per_type,
                balanced_per_level=args.balanced_per_level,
                page_cap=args.page_cap,
            )
            if not args.skip_enrich:
                workers = args.tc_workers if event_type == "TC" else args.workers
                rows = enrich_rows_with_details(rows, workers=max(1, workers))
            else:
                # Strip internal _details_url column added during list fetch
                for r in rows:
                    r.pop("_details_url", None)
            merged_rows.extend(rows)
            print(f"[info] {event_type}: collected {len(rows)} rows")

        merged_rows = dedupe_rows(merged_rows)
        merged_rows = to_csv_rows(merged_rows)
    except Exception as exc:
        print(f"Failed to fetch or parse data: {exc}", file=sys.stderr)
        sys.exit(1)

    if not merged_rows:
        print("No data returned.")
        return

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(merged_rows)

    print(f"Saved {len(merged_rows)} merged rows to {output_path}")


if __name__ == "__main__":
    main()
