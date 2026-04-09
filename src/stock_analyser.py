"""
Module E — Stock Impact Analysis (Event Study)

Estimates the Cumulative Abnormal Return (CAR) for a disaster event using
the market model (OLS regression):

    R_i,t = α + β × R_m,t + ε_t            (estimation window)
    AR_t  = R_i,t − (α̂ + β̂ × R_m,t)      (event window)
    CAR(T1, T2) = Σ AR_t  for t in [T1, T2]

Windows (relative to event date T0):
  - Estimation window : [T0−45, T0−6]  (40 trading days)
  - Event window      : [T0−1,  T0+5]  (7 trading days)
  → Outputs: car_t1 (T0+1), car_t3 (T0+1:T0+3), car_t5 (T0+1:T0+5)

Market proxy:
  - Default: "SPY" (S&P 500 ETF) — used for all events
  - Can be overridden per call

Usage:
    from src.stock_analyser import StockAnalyser
    analyser = StockAnalyser()
    result = analyser.compute_car(
        event_date="2025-09-01",
        ticker="EWJ",
        event_id="EQ_NP_20250901",
    )

Batch usage:
    results_df = analyser.compute_car_batch(events_df)
    # events_df must have columns: event_date, index_ticker, event_id
"""

from __future__ import annotations

import warnings
from datetime import date, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import yfinance as yf
    _YF_AVAILABLE = True
except ImportError:
    _YF_AVAILABLE = False

# Silence yfinance download progress bars
warnings.filterwarnings("ignore", category=FutureWarning)

MARKET_PROXY   = "ACWI"   # MSCI All Country World Index — better proxy for global events
EST_PRE_DAYS   = 45   # calendar days before event to start fetch
EST_POST_DAYS  = 10   # calendar days after event to fetch
EST_WIN_START  = -45  # estimation window start (trading days relative to T0)
EST_WIN_END    = -6   # estimation window end
EVENT_WIN_END  = +5   # event window end

_CACHE: Dict[str, pd.Series] = {}   # ticker → daily return series (session cache)


def _fetch_returns(ticker: str, start: date, end: date) -> Optional[pd.Series]:
    """Download daily close-to-close log returns for `ticker` in [start, end]."""
    if not _YF_AVAILABLE:
        return None
    key = f"{ticker}_{start}_{end}"
    if key in _CACHE:
        return _CACHE[key]
    try:
        raw = yf.download(
            ticker,
            start=start.isoformat(),
            end=(end + timedelta(days=1)).isoformat(),
            auto_adjust=True,
            progress=False,
        )
        if raw.empty:
            return None
        close = raw["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        returns = np.log(close / close.shift(1)).dropna()
        returns.index = pd.to_datetime(returns.index).normalize()
        _CACHE[key] = returns
        return returns
    except Exception:
        return None


def _offset_trading_day(series: pd.Series, t0: date, offset: int) -> Optional[date]:
    """Return the trading date `offset` days from t0 (positive = forward)."""
    idx = pd.to_datetime(t0)
    if offset == 0:
        loc = series.index.searchsorted(idx)
        if loc < len(series.index):
            return series.index[loc].date()
        return None
    sorted_idx = sorted(series.index)
    # Find nearest trading day at or after t0
    start_pos = None
    for i, d in enumerate(sorted_idx):
        if d.date() >= t0:
            start_pos = i
            break
    if start_pos is None:
        return None
    target_pos = start_pos + offset
    if 0 <= target_pos < len(sorted_idx):
        return sorted_idx[target_pos].date()
    return None


def compute_car(
    event_date: str,
    ticker: str,
    market_proxy: str = MARKET_PROXY,
    event_id: str = "",
) -> dict:
    """
    Compute CAR for a single event.

    Parameters
    ----------
    event_date   : ISO date string "YYYY-MM-DD"
    ticker       : stock/ETF ticker for the affected country (from EntityLinker)
    market_proxy : market index ticker (default SPY)
    event_id     : string label for this event (used only for reporting)

    Returns
    -------
    dict with keys:
        event_id, ticker, event_date,
        car_t1, car_t3, car_t5,          (float or None)
        beta, alpha,                      (OLS coefficients or None)
        n_est,                            (number of estimation-window obs)
        error                             (None if successful, else str)
    """
    base = {
        "event_id":   event_id,
        "ticker":     ticker,
        "event_date": event_date,
        "car_t1":     None,
        "car_t3":     None,
        "car_t5":     None,
        "beta":       None,
        "alpha":      None,
        "n_est":      None,
        "error":      None,
    }

    if not _YF_AVAILABLE:
        base["error"] = "yfinance not installed"
        return base

    if not ticker or not event_date:
        base["error"] = "missing ticker or event_date"
        return base

    try:
        t0 = date.fromisoformat(str(event_date)[:10])
    except (ValueError, TypeError):
        base["error"] = f"invalid event_date: {event_date}"
        return base

    # Fetch window: [T0−60d, T0+15d] calendar — ensures enough trading days
    fetch_start = t0 - timedelta(days=60)
    fetch_end   = t0 + timedelta(days=15)

    r_stock  = _fetch_returns(ticker,       fetch_start, fetch_end)
    r_market = _fetch_returns(market_proxy, fetch_start, fetch_end)

    if r_stock is None or r_market is None or r_stock.empty or r_market.empty:
        base["error"] = "download failed or no data"
        return base

    # Align on common trading dates
    common_idx = r_stock.index.intersection(r_market.index)
    r_stock  = r_stock[common_idx]
    r_market = r_market[common_idx]

    if len(common_idx) == 0:
        base["error"] = "no overlapping trading dates"
        return base

    # Identify T0 (nearest trading day at or after event_date)
    t0_ts = pd.Timestamp(t0)
    t0_locs = [i for i, d in enumerate(common_idx) if d >= t0_ts]
    if not t0_locs:
        base["error"] = "event_date after all available data"
        return base
    t0_pos = t0_locs[0]

    # Estimation window: positions [t0_pos + EST_WIN_START, t0_pos + EST_WIN_END]
    est_start = max(0, t0_pos + EST_WIN_START)
    est_end   = max(0, t0_pos + EST_WIN_END)

    if est_end <= est_start:
        base["error"] = "estimation window too short"
        return base

    est_dates  = common_idx[est_start: est_end + 1]
    r_s_est    = r_stock[est_dates].values
    r_m_est    = r_market[est_dates].values
    n_est      = len(r_s_est)
    base["n_est"] = n_est

    if n_est < 10:
        base["error"] = f"too few estimation obs: {n_est}"
        return base

    # OLS: R_stock = alpha + beta * R_market
    X = np.column_stack([np.ones(n_est), r_m_est])
    try:
        coeffs, _, _, _ = np.linalg.lstsq(X, r_s_est, rcond=None)
        alpha, beta = float(coeffs[0]), float(coeffs[1])
    except np.linalg.LinAlgError as e:
        base["error"] = f"OLS failed: {e}"
        return base

    base["alpha"] = round(alpha, 6)
    base["beta"]  = round(beta,  6)

    # Event window: [T0, T0+EVENT_WIN_END]
    ev_start = t0_pos
    ev_end   = min(t0_pos + EVENT_WIN_END, len(common_idx) - 1)
    ev_dates = common_idx[ev_start: ev_end + 1]

    r_s_ev = r_stock[ev_dates].values
    r_m_ev = r_market[ev_dates].values
    ar_ev  = r_s_ev - (alpha + beta * r_m_ev)

    n_ev = len(ar_ev)

    # CAR at T+1, T+3, T+5 relative positions (1-indexed from T0+1)
    def _car(up_to: int) -> Optional[float]:
        if up_to < 1 or up_to >= n_ev:
            return None
        return float(np.sum(ar_ev[1: up_to + 1]))

    base["car_t1"] = _car(1)
    base["car_t3"] = _car(3)
    base["car_t5"] = _car(5)

    return base


class StockAnalyser:
    """
    Compute CAR for a list of event cluster records.

    Usage:
        analyser = StockAnalyser()
        results  = analyser.compute_car_batch(events)
    """

    def __init__(self, market_proxy: str = MARKET_PROXY):
        self.market_proxy = market_proxy

    def compute_car_for_event(self, event: dict) -> dict:
        """
        Compute CAR for a single event dict using one ticker.
        Kept for backwards compatibility; prefer compute_car_multi_event.
        """
        import ast
        raw = event.get("sector_etfs") or []
        if isinstance(raw, str):
            try: raw = ast.literal_eval(raw)
            except Exception: raw = []
        ticker = event.get("index_ticker") or (raw[0] if raw else None)
        return compute_car(
            event_date   = event.get("event_date", ""),
            ticker       = ticker or "",
            market_proxy = self.market_proxy,
            event_id     = str(event.get("event_id", "")),
        )

    def compute_car_multi_event(self, event: dict) -> List[dict]:
        """
        Compute CAR for each sector ETF associated with one event.

        Uses event['sector_etfs'] (list from EntityLinker).
        Returns one row per (event_id, sector, ticker).
        """
        import ast
        event_id   = str(event.get("event_id", ""))
        event_date = event.get("event_date", "")
        raw_etfs   = event.get("sector_etfs") or []
        raw_ind    = event.get("key_industries") or []
        if isinstance(raw_etfs, str):
            try:
                raw_etfs = ast.literal_eval(raw_etfs)
            except Exception:
                raw_etfs = []
        if isinstance(raw_ind, str):
            try:
                raw_ind = ast.literal_eval(raw_ind)
            except Exception:
                raw_ind = []
        etfs       = raw_etfs if isinstance(raw_etfs, list) else []
        industries = raw_ind  if isinstance(raw_ind,  list) else []

        results = []
        for i, ticker in enumerate(etfs):
            sector = industries[i] if i < len(industries) else ticker
            r = compute_car(
                event_date   = event_date,
                ticker       = ticker,
                market_proxy = self.market_proxy,
                event_id     = event_id,
            )
            r["sector"] = sector
            results.append(r)
        return results

    def compute_car_batch(self, events: List[dict]) -> pd.DataFrame:
        """
        Compute CAR for a list of events across all sector ETFs.
        Returns one row per (event, sector_etf).

        Parameters
        ----------
        events : list of dicts with 'event_date' and 'sector_etfs' (from EntityLinker)
        """
        rows = []
        for i, ev in enumerate(events):
            if "event_id" not in ev:
                ev = {**ev, "event_id": str(i)}
            multi = self.compute_car_multi_event(ev)
            if multi:
                rows.extend(multi)
            else:
                # No ETFs available — record a null row for completeness
                rows.append({
                    "event_id": ev["event_id"], "ticker": None,
                    "sector": None, "event_date": ev.get("event_date"),
                    "car_t1": None, "car_t3": None, "car_t5": None,
                    "beta": None, "alpha": None, "n_est": None,
                    "error": "no_sector_etfs",
                })
        return pd.DataFrame(rows)

    def compute_random_baseline(
        self,
        car_df: pd.DataFrame,
        n_samples: Optional[int] = None,
        seed: int = 42,
    ) -> pd.DataFrame:
        """
        Generate a random-date baseline by replacing each real event date with a
        randomly sampled date from the same date pool, excluding a ±10-day window
        around any real event date.

        Parameters
        ----------
        car_df    : output of compute_car_batch (must have 'event_date', 'ticker')
        n_samples : number of pseudo-events to draw; defaults to len(car_df)
        seed      : random seed for reproducibility

        Returns
        -------
        DataFrame with same columns as compute_car_batch output, plus 'source'='random'.
        """
        import random as _random
        from datetime import timedelta

        rng = _random.Random(seed)

        real_dates = pd.to_datetime(car_df["event_date"].dropna()).dt.date.tolist()
        date_min   = min(real_dates)
        date_max   = max(real_dates)

        # Build exclusion set: all real event dates ± 10 days
        excluded = set()
        for d in real_dates:
            for delta in range(-10, 11):
                excluded.add(d + timedelta(days=delta))

        # Full candidate pool: every calendar day in range, not excluded
        total_days = (date_max - date_min).days + 1
        candidate_dates = [
            date_min + timedelta(days=i)
            for i in range(total_days)
            if (date_min + timedelta(days=i)) not in excluded
        ]

        if not candidate_dates:
            return pd.DataFrame()

        valid_rows = car_df[car_df["error"].isna()].copy()
        if n_samples is None:
            n_samples = len(valid_rows)

        # Sample (ticker, random_date) pairs from the real ticker distribution
        tickers = valid_rows["ticker"].tolist()
        rows = []
        for ticker in rng.choices(tickers, k=n_samples):
            rand_date = rng.choice(candidate_dates).isoformat()
            r = compute_car(
                event_date   = rand_date,
                ticker       = ticker,
                market_proxy = self.market_proxy,
                event_id     = f"RAND_{rand_date}_{ticker}",
            )
            r["source"] = "random"
            rows.append(r)
        result = pd.DataFrame(rows)
        result["source"] = "random"
        return result

    def baseline_comparison(
        self, car_df: pd.DataFrame, random_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compare real-event CAR vs random-baseline CAR.

        Returns a summary DataFrame with mean, std, and two-sample t-test p-value
        for each (window, source) pair, plus an overall row.
        """
        from scipy import stats as scipy_stats

        rows = []
        real   = car_df[car_df["error"].isna()].copy()
        random = random_df[random_df["error"].isna()].copy()
        real["source"]   = "event"
        combined = pd.concat([real, random], ignore_index=True)

        for window in ["car_t1", "car_t3", "car_t5"]:
            ev_vals  = real[window].dropna().values
            rnd_vals = random[window].dropna().values
            if len(ev_vals) < 3 or len(rnd_vals) < 3:
                continue
            _, p_two_sample = scipy_stats.ttest_ind(ev_vals, rnd_vals)
            _, p_event_zero = scipy_stats.ttest_1samp(ev_vals, 0.0)
            _, p_rand_zero  = scipy_stats.ttest_1samp(rnd_vals, 0.0)
            rows.append({
                "window":              window,
                "event_mean":          round(float(np.mean(ev_vals)),  6),
                "event_std":           round(float(np.std(ev_vals)),   6),
                "event_n":             len(ev_vals),
                "random_mean":         round(float(np.mean(rnd_vals)), 6),
                "random_std":          round(float(np.std(rnd_vals)),  6),
                "random_n":            len(rnd_vals),
                "p_event_vs_zero":     round(float(p_event_zero), 4),
                "p_random_vs_zero":    round(float(p_rand_zero),  4),
                "p_event_vs_random":   round(float(p_two_sample), 4),
                "event_sig_5pct":      bool(p_event_zero  < 0.05),
                "random_sig_5pct":     bool(p_rand_zero   < 0.05),
                "diff_sig_5pct":       bool(p_two_sample  < 0.05),
            })
        return pd.DataFrame(rows)

    def group_analysis(self, car_df: pd.DataFrame, events_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge CAR results with event metadata and compute group-level statistics.

        Parameters
        ----------
        car_df     : output of compute_car_batch
        events_df  : event cluster records with 'event_type' and 'predicted_alert'

        Returns
        -------
        DataFrame with group means and t-test p-values for each event_type×severity cell.
        """
        from scipy import stats as scipy_stats

        merged = car_df.merge(
            events_df[["event_id", "event_type", "predicted_alert"]],
            on="event_id", how="left",
        )
        merged = merged[merged["error"].isna()].copy()

        rows = []
        group_cols = ["event_type", "predicted_alert", "sector"]
        for col in group_cols:
            if col not in merged.columns:
                merged[col] = None
        for (etype, alert, sector), grp in merged.groupby(group_cols, dropna=False):
            for car_col in ["car_t1", "car_t3", "car_t5"]:
                vals = grp[car_col].dropna().values
                n = len(vals)
                mean_car = float(np.mean(vals)) if n > 0 else None
                p_val = None
                if n >= 3:
                    _, p_val = scipy_stats.ttest_1samp(vals, 0.0)
                rows.append({
                    "event_type":       etype,
                    "severity":         alert,
                    "sector":           sector,
                    "window":           car_col,
                    "n":                n,
                    "mean_car":         round(mean_car, 5) if mean_car is not None else None,
                    "p_value":          round(float(p_val), 4) if p_val is not None else None,
                    "significant_5pct": bool(p_val < 0.05) if p_val is not None else None,
                })
        return pd.DataFrame(rows)
