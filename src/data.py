"""
Stock data fetching — hardened against Yahoo Finance rate-limiting.

Strategy (in order):
  1. Global rate limiter (min 1.5 s between any Yahoo request)
  2. yf.Ticker().history()  ← primary, uses yfinance's own crumb auth
  3. yf.download()          ← secondary fallback
  4. Shorter period fallback (1y → 6mo) if 2y returns empty
  5. Exponential back-off with random jitter between retries
"""

import random
import time
import threading
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
import yaml
from typing import Any, Dict

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
try:
    with open("configs/config.yaml", "r") as f:
        CONFIG = yaml.safe_load(f)
except FileNotFoundError:
    CONFIG = {"data": {"period": "2y"}}

_PRIMARY_PERIOD   = CONFIG["data"]["period"]    # "2y"
_FALLBACK_PERIODS = ["1y", "6mo"]

# ---------------------------------------------------------------------------
# Global rate limiter — enforces minimum gap between Yahoo requests
# Prevents IP-level rate limiting when scanning universes
# ---------------------------------------------------------------------------
_RATE_LOCK       = threading.Lock()
_LAST_REQUEST_TS = 0.0
_MIN_GAP_SECONDS = 1.5          # at least 1.5 s between any Yahoo call


def _rate_limited_sleep():
    """Block until at least _MIN_GAP_SECONDS have passed since the last request."""
    global _LAST_REQUEST_TS
    with _RATE_LOCK:
        elapsed = time.time() - _LAST_REQUEST_TS
        wait    = _MIN_GAP_SECONDS - elapsed
        if wait > 0:
            time.sleep(wait)
        _LAST_REQUEST_TS = time.time()


# ---------------------------------------------------------------------------
# Fetch strategies
# ---------------------------------------------------------------------------

def _try_ticker_history(ticker: str, period: str) -> pd.DataFrame:
    """Primary: yf.Ticker().history() — uses yfinance's own crumb/cookie auth."""
    _rate_limited_sleep()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stock = yf.Ticker(ticker)
        df    = stock.history(period=period, auto_adjust=True, timeout=15)
    return df if (df is not None and not df.empty) else pd.DataFrame()


def _try_yf_download(ticker: str, period: str) -> pd.DataFrame:
    """Secondary: yf.download() — different internal code path, worth trying."""
    _rate_limited_sleep()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = yf.download(
            ticker,
            period=period,
            progress=False,
            auto_adjust=True,
            threads=False,
        )
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df if (df is not None and not df.empty) else pd.DataFrame()


def _fetch_with_retry(
    fetch_fn,
    ticker: str,
    period: str,
    max_retries: int = 3,
    label: str = "",
) -> pd.DataFrame:
    """Run *fetch_fn(ticker, period)* with exponential back-off + jitter."""
    for attempt in range(max_retries):
        try:
            df = fetch_fn(ticker, period)
            if not df.empty:
                return df
        except Exception as e:
            print(f"[WARN] {label} attempt {attempt + 1} for {ticker}: {e}")

        if attempt < max_retries - 1:
            # jitter prevents all retries firing at the same time during scans
            wait = (2 ** attempt) + random.uniform(0.5, 1.5)
            print(
                f"[WARN] No data via {label} for {ticker} "
                f"(attempt {attempt + 1}/{max_retries}). Retrying in {wait:.1f}s ..."
            )
            time.sleep(wait)

    return pd.DataFrame()


def _fetch_history(ticker: str) -> pd.DataFrame:
    """
    Full waterfall:
      1. Ticker.history (2y)
      2. yf.download    (2y)
      3. Ticker.history (1y)
      4. yf.download    (1y)
      5. Ticker.history (6mo)
    """
    # Step 1 & 2: primary period
    for fn, label in [
        (_try_ticker_history, "Ticker.history"),
        (_try_yf_download,    "yf.download"),
    ]:
        df = _fetch_with_retry(fn, ticker, _PRIMARY_PERIOD, max_retries=3, label=label)
        if not df.empty:
            return df

    # Steps 3–5: shorter periods
    for period in _FALLBACK_PERIODS:
        print(f"[INFO] Trying fallback period='{period}' for {ticker} ...")
        for fn, label in [
            (_try_ticker_history, f"Ticker.history({period})"),
            (_try_yf_download,    f"yf.download({period})"),
        ]:
            df = _fetch_with_retry(fn, ticker, period, max_retries=2, label=label)
            if not df.empty:
                print(f"[INFO] Got data for {ticker} with period='{period}'.")
                return df

    return pd.DataFrame()


def _safe_info(ticker: str) -> Dict[str, Any]:
    """Fetch .info without crashing. Returns {} on any error."""
    _rate_limited_sleep()
    try:
        info = yf.Ticker(ticker).info or {}
        return info if len(info) > 2 else {}
    except Exception:
        return {}


def _clean_hist(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise columns and ensure 'Close' exists."""
    df.columns = [str(c) for c in df.columns]
    if "Close" not in df.columns and "Adj Close" in df.columns:
        df = df.rename(columns={"Adj Close": "Close"})
    return df.dropna(subset=["Close"])


# ---------------------------------------------------------------------------
# Public API (signatures unchanged)
# ---------------------------------------------------------------------------

def get_stock_data(ticker: str) -> Dict[str, Any]:
    """
    Return historical OHLCV + fundamentals for *ticker*.
    Never raises — returns {"error": "..."} on failure.
    """
    raw = _fetch_history(ticker)

    if raw.empty:
        return {
            "historical":   None,
            "fundamentals": None,
            "ticker":       ticker,
            "error": (
                f"Could not fetch data for '{ticker}' after multiple attempts. "
                "Yahoo Finance may be rate-limiting your IP. "
                "Wait 60 s and retry, or check that the symbol is correct "
                "(e.g. RELIANCE.NS, TCS.NS, AAPL)."
            ),
        }

    hist = _clean_hist(raw)

    current_price   = float(hist["Close"].iloc[-1])
    first_price     = float(hist["Close"].iloc[0])
    price_change_1y = ((current_price / first_price) - 1) * 100 if first_price else 0.0

    info = _safe_info(ticker)

    fundamentals = {
        "pe_ratio":        float(info.get("trailingPE") or 0),
        "dividend_yield":  float(info.get("dividendYield") or 0) * 100,
        "market_cap":      info.get("marketCap") or "N/A",
        "current_price":   current_price,
        "price_change_1y": price_change_1y,
    }

    return {
        "historical":   hist,
        "fundamentals": fundamentals,
        "ticker":       ticker,
        "error":        None,
    }


def calculate_risk_metrics(hist: pd.DataFrame) -> Dict[str, float]:
    """Annualised volatility, Sharpe ratio, max drawdown."""
    returns  = hist["Close"].pct_change().dropna()
    std      = returns.std()
    vol      = float(std * np.sqrt(252) * 100)
    sharpe   = float((returns.mean() / std) * np.sqrt(252)) if std != 0 else 0.0
    peak     = hist["Close"].cummax()
    drawdown = float(((peak - hist["Close"]) / peak).max() * 100)
    return {"volatility": vol, "sharpe_ratio": sharpe, "max_drawdown": drawdown}