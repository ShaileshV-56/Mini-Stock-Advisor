"""
Mini Stock Advisor — FastAPI Backend  (v1.1 — rate-limit hardened)

Key fixes vs v1.0:
  - TTLCache wraps every get_stock_data() call → same ticker cached 5 min
  - No more repeated Yahoo requests on each Streamlit page rerun
  - Clear, user-friendly 404 messages instead of raw library errors
"""

import sys
import numpy as np

if "numpy" in sys.modules:
    np.float_ = np.float64

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel
from typing import List, Dict, Any
import pandas as pd
import joblib
import os
from cachetools import TTLCache
from threading import Lock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data import get_stock_data, calculate_risk_metrics
from src.agent import get_stock_advice
from src.report import generate_pdf_report
from src.universes import UNIVERSES

# ---------------------------------------------------------------------------
app = FastAPI(
    title="Mini Stock Advisor API",
    version="1.1.0",
    description="Backend API: ML forecasting, AI advice, screener, watchlist, portfolio",
)

# ===========================================================================
# TTL CACHE — 128 tickers, 5-minute TTL
# Prevents hammering Yahoo Finance on every Streamlit rerun
# ===========================================================================
_stock_cache: TTLCache = TTLCache(maxsize=128, ttl=300)
_cache_lock  = Lock()


def _get_stock_cached(ticker: str):
    """Thread-safe TTL-cached wrapper around get_stock_data."""
    with _cache_lock:
        if ticker in _stock_cache:
            return _stock_cache[ticker]

    data = get_stock_data(ticker)          # network call — outside the lock

    with _cache_lock:
        _stock_cache[ticker] = data
    return data


def _require_data(ticker: str):
    """Fetch cached data; raise HTTP 404 with a clear message on failure."""
    data = _get_stock_cached(ticker)
    if data.get("error") or data.get("historical") is None:
        detail = data.get("error") or (
            f"No price data found for '{ticker}'. "
            "Check the symbol (e.g. GOOGL, RELIANCE.NS) and try again. "
            "Yahoo Finance may also be rate-limiting — wait 30 s and retry."
        )
        raise HTTPException(status_code=404, detail=detail)
    return data


# ===========================================================================
# MODEL — loaded once at startup
# ===========================================================================
forecaster = None


@app.on_event("startup")
def load_model():
    global forecaster

    # Clear stale yfinance crumb/cookie cache — prevents JSONDecodeError on .NS stocks
    try:
        import shutil, pathlib
        for cache_dir in [
            pathlib.Path.home() / ".cache" / "py-yfinance",
            pathlib.Path("/tmp/py-yfinance"),
        ]:
            if cache_dir.exists():
                shutil.rmtree(cache_dir, ignore_errors=True)
        print("🧹 yfinance cache cleared.")
    except Exception:
        pass

    model_path = os.path.join(os.path.dirname(__file__), "..", "models", "ensemble.pkl")
    try:
        forecaster = joblib.load(model_path)
        print("✅ Ensemble model loaded.")
    except Exception as e:
        print(f"⚠️  Could not load model: {e}. Momentum fallback will be used.")


# ===========================================================================
# HELPERS
# ===========================================================================

def safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        if isinstance(value, float) and pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def forecast_from_model(hist: pd.DataFrame, horizon: int = 7) -> float:
    if forecaster is not None:
        try:
            pred = forecaster.predict(hist)
            if isinstance(pred, (list, tuple, np.ndarray)):
                pred = np.array(pred).flatten()[-1]
            return float(pred)
        except Exception:
            pass
    close   = hist["Close"].dropna()
    if len(close) < 10:
        return float(close.iloc[-1])
    avg_ret = close.pct_change().dropna().tail(10).mean()
    return float(close.iloc[-1] * ((1 + avg_ret) ** horizon))


def build_technical_indicators(hist: pd.DataFrame) -> pd.DataFrame:
    hist = hist.copy()
    hist["SMA20"] = hist["Close"].rolling(20).mean()
    hist["SMA50"] = hist["Close"].rolling(50).mean()
    delta = hist["Close"].diff()
    gain  = delta.where(delta > 0, 0).rolling(14).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    hist["RSI"] = 100 - (100 / (1 + rs))
    return hist


def rule_based_advice(funds: Dict, forecast_price: float, current_price: float) -> str:
    score, reasons = 0, []
    if safe_float(funds.get("pe_ratio"), 999) < 25:
        score += 1; reasons.append("Low P/E")
    if safe_float(funds.get("dividend_yield"), 0) > 1.0:
        score += 1; reasons.append("Good dividend")
    if current_price and forecast_price > current_price * 1.02:
        score += 1; reasons.append("Price upside")
    if score >= 2:
        return f"BUY ({score}/3): {', '.join(reasons)}"
    if score == 1:
        return f"HOLD ({score}/3): {', '.join(reasons)}"
    return "SELL (0/3): Weak signals"


def hist_to_records(hist: pd.DataFrame) -> List[Dict]:
    df = hist.reset_index()
    df.columns = [str(c) for c in df.columns]
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col]).dt.strftime("%Y-%m-%d")
    return df.fillna(0).to_dict(orient="records")


# ===========================================================================
# SCHEMAS
# ===========================================================================

class ScanRequest(BaseModel):
    universe_name: str
    horizon: int = 7

class ReportRequest(BaseModel):
    ticker: str
    forecast_price: float
    advice: str

class WatchlistRequest(BaseModel):
    watchlist: List[Dict[str, Any]]

class PortfolioRequest(BaseModel):
    holdings: List[Dict[str, Any]]


# ===========================================================================
# ROUTES
# ===========================================================================

@app.get("/health")
def health():
    return {
        "status":       "ok",
        "model_loaded": forecaster is not None,
        "cache_size":   len(_stock_cache),
    }


@app.get("/universes")
def get_universes():
    return {"universes": {k: v for k, v in UNIVERSES.items()}}


# ── Stock analysis ──────────────────────────────────────────────────────────

@app.get("/stock/{ticker}")
def analyze_stock(
    ticker: str,
    horizon: int = Query(default=7, ge=1, le=14),
):
    ticker = ticker.upper().strip()
    data   = _require_data(ticker)

    hist   = build_technical_indicators(data["historical"])
    funds  = data["fundamentals"]
    risk   = calculate_risk_metrics(hist)

    forecast_price = forecast_from_model(hist, horizon=horizon)
    current_price  = safe_float(funds.get("current_price"), hist["Close"].dropna().iloc[-1])
    upside_pct     = ((forecast_price / current_price) - 1) * 100 if current_price else 0.0

    return {
        "ticker":         ticker,
        "current_price":  round(current_price, 2),
        "forecast_price": round(forecast_price, 2),
        "upside_pct":     round(upside_pct, 2),
        "rule_advice":    rule_based_advice(funds, forecast_price, current_price),
        "fundamentals": {
            "pe_ratio":        safe_float(funds.get("pe_ratio")),
            "dividend_yield":  safe_float(funds.get("dividend_yield")),
            "market_cap":      str(funds.get("market_cap", "N/A")),
            "price_change_1y": safe_float(funds.get("price_change_1y")),
        },
        "risk": {
            "volatility":   round(safe_float(risk.get("volatility")), 2),
            "sharpe_ratio": round(safe_float(risk.get("sharpe_ratio")), 2),
            "max_drawdown": round(safe_float(risk.get("max_drawdown")), 2),
        },
        "historical": hist_to_records(hist),
    }


@app.get("/advice/{ticker}")
def ai_advice(ticker: str):
    ticker = ticker.upper().strip()
    try:
        advice = get_stock_advice(ticker)
        return {"ticker": ticker, "advice": advice}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {e}")


# ── Universe screener ───────────────────────────────────────────────────────

@app.post("/scan")
def scan_universe(req: ScanRequest):
    if req.universe_name not in UNIVERSES:
        raise HTTPException(status_code=400, detail=f"Unknown universe '{req.universe_name}'.")

    rows = []
    for ticker in UNIVERSES[req.universe_name]:
        try:
            data = _get_stock_cached(ticker)
            if data.get("error") or data.get("historical") is None:
                continue

            hist  = data["historical"]
            funds = data["fundamentals"]
            risk  = calculate_risk_metrics(hist)
            close = hist["Close"].dropna()

            current_price  = safe_float(funds.get("current_price"), close.iloc[-1])
            forecast_price = forecast_from_model(hist, horizon=req.horizon)
            ret_5d = ((close.iloc[-1] / close.iloc[-6])  - 1) * 100 if len(close) > 6  else 0
            ret_1m = ((close.iloc[-1] / close.iloc[-22]) - 1) * 100 if len(close) > 22 else 0
            upside = ((forecast_price / current_price) - 1) * 100   if current_price    else 0

            rows.append({
                "Ticker":           ticker,
                "Current Price":    round(current_price, 2),
                "5D Return %":      round(ret_5d, 2),
                "1M Return %":      round(ret_1m, 2),
                "Forecast Price":   round(forecast_price, 2),
                "Upside %":         round(upside, 2),
                "P/E":              round(safe_float(funds.get("pe_ratio")), 2),
                "Dividend Yield %": round(safe_float(funds.get("dividend_yield")), 2),
                "Volatility %":     round(safe_float(risk.get("volatility")), 2),
                "Sharpe Ratio":     round(safe_float(risk.get("sharpe_ratio")), 2),
                "Max Drawdown %":   round(safe_float(risk.get("max_drawdown")), 2),
            })
        except Exception:
            continue

    rows.sort(key=lambda x: x["Upside %"], reverse=True)
    return {"universe": req.universe_name, "count": len(rows), "results": rows}


# ── PDF report ──────────────────────────────────────────────────────────────

@app.post("/report")
def download_report(req: ReportRequest):
    ticker = req.ticker.upper().strip()
    data   = _require_data(ticker)
    pdf    = generate_pdf_report(ticker, data["fundamentals"], req.forecast_price, req.advice)
    return Response(
        content=pdf,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{ticker}_report.pdf"'},
    )


# ── Watchlist ───────────────────────────────────────────────────────────────

@app.post("/watchlist/prices")
def watchlist_prices(req: WatchlistRequest):
    rows = []
    for item in req.watchlist:
        ticker = item["ticker"]
        target = safe_float(item.get("target_price"), 0)
        try:
            data = _get_stock_cached(ticker)
            if data.get("error") or data.get("historical") is None:
                continue
            close    = data["historical"]["Close"].dropna()
            current  = float(close.iloc[-1])
            day_chg  = ((close.iloc[-1] / close.iloc[-2]) - 1) * 100 if len(close) > 1 else 0
            distance = ((target / current) - 1) * 100 if target > 0 and current > 0 else None
            rows.append({
                "Ticker":               ticker,
                "Current Price":        round(current, 2),
                "Daily Change %":       round(day_chg, 2),
                "Target Price":         round(target, 2),
                "Distance to Target %": round(distance, 2) if distance is not None else None,
            })
        except Exception:
            continue
    return {"prices": rows}


@app.post("/watchlist/alerts")
def watchlist_alerts(req: WatchlistRequest):
    alerts = []
    for item in req.watchlist:
        ticker = item["ticker"]
        target = safe_float(item.get("target_price"), 0)
        try:
            data = _get_stock_cached(ticker)
            if data.get("error") or data.get("historical") is None:
                continue
            close   = data["historical"]["Close"].dropna()
            current = float(close.iloc[-1])
            daily   = ((close.iloc[-1] / close.iloc[-2]) - 1) * 100 if len(close) > 1 else 0
            if target > 0 and current >= target:
                alerts.append(f"✅ {ticker}: above target ({target:.2f})")
            if target > 0 and current <= target * 0.95:
                alerts.append(f"⚠️  {ticker}: 5% below target ({target:.2f})")
            if daily >= 3:
                alerts.append(f"📈 {ticker}: up {daily:.2f}% today")
            if daily <= -3:
                alerts.append(f"📉 {ticker}: down {daily:.2f}% today")
        except Exception:
            continue
    return {"alerts": alerts}


# ── Portfolio ───────────────────────────────────────────────────────────────

@app.post("/portfolio/summary")
def portfolio_summary(req: PortfolioRequest):
    rows         = []
    returns_map  = {}
    value_weights: List[float] = []

    for item in req.holdings:
        ticker    = item["ticker"]
        qty       = safe_float(item.get("quantity"), 0)
        buy_price = safe_float(item.get("buy_price"), 0)
        try:
            data = _get_stock_cached(ticker)
            if data.get("error") or data.get("historical") is None:
                continue
            close    = data["historical"]["Close"].dropna()
            current  = float(close.iloc[-1])
            invested = qty * buy_price
            curr_val = qty * current
            pnl      = curr_val - invested
            ret_pct  = (pnl / invested) * 100 if invested > 0 else 0
            rows.append({
                "Ticker":        ticker,
                "Quantity":      qty,
                "Buy Price":     round(buy_price, 2),
                "Current Price": round(current, 2),
                "Invested":      round(invested, 2),
                "Current Value": round(curr_val, 2),
                "P/L":           round(pnl, 2),
                "Return %":      round(ret_pct, 2),
            })
            returns_map[ticker] = close.pct_change().dropna().tolist()
            value_weights.append(curr_val)
        except Exception:
            continue

    if rows:
        total = sum(r["Current Value"] for r in rows)
        for r in rows:
            r["Allocation %"] = round((r["Current Value"] / total) * 100, 2) if total else 0

    portfolio_risk = {"volatility": 0.0, "sharpe_ratio": 0.0}
    if returns_map and len(value_weights) > 1:
        try:
            min_len    = min(len(v) for v in returns_map.values())
            ret_matrix = np.array([v[-min_len:] for v in returns_map.values()])
            w          = np.array(value_weights) / sum(value_weights)
            cov        = np.cov(ret_matrix) * 252
            mean_ret   = ret_matrix.mean(axis=1) * 252
            vol        = float(np.sqrt(w @ cov @ w)) * 100
            ret        = float(w @ mean_ret) * 100
            sharpe     = ret / vol if vol != 0 else 0.0
            portfolio_risk = {"volatility": round(vol, 2), "sharpe_ratio": round(sharpe, 2)}
        except Exception:
            pass

    return {"holdings": rows, "portfolio_risk": portfolio_risk}