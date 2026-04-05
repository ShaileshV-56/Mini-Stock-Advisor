"""
Run this ONCE before starting the backend if you keep getting JSONDecodeError.

  python fix_yfinance.py

What it does:
  1. Upgrades yfinance to latest
  2. Clears all stale yfinance cache (crumb tokens, cookies, tz data)
  3. Verifies that ASIANPAINT.NS (a commonly failing Indian stock) can be fetched
"""

import subprocess
import sys
import shutil
import pathlib
import time

print("=" * 60)
print("Step 1: Upgrading yfinance to latest version ...")
print("=" * 60)
subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "yfinance"])

print("\n" + "=" * 60)
print("Step 2: Clearing yfinance cache ...")
print("=" * 60)

cache_paths = [
    pathlib.Path.home() / ".cache" / "py-yfinance",
    pathlib.Path("/tmp/py-yfinance"),
    pathlib.Path.home() / "AppData" / "Local" / "py-yfinance",   # Windows
]

for p in cache_paths:
    if p.exists():
        shutil.rmtree(p, ignore_errors=True)
        print(f"  Deleted: {p}")
    else:
        print(f"  Not found (OK): {p}")

print("\n" + "=" * 60)
print("Step 3: Test fetch for ASIANPAINT.NS ...")
print("=" * 60)
time.sleep(2)   # brief pause after cache clear

import yfinance as yf
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    stock = yf.Ticker("ASIANPAINT.NS")
    hist  = stock.history(period="1mo", auto_adjust=True, timeout=15)

if hist is not None and not hist.empty:
    print(f"  ✅ SUCCESS — got {len(hist)} rows for ASIANPAINT.NS")
    print(f"  Latest close: ₹{hist['Close'].iloc[-1]:.2f}")
else:
    print("  ⚠️  Still empty. Yahoo Finance may be rate-limiting your IP.")
    print("  Wait 60 seconds and restart the backend — the cache is now clean.")

print("\n✅ Done. You can now start the backend:")
print("   cd backend && uvicorn main:app --reload")