"""Download and cache OHLCV data from Yahoo Finance."""

from __future__ import annotations

import hashlib
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf

from paths import RAW_DIR


def _default_start() -> str:
    """Return a start date safely within Yahoo Finance's 730-day 1h limit."""
    return (datetime.now(timezone.utc) - timedelta(days=720)).strftime("%Y-%m-%d")


def _default_end() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


TICKERS = [
    # ETFs — ultra-liquid market baselines
    "SPY", "QQQ", "IWM",
    # Mega-cap tech — retained from v2 universe
    "AAPL", "MSFT", "GOOGL",
    # Financials — JPM (ultra-liquid) + GS (rate-sensitive)
    "JPM", "GS",
    # Energy / commodity-adjacent
    "XOM", "SLB",
    # Small-cap illiquid stress test (falls back to MGNI if PRCT bars are short)
    "PRCT",
]

# If a primary ticker returns fewer than MIN_BARS after cleanup, the downloader
# transparently swaps in its fallback. Only one swap per primary, no chaining.
TICKER_FALLBACKS: dict[str, str] = {"PRCT": "MGNI"}

MIN_BARS = 500  # minimum cleaned 1h bars to keep a ticker (enough for all walk-forward folds)


def _cache_path(ticker: str, interval: str, start: str, end: str) -> Path:
    key = f"{ticker}_{interval}_{start}_{end}"
    slug = hashlib.md5(key.encode()).hexdigest()[:8]
    return RAW_DIR / f"{ticker}_{interval}_{slug}.parquet"


def _fetch_one(
    symbol: str,
    start: str,
    end: str,
    interval: str,
) -> pd.DataFrame | None:
    """Fetch + cache one symbol's cleaned OHLCV. Returns None if yfinance returns empty."""
    path = _cache_path(symbol, interval, start, end)
    if path.exists():
        return pd.read_parquet(path)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = yf.download(
            symbol,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=True,
            progress=False,
        )
    if raw.empty:
        return None

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.columns = ["open", "high", "low", "close", "volume"]
    df.index = pd.to_datetime(df.index, utc=True)
    df = _clean(df)
    df.to_parquet(path)
    return df


def download_ohlcv(
    tickers: list[str] = TICKERS,
    start: str | None = None,
    end: str | None = None,
    interval: str = "1h",
) -> dict[str, pd.DataFrame]:
    """Download and cache OHLCV data for each ticker.

    Tickers with fewer than ``MIN_BARS`` cleaned bars are dropped from the
    result. If a primary ticker has a fallback registered in
    ``TICKER_FALLBACKS`` (e.g. PRCT → MGNI), the fallback is fetched
    automatically and stored under the fallback symbol's key.

    Parameters
    ----------
    tickers:
        List of Yahoo Finance ticker symbols.
    start, end:
        ISO-format date strings (YYYY-MM-DD).
    interval:
        Bar interval. Use "1h" for 2-year coverage; "5m" is limited to ~60 days
        by Yahoo Finance's API.

    Returns
    -------
    dict mapping ticker -> cleaned DataFrame with columns
    [open, high, low, close, volume] and a DatetimeIndex.
    """
    if start is None:
        start = _default_start()
    if end is None:
        end = _default_end()

    result: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        df = _fetch_one(ticker, start, end, interval)
        if df is None or len(df) < MIN_BARS:
            n = 0 if df is None else len(df)
            fallback = TICKER_FALLBACKS.get(ticker)
            if fallback and fallback not in result:
                print(f"[warn] {ticker}: {n} bars (<{MIN_BARS}) — trying fallback {fallback}")
                df = _fetch_one(fallback, start, end, interval)
                if df is None or len(df) < MIN_BARS:
                    n2 = 0 if df is None else len(df)
                    print(f"[warn] fallback {fallback}: {n2} bars — skipping")
                    continue
                key = fallback
            else:
                print(f"[warn] {ticker}: {n} bars — skipping (threshold {MIN_BARS})")
                continue
        else:
            key = ticker

        result[key] = df
        print(f"[data] {key}: {len(df)} bars ({df.index[0].date()} → {df.index[-1].date()})")
    return result


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Remove zero-volume bars and rows with NaN prices."""
    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df[df["volume"] > 0]
    df = df.sort_index()
    return df
