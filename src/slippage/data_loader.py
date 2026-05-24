"""Download and cache OHLCV data from Yahoo Finance."""

from __future__ import annotations

import hashlib
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf

from slippage.paths import RAW_DIR


def _default_start() -> str:
    """Return a start date safely within Yahoo Finance's 730-day 1h limit."""
    return (datetime.now(timezone.utc) - timedelta(days=720)).strftime("%Y-%m-%d")


def _default_end() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


TICKERS = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "JPM"]


def _cache_path(ticker: str, interval: str, start: str, end: str) -> Path:
    key = f"{ticker}_{interval}_{start}_{end}"
    slug = hashlib.md5(key.encode()).hexdigest()[:8]
    return RAW_DIR / f"{ticker}_{interval}_{slug}.parquet"


def download_ohlcv(
    tickers: list[str] = TICKERS,
    start: str | None = None,
    end: str | None = None,
    interval: str = "1h",
) -> dict[str, pd.DataFrame]:
    """Download and cache OHLCV data for each ticker.

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
        path = _cache_path(ticker, interval, start, end)
        if path.exists():
            df = pd.read_parquet(path)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw = yf.download(
                    ticker,
                    start=start,
                    end=end,
                    interval=interval,
                    auto_adjust=True,
                    progress=False,
                )
            if raw.empty:
                print(f"[warn] {ticker}: no data returned")
                continue

            # Flatten multi-level columns if present
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)

            df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
            df.columns = ["open", "high", "low", "close", "volume"]
            df.index = pd.to_datetime(df.index, utc=True)
            df = _clean(df)
            df.to_parquet(path)

        result[ticker] = df
        print(f"[data] {ticker}: {len(df)} bars ({df.index[0].date()} → {df.index[-1].date()})")
    return result


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Remove zero-volume bars and rows with NaN prices."""
    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df[df["volume"] > 0]
    df = df.sort_index()
    return df
