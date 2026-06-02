"""Download and cache OHLCV data from Yahoo Finance."""

from __future__ import annotations

import hashlib
import time
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf

from slippage.logging import get_logger
from slippage.paths import RAW_DIR

logger = get_logger(__name__)

_EXPECTED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


class DataDownloadError(RuntimeError):
    """Raised when market data cannot be obtained or is structurally invalid."""


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


def _download_with_retry(
    symbol: str, start: str, end: str, interval: str, retries: int = 3, backoff: float = 2.0
) -> pd.DataFrame:
    """Call ``yf.download`` with retry + exponential backoff on transient errors.

    Returns the (possibly empty) raw DataFrame. Raises :class:`DataDownloadError`
    only if every attempt raises an exception (network/API failure).
    """
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return yf.download(
                    symbol, start=start, end=end, interval=interval,
                    auto_adjust=True, progress=False,
                )
        except Exception as exc:  # noqa: BLE001 — yfinance raises a variety of errors
            last_exc = exc
            wait = backoff ** (attempt - 1)
            logger.warning(
                "%s: download attempt %d/%d failed (%s); retrying in %.0fs",
                symbol, attempt, retries, exc, wait,
            )
            if attempt < retries:
                time.sleep(wait)
    raise DataDownloadError(
        f"failed to download {symbol} after {retries} attempts"
    ) from last_exc


def _fetch_one(
    symbol: str,
    start: str,
    end: str,
    interval: str,
) -> pd.DataFrame | None:
    """Fetch + cache one symbol's cleaned OHLCV. Returns None if yfinance returns empty.

    Raises :class:`DataDownloadError` on persistent network failure or if the
    response is missing the expected OHLCV columns (a schema change worth
    surfacing rather than swallowing).
    """
    path = _cache_path(symbol, interval, start, end)
    if path.exists():
        try:
            return pd.read_parquet(path)
        except Exception as exc:  # noqa: BLE001 — corrupt cache: refetch
            logger.warning("%s: corrupt cache %s (%s); refetching", symbol, path, exc)
            path.unlink(missing_ok=True)

    raw = _download_with_retry(symbol, start, end, interval)
    if raw.empty:
        return None

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    missing = [c for c in _EXPECTED_COLUMNS if c not in raw.columns]
    if missing:
        raise DataDownloadError(
            f"{symbol}: response missing expected columns {missing} "
            f"(got {list(raw.columns)})"
        )

    df = raw[_EXPECTED_COLUMNS].copy()
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
                logger.warning(
                    "%s: %d bars (<%d) — trying fallback %s", ticker, n, MIN_BARS, fallback
                )
                df = _fetch_one(fallback, start, end, interval)
                if df is None or len(df) < MIN_BARS:
                    n2 = 0 if df is None else len(df)
                    logger.warning("fallback %s: %d bars — skipping", fallback, n2)
                    continue
                key = fallback
            else:
                logger.warning("%s: %d bars — skipping (threshold %d)", ticker, n, MIN_BARS)
                continue
        else:
            key = ticker

        result[key] = df
        logger.info(
            "%s: %d bars (%s → %s)", key, len(df), df.index[0].date(), df.index[-1].date()
        )

    if not result:
        raise DataDownloadError(
            f"no tickers produced >= {MIN_BARS} usable bars (requested {len(tickers)}); "
            "check connectivity, the date range, or MIN_BARS."
        )
    return result


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Remove zero-volume bars and rows with NaN prices."""
    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df[df["volume"] > 0]
    df = df.sort_index()
    return df


def main() -> None:
    """CLI entry point: download OHLCV data for the canonical ticker universe."""
    import argparse

    from slippage.logging import configure_logging
    from slippage.paths import ensure_dirs

    configure_logging()
    ensure_dirs()

    parser = argparse.ArgumentParser(description="Download OHLCV data via yfinance")
    parser.add_argument("--tickers", nargs="+", default=TICKERS)
    parser.add_argument("--start", default=_default_start())
    parser.add_argument("--end", default=_default_end())
    parser.add_argument(
        "--interval",
        default="1h",
        help="Bar interval. Use 1h for 2-year coverage; 5m is limited to ~60 days.",
    )
    args = parser.parse_args()
    data = download_ohlcv(args.tickers, args.start, args.end, args.interval)
    total = sum(len(df) for df in data.values())
    logger.info("Downloaded %d bars across %d tickers.", total, len(data))


if __name__ == "__main__":
    main()
