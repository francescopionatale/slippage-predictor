#!/usr/bin/env python
"""CLI script to download and cache OHLCV data.

Usage:
    python scripts/download_data.py
    python scripts/download_data.py --interval 5m --start 2024-11-01 --end 2025-01-01
"""

from __future__ import annotations

import argparse

from slippage.data import TICKERS, download_ohlcv
from slippage.data.loader import _default_end, _default_start


def main() -> None:
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
    print(f"\nDownloaded {total:,} bars across {len(data)} tickers.")


if __name__ == "__main__":
    main()
