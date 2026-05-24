#!/usr/bin/env python
"""CLI script to download and cache OHLCV data.

Usage:
    python data/download.py
    python data/download.py --interval 5m --start 2024-11-01 --end 2025-01-01
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_loader import TICKERS, download_ohlcv, _default_start, _default_end


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
