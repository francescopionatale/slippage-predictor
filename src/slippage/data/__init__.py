"""Data loading and chronological split helpers."""

from slippage.data.loader import (
    MIN_BARS,
    TICKER_FALLBACKS,
    TICKERS,
    download_ohlcv,
)
from slippage.data.splits import SplitData, temporal_split

# Backwards-compatible re-export: SlippageDataset moved to slippage.models.dataset
# in the package refactor (Block 2.5). Re-export here so legacy
# `from slippage.data import SlippageDataset` continues to work.
from slippage.models.dataset import SlippageDataset

__all__ = [
    "TICKERS",
    "TICKER_FALLBACKS",
    "MIN_BARS",
    "download_ohlcv",
    "SplitData",
    "SlippageDataset",
    "temporal_split",
]
