"""End-to-end helpers that orchestrate data → features → proxy → split.

Centralised here so notebooks, train.py, and evaluate.py don't drift apart.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from slippage.data_loader import TICKERS, download_ohlcv
from slippage.dataset import SplitData, temporal_split
from slippage.features import add_synthetic_orders, compute_market_features
from slippage.proxy import build_proxy


def _ticker_seed(ticker: str, base_seed: int = 42) -> int:
    """Deterministic per-ticker seed so each ticker gets independent synthetic orders."""
    digest = hashlib.md5(f"{base_seed}:{ticker}".encode()).hexdigest()
    return int(digest[:8], 16) % (2**32)


def build_full_proxy(
    data: dict[str, pd.DataFrame],
    alpha: float = 2.0,
    base_seed: int = 42,
) -> pd.DataFrame:
    """Build the concatenated multi-ticker proxy DataFrame.

    Each ticker gets its own deterministic RNG so the result is independent
    of dict iteration order.
    """
    parts = []
    for ticker, df in data.items():
        rng = np.random.default_rng(_ticker_seed(ticker, base_seed))
        mf = compute_market_features(df)
        feats = add_synthetic_orders(mf, rng=rng)
        proxy = build_proxy(df, feats, alpha=alpha)
        proxy["ticker"] = ticker
        parts.append(proxy)
    return pd.concat(parts).sort_index()


def build_full_dataset(
    tickers: list[str] | None = None,
    alpha: float = 2.0,
    base_seed: int = 42,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, SplitData]:
    """Download, build proxy, and split. Used by train/evaluate CLIs and notebooks."""
    if tickers is None:
        tickers = TICKERS
    data = download_ohlcv(tickers)
    proxy_all = build_full_proxy(data, alpha=alpha, base_seed=base_seed)
    split = temporal_split(proxy_all)
    return data, proxy_all, split
