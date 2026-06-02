"""End-to-end helpers that orchestrate data → features → proxy → split.

Centralised here so notebooks, train.py, and evaluate.py don't drift apart.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from slippage.data import TICKERS, SplitData, download_ohlcv, temporal_split
from slippage.features import add_synthetic_orders, compute_market_features
from slippage.proxy import build_proxy

if TYPE_CHECKING:
    from slippage.config import Config


def _ticker_seed(ticker: str, base_seed: int = 42) -> int:
    """Deterministic per-ticker seed so each ticker gets independent synthetic orders."""
    digest = hashlib.md5(f"{base_seed}:{ticker}".encode()).hexdigest()
    return int(digest[:8], 16) % (2**32)


def build_full_proxy(
    data: dict[str, pd.DataFrame],
    alpha: float = 2.0,
    base_seed: int = 42,
    impact_noise: float = 0.20,
    config: "Config | None" = None,
) -> pd.DataFrame:
    """Build the concatenated multi-ticker proxy DataFrame.

    Each ticker gets its own deterministic RNG so the result is independent
    of dict iteration order. Two separate RNGs are used per ticker: one for
    synthetic order sampling, one for impact noise.

    When ``config`` is provided, feature and proxy hyperparameters
    (``vol_window``, ``orders_per_bar``, size bounds, the proxy modulators)
    are read from it; ``alpha`` / ``impact_noise`` from ``config.proxy``
    override the positional defaults.
    """
    feat_kwargs: dict = {}
    proxy_kwargs: dict = {}
    market_kwargs: dict = {}
    if config is not None:
        alpha = config.proxy.alpha
        impact_noise = config.proxy.impact_noise
        market_kwargs = {"vol_window": config.features.vol_window}
        feat_kwargs = {
            "orders_per_bar": config.features.orders_per_bar,
            "size_low": config.features.size_low,
            "size_high": config.features.size_high,
        }
        proxy_kwargs = {
            "urgency_exp": config.proxy.urgency_exp,
            "spread_mult": config.proxy.spread_mult,
            "tod_mult": config.proxy.tod_mult,
            "tod_open_thresh": config.proxy.tod_open_thresh,
            "tod_close_thresh": config.proxy.tod_close_thresh,
        }

    parts = []
    for ticker, df in data.items():
        order_rng = np.random.default_rng(_ticker_seed(ticker, base_seed))
        proxy_rng = np.random.default_rng(_ticker_seed(ticker, base_seed + 1))
        mf = compute_market_features(df, **market_kwargs)
        feats = add_synthetic_orders(mf, rng=order_rng, **feat_kwargs)
        proxy = build_proxy(
            df, feats, alpha=alpha, impact_noise=impact_noise, rng=proxy_rng, **proxy_kwargs
        )
        proxy["ticker"] = ticker
        parts.append(proxy)
    return pd.concat(parts).sort_index()


def build_full_dataset(
    tickers: list[str] | None = None,
    alpha: float = 2.0,
    base_seed: int = 42,
    impact_noise: float = 0.20,
    config: "Config | None" = None,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, SplitData]:
    """Download, build proxy, and split. Used by train/evaluate CLIs and notebooks."""
    if config is not None and tickers is None:
        tickers = config.data.tickers
    if tickers is None:
        tickers = TICKERS
    data = download_ohlcv(tickers)
    proxy_all = build_full_proxy(
        data, alpha=alpha, base_seed=base_seed, impact_noise=impact_noise, config=config
    )
    split = temporal_split(proxy_all)
    return data, proxy_all, split
