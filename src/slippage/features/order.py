"""Synthetic order features + the combined feature-name catalogue."""

from __future__ import annotations

import numpy as np
import pandas as pd

from slippage.features.market import FEATURE_NAMES_MARKET

FEATURE_NAMES_ORDER = ["side", "order_size_fraction", "urgency"]
FEATURE_NAMES = FEATURE_NAMES_MARKET + FEATURE_NAMES_ORDER

# Features actually fed to the model — `side` is excluded because it
# cancels algebraically in the proxy formula (see proxy.build_proxy) and
# the model should not be allowed to learn anything from it.
FEATURE_NAMES_TRAINING = [f for f in FEATURE_NAMES if f != "side"]


def add_synthetic_orders(
    market_feats: pd.DataFrame,
    rng: np.random.Generator | None = None,
    orders_per_bar: int = 4,
    size_low: float = 0.001,
    size_high: float = 0.05,
) -> pd.DataFrame:
    """Expand market features by generating synthetic order features.

    For each input bar, ``orders_per_bar`` synthetic orders are created with
    alternating buy/sell sides and uniformly sampled size and urgency. The
    default 4 = 2 buy + 2 sell per bar gives wider per-bar coverage of the
    (size, urgency) plane and ~2× the training rows of the previous default.

    Parameters
    ----------
    market_feats:
        Output of ``compute_market_features``.
    rng:
        NumPy random Generator for reproducibility.
    orders_per_bar:
        Number of synthetic orders per bar. Must be even for balanced sides.
    size_low, size_high:
        Uniform bounds for ``order_size_fraction`` (fraction of avg volume).

    Returns
    -------
    DataFrame with ``len(market_feats) * orders_per_bar`` rows containing
    all market + order features.
    """
    if rng is None:
        raise ValueError(
            "add_synthetic_orders requires an explicit `rng` to keep "
            "synthetic-order generation reproducible and independent per ticker."
        )

    n = len(market_feats)
    rows = []
    for _ in range(orders_per_bar // 2):
        for side in [+1.0, -1.0]:
            chunk = market_feats.copy()
            chunk["side"] = side
            chunk["order_size_fraction"] = rng.uniform(size_low, size_high, size=n)
            chunk["urgency"] = rng.uniform(0.0, 1.0, size=n)
            rows.append(chunk)

    return pd.concat(rows).sort_index()
