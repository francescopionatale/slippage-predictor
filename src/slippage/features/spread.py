"""Corwin-Schultz bid-ask spread estimator."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _corwin_schultz_spread(
    high: pd.Series,
    low: pd.Series,
) -> pd.Series:
    """Estimate bid-ask spread via the Corwin-Schultz (2012) formula.

    Uses pairs of consecutive bars (t-1, t). Returns a Series aligned with
    the input index. Negative estimates (noise artefacts) are clipped to 0.

    Reference: Corwin & Schultz (2012), "A Simple Way to Estimate Bid-Ask
    Spreads from Daily High and Low Prices."
    """
    ln_h = np.log(high)
    ln_l = np.log(low)

    beta = (ln_h - ln_l) ** 2 + (ln_h.shift(1) - ln_l.shift(1)) ** 2
    gamma = (
        np.log(np.maximum(high, high.shift(1)) / np.minimum(low, low.shift(1)))
    ) ** 2

    k = 3.0 - 2.0 * np.sqrt(2.0)  # = 3 - 2√2 ≈ 0.172
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)

    spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    spread = spread.clip(lower=0)
    return spread
