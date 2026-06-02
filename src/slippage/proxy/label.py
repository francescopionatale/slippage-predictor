"""Synthetic slippage proxy label construction.

Since real execution data is unavailable, slippage is constructed as the
**expected execution cost**: a market-impact penalty scaled by volatility,
order size, and a tunable α, with several multiplicative modulators
(urgency, spread, time-of-day) and multiplicative log-normal noise.

Price drift between the arrival bar and a future bar is deliberately
excluded — it is market risk during the holding period, not execution
cost, and conflating the two drowns the impact signal in random-walk
noise. See README "Why no price drift?" for the full rationale.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from slippage.features import eastern_hours


def build_proxy(
    df: pd.DataFrame,
    features: pd.DataFrame,
    alpha: float = 2.0,
    impact_noise: float = 0.20,
    rng: np.random.Generator | None = None,
    urgency_exp: float = 1.5,
    spread_mult: float = 50.0,
    tod_mult: float = 1.3,
    tod_open_thresh: float = 10.5,
    tod_close_thresh: float = 15.0,
) -> pd.DataFrame:
    """Build the synthetic slippage label for every row in ``features``.

    Follows the empirical square-root law for temporary market impact
    (Almgren et al. 2005; Gatheral & Schied 2013): I(q) ∝ σ · √(q/V).

    Formula (all on bar t, no lookahead to t+1):

        arrival          = close_t
        urgency_factor   = 1 + urgency ** 1.5                          # ∈ [1, 2]
        spread_penalty   = 1 + 50 × spread_cs × order_size_fraction    # ≥ 1
        tod_mult         = 1.3 if hour_ET < 10.5 or hour_ET ≥ 15.0     # opening/close premium
                           else 1.0
        impact           = α × σ_t × √(order_size_fraction) × arrival
                           × urgency_factor × spread_penalty × tod_mult × exp(η)
        exec_price       = arrival + side × impact
        slippage_bps     = 10_000 × side × (exec_price − arrival) / arrival
                         = 10_000 × impact / arrival

    The three new multipliers make the label a genuine function of
    ``urgency``, ``spread_cs``, ``order_size_fraction`` (via the spread
    interaction), and time-of-day (via the index). The ``side`` column
    still cancels algebraically. The closed-form heuristic baseline
    (``β × √size × vol × 10_000``) can only approximate the *average* of
    the modulators, leaving headroom for the MLP.

    The noise is log-normal so that ``impact`` is always positive (real
    execution costs cannot be negative). With ``impact_noise = σ_log = 0.20``
    (≈ 21% multiplicative std), the implied R² ceiling is approximately 0.96.
    The multiplicative factor ``exp(η)`` has median 1 and is broadly
    symmetric on a log scale, matching the standard volatility model for
    multiplicative shocks in finance.

    Parameters
    ----------
    df:
        Raw OHLCV DataFrame with a DatetimeIndex. Only ``close`` is used.
    features:
        Output of ``add_synthetic_orders``; must contain columns
        ``side``, ``order_size_fraction``, ``vol_rolling``, ``urgency``,
        ``spread_cs``.
    alpha:
        Market-impact scale factor. Higher alpha → larger impact penalty.
    impact_noise:
        σ_log of multiplicative log-normal noise on the impact term
        (``η ~ N(0, impact_noise)``). Default 0.20 ≈ 21% multiplicative std.
        Set to 0.0 for a deterministic proxy.
    rng:
        Required when ``impact_noise > 0`` for reproducibility.

    Returns
    -------
    DataFrame equal to ``features`` with added ``slippage_bps`` and
    ``arrival_price`` columns. All rows are preserved (no last-bar drop,
    since the proxy no longer depends on bar t+1).
    """
    # Align features to the OHLCV index (features may be a subset after dropna)
    shared = features.index.intersection(df.index)
    feats = features.loc[shared].copy()
    arrival = df["close"].loc[shared]

    urgency_factor = 1.0 + feats["urgency"] ** urgency_exp
    spread_penalty = 1.0 + spread_mult * feats["spread_cs"] * feats["order_size_fraction"]

    hours = eastern_hours(feats.index)
    tod_factor = pd.Series(
        np.where((hours < tod_open_thresh) | (hours >= tod_close_thresh), tod_mult, 1.0),
        index=feats.index,
    )

    impact = (
        alpha
        * np.sqrt(np.maximum(feats["order_size_fraction"], 0))
        * feats["vol_rolling"]
        * arrival
        * urgency_factor
        * spread_penalty
        * tod_factor
    )

    if impact_noise > 0:
        if rng is None:
            raise ValueError(
                "build_proxy requires an explicit rng when impact_noise > 0, "
                "for reproducibility."
            )
        log_noise = rng.normal(0.0, impact_noise, size=len(impact))
        impact = impact * np.exp(log_noise)

    # slippage = side × (exec_price − arrival) = side × side × impact = impact.
    # Slippage is always a positive cost regardless of side.
    feats["slippage_bps"] = 10_000.0 * impact / arrival
    feats["arrival_price"] = arrival
    return feats
