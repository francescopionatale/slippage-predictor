"""Empirical, non-circular slippage target.

The synthetic proxy (``proxy.label``) is a closed-form function of the
model's own features, so the model can only re-derive arithmetic. This
module builds a target from *empirically estimated* market-impact
quantities instead:

* **Amihud illiquidity** — ``mean(|return| / dollar_volume)``, the
  canonical low-frequency price-impact measure (Amihud, 2002).
* **Kyle's lambda** — the regression slope of returns on signed order
  flow (Kyle, 1985), estimated on a rolling window via the tick rule.

Crucially the target is the illiquidity realised ``horizon`` bars **in the
future**, predicted from features known **now**. That makes it a genuine
forecasting problem: the label is a noisy estimate of a real quantity the
model does not directly observe, so it can actually be wrong — unlike the
synthetic proxy. The absolute scale is normalised to a bps-like range and
is *relative* (not a broker-calibrated cost); the scientific point is that
the circularity is broken, not the exact units.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-12


def amihud_illiquidity(
    close: pd.Series, volume: pd.Series, window: int = 20
) -> pd.Series:
    """Rolling Amihud illiquidity: mean over ``window`` of |ret| / dollar_volume.

    Higher values mean a given amount of trading moves the price more, i.e.
    the asset is less liquid and execution is more expensive.
    """
    ret = np.log(close / close.shift(1))
    dollar_volume = (close * volume).clip(lower=EPS)
    impact = ret.abs() / dollar_volume
    return impact.rolling(window, min_periods=window).mean()


def rolling_kyle_lambda(
    close: pd.Series, volume: pd.Series, window: int = 20
) -> pd.Series:
    """Rolling Kyle's lambda: slope of returns on signed dollar order flow.

    Order-flow sign is approximated with the tick rule (sign of the bar
    return). ``lambda = cov(ret, signed_flow) / var(signed_flow)`` over the
    window — the marginal price impact per unit of net flow.
    """
    ret = np.log(close / close.shift(1))
    signed_flow = np.sign(ret) * (close * volume)

    def _slope(idx: np.ndarray) -> float:
        r = ret.values[idx]
        f = signed_flow.values[idx]
        var = np.var(f)
        if var <= EPS:
            return np.nan
        return float(np.cov(r, f, ddof=0)[0, 1] / var)

    # Rolling regression slope (small windows → cheap).
    out = pd.Series(np.nan, index=close.index)
    n = len(close)
    positions = np.arange(n)
    for end in range(window, n + 1):
        idx = positions[end - window:end]
        out.iloc[end - 1] = _slope(idx)
    return out


def build_empirical_target(
    df: pd.DataFrame,
    features: pd.DataFrame,
    window: int = 20,
    horizon: int = 1,
    base_bps: float = 5.0,
) -> pd.DataFrame:
    """Attach a future-illiquidity target to ``features``.

    The target is the Amihud illiquidity realised ``horizon`` bars ahead,
    median-normalised and scaled to ``base_bps`` so it sits in a familiar
    range. Because it looks *forward*, predicting it from current features
    is a real (falsifiable) task — there is no closed-form shortcut.

    Parameters
    ----------
    df:
        Raw OHLCV with a DatetimeIndex (uses ``close``, ``volume``).
    features:
        Market features aligned to ``df`` (typically a subset after dropna).
    window:
        Rolling window for the illiquidity estimate.
    horizon:
        How many bars ahead the target illiquidity is measured.
    base_bps:
        Scale of the median target (relative cost, not calibrated absolute).

    Returns
    -------
    Copy of ``features`` (restricted to shared rows) with added columns
    ``slippage_bps`` (the forward illiquidity target) and ``kyle_lambda``
    (an explanatory liquidity feature). Rows without a future target are
    dropped.
    """
    illiq = amihud_illiquidity(df["close"], df["volume"], window=window)
    kyle = rolling_kyle_lambda(df["close"], df["volume"], window=window)

    # Forward-looking target: illiquidity `horizon` bars ahead.
    future_illiq = illiq.shift(-horizon)
    median = np.nanmedian(illiq.values)
    if not np.isfinite(median) or median <= 0:
        median = np.nanmean(illiq.values)
    target_bps = base_bps * future_illiq / (median + EPS)

    shared = features.index.intersection(df.index)
    out = features.loc[shared].copy()
    out["kyle_lambda"] = kyle.loc[shared]
    out["slippage_bps"] = target_bps.loc[shared]
    return out.dropna(subset=["slippage_bps", "kyle_lambda"])
