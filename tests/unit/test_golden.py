"""Golden-file numerical regression tests.

Pin exact numeric outputs so a refactor that silently changes results is
caught as a failing diff rather than going unnoticed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from slippage.evaluation import mae, med_ae, rmse
from slippage.proxy import build_proxy


def test_metrics_golden():
    y_true = np.array([0.0, 1.0, 2.0, 3.0, 10.0])
    y_pred = np.array([0.0, 2.0, 0.0, 3.0, 6.0])
    # abs errors: [0, 1, 2, 0, 4]
    assert mae(y_true, y_pred) == pytest.approx(7 / 5)
    assert rmse(y_true, y_pred) == pytest.approx(np.sqrt((0 + 1 + 4 + 0 + 16) / 5))
    assert med_ae(y_true, y_pred) == pytest.approx(1.0)


def test_proxy_formula_golden():
    """With impact_noise=0 the proxy is a deterministic closed form; check it
    matches the documented formula exactly for a hand-computed row."""
    # A single bar at 14:00 ET (no tod premium: 10.5 <= 14 < 15).
    idx = pd.DatetimeIndex(["2024-01-03 14:00"], tz="America/New_York")
    df = pd.DataFrame({"close": [100.0]}, index=idx)
    feats = pd.DataFrame(
        {
            "side": [1.0],
            "order_size_fraction": [0.04],
            "vol_rolling": [0.01],
            "urgency": [0.5],
            "spread_cs": [0.002],
        },
        index=idx,
    )
    out = build_proxy(df, feats, alpha=2.0, impact_noise=0.0)

    alpha, size, vol, urgency, spread_cs = 2.0, 0.04, 0.01, 0.5, 0.002
    urgency_factor = 1.0 + urgency ** 1.5
    spread_penalty = 1.0 + 50.0 * spread_cs * size
    tod_mult = 1.0  # 14:00 ET is mid-session
    impact = alpha * np.sqrt(size) * vol * 100.0 * urgency_factor * spread_penalty * tod_mult
    expected_bps = 10_000.0 * impact / 100.0

    assert out["slippage_bps"].iloc[0] == pytest.approx(expected_bps, rel=1e-9)


def test_proxy_tod_premium_golden():
    """A 09:30 ET bar (< 10.5) gets the 1.3 opening premium."""
    idx = pd.DatetimeIndex(["2024-01-03 09:30"], tz="America/New_York")
    df = pd.DataFrame({"close": [100.0]}, index=idx)
    feats = pd.DataFrame(
        {"side": [1.0], "order_size_fraction": [0.04], "vol_rolling": [0.01],
         "urgency": [0.0], "spread_cs": [0.0]},
        index=idx,
    )
    out = build_proxy(df, feats, alpha=2.0, impact_noise=0.0)
    base = 2.0 * np.sqrt(0.04) * 0.01 * 1.0  # urgency_factor=1, spread_penalty=1
    expected_bps = 10_000.0 * base * 1.3
    assert out["slippage_bps"].iloc[0] == pytest.approx(expected_bps, rel=1e-9)
