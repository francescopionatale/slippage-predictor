"""Tests for the walk-forward cross-validation routine."""

import numpy as np
import pandas as pd
import pytest

from slippage.features import FEATURE_NAMES_TRAINING
from slippage.walk_forward import summarise, walk_forward_cv


@pytest.fixture
def long_proxy_df():
    """A multi-month synthetic proxy long enough for walk-forward folds.

    The standard ``proxy_df`` fixture only covers ~12 days of hourly bars
    (n=300), which isn't enough for a 2-fold × 1-month CV. This fixture
    spans ~6 months so folds can carve out a real test window each.
    """
    n = 4_400  # ~6 months of hourly bars
    rng = np.random.default_rng(0)
    index = pd.date_range("2024-01-02 09:30", periods=n, freq="1h", tz="UTC")
    data = {col: rng.standard_normal(n) for col in FEATURE_NAMES_TRAINING}
    # Synthetic slippage: noisy linear combo + offset (kept ≥ 0 conceptually,
    # though the regression target is unconstrained here — we only test the
    # CV machinery, not predictive quality).
    data["slippage_bps"] = (
        2.0 * data["vol_rolling"]
        + 1.5 * data["order_size_fraction"]
        + rng.standard_normal(n) * 0.5
        + 3.0
    )
    return pd.DataFrame(data, index=index)


EXPECTED_MODELS = {"mlp", "mean", "linear", "heuristic", "gbm"}


def test_walk_forward_cv_shape(long_proxy_df):
    results = walk_forward_cv(
        long_proxy_df, n_folds=2, test_months=1.0,
        epochs=2, seed=0, dropout=0.0,
    )
    # 2 folds × len(EXPECTED_MODELS) rows
    assert len(results) == 2 * len(EXPECTED_MODELS)
    assert set(results["model"].unique()) == EXPECTED_MODELS
    assert set(results["fold"].unique()) == {0, 1}


def test_walk_forward_cv_metrics_finite(long_proxy_df):
    results = walk_forward_cv(
        long_proxy_df, n_folds=2, test_months=1.0,
        epochs=2, seed=0, dropout=0.0,
    )
    for col in ("mae_bps", "rmse_bps", "med_ae_bps"):
        assert np.isfinite(results[col]).all(), f"{col} contains non-finite values"
        assert (results[col] >= 0).all(), f"{col} must be non-negative"


def test_walk_forward_expanding_window(long_proxy_df):
    """n_train must grow monotonically across folds (expanding window)."""
    results = walk_forward_cv(
        long_proxy_df, n_folds=3, test_months=1.0,
        epochs=2, seed=0, dropout=0.0,
    )
    n_train_per_fold = results.groupby("fold")["n_train"].first().sort_index()
    diffs = n_train_per_fold.diff().dropna()
    assert (diffs > 0).all(), (
        f"Expected expanding window; n_train series: {n_train_per_fold.tolist()}"
    )


def test_walk_forward_summarise(long_proxy_df):
    results = walk_forward_cv(
        long_proxy_df, n_folds=2, test_months=1.0,
        epochs=2, seed=0, dropout=0.0,
    )
    summary = summarise(results)
    assert len(summary) == len(EXPECTED_MODELS)
    for col in ("mae_bps_mean", "mae_bps_std", "rmse_bps_mean", "med_ae_bps_mean"):
        assert col in summary.columns
        assert np.isfinite(summary[col]).all()
