"""Tests for the quantile / distributional slippage head."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from slippage.features import FEATURE_NAMES_TRAINING
from slippage.training import (
    interval_coverage,
    pinball_loss,
    predict_quantiles,
    train_quantile,
)

N_FEATURES = len(FEATURE_NAMES_TRAINING)


def test_pinball_loss_median_equals_half_mae():
    """At tau=0.5 the pinball loss is half the absolute error."""
    preds = torch.tensor([[1.0], [2.0], [3.0]])
    target = torch.tensor([2.0, 2.0, 2.0])
    loss = pinball_loss(preds, target, [0.5]).item()
    expected = 0.5 * np.mean(np.abs(np.array([1.0, 2.0, 3.0]) - 2.0))
    assert loss == pytest.approx(expected, rel=1e-5)


def test_pinball_loss_asymmetry():
    """A high quantile penalises under-prediction more than over-prediction."""
    target = torch.tensor([10.0])
    under = pinball_loss(torch.tensor([[5.0]]), target, [0.9]).item()
    over = pinball_loss(torch.tensor([[15.0]]), target, [0.9]).item()
    assert under > over


def test_predict_quantiles_monotone(small_split):
    model, hist = train_quantile(
        small_split, n_features=N_FEATURES, quantiles=(0.1, 0.5, 0.9),
        epochs=3, seed=0,
    )
    q = predict_quantiles(model, small_split.X_test)
    assert q.shape == (len(small_split.X_test), 3)
    # Non-crossing: each row sorted ascending.
    assert (np.diff(q, axis=1) >= 0).all()


def test_interval_coverage_basic():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    lower = np.array([0.0, 0.0, 0.0, 0.0])
    upper = np.array([2.5, 2.5, 2.5, 2.5])
    # 1,2 inside; 3,4 outside -> 0.5
    assert interval_coverage(y, lower, upper) == 0.5


def test_quantile_interval_roughly_calibrated():
    """On a simple homoscedastic signal the 10–90 interval should cover
    most points (loose bound — we only check it's a sane interval)."""
    rng = np.random.default_rng(0)
    n = 1500
    from sklearn.preprocessing import StandardScaler

    from slippage.data import SplitData

    X = rng.standard_normal((n, N_FEATURES)).astype("float32")
    signal = 5.0 + 2.0 * X[:, 0]
    y = np.abs(signal + rng.standard_normal(n) * 1.0).astype("float32")
    sc = StandardScaler().fit(X[:1000])
    split = SplitData(
        sc.transform(X[:1000]), y[:1000],
        sc.transform(X[1000:1250]), y[1000:1250],
        sc.transform(X[1250:]), y[1250:],
        sc, None, None, None,
    )
    model, _ = train_quantile(split, n_features=N_FEATURES,
                              quantiles=(0.1, 0.5, 0.9), epochs=40, seed=0)
    q = predict_quantiles(model, split.X_test)
    cov = interval_coverage(split.y_test, q[:, 0], q[:, 2])
    # Nominal 80% interval; allow a wide tolerance for the small/quick fit.
    assert 0.6 <= cov <= 0.95
