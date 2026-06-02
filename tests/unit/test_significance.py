"""Tests for Diebold-Mariano and the block-bootstrap MAE CI."""

from __future__ import annotations

import numpy as np
import pytest

from slippage.evaluation import (
    block_bootstrap_mae_ci,
    compare_against_reference,
    diebold_mariano,
)


def test_dm_detects_clearly_better_model():
    rng = np.random.default_rng(0)
    n = 500
    y = rng.standard_normal(n) * 10
    good = y + rng.standard_normal(n) * 0.5   # small error
    bad = y + rng.standard_normal(n) * 5.0    # large error
    res = diebold_mariano(y, good, bad)
    assert res.p_value < 0.05
    assert res.favored == "A"          # A (good) has lower loss
    assert res.mean_loss_diff < 0


def test_dm_symmetric_under_swap():
    rng = np.random.default_rng(1)
    n = 400
    y = rng.standard_normal(n)
    a = y + rng.standard_normal(n) * 0.5
    b = y + rng.standard_normal(n) * 2.0
    r_ab = diebold_mariano(y, a, b)
    r_ba = diebold_mariano(y, b, a)
    assert r_ab.statistic == pytest.approx(-r_ba.statistic, rel=1e-6)
    assert r_ab.p_value == pytest.approx(r_ba.p_value, rel=1e-6)


def test_dm_tie_for_identical_forecasts():
    rng = np.random.default_rng(2)
    y = rng.standard_normal(300)
    pred = y + rng.standard_normal(300)
    res = diebold_mariano(y, pred, pred)
    assert res.favored == "tie"
    assert res.p_value == pytest.approx(1.0)


def test_block_bootstrap_ci_contains_point_mae():
    rng = np.random.default_rng(3)
    n = 600
    y = rng.standard_normal(n) * 4
    pred = y + rng.standard_normal(n) * 2
    mae, lo, hi = block_bootstrap_mae_ci(y, pred, n_boot=500, seed=0)
    assert lo <= mae <= hi
    assert hi > lo > 0


def test_block_bootstrap_reproducible():
    rng = np.random.default_rng(4)
    y = rng.standard_normal(300)
    pred = y + rng.standard_normal(300)
    r1 = block_bootstrap_mae_ci(y, pred, n_boot=200, seed=42)
    r2 = block_bootstrap_mae_ci(y, pred, n_boot=200, seed=42)
    assert r1 == r2


def test_compare_against_reference_structure():
    rng = np.random.default_rng(5)
    n = 400
    y = rng.standard_normal(n) * 5
    preds = {
        "mlp": y + rng.standard_normal(n) * 0.5,
        "mean": np.full(n, y.mean()),
    }
    table = compare_against_reference(y, preds, reference="mlp", n_boot=200)
    assert set(table) == {"mlp", "mean"}
    assert "dm_vs_reference" in table["mean"]
    assert "dm_vs_reference" not in table["mlp"]
    # mlp should significantly beat the mean predictor
    assert table["mean"]["dm_vs_reference"]["favored_reference"]
