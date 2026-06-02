"""Integration test: the offline demo path on the committed sample.

Guards `make demo` / scripts/demo.py against regressions without any
network access.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from slippage.data import temporal_split
from slippage.evaluation import build_baselines, global_metrics
from slippage.features import FEATURE_NAMES_TRAINING, add_synthetic_orders, compute_market_features
from slippage.paths import PROJECT_ROOT
from slippage.proxy import build_proxy
from slippage.training import predict, train

SAMPLE = PROJECT_ROOT / "data" / "sample" / "sample_ohlcv.parquet"


def test_sample_data_is_committed():
    assert SAMPLE.exists(), "offline demo sample data is missing"


def test_demo_offline_pipeline_runs():
    df = pd.read_parquet(SAMPLE)
    mf = compute_market_features(df)
    feats = add_synthetic_orders(mf, rng=np.random.default_rng(42))
    proxy = build_proxy(df, feats, alpha=2.0, impact_noise=0.2, rng=np.random.default_rng(43))
    proxy["ticker"] = "DEMO"
    split = temporal_split(proxy.sort_index())

    model, _ = train(
        split, n_features=len(FEATURE_NAMES_TRAINING),
        epochs=5, verbose=False, checkpoint_path=None,
    )
    preds = {"mlp": predict(model, split.X_test)}
    preds.update({n: fn(split.X_test) for n, fn in build_baselines(split).items()})

    for name, p in preds.items():
        m = global_metrics(split.y_test, p)
        assert np.isfinite(m["mae_bps"]) and m["mae_bps"] >= 0, name
    # Deterministic, epoch-independent invariant: the linear and GBM learners
    # both extract signal the trivial mean predictor cannot.
    mean_mae = global_metrics(split.y_test, preds["mean"])["mae_bps"]
    assert global_metrics(split.y_test, preds["linear"])["mae_bps"] < mean_mae
    assert global_metrics(split.y_test, preds["gbm"])["mae_bps"] < mean_mae
