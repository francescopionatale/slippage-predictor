"""End-to-end pipeline smoke test (no network access).

Exercises the wiring across data → features → proxy → split → models →
training → evaluation using a synthetic OHLCV fixture. Catches the kind
of cross-module breakage that pure unit tests miss (e.g., a function
signature change in slippage.features that breaks slippage.proxy).
"""

from __future__ import annotations

import numpy as np

from slippage.data import temporal_split
from slippage.evaluation import evaluate_all, global_metrics
from slippage.features import (
    FEATURE_NAMES_TRAINING,
    add_synthetic_orders,
    compute_market_features,
)
from slippage.models import HeuristicBaseline, MeanPredictor
from slippage.proxy import build_proxy
from slippage.training import predict, train


def test_pipeline_end_to_end(ohlcv_df):
    """Run the entire pipeline on a single synthetic ticker."""
    df = ohlcv_df(n=500, seed=0, min_hl=0.001)

    # 1. market features
    market_feats = compute_market_features(df)
    assert len(market_feats) > 100

    # 2. synthetic orders
    feats = add_synthetic_orders(market_feats, rng=np.random.default_rng(0))
    assert "side" in feats.columns
    assert len(feats) == len(market_feats) * 4  # default orders_per_bar=4

    # 3. proxy label
    proxy = build_proxy(
        df, feats, alpha=2.0, impact_noise=0.0,  # deterministic for speed
    )
    assert "slippage_bps" in proxy.columns
    assert (proxy["slippage_bps"] >= 0).all()

    # 4. temporal split + scaling
    split = temporal_split(proxy)
    assert split.X_train.shape[1] == len(FEATURE_NAMES_TRAINING)

    # 5. tiny training run (2 epochs only — we test wiring, not convergence)
    model, history = train(
        split,
        n_features=len(FEATURE_NAMES_TRAINING),
        epochs=2,
        patience=5,
        verbose=False,
        checkpoint_path=None,  # don't pollute artifacts/ during tests
    )
    assert len(history["train_loss"]) >= 1
    assert history["best_epoch"] >= 1

    # 6. inference + metrics
    y_pred = predict(model, split.X_test)
    metrics = global_metrics(split.y_test, y_pred)
    assert np.isfinite(metrics["mae_bps"])
    assert metrics["mae_bps"] > 0


def test_evaluate_all_with_baselines(ohlcv_df):
    """evaluate_all wires through MLP + baselines + per_ticker correctly."""
    df = ohlcv_df(n=400, seed=1, min_hl=0.001)
    market_feats = compute_market_features(df)
    feats = add_synthetic_orders(market_feats, rng=np.random.default_rng(1))
    proxy = build_proxy(df, feats, alpha=2.0, impact_noise=0.0)
    proxy["ticker"] = "SYNTH"  # per_ticker_metrics needs a ticker column

    split = temporal_split(proxy)
    model, _ = train(
        split,
        n_features=len(FEATURE_NAMES_TRAINING),
        epochs=2,
        patience=5,
        verbose=False,
        checkpoint_path=None,
    )

    mean_pred = MeanPredictor().fit(split.X_train, split.y_train)
    heuristic = HeuristicBaseline(FEATURE_NAMES_TRAINING)
    heuristic.fit(split.X_val, split.y_val, split.scaler.mean_, split.scaler.scale_)

    results = evaluate_all(
        split,
        model_predict_fn=lambda X: predict(model, X),
        baselines={
            "mean": mean_pred.predict,
            "heuristic": lambda X: heuristic.predict(X, split.scaler.mean_, split.scaler.scale_),
        },
    )

    assert {"mlp", "mean", "heuristic", "per_ticker"} <= set(results.keys())
    assert "segment_breakdown" in results["mlp"]
    assert "SYNTH" in results["per_ticker"]
    assert "mlp_mae_bps" in results["per_ticker"]["SYNTH"]
