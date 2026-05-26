"""Unit tests for slippage proxy construction."""

import numpy as np
import pytest

from slippage.features import FEATURE_NAMES_TRAINING, add_synthetic_orders, compute_market_features
from slippage.proxy import build_proxy, calibrate_alpha


def _make_dataset(ohlcv_df, seed=1):
    df = ohlcv_df(n=80, seed=seed, min_hl=0.001)
    mf = compute_market_features(df)
    feats = add_synthetic_orders(mf, rng=np.random.default_rng(seed))
    proxy = build_proxy(df, feats, alpha=2.0, impact_noise=0.0)
    return df, feats, proxy


def test_proxy_no_nan(ohlcv_df):
    df, feats, _ = _make_dataset(ohlcv_df)
    proxy = build_proxy(df, feats, alpha=2.0, rng=np.random.default_rng(0))
    assert proxy["slippage_bps"].isna().sum() == 0


def test_proxy_impact_increases_with_alpha(ohlcv_df):
    """alpha>0 must produce strictly positive slippage; alpha=0 must produce zero."""
    df = ohlcv_df(n=80, seed=1, min_hl=0.001)
    mf = compute_market_features(df)
    feats = add_synthetic_orders(mf, rng=np.random.default_rng(1))
    proxy_zero = build_proxy(df, feats, alpha=0.0, impact_noise=0.0)
    proxy_two = build_proxy(df, feats, alpha=2.0, impact_noise=0.0)
    assert (proxy_zero["slippage_bps"] == 0).all()
    assert (proxy_two["slippage_bps"] > 0).all()
    assert proxy_two["slippage_bps"].mean() > proxy_zero["slippage_bps"].mean()


def test_proxy_always_non_negative(ohlcv_df):
    """Slippage must be >= 0 in both deterministic and stochastic settings."""
    df = ohlcv_df(n=500, seed=1, min_hl=0.001)
    mf = compute_market_features(df)
    feats = add_synthetic_orders(mf, rng=np.random.default_rng(1))
    proxy_det = build_proxy(df, feats, alpha=2.0, impact_noise=0.0)
    proxy_sto = build_proxy(df, feats, alpha=2.0, impact_noise=0.20,
                            rng=np.random.default_rng(99))
    assert (proxy_det["slippage_bps"] >= 0).all()
    assert (proxy_sto["slippage_bps"] >= 0).all()


def test_proxy_monotone_in_alpha(ohlcv_df):
    """Higher alpha → larger average slippage_bps (more market impact)."""
    df = ohlcv_df(n=80, seed=1, min_hl=0.001)
    mf = compute_market_features(df)
    feats = add_synthetic_orders(mf, rng=np.random.default_rng(0))
    prev_mean = None
    for alpha in [0.5, 2.0, 5.0]:
        proxy = build_proxy(df, feats, alpha=alpha, impact_noise=0.0)
        mean_slip = proxy["slippage_bps"].mean()
        if prev_mean is not None:
            assert mean_slip > prev_mean, f"alpha={alpha} should give higher slippage than previous"
        prev_mean = mean_slip


def test_proxy_keeps_all_bars(ohlcv_df):
    """Proxy no longer depends on t+1, so every feature row must be preserved."""
    df = ohlcv_df(n=80, seed=1, min_hl=0.001)
    mf = compute_market_features(df)
    feats = add_synthetic_orders(mf, rng=np.random.default_rng(0))
    proxy = build_proxy(df, feats, alpha=2.0, impact_noise=0.0)
    assert len(proxy) == len(feats)
    # Last feature timestamp must be present in the proxy
    assert feats.index[-1] in proxy.index


def test_proxy_noise_breaks_determinism(ohlcv_df):
    """Stochastic impact (noise>0) must produce different slippage than deterministic."""
    df = ohlcv_df(n=80, seed=1, min_hl=0.001)
    mf = compute_market_features(df)
    feats = add_synthetic_orders(mf, rng=np.random.default_rng(1))
    proxy_det = build_proxy(df, feats, alpha=2.0, impact_noise=0.0)
    proxy_sto = build_proxy(df, feats, alpha=2.0, impact_noise=0.20, rng=np.random.default_rng(99))
    assert not np.allclose(proxy_det["slippage_bps"].values, proxy_sto["slippage_bps"].values)


def test_proxy_monotone_in_urgency(ohlcv_df):
    """At parity of other inputs, higher urgency must yield higher slippage.

    Urgency enters the proxy as ``(1 + urgency**1.5)``, so urgency=1.0
    contributes a 2× multiplier vs urgency=0.0.
    """
    df = ohlcv_df(n=80, seed=1, min_hl=0.001)
    mf = compute_market_features(df)
    feats = add_synthetic_orders(mf, rng=np.random.default_rng(0))

    feats_low = feats.copy()
    feats_low["urgency"] = 0.0
    feats_high = feats.copy()
    feats_high["urgency"] = 1.0

    slip_low = build_proxy(df, feats_low, alpha=2.0, impact_noise=0.0)["slippage_bps"]
    slip_high = build_proxy(df, feats_high, alpha=2.0, impact_noise=0.0)["slippage_bps"]
    # Per-row strict inequality wherever the base impact is non-zero
    nonzero = slip_low > 0
    assert (slip_high.loc[nonzero] > slip_low.loc[nonzero]).all()
    # And the urgency=1.0 contribution roughly doubles the impact
    ratio = slip_high.loc[nonzero] / slip_low.loc[nonzero]
    assert np.allclose(ratio, 2.0, rtol=1e-6)


def test_proxy_noise_requires_rng(ohlcv_df):
    """build_proxy must raise ValueError when impact_noise > 0 and rng is None."""
    df = ohlcv_df(n=80, seed=1, min_hl=0.001)
    mf = compute_market_features(df)
    feats = add_synthetic_orders(mf, rng=np.random.default_rng(1))
    with pytest.raises(ValueError, match="requires an explicit rng"):
        build_proxy(df, feats, alpha=2.0, impact_noise=0.20, rng=None)


def test_calibrate_alpha_runs_and_returns_best(proxy_df):
    from slippage.data import temporal_split

    def _build(alpha):
        return temporal_split(proxy_df(n=300, seed=int(alpha * 10)))

    best, sens = calibrate_alpha(
        _build, n_features=len(FEATURE_NAMES_TRAINING),
        alphas=[1.0, 2.0], epochs=2, seed=0,
    )
    assert best in [1.0, 2.0]
    assert set(sens.keys()) == {1.0, 2.0}
    assert all(np.isfinite(v) for v in sens.values())
