"""Tests for the training loop: config wiring, determinism, guards, persistence."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from slippage.config import Config
from slippage.features import FEATURE_NAMES_TRAINING
from slippage.models import SlippageMLP
from slippage.training import load_model, load_scaler, predict, train, train_from_config

N_FEATURES = len(FEATURE_NAMES_TRAINING)


# ---------------------------------------------------------------------------
# B1 — config actually drives architecture / optimizer / scheduler
# ---------------------------------------------------------------------------

def test_hidden_drives_architecture():
    model = SlippageMLP(n_features=N_FEATURES, hidden=[128, 64, 32])
    widths = [m.out_features for m in model.net if isinstance(m, torch.nn.Linear)]
    assert widths == [128, 64, 32, 1]


def test_default_architecture_is_checkpoint_compatible():
    """Default (64,32)+Softplus must keep the canonical module layout so old
    checkpoints (Linear at indices 0/3/5) still load."""
    model = SlippageMLP(n_features=N_FEATURES)
    kinds = [type(m).__name__ for m in model.net]
    assert kinds == ["Linear", "ReLU", "Dropout", "Linear", "ReLU", "Linear", "Softplus"]


def test_weight_decay_reaches_optimizer(small_split, monkeypatch):
    cfg = Config()
    cfg.training.weight_decay = 0.1
    captured = {}
    real_adam = torch.optim.Adam

    def spy(params, **kw):
        captured["weight_decay"] = kw.get("weight_decay")
        return real_adam(params, **kw)

    monkeypatch.setattr(torch.optim, "Adam", spy)
    train_from_config(
        small_split, n_features=N_FEATURES, config=cfg,
        epochs=2, verbose=False, checkpoint_path=None,
    )
    assert captured["weight_decay"] == 0.1


def test_scheduler_uses_config_values(small_split, monkeypatch):
    cfg = Config()
    cfg.training.scheduler.patience = 3
    cfg.training.scheduler.factor = 0.7
    captured = {}
    real_sched = torch.optim.lr_scheduler.ReduceLROnPlateau

    def spy(opt, **kw):
        captured.update(kw)
        return real_sched(opt, **kw)

    monkeypatch.setattr(torch.optim.lr_scheduler, "ReduceLROnPlateau", spy)
    train_from_config(
        small_split, n_features=N_FEATURES, config=cfg,
        epochs=2, verbose=False, checkpoint_path=None,
    )
    assert captured["patience"] == 3
    assert captured["factor"] == 0.7


# ---------------------------------------------------------------------------
# B3 — NaN/inf guards
# ---------------------------------------------------------------------------

def test_train_raises_on_nan_target(small_split):
    small_split.y_train = small_split.y_train.copy()
    small_split.y_train[0] = np.nan
    with pytest.raises(FloatingPointError):
        train(small_split, n_features=N_FEATURES, epochs=2, verbose=False, checkpoint_path=None)


def test_train_raises_on_non_finite_loss(small_split):
    """A non-finite training loss (here from an inf feature) must abort loudly
    rather than silently leaving the model at random weights."""
    small_split.X_train = small_split.X_train.copy()
    small_split.X_train[0, 0] = np.inf
    with pytest.raises(FloatingPointError):
        train(small_split, n_features=N_FEATURES, epochs=2, verbose=False, checkpoint_path=None)


# ---------------------------------------------------------------------------
# B4 — determinism
# ---------------------------------------------------------------------------

def test_train_two_runs_identical(small_split):
    m1, h1 = train(small_split, n_features=N_FEATURES, epochs=5, seed=123,
                   verbose=False, checkpoint_path=None)
    m2, h2 = train(small_split, n_features=N_FEATURES, epochs=5, seed=123,
                   verbose=False, checkpoint_path=None)
    for a, b in zip(m1.state_dict().values(), m2.state_dict().values()):
        assert torch.equal(a, b)
    assert h1["best_val_mae"] == h2["best_val_mae"]


def test_train_different_seeds_differ(small_split):
    m1, _ = train(small_split, n_features=N_FEATURES, epochs=5, seed=1,
                  verbose=False, checkpoint_path=None)
    m2, _ = train(small_split, n_features=N_FEATURES, epochs=5, seed=2,
                  verbose=False, checkpoint_path=None)
    assert not all(
        torch.equal(a, b)
        for a, b in zip(m1.state_dict().values(), m2.state_dict().values())
    )


# ---------------------------------------------------------------------------
# C2 — checkpoint + scaler round-trip
# ---------------------------------------------------------------------------

def test_checkpoint_roundtrip(small_split, tmp_path):
    ckpt = tmp_path / "ckpt.pt"
    model, _ = train(small_split, n_features=N_FEATURES, hidden=[32, 16],
                     epochs=3, seed=7, verbose=False, checkpoint_path=ckpt)
    reloaded = load_model(ckpt)
    assert reloaded.hidden == [32, 16]
    np.testing.assert_allclose(
        predict(model, small_split.X_test), predict(reloaded, small_split.X_test)
    )


def test_scaler_roundtrip(small_split, tmp_path):
    ckpt = tmp_path / "ckpt.pt"
    train(small_split, n_features=N_FEATURES, epochs=2, verbose=False, checkpoint_path=ckpt)
    scaler = load_scaler(ckpt)
    np.testing.assert_allclose(
        scaler.transform(small_split.test_df[FEATURE_NAMES_TRAINING].values),
        small_split.X_test,
    )


def test_checkpoint_path_none_skips_save(small_split, tmp_path, monkeypatch):
    # When checkpoint_path is None, nothing should be written.
    before = set(tmp_path.iterdir())
    train(small_split, n_features=N_FEATURES, epochs=2, verbose=False, checkpoint_path=None)
    assert set(tmp_path.iterdir()) == before
