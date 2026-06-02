"""Tests that the pydantic config rejects out-of-range / inconsistent values."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from slippage.config import (
    DataConfig,
    FeaturesConfig,
    ModelConfig,
    ProxyConfig,
    SchedulerConfig,
    TrainingConfig,
)


@pytest.mark.parametrize("kwargs", [
    {"dropout": 1.0},          # must be < 1
    {"dropout": -0.1},         # must be >= 0
    {"hidden": []},            # must be non-empty
    {"hidden": [64, 0]},       # widths must be positive
    {"activation": "tanh"},    # unsupported
])
def test_model_config_rejects_bad_values(kwargs):
    with pytest.raises(ValidationError):
        ModelConfig(**kwargs)


@pytest.mark.parametrize("kwargs", [
    {"epochs": 0},
    {"batch_size": -1},
    {"lr": 0.0},
    {"weight_decay": -0.5},
    {"dropout": 1.5},
    {"delta": "fixed"},        # only "adaptive" allowed as a string
    {"delta": -2.0},           # float delta must be > 0
])
def test_training_config_rejects_bad_values(kwargs):
    with pytest.raises(ValidationError):
        TrainingConfig(**kwargs)


def test_training_config_accepts_float_delta():
    assert TrainingConfig(delta=1.5).delta == 1.5


@pytest.mark.parametrize("kwargs", [
    {"orders_per_bar": 3},     # must be even
    {"orders_per_bar": 0},     # must be > 0
    {"size_low": 0.05, "size_high": 0.01},  # low must be < high
    {"vol_window": 0},
])
def test_features_config_rejects_bad_values(kwargs):
    with pytest.raises(ValidationError):
        FeaturesConfig(**kwargs)


@pytest.mark.parametrize("kwargs", [
    {"alpha": 0.0},
    {"impact_noise": -0.1},
    {"tod_open_thresh": 16.0, "tod_close_thresh": 15.0},  # open must be < close
])
def test_proxy_config_rejects_bad_values(kwargs):
    with pytest.raises(ValidationError):
        ProxyConfig(**kwargs)


@pytest.mark.parametrize("kwargs", [
    {"factor": 1.0},   # must be < 1
    {"factor": 0.0},   # must be > 0
    {"min_lr": 0.0},   # must be > 0
])
def test_scheduler_config_rejects_bad_values(kwargs):
    with pytest.raises(ValidationError):
        SchedulerConfig(**kwargs)


def test_data_config_rejects_empty_tickers():
    with pytest.raises(ValidationError):
        DataConfig(tickers=[])
