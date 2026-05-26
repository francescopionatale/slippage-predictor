"""Unit tests for the Config loader.

The configs/*.yaml files are the canonical reference for hyperparameter
defaults. These tests assert that loading them from disk produces a
Config identical to the in-code defaults, so the two never drift.
"""

from __future__ import annotations

import pytest

from slippage.config import (
    Config,
    DataConfig,
    FeaturesConfig,
    ModelConfig,
    ProxyConfig,
    SchedulerConfig,
    TrainingConfig,
)


def test_default_config_construction():
    """Config() with no arguments produces the documented canonical defaults."""
    c = Config()
    assert c.proxy.alpha == 2.0
    assert c.proxy.impact_noise == 0.20
    assert c.features.orders_per_bar == 4
    assert c.training.epochs == 100
    assert c.training.batch_size == 256
    assert c.training.lr == 1e-3
    assert c.model.hidden == [64, 32]
    assert c.model.dropout == 0.1
    assert c.data.min_bars == 500
    assert c.training.scheduler.min_lr == pytest.approx(1e-5)


def test_yaml_defaults_match_code_defaults():
    """configs/*.yaml must reproduce exactly the in-code defaults."""
    from_yaml = Config.load()
    in_code = Config()
    assert from_yaml.model_dump() == in_code.model_dump()


def test_load_with_override(tmp_path):
    """Writing a single YAML file overrides only that section."""
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    (cfg_dir / "proxy.yaml").write_text("alpha: 5.0\nimpact_noise: 0.1\n")
    c = Config.load(cfg_dir)
    assert c.proxy.alpha == 5.0
    assert c.proxy.impact_noise == 0.1
    # Other sections still take in-code defaults
    assert c.training.epochs == 100
    assert c.model.hidden == [64, 32]


def test_extra_keys_rejected():
    """ConfigDict(extra='forbid') guards against typos in YAML."""
    with pytest.raises(Exception):  # pydantic.ValidationError
        ProxyConfig(alpha=2.0, no_such_field=1.0)


def test_repo_config_dir_exists():
    """Sanity: the configs/ directory bundled with the repo is found."""
    from slippage.config import DEFAULT_CONFIG_DIR

    assert DEFAULT_CONFIG_DIR.exists(), DEFAULT_CONFIG_DIR
    for name in ("data", "features", "proxy", "model", "training"):
        assert (DEFAULT_CONFIG_DIR / f"{name}.yaml").exists()


def test_scheduler_nested_default():
    """The scheduler subsection nests cleanly into TrainingConfig."""
    s = SchedulerConfig()
    t = TrainingConfig()
    assert t.scheduler == s
    assert s.factor == 0.5


# Compactly mirror the in-code constants we expect each section to expose
# so changes anywhere either side flag a test failure.
@pytest.mark.parametrize("config_cls,attr,expected", [
    (DataConfig, "min_bars", 500),
    (FeaturesConfig, "orders_per_bar", 4),
    (FeaturesConfig, "vol_window", 20),
    (ProxyConfig, "calibration_epochs", 15),
    (ModelConfig, "activation", "softplus"),
    (TrainingConfig, "weight_decay", 0.0),
])
def test_individual_defaults(config_cls, attr, expected):
    assert getattr(config_cls(), attr) == expected
