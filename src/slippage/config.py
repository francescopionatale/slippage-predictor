"""Typed configuration objects loaded from ``configs/*.yaml``.

Each subsection (data, features, proxy, model, training) is a Pydantic
``BaseModel`` whose field defaults match the hardcoded defaults in the
corresponding source module. Loading the YAML files via
``Config.load()`` reads them as overrides; calling ``Config()`` with no
arguments yields the same canonical defaults the rest of the codebase
already uses.

Tests can lock the cross-reference by asserting the YAML values match
the Pydantic defaults — see ``tests/test_config.py``.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIG_DIR = ROOT_DIR / "configs"


class DataConfig(BaseModel):
    """Data download + ticker universe."""

    model_config = ConfigDict(extra="forbid")

    tickers: list[str] = Field(default_factory=lambda: [
        "SPY", "QQQ", "IWM", "AAPL", "MSFT", "GOOGL",
        "JPM", "GS", "XOM", "SLB", "PRCT",
    ])
    ticker_fallbacks: dict[str, str] = Field(default_factory=lambda: {"PRCT": "MGNI"})
    min_bars: int = 500
    interval: str = "1h"
    lookback_days: int = 720


class FeaturesConfig(BaseModel):
    """Market + synthetic-order feature engineering."""

    model_config = ConfigDict(extra="forbid")

    vol_window: int = 20
    orders_per_bar: int = 4
    size_low: float = 0.001
    size_high: float = 0.05


class ProxyConfig(BaseModel):
    """Synthetic slippage label construction + alpha calibration."""

    model_config = ConfigDict(extra="forbid")

    alpha: float = 2.0
    impact_noise: float = 0.20
    urgency_exp: float = 1.5
    spread_mult: float = 50.0
    tod_mult: float = 1.3
    tod_open_thresh: float = 10.5
    tod_close_thresh: float = 15.0
    calibration_alphas: list[float] = Field(
        default_factory=lambda: [0.5, 1.0, 2.0, 5.0, 10.0],
    )
    calibration_epochs: int = 15


class ModelConfig(BaseModel):
    """MLP architecture."""

    model_config = ConfigDict(extra="forbid")

    hidden: list[int] = Field(default_factory=lambda: [64, 32])
    dropout: float = 0.1
    activation: str = "softplus"


class SchedulerConfig(BaseModel):
    """ReduceLROnPlateau scheduler settings."""

    model_config = ConfigDict(extra="forbid")

    patience: int = 5
    factor: float = 0.5
    min_lr: float = 1e-5


class TrainingConfig(BaseModel):
    """Optimiser + training loop defaults."""

    model_config = ConfigDict(extra="forbid")

    epochs: int = 100
    batch_size: int = 256
    lr: float = 1e-3
    patience: int = 10
    dropout: float = 0.1
    weight_decay: float = 0.0
    delta: str | float = "adaptive"
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)


class Config(BaseModel):
    """Aggregate of every per-domain config."""

    model_config = ConfigDict(extra="forbid")

    data: DataConfig = Field(default_factory=DataConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    proxy: ProxyConfig = Field(default_factory=ProxyConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)

    @classmethod
    def load(cls, directory: Path | str | None = None) -> "Config":
        """Load Config from a directory of YAML files.

        Each subsection is read from ``<directory>/<name>.yaml`` if present.
        Missing files fall through to the in-code defaults. With no
        ``directory`` argument, falls back to ``<repo-root>/configs/``.
        """
        directory = Path(directory) if directory is not None else DEFAULT_CONFIG_DIR
        sections: dict[str, dict] = {}
        for name, _ in (
            ("data", DataConfig),
            ("features", FeaturesConfig),
            ("proxy", ProxyConfig),
            ("model", ModelConfig),
            ("training", TrainingConfig),
        ):
            path = directory / f"{name}.yaml"
            if path.exists():
                with path.open() as f:
                    sections[name] = yaml.safe_load(f) or {}
        return cls.model_validate(sections)
