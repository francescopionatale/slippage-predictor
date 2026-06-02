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
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIG_DIR = ROOT_DIR / "configs"

# Output activations the MLP head can build (kept here, free of a torch import,
# so config validation stays lightweight). slippage.models.mlp maps each name
# to the concrete nn.Module.
OUTPUT_ACTIVATIONS = {"softplus", "relu", "identity"}


class DataConfig(BaseModel):
    """Data download + ticker universe."""

    model_config = ConfigDict(extra="forbid")

    tickers: list[str] = Field(default_factory=lambda: [
        "SPY", "QQQ", "IWM", "AAPL", "MSFT", "GOOGL",
        "JPM", "GS", "XOM", "SLB", "PRCT",
    ])
    ticker_fallbacks: dict[str, str] = Field(default_factory=lambda: {"PRCT": "MGNI"})
    min_bars: int = Field(500, gt=0)
    interval: str = "1h"
    lookback_days: int = Field(720, gt=0)

    @field_validator("tickers")
    @classmethod
    def _tickers_non_empty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("tickers must be a non-empty list")
        return v


class FeaturesConfig(BaseModel):
    """Market + synthetic-order feature engineering."""

    model_config = ConfigDict(extra="forbid")

    vol_window: int = Field(20, gt=0)
    orders_per_bar: int = Field(4, gt=0)
    size_low: float = Field(0.001, gt=0)
    size_high: float = Field(0.05, gt=0)

    @field_validator("orders_per_bar")
    @classmethod
    def _orders_even(cls, v: int) -> int:
        if v % 2 != 0:
            raise ValueError("orders_per_bar must be even (balanced buy/sell sides)")
        return v

    @model_validator(mode="after")
    def _size_bounds_ordered(self) -> "FeaturesConfig":
        if self.size_low >= self.size_high:
            raise ValueError(
                f"size_low ({self.size_low}) must be < size_high ({self.size_high})"
            )
        return self


class ProxyConfig(BaseModel):
    """Synthetic slippage label construction + alpha calibration."""

    model_config = ConfigDict(extra="forbid")

    alpha: float = Field(2.0, gt=0)
    impact_noise: float = Field(0.20, ge=0)
    urgency_exp: float = Field(1.5, ge=0)
    spread_mult: float = Field(50.0, ge=0)
    tod_mult: float = Field(1.3, ge=1.0)
    tod_open_thresh: float = 10.5
    tod_close_thresh: float = 15.0
    calibration_alphas: list[float] = Field(
        default_factory=lambda: [0.5, 1.0, 2.0, 5.0, 10.0],
    )
    calibration_epochs: int = Field(15, gt=0)

    @model_validator(mode="after")
    def _tod_thresholds_ordered(self) -> "ProxyConfig":
        if self.tod_open_thresh >= self.tod_close_thresh:
            raise ValueError(
                f"tod_open_thresh ({self.tod_open_thresh}) must be < "
                f"tod_close_thresh ({self.tod_close_thresh})"
            )
        return self


class ModelConfig(BaseModel):
    """MLP architecture."""

    model_config = ConfigDict(extra="forbid")

    hidden: list[int] = Field(default_factory=lambda: [64, 32])
    dropout: float = Field(0.1, ge=0.0, lt=1.0)
    activation: str = "softplus"

    @field_validator("hidden")
    @classmethod
    def _hidden_positive(cls, v: list[int]) -> list[int]:
        if not v or any(h <= 0 for h in v):
            raise ValueError("hidden must be a non-empty list of positive layer widths")
        return v

    @field_validator("activation")
    @classmethod
    def _activation_supported(cls, v: str) -> str:
        if v not in OUTPUT_ACTIVATIONS:
            raise ValueError(
                f"activation '{v}' not supported; choose from {sorted(OUTPUT_ACTIVATIONS)}"
            )
        return v


class SchedulerConfig(BaseModel):
    """ReduceLROnPlateau scheduler settings."""

    model_config = ConfigDict(extra="forbid")

    patience: int = Field(5, ge=0)
    factor: float = Field(0.5, gt=0.0, lt=1.0)
    min_lr: float = Field(1e-5, gt=0.0)


class TrainingConfig(BaseModel):
    """Optimiser + training loop defaults."""

    model_config = ConfigDict(extra="forbid")

    epochs: int = Field(100, gt=0)
    batch_size: int = Field(256, gt=0)
    lr: float = Field(1e-3, gt=0.0)
    patience: int = Field(10, ge=0)
    dropout: float = Field(0.1, ge=0.0, lt=1.0)
    weight_decay: float = Field(0.0, ge=0.0)
    delta: str | float = "adaptive"
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)

    @field_validator("delta")
    @classmethod
    def _delta_valid(cls, v: str | float) -> str | float:
        if isinstance(v, str):
            if v != "adaptive":
                raise ValueError("delta must be the string 'adaptive' or a positive float")
            return v
        if v <= 0:
            raise ValueError("delta must be > 0 when given as a float")
        return v


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
