"""Circularity diagnostics for the synthetic-label setup.

The synthetic slippage label is a closed-form function of a handful of
features (``order_size_fraction``, ``vol_rolling``, ``urgency``,
``spread_cs`` and time-of-day). A model trained and scored on that label
can score well simply by re-deriving the formula — not by learning
anything transferable. These diagnostics *quantify* that effect honestly:

* :func:`feature_ablation` removes the label-driving features from the
  *training input* (while leaving them in the label). If the test error
  barely moves, the model wasn't relying on them; if it explodes, the
  reported skill was largely formula re-derivation.
* :func:`label_perturbation_sensitivity` scores a trained model against a
  label generated with perturbed proxy parameters. A model that learned a
  robust relationship degrades gracefully; one that memorised the exact
  formula degrades sharply.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from slippage.data import temporal_split
from slippage.evaluation.metrics import mae
from slippage.features import FEATURE_NAMES_TRAINING
from slippage.training import predict, train

# Features that the proxy formula is literally built from (and that are in
# the model's training set). Time-of-day enters via the index, not a column.
LABEL_DRIVING_FEATURES = ["order_size_fraction", "vol_rolling", "urgency", "spread_cs"]


@dataclass
class AblationResult:
    full_test_mae: float
    reduced_test_mae: float
    dropped: list[str]
    kept: list[str]

    @property
    def abs_degradation(self) -> float:
        return self.reduced_test_mae - self.full_test_mae

    @property
    def rel_degradation(self) -> float:
        return self.abs_degradation / self.full_test_mae if self.full_test_mae else float("inf")

    def __str__(self) -> str:
        return (
            f"full MAE={self.full_test_mae:.4f}  "
            f"reduced MAE={self.reduced_test_mae:.4f}  "
            f"(+{self.rel_degradation:.0%} dropping {self.dropped})"
        )


def feature_ablation(
    proxy_df: pd.DataFrame,
    drop: list[str] | None = None,
    epochs: int = 30,
    seed: int = 42,
    dropout: float = 0.1,
) -> AblationResult:
    """Train with and without the label-driving features and compare test MAE.

    A large positive ``rel_degradation`` means the model's apparent skill
    came from reading off the formula's own inputs — the circularity the
    project documents. A small degradation means the kept features carry
    independent signal.
    """
    if drop is None:
        drop = [f for f in LABEL_DRIVING_FEATURES if f in FEATURE_NAMES_TRAINING]
    kept = [f for f in FEATURE_NAMES_TRAINING if f not in drop]
    if not kept:
        raise ValueError("feature_ablation would drop every feature; choose a smaller `drop`")

    full_split = temporal_split(proxy_df, feature_cols=FEATURE_NAMES_TRAINING)
    reduced_split = temporal_split(proxy_df, feature_cols=kept)

    full_model, _ = train(
        full_split, n_features=len(FEATURE_NAMES_TRAINING),
        epochs=epochs, seed=seed, dropout=dropout, verbose=False, checkpoint_path=None,
    )
    reduced_model, _ = train(
        reduced_split, n_features=len(kept),
        epochs=epochs, seed=seed, dropout=dropout, verbose=False, checkpoint_path=None,
    )

    full_mae = mae(full_split.y_test, predict(full_model, full_split.X_test))
    reduced_mae = mae(reduced_split.y_test, predict(reduced_model, reduced_split.X_test))
    return AblationResult(float(full_mae), float(reduced_mae), drop, kept)


def label_perturbation_sensitivity(
    model,
    base_proxy: pd.DataFrame,
    perturbed_proxy: pd.DataFrame,
    feature_cols: list[str] | None = None,
) -> dict[str, float]:
    """Score a model (trained on the base label) against a perturbed label.

    ``base_proxy`` and ``perturbed_proxy`` must share the same feature rows
    (only the ``slippage_bps`` column differs — e.g. regenerated with a
    different ``alpha``). Returns the MAE under each label and the relative
    increase, which measures how formula-specific the fit is.
    """
    if feature_cols is None:
        feature_cols = FEATURE_NAMES_TRAINING

    base_split = temporal_split(base_proxy, feature_cols=feature_cols)
    y_pred = predict(model, base_split.X_test)
    base_mae = float(mae(base_split.y_test, y_pred))

    # Same features/scaler ordering; only the target changes.
    perturbed_split = temporal_split(perturbed_proxy, feature_cols=feature_cols)
    perturbed_mae = float(mae(perturbed_split.y_test, y_pred))

    rel = (perturbed_mae - base_mae) / base_mae if base_mae else float("inf")
    return {
        "base_mae": base_mae,
        "perturbed_mae": perturbed_mae,
        "rel_increase": rel,
    }
