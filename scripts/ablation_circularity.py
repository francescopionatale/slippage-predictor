"""Quantify how much of the model's apparent skill is label circularity.

Runs two honesty checks and writes ``results/ablation.json``:

1. Feature ablation — retrain after dropping the label-driving features
   from the model's *input* (the label keeps them). A large MAE increase
   means the reported skill was largely re-derivation of the proxy formula.
2. Label perturbation — score the base-trained model against a label
   regenerated with a larger ``alpha``. A sharp MAE increase means the fit
   is specific to the exact synthetic formula, not a robust relationship.

This is the cheapest, highest-signal diagnostic for the project's central
limitation (see docs/methodology.md). It uses no extra data.
"""

from __future__ import annotations

import json

from slippage.config import Config
from slippage.data import temporal_split
from slippage.evaluation.ablation import (
    feature_ablation,
    label_perturbation_sensitivity,
)
from slippage.features import FEATURE_NAMES_TRAINING
from slippage.paths import RESULTS_DIR, ensure_dirs
from slippage.pipeline import build_full_dataset, build_full_proxy
from slippage.training import train


def main() -> None:
    ensure_dirs()
    config = Config.load()
    print("Building dataset...")
    data, proxy_all, _ = build_full_dataset(config=config)

    print("\n[1/2] Feature ablation (dropping label-driving features)...")
    ablation = feature_ablation(proxy_all, epochs=config.training.epochs)
    print("   ", ablation)

    print("\n[2/2] Label-perturbation sensitivity (alpha x1.5)...")
    base_split = temporal_split(proxy_all, feature_cols=FEATURE_NAMES_TRAINING)
    model, _ = train(
        base_split, n_features=len(FEATURE_NAMES_TRAINING),
        epochs=config.training.epochs, verbose=False, checkpoint_path=None,
    )
    # Rebuild with a larger alpha only (no config, so the explicit alpha wins);
    # all other params match the canonical defaults.
    perturbed = build_full_proxy(data, alpha=config.proxy.alpha * 1.5)
    sensitivity = label_perturbation_sensitivity(model, proxy_all, perturbed)
    print(
        f"    base MAE={sensitivity['base_mae']:.4f}  "
        f"perturbed MAE={sensitivity['perturbed_mae']:.4f}  "
        f"(+{sensitivity['rel_increase']:.0%})"
    )

    out = {
        "feature_ablation": {
            "full_test_mae": ablation.full_test_mae,
            "reduced_test_mae": ablation.reduced_test_mae,
            "rel_degradation": ablation.rel_degradation,
            "dropped": ablation.dropped,
            "kept": ablation.kept,
        },
        "label_perturbation": sensitivity,
    }
    path = RESULTS_DIR / "ablation.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
