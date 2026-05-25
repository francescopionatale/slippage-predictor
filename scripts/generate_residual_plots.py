"""Generate residual diagnostic plots for the canonical MLP checkpoint.

Loads results/model_checkpoint.pt, rebuilds the canonical test split via
build_full_dataset, predicts on the test set, and writes three diagnostic
figures to results/figures/:

  - residuals_vs_predicted.png
  - residual_distribution.png
  - qq_residuals.png
"""

from __future__ import annotations

import torch

from model import SlippageMLP
from paths import RESULTS_DIR
from pipeline import build_full_dataset
from viz import plot_qq_residuals, plot_residual_distribution, plot_residuals_vs_predicted

CHECKPOINT = RESULTS_DIR / "model_checkpoint.pt"


def main() -> None:
    if not CHECKPOINT.exists():
        print(f"Checkpoint not found at {CHECKPOINT}. Run python -m train first.")
        return

    print("Building dataset (using cached parquet)...")
    _, _, split = build_full_dataset()

    ckpt = torch.load(CHECKPOINT, weights_only=True)
    model = SlippageMLP(n_features=ckpt["n_features"], dropout=0.1)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    with torch.no_grad():
        y_pred = model(torch.tensor(split.X_test, dtype=torch.float32)).squeeze(1).numpy()

    y_true = split.y_test

    print("Writing residuals_vs_predicted.png ...")
    plot_residuals_vs_predicted(y_true, y_pred)
    print("Writing residual_distribution.png ...")
    plot_residual_distribution(y_true, y_pred)
    print("Writing qq_residuals.png ...")
    plot_qq_residuals(y_true, y_pred)
    print("Done — figures written to results/figures/")


if __name__ == "__main__":
    main()
