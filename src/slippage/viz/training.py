"""Plots tracking training progress (loss and validation metrics)."""

from __future__ import annotations

import matplotlib.pyplot as plt

from slippage.paths import FIGURES_DIR


def plot_training_history(
    history: dict,
    save_as: str | None = "training_history.png",
) -> plt.Figure:
    """Plot train loss and val MAE over epochs."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(history["train_loss"], label="Train Huber loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss")
    ax1.legend()

    ax2.plot(history["val_mae"], color="orange", label="Val MAE (bps)")
    if "best_epoch" in history:
        ax2.axvline(
            history["best_epoch"] - 1, color="red", ls="--", lw=1,
            label=f"best epoch {history['best_epoch']}",
        )
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("MAE (bps)")
    ax2.set_title("Validation MAE")
    ax2.legend()

    fig.tight_layout()
    if save_as:
        fig.savefig(FIGURES_DIR / save_as, dpi=150)
    return fig
