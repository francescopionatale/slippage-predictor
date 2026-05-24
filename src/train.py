"""Training loop for the slippage MLP.

Can also be run as a module:
    python -m slippage.train
"""

from __future__ import annotations

import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import SlippageDataset, SplitData
from model import SlippageMLP
from paths import RESULTS_DIR


def train(
    split: SplitData,
    n_features: int,
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-3,
    patience: int = 10,
    seed: int = 42,
    verbose: bool = True,
) -> tuple[SlippageMLP, dict]:
    """Train MLP with Huber loss, Adam, ReduceLROnPlateau, and early stopping.

    Returns
    -------
    model:
        Best model (lowest val MAE).
    history:
        Dict with lists 'train_loss', 'val_mae' and scalars
        'best_epoch', 'best_val_mae'.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_ds = SlippageDataset(split.X_train, split.y_train)
    val_ds = SlippageDataset(split.X_val, split.y_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = SlippageMLP(n_features=n_features)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5, min_lr=1e-5
    )
    criterion = nn.HuberLoss(delta=1.0)

    history = {"train_loss": [], "val_mae": []}
    best_val_mae = float("inf")
    best_state = None
    best_epoch = 0
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for X_b, y_b in train_loader:
            optimizer.zero_grad()
            pred = model(X_b)
            loss = criterion(pred, y_b)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(y_b)
        epoch_loss /= len(train_ds)

        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for X_b, y_b in val_loader:
                val_preds.append(model(X_b).squeeze(1))
                val_targets.append(y_b.squeeze(1))
        val_mae = (torch.cat(val_preds) - torch.cat(val_targets)).abs().mean().item()

        scheduler.step(val_mae)
        history["train_loss"].append(epoch_loss)
        history["val_mae"].append(val_mae)

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1

        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(f"Epoch {epoch:03d}  train_loss={epoch_loss:.4f}  val_mae={val_mae:.4f}")

        if no_improve >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch} (best epoch {best_epoch})")
            break

    model.load_state_dict(best_state)
    history["best_epoch"] = best_epoch
    history["best_val_mae"] = best_val_mae

    checkpoint_path = RESULTS_DIR / "model_checkpoint.pt"
    torch.save({"state_dict": model.state_dict(), "n_features": n_features}, checkpoint_path)
    if verbose:
        print(f"Model saved to {checkpoint_path}")

    return model, history


def predict(model: SlippageMLP, X: np.ndarray) -> np.ndarray:
    """Run model inference on a numpy feature matrix."""
    model.eval()
    with torch.no_grad():
        t = torch.tensor(X, dtype=torch.float32)
        return model(t).squeeze(1).numpy()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _main() -> None:
    from features import FEATURE_NAMES
    from pipeline import build_full_dataset

    print("Downloading data and building split...")
    _, _, split = build_full_dataset()
    print(f"Train: {len(split.X_train):,}  Val: {len(split.X_val):,}  Test: {len(split.X_test):,}")

    model, history = train(split, n_features=len(FEATURE_NAMES))

    metrics = {"best_epoch": history["best_epoch"], "best_val_mae": history["best_val_mae"]}
    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("Done.")


if __name__ == "__main__":
    _main()
