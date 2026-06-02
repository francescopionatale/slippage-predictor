"""Training loop for the slippage MLP.

Can also be run as a module:
    python -m slippage.train
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from slippage.data import SlippageDataset, SplitData
from slippage.logging import get_logger
from slippage.models import SlippageMLP
from slippage.paths import CHECKPOINTS_DIR, RESULTS_DIR

logger = get_logger(__name__)


def _scaler_path(checkpoint_path: str | Path) -> Path:
    """Sibling path where the fitted scaler is persisted next to a checkpoint."""
    p = Path(checkpoint_path)
    return p.with_name(p.stem + ".scaler.joblib")

if TYPE_CHECKING:
    from slippage.config import Config

_DEFAULT_CHECKPOINT: Any = object()  # sentinel: "use CHECKPOINTS_DIR/model_checkpoint.pt"


def train(
    split: SplitData,
    n_features: int,
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-3,
    patience: int = 10,
    seed: int = 42,
    verbose: bool = True,
    delta: float | None = None,
    dropout: float = 0.1,
    weight_decay: float = 0.0,
    hidden: Sequence[int] = (64, 32),
    activation: str = "softplus",
    scheduler_patience: int = 5,
    scheduler_factor: float = 0.5,
    scheduler_min_lr: float = 1e-5,
    checkpoint_path: Any = _DEFAULT_CHECKPOINT,
) -> tuple[SlippageMLP, dict]:
    """Train MLP with Huber loss, Adam, ReduceLROnPlateau, and early stopping.

    Parameters
    ----------
    delta:
        Huber loss delta in bps. If ``None`` (default), computed adaptively
        as ``max(1.0, median(|y_train|))`` so most residuals fall in the
        quadratic region and the gradient scales with error magnitude.
    dropout:
        Dropout probability for the MLP's first hidden layer.
    weight_decay:
        L2 penalty passed to the Adam optimizer.
    hidden, activation:
        MLP architecture — hidden-layer widths and output activation.
        Defaults reproduce the canonical ``(64, 32)`` + Softplus model.
    scheduler_patience, scheduler_factor, scheduler_min_lr:
        ReduceLROnPlateau settings.
    checkpoint_path:
        Where to persist the best model. Defaults to
        ``artifacts/checkpoints/model_checkpoint.pt``; pass an explicit path to write
        elsewhere, or ``None`` to skip saving entirely (useful for
        walk-forward CV where dozens of folds would otherwise overwrite
        each other). The saved checkpoint is self-describing
        (architecture + feature count) so ``load_model`` can rebuild it.

    Returns
    -------
    model:
        Best model (lowest val MAE).
    history:
        Dict with lists 'train_loss', 'val_mae' and scalars
        'best_epoch', 'best_val_mae'.
    """
    # Determinism: seed every RNG that influences training and pin the
    # DataLoader shuffle generator so two same-seed runs are bit-identical.
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if not np.all(np.isfinite(split.y_train)):
        raise FloatingPointError(
            "split.y_train contains non-finite values (NaN/inf); cannot train."
        )

    train_ds = SlippageDataset(split.X_train, split.y_train)
    val_ds = SlippageDataset(split.X_val, split.y_val)

    loader_generator = torch.Generator()
    loader_generator.manual_seed(seed)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, generator=loader_generator
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = SlippageMLP(
        n_features=n_features, hidden=hidden, dropout=dropout, activation=activation
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=scheduler_patience,
        factor=scheduler_factor,
        min_lr=scheduler_min_lr,
    )

    if delta is None:
        delta = max(1.0, float(np.median(np.abs(split.y_train))))
    criterion = nn.HuberLoss(delta=delta)
    if verbose:
        logger.info("Adaptive Huber delta = %.2f bps (median|y_train|)", delta)

    history: dict[str, Any] = {"train_loss": [], "val_mae": []}
    best_val_mae = float("inf")
    best_state: dict[str, Any] = {k: v.clone() for k, v in model.state_dict().items()}
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

        if not math.isfinite(epoch_loss):
            raise FloatingPointError(
                f"Training loss became non-finite ({epoch_loss}) at epoch {epoch}. "
                "Likely a diverging learning rate or NaN/inf in the inputs — "
                "aborting so the model isn't silently left at random weights."
            )

        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for X_b, y_b in val_loader:
                val_preds.append(model(X_b).squeeze(1))
                val_targets.append(y_b.squeeze(1))
        val_mae = (torch.cat(val_preds) - torch.cat(val_targets)).abs().mean().item()

        if not math.isfinite(val_mae):
            raise FloatingPointError(
                f"Validation MAE became non-finite ({val_mae}) at epoch {epoch}."
            )

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
            logger.info("Epoch %03d  train_loss=%.4f  val_mae=%.4f", epoch, epoch_loss, val_mae)

        if no_improve >= patience:
            if verbose:
                logger.info("Early stopping at epoch %d (best epoch %d)", epoch, best_epoch)
            break

    model.load_state_dict(best_state)
    history["best_epoch"] = best_epoch
    history["best_val_mae"] = best_val_mae

    if checkpoint_path is _DEFAULT_CHECKPOINT:
        checkpoint_path = CHECKPOINTS_DIR / "model_checkpoint.pt"
    if checkpoint_path is not None:
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "n_features": n_features,
                "hidden": list(model.hidden),
                "activation": model.activation,
                "dropout": dropout,
                "n_outputs": model.n_outputs,
            },
            checkpoint_path,
        )
        # Persist the train-only scaler beside the checkpoint so inference can
        # reproduce the exact feature normalisation (closes the train/serve gap).
        joblib.dump(split.scaler, _scaler_path(checkpoint_path))
        if verbose:
            logger.info("Model saved to %s", checkpoint_path)

    return model, history


def load_model(checkpoint_path: str | Path | None = None) -> SlippageMLP:
    """Rebuild a ``SlippageMLP`` from a self-describing checkpoint.

    The checkpoint written by :func:`train` stores the architecture
    (``hidden``, ``activation``, ``n_outputs``) alongside the weights, so
    a model can be reconstructed without knowing how it was trained.
    Older checkpoints that only carry ``state_dict`` + ``n_features`` fall
    back to the canonical ``(64, 32)`` + Softplus architecture.
    """
    if checkpoint_path is None:
        checkpoint_path = CHECKPOINTS_DIR / "model_checkpoint.pt"
    ckpt = torch.load(checkpoint_path, weights_only=True)
    model = SlippageMLP(
        n_features=ckpt["n_features"],
        hidden=ckpt.get("hidden", (64, 32)),
        dropout=ckpt.get("dropout", 0.1),
        activation=ckpt.get("activation", "softplus"),
        n_outputs=ckpt.get("n_outputs", 1),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def load_scaler(checkpoint_path: str | Path | None = None) -> StandardScaler:
    """Load the ``StandardScaler`` persisted alongside a checkpoint by :func:`train`."""
    if checkpoint_path is None:
        checkpoint_path = CHECKPOINTS_DIR / "model_checkpoint.pt"
    return joblib.load(_scaler_path(checkpoint_path))


def predict(model: SlippageMLP, X: np.ndarray) -> np.ndarray:
    """Run model inference on a numpy feature matrix.

    For a single-output (point) model this returns a 1-D array of length
    ``len(X)``. For a multi-output head the raw ``(n, n_outputs)`` matrix
    is returned unchanged.
    """
    model.eval()
    with torch.no_grad():
        t = torch.tensor(np.asarray(X), dtype=torch.float32)
        out = model(t).numpy()
    if out.ndim == 2 and out.shape[1] == 1:
        return out.squeeze(1)
    return out


def train_from_config(
    split: SplitData,
    n_features: int,
    config: "Config",
    *,
    seed: int = 42,
    verbose: bool = True,
    checkpoint_path: Any = _DEFAULT_CHECKPOINT,
    **overrides: Any,
) -> tuple[SlippageMLP, dict]:
    """Train using hyperparameters sourced from a :class:`~slippage.config.Config`.

    Every relevant field (architecture, optimizer, scheduler, Huber delta)
    is read from ``config`` so editing ``configs/*.yaml`` actually changes
    behaviour. Pass keyword ``overrides`` to override individual ``train``
    arguments (handy for tests and sweeps).
    """
    t = config.training
    m = config.model
    delta = None if t.delta == "adaptive" else float(t.delta)
    kwargs: dict[str, Any] = dict(
        epochs=t.epochs,
        batch_size=t.batch_size,
        lr=t.lr,
        patience=t.patience,
        dropout=t.dropout,
        weight_decay=t.weight_decay,
        delta=delta,
        hidden=tuple(m.hidden),
        activation=m.activation,
        scheduler_patience=t.scheduler.patience,
        scheduler_factor=t.scheduler.factor,
        scheduler_min_lr=t.scheduler.min_lr,
    )
    kwargs.update(overrides)
    return train(
        split,
        n_features=n_features,
        seed=seed,
        verbose=verbose,
        checkpoint_path=checkpoint_path,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    from slippage.config import Config
    from slippage.features import FEATURE_NAMES_TRAINING
    from slippage.paths import ensure_dirs
    from slippage.pipeline import build_full_dataset

    ensure_dirs()
    config = Config.load()
    logger.info("Downloading data and building split...")
    _, _, split = build_full_dataset(config=config)
    logger.info(
        "Train: %d  Val: %d  Test: %d",
        len(split.X_train), len(split.X_val), len(split.X_test),
    )

    model, history = train_from_config(
        split, n_features=len(FEATURE_NAMES_TRAINING), config=config
    )

    metrics = {"best_epoch": history["best_epoch"], "best_val_mae": history["best_val_mae"]}
    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Done.")


if __name__ == "__main__":
    main()
