"""Quantile (distributional) slippage head.

Slippage is a strictly-positive, right-skewed cost: a trading desk cares
about the *tail* (worst-case execution cost), not just the mean. A point
MAE throws that away. Here we train the same MLP backbone with a
multi-output head under the **pinball (quantile) loss**, so the model
emits a set of conditional quantiles — a native prediction interval and a
VaR-style "cost-at-risk" — instead of a single number.

This is deliberately a separate, opt-in code path: the canonical point
model in :mod:`slippage.training.trainer` is unchanged.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from slippage.data import SlippageDataset, SplitData
from slippage.models import SlippageMLP

DEFAULT_QUANTILES: tuple[float, ...] = (0.1, 0.5, 0.9, 0.95)


def pinball_loss(
    preds: torch.Tensor, target: torch.Tensor, quantiles: Sequence[float]
) -> torch.Tensor:
    """Mean pinball loss across the requested quantiles.

    Parameters
    ----------
    preds:
        Predicted quantiles, shape ``(N, Q)``.
    target:
        Ground truth, shape ``(N, 1)`` or ``(N,)``.
    quantiles:
        The ``Q`` quantile levels in ``(0, 1)``.
    """
    if target.ndim == 1:
        target = target.unsqueeze(1)
    q = torch.tensor(quantiles, dtype=preds.dtype, device=preds.device).view(1, -1)
    errors = target - preds
    # pinball: max(q * e, (q - 1) * e)
    loss = torch.maximum(q * errors, (q - 1.0) * errors)
    return loss.mean()


def train_quantile(
    split: SplitData,
    n_features: int,
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-3,
    patience: int = 10,
    seed: int = 42,
    dropout: float = 0.1,
    hidden: Sequence[int] = (64, 32),
    weight_decay: float = 0.0,
    verbose: bool = False,
) -> tuple[SlippageMLP, dict]:
    """Train a multi-output MLP under pinball loss, early-stopping on val loss.

    Returns the best model (lowest validation pinball loss) and a history
    dict with ``train_loss``, ``val_loss``, ``best_epoch``, ``best_val_loss``
    and the ``quantiles`` used.
    """
    quantiles = tuple(quantiles)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if not np.all(np.isfinite(split.y_train)):
        raise FloatingPointError("split.y_train contains non-finite values; cannot train.")

    train_ds = SlippageDataset(split.X_train, split.y_train)
    val_ds = SlippageDataset(split.X_val, split.y_val)
    gen = torch.Generator()
    gen.manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=gen)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = SlippageMLP(
        n_features=n_features,
        hidden=hidden,
        dropout=dropout,
        activation="softplus",  # quantiles of a non-negative cost
        n_outputs=len(quantiles),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    history: dict = {"train_loss": [], "val_loss": [], "quantiles": list(quantiles)}
    best_val = float("inf")
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    best_epoch, no_improve = 0, 0

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for X_b, y_b in train_loader:
            optimizer.zero_grad()
            loss = pinball_loss(model(X_b), y_b, quantiles)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(y_b)
        train_loss = running / len(train_ds)
        if not math.isfinite(train_loss):
            raise FloatingPointError(
                f"Quantile training loss became non-finite at epoch {epoch}."
            )

        model.eval()
        with torch.no_grad():
            vloss, n = 0.0, 0
            for X_b, y_b in val_loader:
                vloss += pinball_loss(model(X_b), y_b, quantiles).item() * len(y_b)
                n += len(y_b)
        val_loss = vloss / n

        scheduler.step(val_loss)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch, no_improve = epoch, 0
        else:
            no_improve += 1
        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(f"Epoch {epoch:03d}  train={train_loss:.4f}  val={val_loss:.4f}")
        if no_improve >= patience:
            break

    model.load_state_dict(best_state)
    history["best_epoch"] = best_epoch
    history["best_val_loss"] = best_val
    return model, history


def predict_quantiles(model: SlippageMLP, X: np.ndarray) -> np.ndarray:
    """Predict quantiles for each row, sorted ascending to avoid crossing.

    Returns an ``(N, Q)`` array.
    """
    model.eval()
    with torch.no_grad():
        out = model(torch.tensor(np.asarray(X), dtype=torch.float32)).numpy()
    # Enforce monotone (non-crossing) quantiles row-wise.
    return np.sort(out, axis=1)


def interval_coverage(
    y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> float:
    """Empirical fraction of observations falling within ``[lower, upper]``."""
    y_true = np.asarray(y_true).ravel()
    inside = (y_true >= np.asarray(lower).ravel()) & (y_true <= np.asarray(upper).ravel())
    return float(inside.mean())
