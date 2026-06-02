"""Statistical-significance tooling for model comparison.

A lower MAE is not, by itself, evidence that one model genuinely beats
another — the difference could be noise. Two complementary tools:

* :func:`diebold_mariano` — the standard test for "is forecaster A's loss
  significantly different from B's", computed on the *per-observation*
  loss differential (with a small-sample / autocorrelation correction).
* :func:`block_bootstrap_mae_ci` — a confidence interval on the MAE that
  respects the serial correlation of hourly bars via a moving-block
  bootstrap (an i.i.d. bootstrap would badly understate the uncertainty).

These replace the misleading train-seed standard deviation previously used
as the only notion of uncertainty.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy import stats


def _loss(name_or_fn: str | Callable, err: np.ndarray) -> np.ndarray:
    if callable(name_or_fn):
        return name_or_fn(err)
    if name_or_fn == "abs":
        return np.abs(err)
    if name_or_fn == "sq":
        return err ** 2
    raise ValueError(f"unknown loss '{name_or_fn}' (use 'abs', 'sq', or a callable)")


@dataclass
class DMResult:
    """Result of a Diebold-Mariano test (A vs B)."""

    statistic: float
    p_value: float
    mean_loss_diff: float  # mean(loss_A - loss_B); negative => A has lower loss
    favored: str  # "A", "B", or "tie"

    def __str__(self) -> str:
        return (
            f"DM={self.statistic:+.3f}  p={self.p_value:.4f}  "
            f"Δloss={self.mean_loss_diff:+.4f}  favored={self.favored}"
        )


def diebold_mariano(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    loss: str | Callable = "abs",
    h: int = 1,
) -> DMResult:
    """Diebold-Mariano test comparing the loss of forecast A vs forecast B.

    Uses a Newey-West long-run variance (lags ``h-1``) and the
    Harvey-Leybourne-Newbold small-sample correction with Student-t
    p-values. The null is "equal predictive accuracy"; a small p-value
    means the loss difference is significant.

    Parameters
    ----------
    y_true:
        Realised values.
    pred_a, pred_b:
        The two competing prediction arrays.
    loss:
        ``"abs"`` (MAE-style), ``"sq"`` (MSE-style), or a callable mapping
        an error array to a loss array.
    h:
        Forecast horizon (autocorrelation lags in the variance). 1 for the
        one-step case here.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    err_a = y_true - np.asarray(pred_a, dtype=float).ravel()
    err_b = y_true - np.asarray(pred_b, dtype=float).ravel()
    d = _loss(loss, err_a) - _loss(loss, err_b)
    n = len(d)
    if n < 3:
        raise ValueError("diebold_mariano needs at least 3 observations")

    d_bar = float(d.mean())
    d_centered = d - d_bar
    gamma0 = float(np.mean(d_centered ** 2))
    lrv = gamma0
    for lag in range(1, h):
        cov = float(np.mean(d_centered[lag:] * d_centered[:-lag]))
        lrv += 2.0 * (1.0 - lag / h) * cov

    if lrv <= 0:
        # Degenerate (identical or perfectly anti-correlated losses).
        return DMResult(0.0, 1.0, d_bar, "tie")

    dm = d_bar / math.sqrt(lrv / n)
    # Harvey, Leybourne & Newbold (1997) small-sample correction.
    correction = math.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_corrected = dm * correction
    p_value = float(2.0 * stats.t.sf(abs(dm_corrected), df=n - 1))

    favored = "tie"
    if p_value < 0.05:
        favored = "A" if d_bar < 0 else "B"
    return DMResult(dm_corrected, p_value, d_bar, favored)


def block_bootstrap_mae_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_boot: int = 1000,
    block_size: int | None = None,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Moving-block-bootstrap confidence interval for the MAE.

    Returns ``(mae, lower, upper)`` at the ``1 - alpha`` level. The block
    bootstrap resamples contiguous blocks (length ``~n**(1/3)`` by default)
    so the serial correlation of intraday errors is preserved.
    """
    err = np.abs(np.asarray(y_true, dtype=float).ravel() - np.asarray(y_pred, dtype=float).ravel())
    n = len(err)
    if n == 0:
        raise ValueError("block_bootstrap_mae_ci: empty input")
    if block_size is None:
        block_size = max(1, int(round(n ** (1 / 3))))
    block_size = min(block_size, n)
    n_blocks = math.ceil(n / block_size)
    rng = np.random.default_rng(seed)
    offsets = np.arange(block_size)

    boot_maes = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n - block_size + 1, size=n_blocks)
        idx = (starts[:, None] + offsets).ravel()[:n]
        boot_maes[b] = err[idx].mean()

    lo, hi = np.percentile(boot_maes, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(err.mean()), float(lo), float(hi)


def compare_against_reference(
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    reference: str = "mlp",
    loss: str | Callable = "abs",
    n_boot: int = 1000,
    seed: int = 0,
) -> dict[str, dict]:
    """Full comparison table: MAE + bootstrap CI per model, plus a DM test of
    ``reference`` against every other model.

    Returns a dict keyed by model name with ``mae``, ``mae_ci`` and (for the
    non-reference models) ``dm`` results.
    """
    if reference not in predictions:
        raise KeyError(f"reference '{reference}' not in predictions {list(predictions)}")

    out: dict[str, dict] = {}
    for name, pred in predictions.items():
        mae, lo, hi = block_bootstrap_mae_ci(y_true, pred, n_boot=n_boot, seed=seed)
        entry: dict = {"mae": mae, "mae_ci": (lo, hi)}
        if name != reference:
            dm = diebold_mariano(y_true, predictions[reference], pred, loss=loss)
            entry["dm_vs_reference"] = {
                "statistic": dm.statistic,
                "p_value": dm.p_value,
                "mean_loss_diff": dm.mean_loss_diff,
                "favored_reference": dm.favored == "A",
            }
        out[name] = entry
    return out
