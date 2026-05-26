"""Alpha calibration for the synthetic slippage proxy.

Sweep candidate α values, train a short MLP per candidate against the
proxy rebuilt with that α, and return the α with the lowest validation
MAE.
"""

from __future__ import annotations


def calibrate_alpha(
    build_split_fn,
    n_features: int,
    alphas: list[float] | None = None,
    epochs: int = 15,
    seed: int = 42,
) -> tuple[float, dict[float, float]]:
    """For each alpha, rebuild the dataset, train a short MLP, return val MAE.

    Returns the alpha with the lowest validation MAE.

    Parameters
    ----------
    build_split_fn:
        Callable ``alpha -> SplitData``. Must rebuild the proxy with the
        given alpha so each candidate is evaluated on a fair dataset.
    n_features:
        Number of input features (used to instantiate the MLP).
    alphas:
        Candidate alpha values. Defaults to [0.5, 1.0, 2.0, 5.0, 10.0].
    epochs:
        Maximum epochs per candidate (short training is sufficient).
    seed:
        Random seed passed to ``train()`` for reproducibility.

    Returns
    -------
    best_alpha, dict mapping alpha -> best_val_mae
    """
    from slippage.training import train  # lazy import to avoid circular dependency

    if alphas is None:
        alphas = [0.5, 1.0, 2.0, 5.0, 10.0]

    sensitivity: dict[float, float] = {}
    for a in alphas:
        split = build_split_fn(a)
        _, history = train(
            split,
            n_features=n_features,
            epochs=epochs,
            patience=5,
            seed=seed,
            verbose=False,
        )
        sensitivity[a] = float(history["best_val_mae"])

    best = min(sensitivity, key=lambda a: sensitivity[a])
    return best, sensitivity
