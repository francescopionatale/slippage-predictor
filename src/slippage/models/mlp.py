"""MLP architecture for slippage regression."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

# Name → output-activation factory. The valid names are mirrored in
# slippage.config.OUTPUT_ACTIVATIONS (kept torch-free for lightweight
# config validation).
_OUTPUT_ACTIVATIONS = {
    "softplus": nn.Softplus,
    "relu": nn.ReLU,
    "identity": nn.Identity,
}


class SlippageMLP(nn.Module):
    """Feed-forward MLP with a configurable hidden stack and output head.

    The default ``hidden=(64, 32)`` with a Softplus head reproduces the
    canonical architecture exactly (Input → Dense(64)+ReLU+Dropout →
    Dense(32)+ReLU → Dense(1) → Softplus), so existing checkpoints load
    unchanged. Dropout is applied after the first hidden layer only.

    The Softplus activation on the output enforces non-negative
    predictions: slippage is always a cost ≥ 0, so emitting negative
    values would be physically meaningless and waste capacity learning
    to clip them. Softplus is smooth (unlike ReLU), keeping gradients
    well-conditioned near zero.

    Parameters
    ----------
    n_features:
        Number of input features.
    hidden:
        Widths of the hidden layers. Defaults to ``(64, 32)``.
    dropout:
        Dropout probability applied after the first hidden layer.
    activation:
        Output activation name (``softplus`` | ``relu`` | ``identity``).
    n_outputs:
        Size of the output layer. 1 for a point estimate; >1 for a
        quantile/distributional head (see ``slippage.training``).
    """

    def __init__(
        self,
        n_features: int,
        hidden: Sequence[int] = (64, 32),
        dropout: float = 0.1,
        activation: str = "softplus",
        n_outputs: int = 1,
    ) -> None:
        super().__init__()
        if activation not in _OUTPUT_ACTIVATIONS:
            raise ValueError(
                f"activation '{activation}' not supported; "
                f"choose from {sorted(_OUTPUT_ACTIVATIONS)}"
            )
        self.hidden = list(hidden)
        self.activation = activation
        self.n_features = n_features
        self.n_outputs = n_outputs

        layers: list[nn.Module] = []
        in_dim = n_features
        for i, width in enumerate(hidden):
            layers.append(nn.Linear(in_dim, width))
            layers.append(nn.ReLU())
            if i == 0:
                # Dropout after the first hidden layer only (matches the
                # canonical architecture and keeps module indices stable
                # so legacy checkpoints stay loadable).
                layers.append(nn.Dropout(dropout))
            in_dim = width
        layers.append(nn.Linear(in_dim, n_outputs))
        layers.append(_OUTPUT_ACTIVATIONS[activation]())

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
