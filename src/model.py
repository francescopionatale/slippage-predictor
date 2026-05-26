"""MLP architecture for slippage regression."""

from __future__ import annotations

import torch
import torch.nn as nn


class SlippageMLP(nn.Module):
    """Two-hidden-layer MLP with a Softplus head.

    Architecture: Input → Dense(64)+ReLU+Dropout → Dense(32)+ReLU →
    Dense(1) → Softplus.

    The Softplus activation on the output enforces non-negative
    predictions: slippage is always a cost ≥ 0, so emitting negative
    values would be physically meaningless and waste capacity learning
    to clip them. Softplus is smooth (unlike ReLU), keeping gradients
    well-conditioned near zero.
    """

    def __init__(self, n_features: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
