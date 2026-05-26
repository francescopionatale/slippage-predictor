"""PyTorch Dataset wrapper for scaled feature matrices + slippage labels."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class SlippageDataset(Dataset):
    """PyTorch Dataset wrapping feature matrix and labels."""

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]
