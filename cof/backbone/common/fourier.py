"""Cross-backbone building blocks shared by backbone implementations.

Author: Qing Yao
Date: 2026/9/23
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class GaussianFourierProjection(nn.Module):
    """Gaussian Fourier embeddings for noise levels.

    Projects a scalar time/sigma level onto a fixed set of random frequencies
    and emits sine/cosine features, giving conditioning MLPs a high-frequency
    embedding that stays bounded for very small or very large levels. The
    frequency table ``W`` is drawn once and frozen, so training and sampling
    share the same random feature map.
    """

    def __init__(self, embedding_size: int = 256, scale: float = 1.0) -> None:
        """Draw the frozen random frequency table.

        Args:
            embedding_size: Number of random frequencies.
            scale: Standard deviation of the drawn frequencies.
        """
        super().__init__()
        self.W = nn.Parameter(torch.randn(embedding_size) * scale, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project noise levels onto the frozen random frequencies.

        Args:
            x: Tensor of time or sigma levels broadcast against the frequency
                axis (typically shape ``[batch]``).

        Returns:
            Tensor of shape ``[batch, 2 * embedding_size]`` with sine features
            in the first half and cosine features in the second half.
        """
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
