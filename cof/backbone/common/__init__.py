"""Building blocks shared across backbone implementations.

Re-exports the blocks so backbones can import them from one module instead of
reaching into individual implementation files.

Author: Qing Yao
Date: 2026/9/23
"""

from cof.backbone.common.fourier import GaussianFourierProjection

__all__ = ["GaussianFourierProjection"]
