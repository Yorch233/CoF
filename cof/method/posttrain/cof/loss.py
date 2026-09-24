"""Clean-endpoint discrepancy shared by DRC and CTC.

The class and its configuration binding are part of the released contract; the
objective implementation is withheld during peer review and will be released
upon acceptance.
"""

from __future__ import annotations

from collections.abc import Callable

from torch import Tensor

from cof.config.schemas import EndpointLossConfig
from cof.data.stft import StftTransform
from cof.method.base import WITHHELD_MESSAGE


def negative_si_sdr_per_sample(prediction: Tensor, target: Tensor) -> Tensor:
    """Withheld: per-sample waveform objective term."""
    raise NotImplementedError(WITHHELD_MESSAGE)


def power_compressed_spectrum(spectrum: Tensor) -> tuple[Tensor, Tensor]:
    """Withheld: spectral feature transform for the endpoint objective."""
    raise NotImplementedError(WITHHELD_MESSAGE)


class CoFCleanLoss:
    """Weighted clean-endpoint objective bound to the shared representation."""

    def __init__(
        self, transform: StftTransform, weight_fn: Callable[[Tensor], Tensor], config: EndpointLossConfig | None = None
    ) -> None:
        """Bind the representation, per-time weights and validated objective knobs."""
        self.transform = transform
        self.weight_fn = weight_fn
        self.config = config or EndpointLossConfig()

    def __call__(self, prediction: Tensor, clean: Tensor, time: Tensor | None = None) -> dict[str, Tensor]:
        """Withheld: the differentiable total and independently named terms."""
        raise NotImplementedError(WITHHELD_MESSAGE)
