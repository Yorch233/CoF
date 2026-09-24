"""Dynamic Rollout Correction on a randomly selected inference-schedule state.

The result record below is part of the released contract; the rollout and
correction implementation is withheld during peer review and will be released
upon acceptance.
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

from cof.config.schemas import DrcConfig
from cof.method.base import WITHHELD_MESSAGE, EndpointLoss
from cof.model import GenerativeModel4SE


@dataclass(frozen=True)
class DrcResult:
    """The DRC objective record with the detached state receiving correction."""

    loss: Tensor
    corrected_state: Tensor
    corrected_time: Tensor
    native_prediction: Tensor
    clean_prediction: Tensor
    rollout_steps: int
    diagnostics: dict[str, Tensor]
    terms: dict[str, Tensor]


def compute_drc(
    model: GenerativeModel4SE,
    clean: Tensor,
    noisy: Tensor,
    *,
    config: DrcConfig,
    endpoint_loss: EndpointLoss,
    solver: str,
    skip_type: str,
) -> DrcResult:
    """Withheld: roll out along an inference schedule and correct one state."""
    raise NotImplementedError(WITHHELD_MESSAGE)
