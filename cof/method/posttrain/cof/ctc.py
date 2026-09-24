"""Counterfactual Transition Consistency with explicit model and noise pairing.

The result record below is part of the released contract; the transition
construction and consistency implementation is withheld during peer review and
will be released upon acceptance.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from torch import Tensor

from cof.config.schemas import CtcConfig
from cof.method.base import WITHHELD_MESSAGE, EndpointLoss, TrainingMethod
from cof.method.posttrain.cof.drc import DrcResult
from cof.model import GenerativeModel4SE


@dataclass(frozen=True)
class CtcResult:
    """The CTC objective record over matched factual and counterfactual states."""

    loss: Tensor
    time: Tensor
    factual_state: Tensor
    counterfactual_state: Tensor
    factual_prediction: Tensor
    counterfactual_prediction: Tensor


def ctc_substep_times(time: Tensor, target_time: Tensor, num_substeps: int) -> list[Tensor]:
    """Withheld: build the transition targets for one CTC term."""
    raise NotImplementedError(WITHHELD_MESSAGE)


def transition_branch(
    model: GenerativeModel4SE,
    corrected_state: Tensor,
    time: Tensor,
    target_time: Tensor,
    noisy: Tensor,
    noises: list[Tensor | None],
    endpoint_at: Callable[[Tensor, Tensor, int], Tensor],
    *,
    solver: str,
) -> Tensor:
    """Withheld: build one matched transition branch."""
    raise NotImplementedError(WITHHELD_MESSAGE)


def compute_ctc(
    method: TrainingMethod,
    drc: DrcResult,
    clean: Tensor,
    noisy: Tensor,
    *,
    config: CtcConfig,
    endpoint_loss: EndpointLoss,
    solver: str,
    t_min: float,
) -> CtcResult:
    """Withheld: match factual and oracle transitions and supervise the factual branch."""
    raise NotImplementedError(WITHHELD_MESSAGE)
