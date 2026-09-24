"""Trainer-independent method contracts and per-update result records."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from torch import Tensor, nn

from cof.model import GenerativeModel4SE

NativePredictor = Callable[[Tensor, Tensor, Tensor], Tensor]
EndpointLoss = Callable[[Tensor, Tensor, Tensor], dict[str, Tensor]]

WITHHELD_MESSAGE = (
    "The CoF post-training implementation (DRC + CTC) is withheld during peer review "
    "and will be released upon acceptance."
)


@dataclass(frozen=True)
class StepContext:
    """Small runtime context without model, callback or configuration access."""

    global_step: int = 0
    split: str = "train"


@dataclass
class StepResult:
    """Differentiable total with independently named loss terms and diagnostics."""

    loss: Tensor
    terms: dict[str, Tensor] = field(default_factory=dict)
    diagnostics: dict[str, Tensor] = field(default_factory=dict)

    def logging_values(self) -> dict[str, Tensor]:
        """Return disjoint scalar fields, rejecting accidentally shadowed losses."""
        if "loss" in self.terms or "loss" in self.diagnostics or self.terms.keys() & self.diagnostics.keys():
            raise ValueError("StepResult fields must have unique names")
        return {"loss": self.loss, **self.terms, **self.diagnostics}


class TrainingMethod(ABC):
    """An algorithm injected into a training pipeline without Lightning inheritance.

    The pipeline is the single parameter owner. Methods may declare additional
    trainable modules explicitly and checkpoint their non-parameter algorithm state.
    """

    name: str
    stage: str
    required_capabilities: frozenset[str] = frozenset()

    def __init__(self, model: GenerativeModel4SE) -> None:
        """Bind the shared model and validate mathematical capabilities."""
        self.model = model
        self.references: dict[str, NativePredictor] = {}
        missing = self.required_capabilities - model.formulation.capabilities
        if missing:
            raise ValueError(f"{self.name} requires missing formulation capabilities: {sorted(missing)}")

    def reference_prediction(self, name: str, state: Tensor, time: Tensor, condition: Tensor) -> Tensor:
        """Query an online or explicitly injected reference predictor."""
        if name == "online":
            return self.model(state, time, [condition])
        if name not in self.references:
            raise ValueError(f"Reference predictor {name!r} is not configured")
        return self.references[name](state, time, condition)

    def required_references(self) -> set[str]:
        """Return external reference names required before training starts."""
        return set()

    def auxiliary_modules(self) -> dict[str, nn.Module]:
        """Return explicitly owned extra trainable modules for optimizer registration."""
        return {}

    def state_dict(self) -> dict[str, Any]:
        """Return method-specific resumable state, empty for stateless algorithms."""
        return {}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore method state and reject unknown state in stateless methods."""
        if state:
            raise ValueError(f"Unexpected state for stateless method {self.name}")

    @abstractmethod
    def step(self, batch: tuple[Tensor, Tensor], context: StepContext) -> StepResult:
        """Compute one method update without invoking backward or optimizer.step."""
