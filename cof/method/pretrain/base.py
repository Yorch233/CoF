"""Default supervised pretraining through formulation-defined states and loss."""

from __future__ import annotations

import math
from collections.abc import Callable
from functools import partial

import torch
from torch import Tensor, nn

from cof.config.manager import Config
from cof.method.base import StepContext, StepResult, TrainingMethod
from cof.method.registry import PretrainingRegister
from cof.model import GenerativeModel4SE


@PretrainingRegister.register("base")
class BasePretraining(TrainingMethod):
    """Train on analytic path samples using the formulation's native objective."""

    name = "base"
    stage = "pretrain"
    required_capabilities = frozenset({"supervised"})

    def __init__(
        self,
        model: GenerativeModel4SE,
        *,
        reduction: str = "sum",
        time_loss_weight: float = 1e-3,
        loss_fn: Callable[..., dict[str, Tensor]] | None = None,
        t_min: float | None = None,
        t_max: float | None = None,
    ) -> None:
        """Inject the native objective and intersect time limits with the formulation."""
        super().__init__(model)
        if reduction not in {"sum", "mean"} or not math.isfinite(time_loss_weight) or time_loss_weight < 0:
            raise ValueError("Invalid base pretraining loss configuration")
        form = model.formulation
        self.t_min = max(form.training_time_start, t_min if t_min is not None else form.training_time_start)
        self.t_max = min(form.training_time_end, t_max if t_max is not None else form.training_time_end)
        if not 0 <= self.t_min < self.t_max <= 1:
            raise ValueError("Training times must satisfy 0 <= t_min < t_max <= 1")
        self.loss_fn = loss_fn or partial(
            form.training_loss, transform=model.transform, reduction=reduction, time_loss_weight=time_loss_weight
        )

    @classmethod
    def from_config(cls, model: GenerativeModel4SE, config: Config) -> BasePretraining:
        """Resolve the base method's own knobs while preserving preset namespaces."""
        return cls(
            model,
            reduction=config.get("optimization.reduction", "sum"),
            time_loss_weight=float(config.get("formulation.kwargs.time_loss_weight", 1e-3)),
            t_min=config.get("formulation.sampling.t_min"),
            t_max=config.get("formulation.sampling.t_max"),
        )

    def auxiliary_modules(self) -> dict[str, nn.Module]:
        """Register a trainable injected objective when one is supplied."""
        return {"loss": self.loss_fn} if isinstance(self.loss_fn, nn.Module) else {}

    def step(self, batch: tuple[Tensor, Tensor], context: StepContext) -> StepResult:
        """Sample a path state, predict once and evaluate the native supervised loss."""
        if len(batch) != 2:
            raise ValueError("Base pretraining requires exactly clean/noisy spectra")
        clean, noisy = batch
        time = torch.rand(clean.shape[0], device=clean.device) * (self.t_max - self.t_min) + self.t_min
        state = self.model.formulation.sample_training_state(clean, noisy, time)
        prediction = self.model(state, time, [noisy])
        losses = self.loss_fn(prediction, state, time, clean, noisy)
        if not all(torch.isfinite(value).all() for value in losses.values()):
            raise FloatingPointError("Non-finite base pretraining objective")
        return StepResult(
            losses["loss"],
            {key: value for key, value in losses.items() if key != "loss"},
            {"model_calls": clean.real.new_tensor(1.0)},
        )
