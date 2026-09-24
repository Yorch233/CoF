"""Shared Lightning mechanics for independently injected training algorithms."""

from __future__ import annotations

from typing import Any

import lightning as lightning
import torch
from torch import Tensor, nn

from cof.config.schemas import OptimizationConfig
from cof.method.base import StepContext, StepResult, TrainingMethod
from cof.model import GenerativeModel4SE


def training_stage_metric_name(stage: str) -> str:
    """Return the public metric namespace for an explicit training stage."""
    if stage == "pretrain":
        return "pretrain"
    if stage == "post_training":
        return "posttrain"
    raise ValueError(f"Unsupported training stage: {stage}")


class BaseTrainingPipeline(lightning.LightningModule):
    """Execute method steps and own parameters, optimizers and method checkpoints.

    Mathematical targets and method-specific knobs never enter this class.
    Each stage subclass provides only its stage identity and compatibility rule.
    """

    stage_name: str

    def __init__(
        self, model: GenerativeModel4SE, method: TrainingMethod, optimization: OptimizationConfig | None = None
    ) -> None:
        """Bind a stage-compatible method with exactly one registered model owner."""
        super().__init__()
        if method.stage != self.stage_name or method.model is not model:
            raise ValueError("Pipeline stage/model does not match the injected method")
        self.model = model
        self.method = method
        self.method_modules = nn.ModuleDict(method.auxiliary_modules())
        self.optimization = optimization or OptimizationConfig()

    def forward(self, state: Tensor, time: Tensor, condition: list[Tensor]) -> Tensor:
        """Delegate differentiable native prediction to the complete model."""
        return self.model(state, time, condition)

    def _shared_step(self, batch: tuple[Tensor, Tensor], split: str) -> StepResult:
        """Execute one update and log arbitrary disjoint scalar result fields."""
        result = self.method.step(batch, StepContext(self.global_step, split))
        values = result.logging_values()
        for name, value in values.items():
            if value.numel() != 1 or not torch.isfinite(value).all():
                raise FloatingPointError(f"Method produced a non-finite or non-scalar {name}")
            self.log(
                f"{training_stage_metric_name(self.stage_name)}/{split}/{name}",
                value,
                on_step=split == "train",
                on_epoch=split != "train",
                sync_dist=True,
                batch_size=batch[0].shape[0],
                prog_bar=name == "loss",
            )
        return result

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        """Return the differentiable objective for Lightning automatic optimization."""
        return self._shared_step(batch, "train").loss

    def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        """Evaluate the same method with validation logging and no backward step."""
        return self._shared_step(batch, "valid").loss

    def configure_optimizers(self) -> Any:
        """Optimize all explicitly registered trainable modules with optional warmup."""
        optimizer_type = {"Adam": torch.optim.Adam, "AdamW": torch.optim.AdamW}[self.optimization.optimizer]
        optimizer = optimizer_type(
            (parameter for parameter in self.parameters() if parameter.requires_grad),
            lr=self.optimization.learning_rate,
        )
        warmup = self.optimization.warmup_steps
        if not warmup:
            return optimizer
        schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min((step + 1) / warmup, 1.0))
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": schedule, "interval": "step"}}

    def configure_gradient_clipping(
        self,
        optimizer: torch.optim.Optimizer,
        gradient_clip_val: float | None = None,
        gradient_clip_algorithm: str | None = None,
    ) -> None:
        """Apply the pipeline's configured gradient norm bound."""
        value = self.optimization.gradient_clip if gradient_clip_val is None else gradient_clip_val
        if value > 0:
            self.clip_gradients(optimizer, gradient_clip_val=value, gradient_clip_algorithm="norm")

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Record method identity and algorithm state alongside Lightning state."""
        checkpoint["method"] = {"stage": self.stage_name, "name": self.method.name, "state": self.method.state_dict()}

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Reject mismatched method/stage resumes and restore algorithm state."""
        identity = checkpoint.get("method", {})
        if identity.get("stage") != self.stage_name or identity.get("name") != self.method.name:
            raise ValueError("Checkpoint method/stage does not match the requested pipeline")
        self.method.load_state_dict(identity.get("state", {}))
