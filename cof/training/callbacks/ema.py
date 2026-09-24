"""Maintain exponential-moving-average weights for training and validation.

The callback keeps a shadow copy of every trainable parameter, updates it
after each optimizer step, and swaps it in for validation so all exported
weights and validation metrics reflect the averaged model rather than the
noisy online one.

Author: Qing Yao
Date: 2026/9/23
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from lightning.pytorch.callbacks import Callback
from torch import Tensor

if TYPE_CHECKING:
    from lightning.pytorch import LightningModule, Trainer


class EmaCallback(Callback):
    """Track model parameters with EMA and evaluate using the averaged weights."""

    def __init__(self, decay: float = 0.999) -> None:
        """Initialize EMA.

        Args:
            decay: Weight assigned to the previous moving average.
        """
        super().__init__()
        if not 0.0 <= decay < 1.0:
            raise ValueError("EMA decay must be in [0, 1)")
        self.decay = decay
        self.shadow_parameters: list[Tensor] = []
        self.backup_parameters: list[Tensor] = []
        self.num_updates = 0
        self.last_optimizer_step = 0

    @staticmethod
    def _parameters(pl_module: LightningModule) -> list[Tensor]:
        """Return the module's trainable parameters in registration order."""
        return [parameter for parameter in pl_module.parameters() if parameter.requires_grad]

    def _initialize(self, pl_module: LightningModule) -> None:
        """Create the shadow copies on first use, leaving existing state alone."""
        if not self.shadow_parameters:
            self.shadow_parameters = [parameter.detach().clone() for parameter in self._parameters(pl_module)]

    def shadow_parameter_dict(
        self,
        pl_module: LightningModule,
        *,
        prefix: str = "",
    ) -> dict[str, Tensor]:
        """Return detached EMA tensors keyed like a selected child module.

        This lets posttraining evaluate an EMA target through
        ``torch.func.functional_call`` without swapping live parameters or
        allocating a second trainable network.

        Args:
            pl_module: Module whose trainable parameters the shadows mirror.
            prefix: Parameter-name prefix selecting the child module; keys in
                the result have the prefix removed.

        Returns:
            Detached EMA tensors on each parameter's device and dtype.
        """
        self._initialize(pl_module)
        selected: dict[str, Tensor] = {}
        shadow_index = 0
        for name, parameter in pl_module.named_parameters():
            if not parameter.requires_grad:
                continue
            shadow = self.shadow_parameters[shadow_index]
            shadow_index += 1
            if name.startswith(prefix):
                selected[name.removeprefix(prefix)] = shadow.to(
                    device=parameter.device,
                    dtype=parameter.dtype,
                ).detach()
        return selected

    def state_dict(self) -> dict[str, Any]:
        """Save EMA weights and update count in the Lightning checkpoint.

        Returns:
            Mapping with the decay, update count, and shadow parameters.
        """
        return {
            "decay": self.decay,
            "num_updates": self.num_updates,
            "last_optimizer_step": self.last_optimizer_step,
            "shadow_parameters": self.shadow_parameters,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore EMA state when training resumes.

        Args:
            state_dict: State written by ``state_dict``.
        """
        self.decay = float(state_dict["decay"])
        self.num_updates = int(state_dict["num_updates"])
        self.last_optimizer_step = int(state_dict["last_optimizer_step"])
        self.shadow_parameters = list(state_dict["shadow_parameters"])

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Initialize EMA after Lightning places the model on its training device.

        Args:
            trainer: Lightning trainer starting the fit.
            pl_module: Module whose parameters seed the shadow copies.
        """
        del trainer
        self._initialize(pl_module)

    @torch.no_grad()
    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Update EMA after each optimizer step.

        Args:
            trainer: Lightning trainer driving the fit.
            pl_module: Module whose parameters update the shadow copies.
            outputs: Step outputs from the training step.
            batch: Current batch.
            batch_idx: Index of the current batch.
        """
        del outputs, batch, batch_idx
        if trainer.global_step <= self.last_optimizer_step:
            return
        self.last_optimizer_step = trainer.global_step
        parameters = self._parameters(pl_module)
        self._initialize(pl_module)
        for shadow, parameter in zip(self.shadow_parameters, parameters, strict=True):
            shadow.data = shadow.to(device=parameter.device, dtype=parameter.dtype)
            shadow.lerp_(parameter.detach(), 1.0 - self.decay)
        self.num_updates += 1

    @torch.no_grad()
    def on_validation_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Swap EMA parameters in for validation.

        Args:
            trainer: Lightning trainer starting validation.
            pl_module: Module whose trainable parameters are replaced.
        """
        del trainer
        parameters = self._parameters(pl_module)
        self._initialize(pl_module)
        self.backup_parameters = [parameter.detach().clone() for parameter in parameters]
        for parameter, shadow in zip(parameters, self.shadow_parameters, strict=True):
            parameter.copy_(shadow.to(device=parameter.device, dtype=parameter.dtype))

    @torch.no_grad()
    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Restore trainable parameters after validation.

        Args:
            trainer: Lightning trainer finishing validation.
            pl_module: Module whose parameters are restored.
        """
        del trainer
        if not self.backup_parameters:
            return
        for parameter, backup in zip(self._parameters(pl_module), self.backup_parameters, strict=True):
            parameter.copy_(backup)
        self.backup_parameters = []

    def on_exception(self, trainer: Trainer, pl_module: LightningModule, exception: BaseException) -> None:
        """Restore trainable weights if validation exits with an exception.

        Args:
            trainer: Lightning trainer that observed the exception.
            pl_module: Module whose parameters are restored.
            exception: The raised exception (ignored).
        """
        del exception
        self.on_validation_end(trainer, pl_module)
