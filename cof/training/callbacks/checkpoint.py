"""Persist full-state checkpoints, best-model exports, and the final model.

The callbacks separate two artifact kinds: bounded, fully resumable
Lightning states under ``checkpoints/``, and EMA safetensors exports in the
run directory whose filenames and ``config.yml`` fields double as the
run's validation-selected model provenance.

Author: Qing Yao
Date: 2026/9/23
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from torch import Tensor

from cof.config.manager import Config
from cof.training.callbacks.ema import EmaCallback

if TYPE_CHECKING:
    from lightning.pytorch import LightningModule, Trainer


def _atomic_save_model(model: Any, model_path: Path) -> None:
    """Atomically replace one safetensors model artifact.

    Args:
        model: Model whose state is serialized.
        model_path: Final destination path.
    """
    model.save_checkpoint(model_path)


class FullStateModelCheckpoint(ModelCheckpoint):
    """Periodically save bounded, fully resumable Lightning states."""

    def __init__(self, run_path: str | Path, *, save_state_steps: int, checkpoints_total_limit: int) -> None:
        """Initialize checkpoint storage below a run directory.

        Args:
            run_path: Run directory receiving a ``checkpoints`` subdirectory.
            save_state_steps: Interval, in optimizer steps, between full
                state saves.
            checkpoints_total_limit: Number of most recent intermediate
                ``step=*.ckpt`` files to keep.

        Raises:
            ValueError: If either limit is not positive.
        """
        if save_state_steps < 1:
            raise ValueError("save_state_steps must be positive")
        if checkpoints_total_limit < 1:
            raise ValueError("checkpoints_total_limit must be positive")
        self.checkpoints_total_limit = checkpoints_total_limit
        super().__init__(
            dirpath=Path(run_path) / "checkpoints",
            filename="step={step}",
            monitor=None,
            save_top_k=-1,
            save_last=True,
            save_weights_only=False,
            save_on_exception=True,
            every_n_train_steps=save_state_steps,
            save_on_train_epoch_end=False,
            auto_insert_metric_name=False,
        )

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Save on schedule, then prune intermediate checkpoints beyond the configured limit.

        Args:
            trainer: Lightning trainer driving the fit.
            pl_module: Training module.
            outputs: Step outputs from the training step.
            batch: Current batch.
            batch_idx: Index of the current batch.
        """
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        if trainer.is_global_zero:
            self._prune_intermediate_checkpoints()

    def _prune_intermediate_checkpoints(self) -> None:
        """Delete the oldest intermediate checkpoints past the retention limit."""
        checkpoints = sorted(Path(self.dirpath).glob("step=*.ckpt"), key=lambda path: path.stat().st_mtime_ns)
        for checkpoint in checkpoints[: -self.checkpoints_total_limit]:
            checkpoint.unlink()


class BestModelExport(Callback):
    """Export the best monitored model and its matching run configuration."""

    def __init__(
        self,
        run_path: str | Path,
        config: Config,
        monitor: str = "valid/loss_per_epoch",
        *,
        mode: str = "min",
        config_field: str = "best_valid_loss",
        filename: str = "model.safetensors",
        epoch_field: str = "best_model_epoch",
        step_field: str = "best_model_step",
        model_field: str | None = None,
        cleanup_patterns: tuple[str, ...] = (),
        strict: bool = True,
    ) -> None:
        """Initialize the safetensors export callback.

        Args:
            run_path: Run directory receiving exported weights.
            config: Run configuration updated with the best-score fields.
            monitor: Logged metric name to watch.
            mode: ``"min"`` or ``"max"`` improvement direction.
            config_field: ``config.yml`` field recording the best score.
            filename: Export filename pattern formatted with the score.
            epoch_field: ``config.yml`` field recording the best epoch.
            step_field: ``config.yml`` field recording the best step.
            model_field: Optional ``config.yml`` field naming the exported
                file (e.g. the default test model).
            cleanup_patterns: Glob patterns of stale exports removed after
                each improved save.
            strict: Whether a missing monitored metric is an error.

        Raises:
            ValueError: If ``mode`` is neither ``"min"`` nor ``"max"``.
        """
        super().__init__()
        if mode not in {"min", "max"}:
            raise ValueError("mode must be 'min' or 'max'")
        self.run_path = Path(run_path)
        self.config = config
        self.monitor = monitor
        self.mode = mode
        self.config_field = config_field
        self.filename = filename
        self.epoch_field = epoch_field
        self.step_field = step_field
        self.model_field = model_field
        self.cleanup_patterns = cleanup_patterns
        self.strict = strict
        self.best_score: float | None = config.get(config_field)

    @property
    def state_key(self) -> str:
        """Keep multiple metric-specific best-model callbacks independently resumable."""
        return self._generate_state_key(
            monitor=self.monitor,
            mode=self.mode,
            filename=self.filename,
        )

    def state_dict(self) -> dict[str, Any]:
        """Persist the current best score in full-state checkpoints.

        Returns:
            Mapping holding the current best score.
        """
        return {"best_score": self.best_score}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore the current best score on resume.

        Args:
            state_dict: State written by ``state_dict``.
        """
        best_score = state_dict.get("best_score")
        self.best_score = float(best_score) if best_score is not None else None

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Atomically export improved EMA weights and update config.yml.

        Args:
            trainer: Lightning trainer driving validation.
            pl_module: Validated module whose EMA weights are exported.
        """
        if trainer.sanity_checking:
            return
        metric = trainer.callback_metrics.get(self.monitor)
        if metric is None:
            if self.strict:
                raise RuntimeError(f"Monitored metric {self.monitor!r} was not logged")
            return
        score = float(metric.detach().cpu()) if isinstance(metric, Tensor) else float(metric)
        improved = self.best_score is None or (
            score < self.best_score if self.mode == "min" else score > self.best_score
        )
        if not improved:
            return

        self.best_score = score
        if not trainer.is_global_zero:
            return
        model = getattr(pl_module, "model", pl_module)
        filename = self.filename.format(score, score=score)
        model_path = self.run_path / filename
        _atomic_save_model(model, model_path)
        for pattern in self.cleanup_patterns:
            for stale_path in self.run_path.glob(pattern):
                if stale_path != model_path:
                    stale_path.unlink()
        fields = {
            self.config_field: score,
            self.epoch_field: trainer.current_epoch,
            self.step_field: trainer.global_step,
        }
        if self.model_field is not None:
            fields[self.model_field] = filename
        self.config.update(fields)
        self.config.save(self.run_path)


class LastModelExport(Callback):
    """Export the final EMA-averaged model as a safetensors artifact.

    With EMA enabled the export swaps the shadow weights in exactly like
    validation does, so ``model_last.safetensors`` always matches the weights
    the exported ``model_valid_*`` artifacts and the validation metrics were
    computed on; without EMA it falls back to the trainable weights.
    """

    def __init__(
        self,
        run_path: str | Path,
        config: Config,
        filename: str = "model_last.safetensors",
    ) -> None:
        """Configure the final model artifact.

        Args:
            run_path: Run directory receiving the export.
            config: Run configuration updated with final-model fields.
            filename: Export filename.
        """
        super().__init__()
        self.run_path = Path(run_path)
        self.config = config
        self.filename = filename

    @staticmethod
    def _ema_callback(trainer: Trainer) -> EmaCallback | None:
        """Return the trainer's EMA callback, or ``None`` when absent."""
        return next(
            (callback for callback in getattr(trainer, "callbacks", []) if isinstance(callback, EmaCallback)),
            None,
        )

    @staticmethod
    @torch.no_grad()
    def _swap_ema_in(pl_module: LightningModule, ema: EmaCallback | None) -> list[Tensor]:
        """Copy EMA shadow weights into the trainable parameters; return the backup.

        Args:
            pl_module: Module whose trainable parameters are replaced.
            ema: EMA callback providing the shadow parameters.

        Returns:
            Backup of the original trainable parameters, empty when there is
            nothing to swap.
        """
        if ema is None or not ema.shadow_parameters:
            return []
        parameters = [parameter for parameter in pl_module.parameters() if parameter.requires_grad]
        if len(parameters) != len(ema.shadow_parameters):
            raise RuntimeError("EMA shadow parameters do not match the trainable model parameters")
        backup = [parameter.detach().clone() for parameter in parameters]
        for parameter, shadow in zip(parameters, ema.shadow_parameters, strict=True):
            parameter.copy_(shadow.to(device=parameter.device, dtype=parameter.dtype))
        return backup

    @staticmethod
    @torch.no_grad()
    def _restore(pl_module: LightningModule, backup: list[Tensor]) -> None:
        """Restore the trainable parameters backed up by ``_swap_ema_in``.

        Args:
            pl_module: Module whose parameters are restored.
            backup: Parameter copies written back in order.
        """
        if not backup:
            return
        parameters = [parameter for parameter in pl_module.parameters() if parameter.requires_grad]
        for parameter, weights in zip(parameters, backup, strict=True):
            parameter.copy_(weights)

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Atomically export the final EMA weights and record its provenance.

        Args:
            trainer: Lightning trainer finishing the fit.
            pl_module: Module whose weights are exported.
        """
        if not trainer.is_global_zero:
            return
        swapped = self._swap_ema_in(pl_module, self._ema_callback(trainer))
        try:
            model = getattr(pl_module, "model", pl_module)
            _atomic_save_model(model, self.run_path / self.filename)
            self.config.update(
                {
                    "weights.last_model": self.filename,
                    "weights.last_model_epoch": trainer.current_epoch,
                    "weights.last_model_step": trainer.global_step,
                    "weights.last_model_weights": "ema" if swapped else "online",
                }
            )
            self.config.save(self.run_path)
        finally:
            self._restore(pl_module, swapped)
