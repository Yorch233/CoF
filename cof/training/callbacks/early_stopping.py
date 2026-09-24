"""Stop training from a monitored validation metric.

The callback extends Lightning's early stopping with run-config persistence
and explicit logging of the stopping state, so a stopped run's ``config.yml``
records why it ended.

Author: Qing Yao
Date: 2026/9/23
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from lightning.pytorch.callbacks import EarlyStopping

from cof.config.manager import Config

if TYPE_CHECKING:
    from lightning.pytorch import LightningModule, Trainer


class ValidationEarlyStopping(EarlyStopping):
    """Stop training after a configured monitored metric stops improving."""

    def __init__(
        self,
        patience: int = 50,
        monitor: str = "valid/loss_per_epoch",
        *,
        mode: str = "min",
        config: Config | None = None,
        run_path: str | Path | None = None,
        best_field: str | None = None,
    ) -> None:
        """Initialize monitored early stopping and optional run-config persistence.

        Args:
            patience: Number of validations without improvement before
                stopping.
            monitor: Logged metric name to watch; the stopping-state
                diagnostics are namespaced next to it (its parent path).
            mode: ``"min"`` or ``"max"`` improvement direction.
            config: Optional run configuration updated with the stopping
                state.
            run_path: Directory where the run configuration is saved.
            best_field: Optional ``config.yml`` field recording the best
                monitored value.
        """
        super().__init__(
            monitor=monitor,
            mode=mode,
            patience=patience,
            check_finite=True,
            check_on_train_epoch_end=False,
        )
        # Namespace the stopping-state diagnostics next to the monitored
        # metric so they land in the same ``{stage}/valid`` panel group.
        prefix = monitor.rsplit("/", 1)[0]
        self.metric_names = {
            "wait_count": f"{prefix}/early_stopping_wait_count_per_epoch",
            "best_value": f"{prefix}/early_stopping_best_value_per_epoch",
            "best_epoch": f"{prefix}/early_stopping_best_epoch_per_epoch",
        }
        self.run_config = config
        self.run_path = Path(run_path) if run_path is not None else None
        self.best_field = best_field
        self.best_epoch = -1

    def state_dict(self) -> dict[str, Any]:
        """Persist the epoch associated with the best monitored value.

        Returns:
            The parent state extended with ``best_epoch``.
        """
        return {**super().state_dict(), "best_epoch": self.best_epoch}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore early-stopping state, including the best epoch.

        Args:
            state_dict: State written by ``state_dict``.
        """
        parent_state = dict(state_dict)
        self.best_epoch = int(parent_state.pop("best_epoch", -1))
        super().load_state_dict(parent_state)

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Run the stopping check, log its state, and persist summary fields.

        Args:
            trainer: Lightning trainer driving validation.
            pl_module: Validated module.
        """
        previous_best = float(self.best_score.detach().cpu())
        super().on_validation_end(trainer, pl_module)
        if trainer.sanity_checking:
            return
        best_value = float(self.best_score.detach().cpu())
        if best_value != previous_best:
            self.best_epoch = trainer.current_epoch
        if not trainer.is_global_zero:
            return

        logger = trainer.logger
        if logger is not None:
            logger.log_metrics(
                {
                    self.metric_names["wait_count"]: int(self.wait_count),
                    self.metric_names["best_value"]: best_value,
                    self.metric_names["best_epoch"]: self.best_epoch,
                },
                step=trainer.global_step,
            )
        if self.run_config is None or self.run_path is None:
            return
        updates = {"metrics.early_stop_cnt": int(self.wait_count)}
        if self.best_field is not None:
            updates[self.best_field] = float(self.best_score.detach().cpu())
        self.run_config.update(updates)
        self.run_config.save(self.run_path)
