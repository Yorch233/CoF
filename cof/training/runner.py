"""Orchestrate Lightning training for the separate pretrain and posttrain stages.

This module owns the generative run lifecycle: it resolves or creates the run
directory and its persisted configuration, builds the paired dataloaders and
callback suite, instantiates the stage-specific pipeline, and drives
``Trainer.fit``.  The two stages differ in shape — pretraining stops by epoch,
CoF post-training by optimizer step — and the stage contract is enforced here
before any weights are touched.

Author: Qing Yao
Date: 2026/9/23
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import lightning as L
import torch
import wandb
from lightning.pytorch.callbacks import RichProgressBar
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader

from cof.backbone.ncsnpp.ncsnpp_utils.op.backend import PYTORCH_NATIVE_BACKEND, activate_operator_backend
from cof.config.manager import Config, read_config_from_yaml
from cof.data import ComplexSpecDataset
from cof.evaluation import MetricSuite
from cof.formulation.registry import path_type_for
from cof.pipeline.base import training_stage_metric_name
from cof.pipeline.build import build_generative_pipeline
from cof.pipeline.provenance import file_sha256
from cof.training.callbacks import (
    BestModelExport,
    EmaCallback,
    FullStateModelCheckpoint,
    GenerativeSampleMetrics,
    LastModelExport,
    ValidationEarlyStopping,
)
from cof.training.runtime import (
    distributed_rank,
    resolve_resume_checkpoint,
    resolve_trainer_runtime,
    run_path_from_checkpoint,
    wandb_enabled,
)
from cof.utils.sampling import canonical_sampling_values, resolve_sampling_protocol


@dataclass(frozen=True)
class GenerativeRun:
    """Resolved generative training run metadata."""

    config: Config
    run_path: Path
    checkpoint_path: Path | None


def _training_target_for_path(config: Config) -> str:
    """Resolve the network target required by the configured probability path.

    Args:
        config: Effective run configuration.

    Returns:
        ``"vector"`` for OT-CFM paths; otherwise the configured training
        target (default ``"data"``).
    """
    return path_type_for(str(config.get("formulation.name"))).prediction_type


def _training_stage_for_config(config: Config) -> tuple[str, str]:
    """Validate the explicit training stage and return it with its stopping unit.

    The CLI selects an explicit stage, independently of the injected method.

    Args:
        config: Effective run configuration.

    Returns:
        The ``training_stage`` value paired with its ``training_unit``
        (``"epoch"`` for pretraining, ``"step"`` for post-training).

    Raises:
        ValueError: If the stage and post-training method disagree or the
            stage value is unsupported.
    """
    training_stage = str(config.get("stage") or "pretrain")
    if training_stage == "pretrain":
        return "pretrain", "epoch"
    if training_stage == "post_training":
        return "post_training", "step"
    raise ValueError(f"Unsupported training stage: {training_stage!r}")


def _trainer_schedule(config: Config) -> dict[str, int | float | None]:
    """Return stopping and validation controls for the selected stage.

    Args:
        config: Effective run configuration.

    Returns:
        ``max_epochs``/``max_steps`` and validation cadence keywords suitable
        for ``L.Trainer``; pretraining is epoch-based, post-training
        step-based.

    Raises:
        ValueError: If the configured training stage is unsupported.
    """
    training_stage = str(config.get("stage") or _training_stage_for_config(config)[0])
    if training_stage == "pretrain":
        return {
            "max_epochs": int(config.get("pretrain.num_epoch", 1000)),
            "max_steps": -1,
            "val_check_interval": 1.0,
            "check_val_every_n_epoch": 1,
        }
    if training_stage == "post_training":
        return {
            "max_epochs": None,
            "max_steps": int(config.get("posttrain.max_steps", -1)),
            "val_check_interval": int(config.get("posttrain.validation_every_n_steps", 1000)),
            "check_val_every_n_epoch": None,
        }
    raise ValueError(f"Unsupported training stage: {training_stage!r}")


def prepare_generative_run(
    config: Config,
    *,
    checkpoint_path: Path | None = None,
    now: datetime | None = None,
    run_id: str | None = None,
    overrides: dict[str, Any] | None = None,
) -> GenerativeRun:
    """Create a new run or restore the persisted configuration of a run.

    Args:
        config: Effective run configuration to author or reconcile.
        checkpoint_path: Resumable Lightning checkpoint; when given, the run
            directory is derived from it and its persisted ``config.yml`` is
            reloaded and re-canonicalized.
        now: Timestamp override used to name new runs (testing hook).
        run_id: Explicit experiment-tracking run ID for new runs.
        overrides: Per-run CLI overrides merged into a resumed configuration.

    Returns:
        The resolved run metadata: configuration, run directory, and
        checkpoint path (``None`` for a brand-new run).

    Raises:
        ValueError: If a resumed run's stage conflicts with its persisted
            method, if post-training lacks initialization weights, or if a
            distributed worker finds a non-generative run.
    """
    training_stage, training_unit = _training_stage_for_config(config)
    config.update(
        {
            "formulation.training_target": _training_target_for_path(config),
            "stage": training_stage,
        }
    )
    if checkpoint_path is not None:
        checkpoint_path = resolve_resume_checkpoint(checkpoint_path)
        run_path = run_path_from_checkpoint(checkpoint_path)
        resumed_config = read_config_from_yaml(run_path / "config.yml")
        resumed_config.update(overrides or {})
        resumed_stage, resumed_unit = _training_stage_for_config(resumed_config)
        resumed_config.update(
            {
                "formulation.training_target": _training_target_for_path(resumed_config),
                "stage": resumed_stage,
            }
        )
        resumed_config.update(canonical_sampling_values(resumed_config))
        resumed_config.save(run_path)
        return GenerativeRun(resumed_config, run_path, checkpoint_path.resolve())

    config.update(canonical_sampling_values(config))
    init_model_path = config.get("run.init_model_path")
    if training_stage == "post_training" and init_model_path is None:
        raise ValueError(
            "Post-training requires the pretraining weights: pass --pretrain-run (a pretraining run "
            "directory) or resume an existing run with --resume --checkpoint-path."
        )
    if init_model_path is not None:
        resolved_init_path = Path(init_model_path).expanduser().resolve()
        if not resolved_init_path.is_file():
            raise ValueError(f"Initial model weights do not exist: {resolved_init_path}")
        config.update(
            {
                "run.init_model_path": str(resolved_init_path),
                "run.init_model_sha256": file_sha256(resolved_init_path),
                "run.optimizer_restart": True,
            }
        )
    else:
        if config.get("run.optimizer_restart") is None:
            config.update({"run.optimizer_restart": False})

    timestamp = (now or datetime.now()).strftime("%m%d%H%M")
    run_name = config.get("run.run_name")
    if not run_name:
        # Default identity: cof_{formulation}_{stage}_{mmddHHMM} with a
        # lowercase formulation token, e.g. cof_sbve_posttrain_09241740.
        # The stage token uses the metric namespace (pretrain | posttrain),
        # not the internal "post_training".
        stage_token = training_stage_metric_name(training_stage)
        formulation_token = str(config.get("formulation.name") or "generative").lower()
        run_name = f"cof_{formulation_token}_{stage_token}_{timestamp}"
    run_path = Path(config.get("paths.run_dir", "runs")).expanduser().resolve() / run_name
    if distributed_rank() > 0:
        child_config = read_config_from_yaml(run_path / "config.yml")
        return GenerativeRun(child_config, run_path, None)
    run_path.mkdir(parents=True, exist_ok=False)
    config.update(
        {
            "run.run_name": run_name,
            "run.run_id": run_id or (wandb.util.generate_id() if wandb_enabled(config) else None),
            "run.run_path": str(run_path),
            "run.output_path": str(run_path),
        }
    )
    config.save(run_path)
    return GenerativeRun(config, run_path, None)


def build_generative_dataloader(config: Config, subset: str) -> DataLoader[Any]:
    """Build a paired clean/noisy spectrogram dataloader.

    Args:
        config: Effective run configuration.
        subset: Dataset split to load (``"train"``, ``"valid"``, or
            ``"test"``); training splits are shuffled.

    Returns:
        A dataloader over ``ComplexSpecDataset`` for the requested split.
    """
    dataset = ComplexSpecDataset(
        config,
        dataset=config.get("registry.dataset"),
        subset=subset,
        shuffle_spec=subset == "train",
        return_spec=True,
        dummy=bool(config.get("data.dummy", False)),
    )
    num_workers = int(config.get("runtime.num_workers", 0))
    return DataLoader(
        dataset,
        batch_size=int(config.get("optimization.batch_size", 8)),
        shuffle=subset == "train",
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def build_generative_callbacks(config: Config, run_path: Path) -> list[Any]:
    """Build EMA, validation export, resumable checkpoint, and telemetry callbacks.

    Args:
        config: Effective run configuration.
        run_path: Run directory receiving exported weights and checkpoints.

    Returns:
        Callbacks in fit order: optional EMA, sampled validation metrics,
        best-loss and best-PESQ exports, last-model export, full-state
        checkpointing, and optional validation early stopping.
    """
    callbacks: list[Any] = []
    training_stage = str(config.get("stage") or _training_stage_for_config(config)[0])
    metric_namespace = training_stage_metric_name(training_stage)
    valid_loss_monitor = f"{metric_namespace}/valid/loss"
    metric_suite = MetricSuite(
        tuple(config.get("evaluation.metrics", ["pesq", "si_sdr"])), int(config.get("data.sample_rate", 16000))
    )
    selection = str(config.get("evaluation.selection_metric", "PESQ"))
    if selection not in metric_suite.output_names:
        raise ValueError(f"Selection metric {selection!r} is not enabled")
    selection_token = selection.lower()
    selection_monitor = f"{metric_namespace}/valid/{selection}_per_epoch"
    protocol = resolve_sampling_protocol(config)
    if config.get("optimization.ema", True):
        callbacks.append(EmaCallback(decay=float(config.get("optimization.ema_rate", 0.999))))
    callbacks.extend(
        [
            GenerativeSampleMetrics(
                config,
                valid_samples=int(config.get("runtime.valid_metric_samples", 50)),
                num_steps=protocol.num_steps,
                solver=protocol.solver,
                skip_type=protocol.skip_type,
                seed=int(config.get("optimization.seed", 1234)),
            ),
            BestModelExport(
                run_path,
                config,
                monitor=valid_loss_monitor,
                mode="min",
                config_field="metrics.best_valid_loss",
                filename="model_valid_loss={:.6f}.safetensors",
                epoch_field="metrics.best_valid_loss_epoch",
                step_field="metrics.best_valid_loss_step",
                model_field="metrics.best_valid_loss_model",
                cleanup_patterns=("model_valid_loss=*.safetensors",),
            ),
            BestModelExport(
                run_path,
                config,
                monitor=selection_monitor,
                mode=metric_suite.modes[selection],
                config_field=f"metrics.best_{selection_token}",
                filename=f"model_valid_{selection_token}={{:.4f}}.safetensors",
                epoch_field=f"metrics.best_{selection_token}_epoch",
                step_field=f"metrics.best_{selection_token}_step",
                model_field="weights.default_test_model",
                cleanup_patterns=(f"model_valid_{selection_token}=*.safetensors",),
                strict=False,
            ),
            LastModelExport(run_path, config),
            FullStateModelCheckpoint(
                run_path,
                save_state_steps=int(config.get("runtime.save_state_steps", 1000)),
                checkpoints_total_limit=int(config.get("runtime.checkpoints_total_limit", 3)),
            ),
        ]
    )
    if config.get("runtime.early_stopping_enabled", False):
        callbacks.append(
            ValidationEarlyStopping(
                patience=int(config.get("runtime.patience", 20)),
                monitor=valid_loss_monitor,
                mode="min",
                config=config,
                run_path=run_path,
                best_field="metrics.best_valid_loss",
            )
        )
    return callbacks


def start_generative_training(
    config: Config,
    *,
    checkpoint_path: Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> Path:
    """Train or resume a generative CoF model and return its run directory.

    Args:
        config: Effective run configuration.
        checkpoint_path: Optional resumable checkpoint; requires ``--resume``
            semantics upstream and derives the run directory.
        overrides: Per-run overrides merged into a resumed configuration.

    Returns:
        The run directory holding the configuration, exported weights, and
        resumable checkpoints.

    Raises:
        ValueError: If a CTC branch requires EMA while the EMA callback is
            disabled, or if run preparation rejects the configuration.
    """
    run = prepare_generative_run(config, checkpoint_path=checkpoint_path, overrides=overrides)
    backbone = str(run.config.get("model.backbone", "ncsnpp_base") or "ncsnpp_base")
    if backbone.startswith("ncsnpp"):
        activate_operator_backend(run.config.get("host.ncsnpp_operator_backend", PYTORCH_NATIVE_BACKEND))
    L.seed_everything(int(run.config.get("optimization.seed", 1234)), workers=True)
    if wandb_enabled(run.config):
        logs_dir = Path(run.config.get("runtime.logs_dir", run.run_path))
        logs_dir.expanduser().resolve().mkdir(parents=True, exist_ok=True)
        logger: WandbLogger | bool = WandbLogger(
            project=str(run.config.get("runtime.wandb_project", "CoF")),
            name=run.config.get("run.run_name"),
            id=run.config.get("run.run_id"),
            save_dir=str(logs_dir),
            resume="allow" if run.checkpoint_path is not None else None,
        )
        logger.log_hyperparams(run.config.dict())
    else:
        logger = False

    runtime = resolve_trainer_runtime(run.config)
    training_schedule = _trainer_schedule(run.config)
    trainer = L.Trainer(
        accelerator=runtime.accelerator,
        devices=runtime.devices,
        strategy=runtime.strategy,
        num_nodes=int(run.config.get("runtime.num_nodes", 1)),
        precision=runtime.precision,
        max_epochs=training_schedule["max_epochs"],
        max_steps=training_schedule["max_steps"],
        val_check_interval=training_schedule["val_check_interval"],
        check_val_every_n_epoch=training_schedule["check_val_every_n_epoch"],
        default_root_dir=run.run_path,
        logger=logger,
        callbacks=[*build_generative_callbacks(run.config, run.run_path), RichProgressBar()],
        log_every_n_steps=int(run.config.get("runtime.log_steps", 10)),
    )
    pipeline = build_generative_pipeline(run.config, load_initial_weights=run.checkpoint_path is None)
    ema_index = next(
        (index for index, callback in enumerate(trainer.callbacks) if isinstance(callback, EmaCallback)),
        None,
    )
    if ema_index is not None:
        ema = trainer.callbacks[ema_index]

        def ema_predictor(state: torch.Tensor, time: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
            """Evaluate EMA parameters without modifying the live model weights."""
            shadow = ema.shadow_parameter_dict(pipeline, prefix="model.")
            with torch.no_grad():
                return torch.func.functional_call(pipeline.model, shadow, (state, time, [condition])).detach()

        pipeline.method.references["ema"] = ema_predictor
    missing = pipeline.method.required_references() - pipeline.method.references.keys()
    if missing:
        raise ValueError(f"Required reference predictors are not configured: {sorted(missing)}")
    trainer.fit(
        pipeline,
        train_dataloaders=build_generative_dataloader(run.config, "train"),
        val_dataloaders=build_generative_dataloader(run.config, "valid"),
        ckpt_path=str(run.checkpoint_path) if run.checkpoint_path is not None else None,
    )
    return run.run_path
