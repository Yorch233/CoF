"""Single owner of the training-run configuration merge.

Five sources feed one effective run configuration, applied in this order:

1. ``config/<formulation>/pretrain.yml`` — the tracked preset (pretrain only);
2. ``.config/cof.yml`` — host settings, whose ``inherit`` field is ignored;
3. CLI flags — only values the user actually passed;
4. for post-training, the pretraining run's persisted ``config.yml`` supplies
   the model/protocol identity parameters verbatim, overlaid with the
   stage-owned defaults from ``config/<formulation>/posttrain.yml``;
5. explicit stage and registered method selection.

The CLI layer owns argument declarations; every merge rule and stage contract
lives here and reports violations as ``ValueError``.

Author: Qing Yao
Date: 2026/9/23
"""

from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml

from cof.config.manager import Config, read_config_from_yaml
from cof.formulation.registry import path_type_for
from cof.method.registry import PostTrainingRegister, PretrainingRegister, load_builtin_methods
from cof.utils.paths import PROJECT_ROOT, USER_CONFIG_PATH


class TrainingStage(StrEnum):
    """Training stage selected on the CLI."""

    PRETRAIN = "pretrain"
    POSTTRAIN = "posttrain"


# NOTE: Model- and protocol-identity parameters — the pretraining run fixes
# them and post-training inherits them verbatim; overriding them on the CLI is
# an error.
POSTTRAIN_LOCKED_FLAGS = {
    "generative_backbone": "--generative-backbone",
    "backbone_kwargs": "--backbone-kwargs",
    "formulation_kwargs": "--formulation-kwargs",
    "sampling_num_steps": "--sampling-num-steps",
    "sampling_skip_type": "--sampling-skip-type",
}
# NOTE: The rollout solver is a per-run post-training choice (CLI or preset),
# but an interrupted run must resume with the solver it was trained with.
# NOTE: Run-bookkeeping keys that must not leak from a pretraining run into a
# new post-training run.
RUN_ARTIFACT_KEYS = {"run", "metrics", "weights", "host"}

# CLI flag names are flat; the resolver places each value into the nested
# run-configuration hierarchy.
OVERRIDE_PLACEMENT = {
    "generative_backbone": "model.backbone",
    "backbone_kwargs": "model.backbone_kwargs",
    "formulation_kwargs": "formulation.kwargs",
    "sampling_solver": "formulation.sampling.solver",
    "sampling_num_steps": "formulation.sampling.num_steps",
    "sampling_skip_type": "formulation.sampling.skip_type",
    "learning_rate": "optimization.learning_rate",
    "batch_size": "optimization.batch_size",
    "optimizer": "optimization.optimizer",
    "ema": "optimization.ema",
    "ema_rate": "optimization.ema_rate",
    "seed": "optimization.seed",
    "reduction": "optimization.reduction",
    "time_loss_weight": "formulation.kwargs.time_loss_weight",
    "gradient_checkpointing": "optimization.gradient_checkpointing",
    "gradient_clip_val": "optimization.gradient_clip_val",
    "dataset": "registry.dataset",
    "datasets": "registry.datasets",
    "run_name": "run.run_name",
    "run_dir": "paths.run_dir",
    "comment": "run.comment",
    "logger": "runtime.logger",
    "wandb_project": "runtime.wandb_project",
    "num_workers": "runtime.num_workers",
    "precision": "runtime.precision",
    "accelerator": "runtime.accelerator",
    "devices": "runtime.devices",
    "strategy": "runtime.strategy",
    "num_nodes": "runtime.num_nodes",
    "log_steps": "runtime.log_steps",
    "save_state_steps": "runtime.save_state_steps",
    "checkpoints_total_limit": "runtime.checkpoints_total_limit",
    "patience": "runtime.patience",
}

# Stage-owned CLI knobs land in different groups per branch.
PRETRAIN_STAGE_PLACEMENT = {
    "pretraining_method": "pretrain.method",
    "num_epoch": "pretrain.num_epoch",
    "max_steps": "pretrain.max_steps",
}
POSTTRAIN_STAGE_PLACEMENT = {
    "post_training_method": "posttrain.method",
    "max_steps": "posttrain.max_steps",
    "validation_every_n_steps": "posttrain.validation_every_n_steps",
    "post_training_si_sdr_weight": "posttrain.cof.objective.si_sdr_weight",
    "post_training_magnitude_weight": "posttrain.cof.objective.magnitude_weight",
    "post_training_complex_weight": "posttrain.cof.objective.complex_weight",
    "post_training_compression": "posttrain.cof.objective.compression",
    "rollout_n_max": "posttrain.cof.drc.rollout.n_max",
    "rollout_fixed_n": "posttrain.cof.drc.rollout.fixed_n",
    "lambda_ctc": "posttrain.cof.ctc.lambda",
    "ctc_steps": "posttrain.cof.ctc.steps",
    "ctc_cf_model": "posttrain.cof.ctc.cf_model",
    "ctc_fa_model": "posttrain.cof.ctc.fa_model",
    "ctc_shared_noise": "posttrain.cof.ctc.shared_noise",
}

HOST_PLACEMENT = {
    "batch_size": "optimization.batch_size",
    "optimizer": "optimization.optimizer",
    "learning_rate": "optimization.learning_rate",
    "ema": "optimization.ema",
    "ema_rate": "optimization.ema_rate",
    "seed": "optimization.seed",
    "dataset": "registry.dataset",
    "datasets": "registry.datasets",
    "run_dir": "paths.run_dir",
    "results_dir": "paths.results_dir",
    "logger": "runtime.logger",
    "num_workers": "runtime.num_workers",
    "precision": "runtime.precision",
    "mixed_precision": "runtime.mixed_precision",
    "multi_gpu": "runtime.multi_gpu",
    "gpu_ids": "runtime.gpu_ids",
    "accelerator": "runtime.accelerator",
    "devices": "runtime.devices",
    "strategy": "runtime.strategy",
    "log_steps": "runtime.log_steps",
    "save_state_steps": "runtime.save_state_steps",
    "checkpoints_total_limit": "runtime.checkpoints_total_limit",
    "valid_metric_samples": "runtime.valid_metric_samples",
    "patience": "runtime.patience",
    "early_stopping_enabled": "runtime.early_stopping_enabled",
    "require_cuda_jit": "host.require_cuda_jit",
    "ncsnpp_operator_backend": "host.ncsnpp_operator_backend",
    "ncsnpp_cuda_jit_status": "host.ncsnpp_cuda_jit_status",
    "logs_dir": "runtime.logs_dir",
}


def _prune_run_metadata(config: dict[str, Any], keys: set[str]) -> None:
    """Drop run-specific bookkeeping groups from a nested mapping in place."""
    for key in keys:
        config.pop(key, None)


def normalize_explicit_overrides(overrides: dict[str, Any]) -> dict[str, Any]:
    """Keep only CLI values the user actually passed and parse JSON mappings.

    Args:
        overrides: Raw CLI keyword values; ``None`` and empty strings mean
            "not passed".

    Returns:
        The explicit subset of the overrides with ``StrEnum``/``Path``
        normalized to strings and the two kwargs mappings parsed from JSON.

    Raises:
        ValueError: If a kwargs mapping is present but is not a JSON object.
    """
    explicit = {
        key: (value.value if isinstance(value, StrEnum) else str(value) if isinstance(value, Path) else value)
        for key, value in overrides.items()
        if value is not None and value != ""
    }
    for mapping_name in ("formulation_kwargs", "backbone_kwargs"):
        if mapping_name not in explicit:
            continue
        raw_value = explicit[mapping_name]
        if isinstance(raw_value, dict):
            parsed = raw_value
        else:
            try:
                parsed = json.loads(str(raw_value))
            except json.JSONDecodeError as error:
                raise ValueError(f"{mapping_name} must be a JSON object") from error
        if not isinstance(parsed, dict):
            raise ValueError(f"{mapping_name} must be a JSON object")
        explicit[mapping_name] = parsed
    return explicit


def read_host_settings(host_config_path: Path) -> dict[str, Any]:
    """Read the machine-local settings without following their ``inherit`` chain.

    Args:
        host_config_path: Path to the host-local YAML settings file.

    Returns:
        The top-level settings mapping, or an empty mapping when the file is
        absent; the ``inherit`` key is always removed.

    Raises:
        ValueError: If the host configuration root is not a mapping.
    """
    if not Path(host_config_path).exists():
        return {}
    raw = yaml.safe_load(Path(host_config_path).read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Host configuration {host_config_path} must be a mapping")
    raw.pop("inherit", None)
    return raw


def validate_dataset_registered(config: Config) -> None:
    """Reject a run whose selected dataset is absent from the registry.

    Args:
        config: Merged run configuration carrying ``datasets`` and the
            selected ``dataset`` ID.

    Raises:
        ValueError: If no dataset is registered or the selected ID is unknown.
    """
    datasets = Config.unwrap(config.get("registry.datasets") or {})
    if not isinstance(datasets, dict) or not datasets:
        raise ValueError("No datasets are registered; run 'cof dataset add --id ID --path PATH'")
    selected_dataset = config.get("registry.dataset")
    if not isinstance(selected_dataset, str) or selected_dataset not in datasets:
        available = ", ".join(datasets)
        raise ValueError(f"Dataset ID {selected_dataset!r} is not registered; available IDs: {available}")


def _require_registered_pretraining(method: str) -> None:
    """Reject a pretraining method that has no registered framework.

    Args:
        method: Pretraining method name.

    Raises:
        ValueError: If the method is not present in the registry.
    """
    load_builtin_methods()
    try:
        PretrainingRegister.fetch(method)
    except KeyError as error:
        choices = ", ".join(sorted(PretrainingRegister.names()))
        raise ValueError(f"Unsupported pretraining method: {method!r}; registered methods: {choices}") from error


def _require_registered_method(method: str) -> None:
    """Reject a post-training method that has no registered objective.

    Args:
        method: Post-training method name.

    Raises:
        ValueError: If the method is not present in the registry.
    """
    load_builtin_methods()
    try:
        PostTrainingRegister.fetch(method)
    except KeyError as error:
        choices = ", ".join(sorted(PostTrainingRegister.names()))
        raise ValueError(f"Unsupported post-training method: {method!r}; registered methods: {choices}") from error


def validate_stage_args(
    training_stage: TrainingStage,
    formulation: str | None,
    post_training_method: str | None,
    pretrain_run: Path | None,
) -> None:
    """Enforce the stage-specific CLI contract before any configuration loads.

    Args:
        training_stage: Stage selected on the CLI.
        formulation: Formulation flag value (pretraining only).
        post_training_method: Post-training method flag value.
        pretrain_run: Pretraining run directory (post-training only).

    Raises:
        ValueError: If a required flag is missing or a flag is passed that the
            stage does not accept.
    """
    if training_stage == TrainingStage.PRETRAIN:
        if formulation is None:
            raise ValueError("--formulation is required for --training-stage pretrain")
        if post_training_method is not None and post_training_method != "cof":
            raise ValueError("--post-training-method is not allowed for --training-stage pretrain")
        if pretrain_run is not None:
            raise ValueError("--pretrain-run is not allowed for --training-stage pretrain")
        return
    if formulation is not None:
        raise ValueError(
            "--formulation is not allowed for --training-stage posttrain; it is inherited from the pretraining run"
        )
    if post_training_method is None:
        raise ValueError("--post-training-method is required for --training-stage posttrain")
    _require_registered_method(post_training_method)
    if pretrain_run is None:
        raise ValueError("--pretrain-run is required for --training-stage posttrain")


def validate_resume(resume: bool, checkpoint_path: Path | None) -> None:
    """Require ``--resume`` and ``--checkpoint-path`` to be passed together.

    Args:
        resume: Whether resume was requested.
        checkpoint_path: Checkpoint path passed on the CLI, if any.

    Raises:
        ValueError: If only one of the two was provided.
    """
    if resume and checkpoint_path is None:
        raise ValueError("--checkpoint-path is required with --resume")
    if not resume and checkpoint_path is not None:
        raise ValueError("--checkpoint-path requires --resume")


def resolve_training_config(
    *,
    training_stage: TrainingStage,
    formulation: str | None,
    pretrain_run: Path | None,
    host_config_path: Path = USER_CONFIG_PATH,
    resume_checkpoint: Path | None = None,
    **overrides: Any,
) -> tuple[Config, dict[str, Any]]:
    """Resolve the effective run configuration for the requested stage.

    Pretraining starts from ``config/<formulation>/pretrain.yml`` plus the
    machine-local host settings.  Post-training starts from the pretraining
    run's persisted ``config.yml`` — its model and protocol identity parameters
    are inherited verbatim — and overlays only the stage-owned defaults from
    ``config/<formulation>/posttrain.yml`` plus the CLI overrides.

    Args:
        training_stage: Stage selected on the CLI.
        formulation: Formulation selecting the tracked pretraining preset;
            required for pretraining, rejected for post-training.
        pretrain_run: Pretraining run directory; required for post-training.
        resume_checkpoint: Exact full-state checkpoint, bypassing new-run source requirements.
        host_config_path: Host-local settings file read without its
            ``inherit`` chain.
        **overrides: Explicit CLI flag values.

    Returns:
        The effective run configuration and the overrides still owed to the
        trainer runtime.

    Raises:
        ValueError: If a stage contract, dataset registration, JSON mapping,
            or pretraining-run requirement is violated.
    """
    explicit = normalize_explicit_overrides(overrides)
    stage_placement = (
        PRETRAIN_STAGE_PLACEMENT if training_stage == TrainingStage.PRETRAIN else POSTTRAIN_STAGE_PLACEMENT
    )
    placement = {**OVERRIDE_PLACEMENT, **stage_placement}
    placed: dict[str, Any] = {placement.get(key, key): value for key, value in explicit.items()}
    if resume_checkpoint is not None:
        from cof.training.runtime import run_path_from_checkpoint

        config = read_config_from_yaml(run_path_from_checkpoint(resume_checkpoint.resolve()) / "config.yml")
        expected_stage = "pretrain" if training_stage == TrainingStage.PRETRAIN else "post_training"
        if config.get("stage") != expected_stage:
            raise ValueError("Resume stage conflicts with the checkpoint run")
        forbidden = set(explicit) & (
            set(POSTTRAIN_LOCKED_FLAGS)
            | {"sampling_solver", "pretraining_method", "post_training_method", "run_name", "run_dir"}
        )
        # CLI method defaults repeat the configured identity and are harmless.
        for key, path in (("pretraining_method", "pretrain.method"), ("post_training_method", "posttrain.method")):
            if key in forbidden and explicit[key] == config.get(path):
                forbidden.remove(key)
        if forbidden:
            raise ValueError(f"Cannot change component identity while resuming: {sorted(forbidden)}")
        config.update(placed)
        validate_dataset_registered(config)
        return config, placed
    if training_stage == TrainingStage.PRETRAIN:
        preset = PROJECT_ROOT / "config" / path_type_for(str(formulation)).preset_name / "pretrain.yml"
        config = Config(read_config_from_yaml(preset).dict())
        host = read_host_settings(host_config_path)
        config.update({HOST_PLACEMENT.get(key, key): value for key, value in host.items()})
        config.update(placed)
        config.update({"stage": "pretrain"})
        _require_registered_pretraining(str(config.get("pretrain.method", "base")))
    else:
        locked_passed = sorted(POSTTRAIN_LOCKED_FLAGS[flag] for flag in POSTTRAIN_LOCKED_FLAGS if flag in explicit)
        if locked_passed:
            joined = ", ".join(locked_passed)
            raise ValueError(
                f"{joined} are inherited from the pretraining run and cannot be overridden "
                "for --training-stage posttrain"
            )
        pretrain_path = Path(pretrain_run).expanduser().resolve()
        pretrain_config_file = pretrain_path / "config.yml"
        if not pretrain_config_file.is_file():
            raise ValueError(
                f"--pretrain-run must point to a pretraining run directory with a config.yml; got {pretrain_path}"
            )
        pretrain_config = read_config_from_yaml(pretrain_config_file).dict()
        if str(pretrain_config.get("stage", "pretrain")) != "pretrain":
            raise ValueError(f"The run at {pretrain_path} is not a pretraining run")
        formulation_name = str(pretrain_config.get("formulation", {}).get("name", ""))
        formulation = formulation_name.upper().replace("-", "").replace("_", "")
        formulation_dir = path_type_for(formulation).preset_name
        if formulation_dir is None:
            raise ValueError(f"The pretraining run config has no known formulation: {formulation!r}")
        stage_file = (
            yaml.safe_load((PROJECT_ROOT / "config" / formulation_dir / "posttrain.yml").read_text(encoding="utf-8"))
            or {}
        )
        stage_file.pop("inherit", None)
        from cof.pipeline.provenance import resolve_model_path

        init_model_path = resolve_model_path(pretrain_path, Config(pretrain_config))
        _prune_run_metadata(pretrain_config, RUN_ARTIFACT_KEYS)
        pretrain_config.pop("pretrain", None)

        config = Config(pretrain_config)
        config.update(stage_file)
        config.update(placed)
        config.update({"stage": "post_training"})
        config.update({"run.pretrain_run": str(pretrain_path), "run.init_model_path": str(init_model_path)})
    validate_dataset_registered(config)
    return config, placed
