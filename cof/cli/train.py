"""Launch the explicit pretrain and CoF posttrain training stages.

Translate CLI options into a resolved run configuration. Post-training is
locked to the identity of its ``--pretrain-run``: the inherited model and
sampling-protocol parameters are read from that run's ``config.yml``, and
passing them again on the command line is rejected. Per-run budget, data,
and runtime overrides stay available in both stages.

Release status: the CoF post-training implementation is withheld during peer
review. The posttrain stage prints an explanatory notice and exits before any
run directory is created; pretraining is fully available.

Author: Qing Yao
Date: 2026/9/23
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any

import typer

from cof.config.manager import Config
from cof.config.resolver import (
    TrainingStage,
    resolve_training_config,
    validate_resume,
    validate_stage_args,
)


class Optimizer(StrEnum):
    """Supported torch optimizers."""

    ADAM = "Adam"
    ADAMW = "AdamW"


class Logger(StrEnum):
    """Supported experiment loggers."""

    NONE = "none"
    WANDB = "wandb"


class SamplingSolver(StrEnum):
    """Solver used by deployment and training rollouts."""

    SB_SDE = "SB_SDE_Solver"
    SB_ODE = "SB_ODE_Solver"
    OTCFM_ODE = "OTCFM_ODE_Solver"
    AUTO = "AUTO"


class CtcCfModel(StrEnum):
    """Network evaluating the counterfactual CTC target on the oracle state."""

    ONLINE = "online"
    EMA = "ema"


class CtcFaModel(StrEnum):
    """Network supplying the factual CTC branch clean endpoints."""

    ONLINE = "online"
    EMA = "ema"


class SkipType(StrEnum):
    """Timestep schedule used by the rollout sampler."""

    UNIFORM = "time_uniform"
    QUADRATIC = "time_quadratic"


def _prepare_training_command(
    config: Config,
    overrides: dict[str, Any],
    *,
    checkpoint_path: Path | None,
    assume_yes: bool,
) -> tuple[Config, dict[str, Any]]:
    """Run rank-zero CUDA/JIT preflight and the final confirmation.

    Args:
        config: Resolved training configuration about to launch.
        overrides: Per-run CLI overrides to persist with the run record.
        checkpoint_path: Resume checkpoint, whose run directory supplies the
            effective configuration when resuming.
        assume_yes: Skip the interactive preflight confirmation.

    Returns:
        The possibly re-read configuration (resume runs) and the overrides
        extended with the resolved operator backend and CUDA JIT status.

    Raises:
        typer.Abort: If the preflight fails or the operator declines it.
    """
    # NOTE: Deferred imports keep CLI startup and ``--help`` light; the training
    # stack (preflight, runtime) is only needed once a launch is real.
    from cof.config.manager import read_config_from_yaml
    from cof.training.preflight import (
        BACKEND_FIELD,
        JIT_STATUS_FIELD,
        TrainingCancelled,
        prepare_training_launch,
    )
    from cof.training.runtime import distributed_rank, run_path_from_checkpoint

    if distributed_rank() > 0:
        # NOTE: Only rank zero owns the preflight; workers proceed with the
        # broadcast configuration.
        return config, overrides
    if checkpoint_path is not None:
        # IMPORTANT: A resumed run re-reads its original config.yml so the stored
        # run identity wins; only explicit CLI overrides are reapplied on top.
        run_path = run_path_from_checkpoint(checkpoint_path.resolve())
        config = read_config_from_yaml(run_path / "config.yml")
        config.update(overrides)
    try:
        prepare_training_launch(config, assume_yes=assume_yes)
    except TrainingCancelled as error:
        raise typer.Abort() from error
    overrides.update(
        {
            BACKEND_FIELD: config.get(BACKEND_FIELD),
            JIT_STATUS_FIELD: config.get(JIT_STATUS_FIELD),
        }
    )
    return config, overrides


def train(
    training_stage: TrainingStage = typer.Option(
        ...,
        "--training-stage",
        "--training_stage",
        case_sensitive=False,
        help="pretrain trains from scratch; posttrain CoF-post-trains a pretraining run.",
    ),
    formulation: str | None = typer.Option(
        None,
        "--formulation",
        case_sensitive=False,
        help="Pretraining only: the generative formulation to train (SB-VE or OT-CFM).",
    ),
    post_training_method: str = typer.Option(
        "cof",
        "--post-training-method",
        case_sensitive=True,
        help="Post-training only: the post-training method to apply (default: cof).",
    ),
    pretraining_method: str = typer.Option(
        "base",
        "--pretraining-method",
        case_sensitive=True,
        help="Pretraining only: the pretraining method to apply (default: base).",
    ),
    pretrain_run: Path | None = typer.Option(
        None,
        "--pretrain-run",
        "--pretrain_run",
        file_okay=False,
        help=(
            "Post-training only: pretraining run directory. Its config.yml supplies the "
            "inherited model/protocol parameters and its exported weights initialize the run."
        ),
    ),
    max_epoch: int | None = typer.Option(
        None, "--max-epoch", "--max_epoch", min=1, rich_help_panel="Optimization & Budget"
    ),
    max_steps: int | None = typer.Option(
        None, "--max-steps", "--max_steps", min=1, rich_help_panel="Optimization & Budget"
    ),
    learning_rate: float | None = typer.Option(
        None, "--learning-rate", "--learning_rate", min=0.0, rich_help_panel="Optimization & Budget"
    ),
    dataset: str | None = typer.Option(None, rich_help_panel="Run & Data"),
    run_name: str | None = typer.Option(None, "--run-name", "--run_name", rich_help_panel="Run & Data"),
    run_dir: Path | None = typer.Option(None, "--run-dir", "--run_dir", file_okay=False, rich_help_panel="Run & Data"),
    comment: str | None = typer.Option(
        None,
        "--comment",
        help="Free-form note describing the experiment purpose, persisted to the run config.yml.",
        rich_help_panel="Run & Data",
    ),
    batch_size: int | None = typer.Option(
        None, "--batch-size", "--batch_size", min=1, rich_help_panel="Optimization & Budget"
    ),
    optimizer: Optimizer | None = typer.Option(None, case_sensitive=True, rich_help_panel="Optimization & Budget"),
    logger: Logger | None = typer.Option(
        None, case_sensitive=False, rich_help_panel="Validation, Logging & Checkpoints"
    ),
    wandb_project: str | None = typer.Option(
        None, "--wandb-project", rich_help_panel="Validation, Logging & Checkpoints"
    ),
    ema: bool | None = typer.Option(None, "--ema/--no-ema", rich_help_panel="Optimization & Budget"),
    ema_rate: float | None = typer.Option(
        None, "--ema-rate", "--ema_rate", min=0.0, max=0.999999, rich_help_panel="Optimization & Budget"
    ),
    patience: int | None = typer.Option(None, min=1, rich_help_panel="Optimization & Budget"),
    num_workers: int | None = typer.Option(None, "--num-workers", "--num_workers", min=0, rich_help_panel="Run & Data"),
    seed: int | None = typer.Option(None, rich_help_panel="Run & Data"),
    precision: str | None = typer.Option(None, rich_help_panel="Runtime & Hardware"),
    accelerator: str | None = typer.Option(None, rich_help_panel="Runtime & Hardware"),
    devices: str | None = typer.Option(None, rich_help_panel="Runtime & Hardware"),
    strategy: str | None = typer.Option(None, rich_help_panel="Runtime & Hardware"),
    num_nodes: int | None = typer.Option(
        None, "--num-nodes", "--num_nodes", min=1, rich_help_panel="Runtime & Hardware"
    ),
    log_steps: int | None = typer.Option(
        None, "--log-steps", "--log_steps", min=1, rich_help_panel="Validation, Logging & Checkpoints"
    ),
    save_state_steps: int | None = typer.Option(
        None, "--save-state-steps", min=1, rich_help_panel="Validation, Logging & Checkpoints"
    ),
    checkpoints_total_limit: int | None = typer.Option(
        None, "--checkpoints-total-limit", min=1, rich_help_panel="Validation, Logging & Checkpoints"
    ),
    generative_backbone: str | None = typer.Option(
        None,
        "--generative-backbone",
        "--generative_backbone",
        help="Pretraining only; post-training inherits it from the pretraining run.",
        rich_help_panel="Pretraining Model & Sampling (inherited by post-training)",
    ),
    backbone_kwargs: str | None = typer.Option(
        None,
        "--backbone-kwargs",
        help="Pretraining only: backbone constructor keyword arguments as a JSON object.",
        rich_help_panel="Pretraining Model & Sampling (inherited by post-training)",
    ),
    formulation_kwargs: str | None = typer.Option(
        None,
        "--formulation-kwargs",
        help="Pretraining only: formulation-path constructor keyword arguments as a JSON object.",
        rich_help_panel="Pretraining Model & Sampling (inherited by post-training)",
    ),
    reduction: str | None = typer.Option(
        None, help="Loss reduction: mean or sum.", rich_help_panel="Optimization & Budget"
    ),
    sampling_solver: SamplingSolver | None = typer.Option(
        None,
        "--sampling-solver",
        case_sensitive=True,
        help=(
            "Solver for validation-time inference; resolves from the formulation when omitted "
            "(SB-VE: SB_SDE_Solver, OT-CFM: OTCFM_ODE_Solver). Post-training also generates its "
            "DRC/CTC rollouts with it and records it as the recommended inference solver."
        ),
        rich_help_panel="Pretraining Model & Sampling (inherited by post-training)",
    ),
    sampling_num_steps: int | None = typer.Option(
        None,
        "--sampling-num-steps",
        min=1,
        help="Validation-time inference step count; defaults to 4. Post-training inherits it.",
        rich_help_panel="Pretraining Model & Sampling (inherited by post-training)",
    ),
    sampling_skip_type: SkipType | None = typer.Option(
        None,
        "--sampling-skip-type",
        case_sensitive=True,
        help="Validation-time inference time-grid spacing; defaults to time_uniform. Post-training inherits it.",
        rich_help_panel="Pretraining Model & Sampling (inherited by post-training)",
    ),
    validation_every_n_steps: int | None = typer.Option(
        None, "--validation-every-n-steps", min=1, rich_help_panel="Validation, Logging & Checkpoints"
    ),
    rollout_n_max: int | None = typer.Option(
        None, "--rollout-n-max", min=1, rich_help_panel="CoF Post-Training Objective (DRC + CTC)"
    ),
    rollout_fixed_n: int | None = typer.Option(
        None,
        "--rollout-fixed-n",
        min=1,
        help="Pin the rollout deployment depth N; unset (null) keeps the sampled schedule.",
        rich_help_panel="CoF Post-Training Objective (DRC + CTC)",
    ),
    lambda_ctc: float | None = typer.Option(
        None,
        "--lambda-ctc",
        "--lambda_ctc",
        min=0.0,
        help="Weight of the CTC loss relative to DRC.",
        rich_help_panel="CoF Post-Training Objective (DRC + CTC)",
    ),
    ctc_steps: int | None = typer.Option(
        None,
        "--ctc-steps",
        "--ctc_steps",
        min=1,
        help="Number of uniform sub-steps between t and s in the CTC transition.",
        rich_help_panel="CoF Post-Training Objective (DRC + CTC)",
    ),
    ctc_cf_model: CtcCfModel | None = typer.Option(
        None,
        "--ctc-cf-model",
        "--ctc_cf_model",
        case_sensitive=False,
        help="Network evaluating the counterfactual CTC target on the oracle state.",
        rich_help_panel="CoF Post-Training Objective (DRC + CTC)",
    ),
    ctc_fa_model: CtcFaModel | None = typer.Option(
        None,
        "--ctc-fa-model",
        "--ctc_fa_model",
        case_sensitive=False,
        help="Network supplying the factual CTC branch clean endpoints.",
        rich_help_panel="CoF Post-Training Objective (DRC + CTC)",
    ),
    ctc_shared_noise: bool | None = typer.Option(
        None,
        "--ctc-shared-noise/--no-ctc-shared-noise",
        help="Share the k Brownian increments between the two CTC branches.",
        rich_help_panel="CoF Post-Training Objective (DRC + CTC)",
    ),
    gradient_checkpointing: bool | None = typer.Option(
        None,
        "--gradient-checkpointing/--no-gradient-checkpointing",
        rich_help_panel="Optimization & Budget",
    ),
    gradient_clip_val: float | None = typer.Option(
        None, "--gradient-clip-val", min=0.0, rich_help_panel="Optimization & Budget"
    ),
    require_cuda_jit: bool | None = typer.Option(None, "--require-cuda-jit", rich_help_panel="Runtime & Hardware"),
    resume: bool = typer.Option(False),
    checkpoint_path: Path | None = typer.Option(None, "--checkpoint-path", "--checkpoint_path", exists=True),
    yes: bool = typer.Option(False, "--yes", help="Skip preflight confirmation."),
) -> None:
    """Train CoF: epoch-based pretraining or the step-based CoF posttrain stage.

    Release status: the CoF post-training implementation is withheld during peer
    review — the posttrain stage prints an explanatory notice and exits. All
    post-training options below remain part of the released configuration
    contract and take effect once the implementation returns.

    Post-training locks its model/protocol identity to ``--pretrain-run``:
    inherited options are rejected on the CLI, while budget, data, runtime,
    and logging overrides stay available in both stages. Each option's help
    text documents its individual semantics and constraints.

    Args:
        training_stage: Stage; ``pretrain`` trains from scratch, ``posttrain``
            CoF-post-trains a pretraining run.
        formulation: Stage (pretraining only); generative formulation to train.
        post_training_method: Stage (post-training only); method to apply.
        pretraining_method: Stage (pretraining only); method to apply.
        pretrain_run: Stage (post-training only); pretraining run directory
            supplying the inherited configuration and initial weights.
        max_epoch: Budget; epoch cap for the epoch-based pretrain stage.
        max_steps: Budget; step cap.
        learning_rate: Budget; optimizer learning rate.
        batch_size: Budget; samples per step.
        optimizer: Budget; torch optimizer family.
        ema: Budget; exponential moving average switch.
        ema_rate: Budget; EMA decay rate.
        patience: Budget; early-stopping patience.
        gradient_checkpointing: Budget; activation checkpointing switch.
        gradient_clip_val: Budget; gradient clipping norm.
        reduction: Budget; loss reduction, ``mean`` or ``sum``.
        dataset: Data; registered dataset ID for this run.
        run_name: Data; explicit run name override.
        run_dir: Data; run root directory override.
        comment: Data; free-form note persisted to the run config.yml.
        num_workers: Data; dataloader worker count.
        seed: Data; training seed.
        precision: Runtime; trainer precision string.
        accelerator: Runtime; accelerator selection.
        devices: Runtime; device selection.
        strategy: Runtime; distributed strategy.
        num_nodes: Runtime; distributed node count.
        require_cuda_jit: Runtime; abort instead of falling back when the
            NCSN++ CUDA operators cannot be JIT-compiled.
        logger: Logging; experiment logger.
        wandb_project: Logging; Weights & Biases project name.
        log_steps: Logging; loss-logging interval in steps.
        save_state_steps: Logging; resumable-state interval in steps.
        checkpoints_total_limit: Logging; intermediate checkpoint limit.
        validation_every_n_steps: Logging; validation interval in steps.
        generative_backbone: Pretraining model (inherited by post-training);
            backbone architecture name.
        backbone_kwargs: Pretraining model (inherited); backbone constructor
            keyword arguments as a JSON object.
        formulation_kwargs: Pretraining model (inherited);
            formulation-path constructor keyword arguments as a JSON object.
        sampling_solver: Validation-time inference solver; resolves from the
            formulation when omitted. Post-training also generates its
            DRC/CTC rollouts with it and records it as the recommended
            inference solver.
        sampling_num_steps: Validation-time inference step count (default 4);
            inherited by post-training.
        sampling_skip_type: Validation-time inference timestep schedule;
            inherited by post-training.
        rollout_n_max: CoF objective; upper bound of the sampled rollout
            depths.
        rollout_fixed_n: CoF objective; pinned rollout depth N.
        lambda_ctc: CoF objective; CTC loss weight relative to DRC.
        ctc_steps: CoF objective; uniform sub-steps in the CTC transition.
        ctc_cf_model: CoF objective; network evaluating the counterfactual
            target.
        ctc_fa_model: CoF objective; network supplying the factual clean
            endpoints.
        ctc_shared_noise: CoF objective; share the Brownian increments
            between the CTC branches.
        resume: Resume; continue from ``checkpoint_path`` when set.
        checkpoint_path: Resume; resumable checkpoint file, which also
            derives the run directory.
        yes: Confirmation; skip the preflight confirmation.

    Raises:
        typer.BadParameter: If stage selection, resume arguments, or
            inherited post-training parameters are inconsistent.
    """
    from cof.training.runner import start_generative_training

    try:
        validate_resume(resume, checkpoint_path)
        if training_stage == TrainingStage.POSTTRAIN:
            typer.secho(
                "The CoF post-training stage is not runnable in this snapshot: the DRC + CTC\n"
                "implementation is withheld during peer review and will be released upon\n"
                "acceptance. Pretraining, inference, and evaluation are fully available.",
                fg=typer.colors.YELLOW,
                err=True,
            )
            raise typer.Exit(code=1)
        if not resume:
            validate_stage_args(training_stage, formulation, post_training_method, pretrain_run)
        config, overrides = resolve_training_config(
            training_stage=training_stage,
            formulation=formulation,
            pretrain_run=pretrain_run,
            resume_checkpoint=checkpoint_path,
            post_training_method=None if training_stage == TrainingStage.PRETRAIN else post_training_method,
            pretraining_method=None if training_stage == TrainingStage.POSTTRAIN else pretraining_method,
            num_epoch=max_epoch,
            max_steps=max_steps,
            learning_rate=learning_rate,
            dataset=dataset,
            run_name=run_name,
            run_dir=run_dir,
            comment=comment,
            batch_size=batch_size,
            optimizer=optimizer,
            logger=logger,
            wandb_project=wandb_project,
            ema=ema,
            ema_rate=ema_rate,
            patience=patience,
            num_workers=num_workers,
            seed=seed,
            precision=precision,
            accelerator=accelerator,
            devices=devices,
            strategy=strategy,
            num_nodes=num_nodes,
            log_steps=log_steps,
            save_state_steps=save_state_steps,
            checkpoints_total_limit=checkpoints_total_limit,
            generative_backbone=generative_backbone,
            backbone_kwargs=backbone_kwargs,
            formulation_kwargs=formulation_kwargs,
            reduction=reduction,
            sampling_solver=sampling_solver,
            sampling_num_steps=sampling_num_steps,
            sampling_skip_type=sampling_skip_type,
            validation_every_n_steps=validation_every_n_steps,
            rollout_n_max=rollout_n_max,
            rollout_fixed_n=rollout_fixed_n,
            lambda_ctc=lambda_ctc,
            ctc_steps=ctc_steps,
            ctc_cf_model=ctc_cf_model,
            ctc_fa_model=ctc_fa_model,
            ctc_shared_noise=ctc_shared_noise,
            gradient_checkpointing=gradient_checkpointing,
            gradient_clip_val=gradient_clip_val,
            require_cuda_jit=require_cuda_jit,
        )
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error
    config, overrides = _prepare_training_command(
        config,
        overrides,
        checkpoint_path=checkpoint_path,
        assume_yes=yes,
    )
    start_generative_training(config, checkpoint_path=checkpoint_path, overrides=overrides)
