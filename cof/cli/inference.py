"""Run generative inference over a registered dataset split.

Wrap :func:`cof.pipeline.inference.run_generative_inference` with typed CLI
options. The sampler, step count, and timestep schedule are part of the
result variant's identity, so a conflicting re-run fails closed unless
``--overwrite`` is given explicitly; the enhanced model itself is always
resolved from the referenced run rather than downloaded.

Author: Qing Yao
Date: 2026/9/23
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

import typer

from cof.pipeline.inference import run_generative_inference


class DatasetSplit(StrEnum):
    """Supported registered dataset splits."""

    TRAIN = "train"
    VALID = "valid"
    TEST = "test"


class Sampler(StrEnum):
    """Supported CoF sampling solvers."""

    SB_SDE = "SB_SDE_Solver"
    SB_ODE = "SB_ODE_Solver"
    OTCFM_ODE = "OTCFM_ODE_Solver"
    AUTO = "AUTO"


class SkipType(StrEnum):
    """Supported sampling timestep schedules."""

    UNIFORM = "time_uniform"
    QUADRATIC = "time_quadratic"


def inference(
    run: str | None = typer.Option(None, help="Local run name or run directory."),
    checkpoint: Path | None = typer.Option(
        None,
        "--ckpt",
        "--checkpoint",
        exists=True,
        file_okay=True,
        dir_okay=False,
        help="Self-describing model checkpoint file, used directly without a run directory.",
    ),
    dataset: str | None = typer.Option(None, help="Registered dataset ID; defaults to the selected ID."),
    split: DatasetSplit = typer.Option(DatasetSplit.TEST, case_sensitive=False),
    sampler: Sampler | None = typer.Option(
        None,
        case_sensitive=True,
        help="Override the solver stored in the run configuration.",
    ),
    num_steps: int | None = typer.Option(None, "--num-steps", min=1),
    skip_type: SkipType | None = typer.Option(None, "--skip-type", case_sensitive=True),
    t_min: float | None = typer.Option(None, "--t-min", min=0.0, max=0.999999),
    max_samples: int | None = typer.Option(None, "--max-samples", min=1),
    seed: int = typer.Option(1234),
    device: str = typer.Option("auto", help="Torch device, CUDA index, or auto."),
    num_workers: int = typer.Option(0, "--num-workers", min=0),
    overwrite: bool = typer.Option(False, help="Replace conflicting or existing result WAV files."),
    progress: bool = typer.Option(True, "--progress/--no-progress"),
) -> None:
    """Enhance a registered dataset split with a generative CoF model.

    The model export is resolved from the run itself (recorded default test
    model first); the sampling protocol stored in the run configuration is
    used unless explicitly overridden, and each override changes the name of
    the written result variant directory.

    Args:
        run: Target; local generative run name or directory.
        checkpoint: Target; self-describing model file used directly,
            mutually exclusive with ``run``.
        dataset: Target; registered dataset ID, defaulting to the selected
            ID.
        split: Target; registered split to enhance.
        sampler: Protocol override; solver replacing the one stored in the
            run configuration.
        num_steps: Protocol override; sampling step count (NFE).
        skip_type: Protocol override; timestep schedule.
        t_min: Protocol override; starting time of the sampling trajectory.
        max_samples: Limit; optional cap on the number of enhanced files.
        seed: Limit; sampling seed.
        device: Limit; torch device, CUDA index, or auto.
        num_workers: Limit; dataloader worker count.
        overwrite: Output; replace conflicting or existing result WAVs.
        progress: Output; render a progress bar.

    Raises:
        typer.BadParameter: If the run, dataset, or sampling protocol is
            invalid, or a result conflict was left unresolved.
    """
    try:
        output = run_generative_inference(
            run,
            checkpoint=checkpoint,
            dataset_id=dataset,
            split=split.value,
            sampler=sampler.value if sampler is not None else None,
            num_steps=num_steps,
            skip_type=skip_type.value if skip_type is not None else None,
            t_min=t_min,
            max_samples=max_samples,
            seed=seed,
            device=device,
            num_workers=num_workers,
            overwrite=overwrite,
            progress=progress,
        )
    except (OSError, RuntimeError, ValueError) as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(f"Generative results: {output}")
