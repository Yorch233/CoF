"""Enhance individual audio files with a trained CoF run.

Wrap :func:`cof.pipeline.inference.enhance_audio_files` with typed CLI
options. The command serves ad-hoc enhancement outside the paired-dataset
contract: inputs are arbitrary audio files or directories, and the output
directory still carries an ``enhance.json`` manifest recording the run,
model hash, and sampling protocol, so enhanced WAVs remain traceable.

Author: Qing Yao
Date: 2026/9/25
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

import typer

from cof.pipeline.inference import enhance_audio_files


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


def enhance(
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
    inputs: list[Path] = typer.Option(
        ...,
        "--input",
        exists=True,
        readable=True,
        help="Audio file or directory to enhance; repeat for several.",
    ),
    output_dir: Path = typer.Option(
        ...,
        "--output-dir",
        file_okay=False,
        help="Directory receiving the enhanced WAVs and enhance.json manifest.",
    ),
    sampler: Sampler | None = typer.Option(
        None,
        case_sensitive=True,
        help="Override the solver stored in the run configuration.",
    ),
    num_steps: int | None = typer.Option(None, "--num-steps", min=1),
    skip_type: SkipType | None = typer.Option(None, "--skip-type", case_sensitive=True),
    t_min: float | None = typer.Option(None, "--t-min", min=0.0, max=0.999999),
    seed: int = typer.Option(1234),
    device: str = typer.Option("auto", help="Torch device, CUDA index, or auto."),
    num_workers: int = typer.Option(0, "--num-workers", min=0),
    overwrite: bool = typer.Option(False, help="Replace conflicting or existing enhanced WAV files."),
    progress: bool = typer.Option(True, "--progress/--no-progress"),
) -> None:
    """Enhance individual audio files without a registered dataset.

    The model export is resolved from the run itself (recorded default test
    model first) and the sampling protocol stored in the run configuration is
    used unless explicitly overridden.
    """
    try:
        output, count = enhance_audio_files(
            run,
            checkpoint=checkpoint,
            inputs=inputs,
            output_dir=output_dir,
            sampler=None if sampler is None else str(sampler),
            num_steps=num_steps,
            skip_type=None if skip_type is None else str(skip_type),
            t_min=t_min,
            seed=seed,
            device=device,
            num_workers=num_workers,
            overwrite=overwrite,
            progress=progress,
        )
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(f"Enhanced {count} file(s) into {output}")
