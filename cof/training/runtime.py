"""Resolve user-facing training settings into Lightning Trainer arguments.

The module maps the host wizard fields written by ``cof config`` (precision,
multi-GPU, device selection) onto validated ``accelerator``/``devices``/
``strategy``/``precision`` values, while letting explicit advanced CLI
settings win.  Run-directory resolution and distributed-rank detection for
the training entry points also live here.

Author: Qing Yao
Date: 2026/9/23
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from cof.config.manager import Config

MIXED_PRECISION = {"none": "32-true", "fp16": "16-mixed", "bf16": "bf16-mixed"}


def wandb_enabled(config: Config) -> bool:
    """Return whether the configured experiment logger is WandB.

    Args:
        config: Host or run configuration.

    Returns:
        True when ``logger`` is unset or set to ``"wandb"``.
    """
    return config.get("runtime.logger", "wandb") == "wandb"


def resolve_resume_checkpoint(checkpoint_path: Path) -> Path:
    """Resolve directories to last.ckpt and preserve an explicitly selected state.

    Checkpoints must exist inside a run's checkpoints directory. Directory
    inputs require last.ckpt; explicit files do not fall back to another state.
    """
    path = checkpoint_path.expanduser().resolve()
    if path.is_dir():
        if (path / "checkpoints").is_dir():
            path = path / "checkpoints" / "last.ckpt"
        elif path.name == "checkpoints":
            path = path / "last.ckpt"
        else:
            raise ValueError(f"No checkpoints/last.ckpt found under {path}")
    if path.suffix != ".ckpt" or path.parent.name != "checkpoints":
        raise ValueError("A resume checkpoint must be a run directory, checkpoints directory, or checkpoints/*.ckpt")
    if not path.is_file():
        raise ValueError(f"Resume checkpoint does not exist: {path}")
    if not (path.parent.parent / "config.yml").is_file():
        raise ValueError(f"Resume checkpoint has no owning run config.yml: {path}")
    return path


def run_path_from_checkpoint(checkpoint_path: Path) -> Path:
    """Return the run owning a directory-selected or explicit checkpoint."""
    return resolve_resume_checkpoint(checkpoint_path).parent.parent


def distributed_rank() -> int:
    """Return the rank assigned by a distributed launcher, defaulting to zero.

    Returns:
        The ``RANK`` or ``LOCAL_RANK`` environment value as an integer, or
        zero outside a distributed launch.

    Raises:
        ValueError: If the environment value is not an integer.
    """
    value = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))
    try:
        return int(value)
    except ValueError as error:
        raise ValueError(f"Distributed rank must be an integer, got {value!r}") from error


@dataclass(frozen=True)
class TrainerRuntime:
    """Lightning runtime values resolved from local wizard settings."""

    accelerator: str
    devices: str | int | list[int]
    strategy: str
    precision: str


def resolve_trainer_runtime(config: Config) -> TrainerRuntime:
    """Resolve wizard fields while honoring explicit advanced CLI overrides.

    Args:
        config: Merged host and CLI configuration.

    Returns:
        Validated Lightning runtime values.

    Raises:
        ValueError: If the precision choice, ``multi_gpu`` flag, or
            ``gpu_ids`` list is invalid or internally inconsistent.
    """
    mixed_precision = config.get("runtime.mixed_precision", "none")
    if mixed_precision not in MIXED_PRECISION:
        choices = ", ".join(MIXED_PRECISION)
        raise ValueError(f"mixed_precision must be one of: {choices}")
    precision = config.get("runtime.precision") or MIXED_PRECISION[mixed_precision]

    multi_gpu = config.get("runtime.multi_gpu", False)
    if not isinstance(multi_gpu, bool):
        raise ValueError("multi_gpu must be a boolean")
    gpu_ids = config.get("runtime.gpu_ids", "all")
    if gpu_ids == "all":
        devices: str | int | list[int] = "auto" if multi_gpu else 1
    elif isinstance(gpu_ids, list) and gpu_ids and all(isinstance(device_id, int) for device_id in gpu_ids):
        if len(gpu_ids) != len(set(gpu_ids)) or any(device_id < 0 for device_id in gpu_ids):
            raise ValueError("gpu_ids must contain unique, non-negative integer IDs")
        if multi_gpu and len(gpu_ids) < 2:
            raise ValueError("multi_gpu requires at least two GPU IDs")
        if not multi_gpu and len(gpu_ids) != 1:
            raise ValueError("single-GPU training requires exactly one GPU ID")
        devices = gpu_ids
    else:
        raise ValueError("gpu_ids must be 'all' or a non-empty list of integer IDs")

    return TrainerRuntime(
        accelerator=config.get("runtime.accelerator", "gpu"),
        devices=config.get("runtime.devices") or devices,
        strategy=config.get("runtime.strategy") or ("ddp" if multi_gpu else "auto"),
        precision=precision,
    )
