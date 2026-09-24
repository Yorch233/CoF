"""Enhance registered dataset splits with trained generative CoF runs.

The workflow resolves the run, dataset, and sampling protocol, derives a
variant directory whose name carries the protocol identity, writes every
enhanced WAV under a manifested, resumable result directory, and records the
exact model hash and sampling settings in ``inference.json`` so later metric
evaluation can verify provenance.

Author: Qing Yao
Date: 2026/9/23
"""

from __future__ import annotations

import hashlib
import random
import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

import torch
import torchaudio
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from cof.model import GenerativeModel4SE
from cof.pipeline.provenance import (
    RunArtifact,
    finish_manifest,
    paired_wavs,
    prepare_artifact_directory,
    resolve_dataset,
    resolve_model_reference,
    result_directory,
)
from cof.utils.atomic import atomic_output_path
from cof.utils.progress import track_workflow_items
from cof.utils.sampling import resolve_inference_t_min, resolve_sampling_protocol

DeviceName = str | torch.device


class _InferenceAudioDataset(Dataset[tuple[str, Tensor]]):
    """Lazily load and resample named noisy WAV files.

    Args:
        paths: Noisy WAV paths to serve, in order.
        sample_rate: Sample rate each waveform is resampled to.
    """

    def __init__(self, paths: list[Path], sample_rate: int) -> None:
        """Store the path list and target sample rate."""
        self.paths = paths
        self.sample_rate = sample_rate

    def __len__(self) -> int:
        """Return the number of served files."""
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[str, Tensor]:
        """Load one waveform and return it with its filename.

        Args:
            index: Position in the path list.

        Returns:
            The filename and its waveform tensor.
        """
        path = self.paths[index]
        audio, source_rate = torchaudio.load(path)
        if source_rate != self.sample_rate:
            audio = torchaudio.functional.resample(audio, source_rate, self.sample_rate)
        return path.name, audio


def resolve_device(value: DeviceName = "auto") -> torch.device:
    """Resolve an automatic, CUDA-index, or explicit torch device selection.

    Args:
        value: ``"auto"``, a plain CUDA index, or a ``torch.device``
            expression; an existing device object is passed through.

    Returns:
        The resolved ``torch.device``.

    Raises:
        ValueError: If a CUDA device was requested but CUDA is unavailable.
    """
    if isinstance(value, torch.device):
        return value
    normalized = value.strip().lower()
    if normalized == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if normalized.isdigit():
        normalized = f"cuda:{normalized}"
    device = torch.device(normalized)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(f"CUDA device {device} was requested but CUDA is unavailable")
    return device


def _audio_iterator(
    paths: list[Path],
    *,
    sample_rate: int,
    num_workers: int,
    progress: bool,
    description: str,
) -> Iterator[tuple[str, Tensor]]:
    """Yield loaded waveforms, optionally under a progress bar.

    Args:
        paths: Noisy WAV paths to load.
        sample_rate: Sample rate each waveform is resampled to.
        num_workers: Dataloader worker processes; zero loads inline.
        progress: Whether to display a progress bar.
        description: Progress-bar label.

    Yields:
        Filename and waveform pairs in input order.
    """
    loader = DataLoader(
        _InferenceAudioDataset(paths, sample_rate),
        batch_size=None,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )
    yield from track_workflow_items(loader, description, total=len(paths), enabled=progress)


def _set_file_seed(seed: int, filename: str) -> None:
    """Seed the global RNGs for one file's sampling trajectory."""
    # NOTE: The base seed is derived from the filename digest so every file
    # keeps a deterministic, order-independent trajectory for a given run seed.
    digest = hashlib.sha256(filename.encode(), usedforsecurity=False).digest()
    file_seed = (seed + int.from_bytes(digest[:4], "big")) % (2**31)
    torch.manual_seed(file_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(file_seed)


def _warn_off_rollout_solver(run: RunArtifact, solver: str) -> None:
    """Warn when a post-trained run is sampled with a non-rollout solver.

    CoF post-training adapts the model to rollouts of one canonical solver,
    so sampling the result with any other solver departs from the
    post-trained configuration.

    Args:
        run: Resolved run artifact carrying the recorded stage and solver.
        solver: Solver the inference request resolved to.
    """
    if run.config.get("stage") != "post_training":
        return
    rollout_solver = run.config.get("formulation.sampling.solver")
    if rollout_solver is not None and solver != rollout_solver:
        warnings.warn(
            f"Run {run.name!r} was post-trained with the {rollout_solver} rollout solver; "
            f"sampling with {solver} departs from the post-trained configuration.",
            stacklevel=3,
        )


def run_generative_inference(
    run_reference: str | Path | None = None,
    *,
    checkpoint: str | Path | None = None,
    dataset_id: str | None = None,
    split: str = "test",
    sampler: Literal["SB_SDE_Solver", "SB_ODE_Solver", "OTCFM_ODE_Solver", "AUTO"] | None = None,
    num_steps: int | None = None,
    skip_type: Literal["time_uniform", "time_quadratic"] | None = None,
    t_min: float | None = None,
    max_samples: int | None = None,
    seed: int = 1234,
    device: DeviceName = "auto",
    num_workers: int = 0,
    overwrite: bool = False,
    progress: bool = True,
) -> Path:
    """Enhance a registered dataset split with a trained generative CoF run.

    Args:
        run_reference: Run name or directory providing the model.
        checkpoint: Self-describing model file used directly, without a run
            directory; mutually exclusive with ``run_reference``.
        dataset_id: Explicit dataset ID overriding the configured selection.
        split: Dataset split to enhance.
        sampler: Solver override; ``None`` or ``"AUTO"`` resolves the
            configured or path-default solver.
        num_steps: Sampling step count (NFE) override.
        skip_type: Time-skip schedule override.
        t_min: Lower network-evaluation time override.
        max_samples: Optional cap on the number of files, sampled with the
            run seed and recorded in the manifest.
        seed: Base seed; each file derives a deterministic per-file seed.
        device: Device selection understood by ``resolve_device``.
        num_workers: Dataloader worker processes for audio loading.
        overwrite: Clear and rewrite a conflicting result directory.
        progress: Whether to display a progress bar.

    Returns:
        The variant directory holding the WAVs and ``inference.json``.

    Raises:
        ValueError: If ``max_samples`` is below one or the run, dataset, or
            sampling protocol cannot be resolved.
    """
    if max_samples is not None and max_samples < 1:
        raise ValueError("max_samples must be at least 1")
    run = resolve_model_reference(run_reference, checkpoint)
    protocol = resolve_sampling_protocol(
        run.config,
        solver=sampler,
        num_steps=num_steps,
        skip_type=skip_type,
    )
    _warn_off_rollout_solver(run, protocol.solver)
    resolved_t_min = resolve_inference_t_min(run.config, t_min)
    selected_dataset, dataset_root = resolve_dataset(run.config, dataset_id)
    pairs = paired_wavs(dataset_root, split)
    if max_samples is not None and max_samples < len(pairs):
        pairs = sorted(random.Random(seed).sample(pairs, k=max_samples), key=lambda pair: pair[1].name)
    target_device = resolve_device(device)
    sample_rate = int(run.config.get("data.sample_rate", 16_000))
    variant = f"{protocol.solver}_N={protocol.num_steps}"
    output = result_directory(run, variant)
    bridge = GenerativeModel4SE.from_checkpoint(run.model_path, config=run.config, device=target_device)
    time_grid = bridge.sampling_time_grid(
        num_steps=protocol.num_steps, solver=protocol.solver, skip_type=protocol.skip_type, t_min=resolved_t_min
    )
    signature = {
        "artifact_type": "generative_inference",
        "dataset_id": selected_dataset,
        "split": split,
        "source_type": run.source_type,
        "model_path": str(run.model_path),
        "run_name": run.name,
        "run_path": str(run.path),
        "run_id": run.config.get("run.run_id"),
        "model_sha256": run.model_sha256,
        "sample_rate": sample_rate,
        "device": str(target_device),
        "sampler": protocol.solver,
        "num_steps": protocol.num_steps,
        "skip_type": protocol.skip_type,
        "t_min": resolved_t_min,
        "time_grid": time_grid.tolist(),
        "terminal_time": time_grid[-1].item(),
        "max_samples": max_samples,
        "seed": seed,
    }
    expected = [noisy.name for _, noisy in pairs]
    existing = prepare_artifact_directory(output, signature, overwrite=overwrite, expected_files=expected)
    pending = [noisy for _, noisy in pairs if noisy.name not in existing]
    if pending:
        for name, audio in _audio_iterator(
            pending,
            sample_rate=sample_rate,
            num_workers=num_workers,
            progress=progress,
            description="CoF inference",
        ):
            _set_file_seed(seed, name)
            enhanced, _, _ = bridge.enhance(
                audio,
                num_steps=protocol.num_steps,
                solver=protocol.solver,
                skip_type=protocol.skip_type,
                t_min=resolved_t_min,
                time_grid=time_grid,
            )
            with atomic_output_path(output / name) as temporary:
                torchaudio.save(temporary, enhanced.float().reshape(1, -1), sample_rate)
    finish_manifest(output, signature, expected)
    return output


_AUDIO_SUFFIXES = ("*.wav", "*.flac", "*.ogg")


def _collect_inputs(inputs: list[Path]) -> list[Path]:
    """Expand files and directories into an ordered list of audio files.

    Args:
        inputs: Paths as given on the command line; a directory contributes
            its audio files sorted by name.

    Returns:
        The expanded, de-duplicated audio paths in input order.

    Raises:
        ValueError: If nothing resolves to at least one audio file.
    """
    collected: list[Path] = []
    for item in inputs:
        if item.is_dir():
            collected.extend(sorted(path for pattern in _AUDIO_SUFFIXES for path in item.glob(pattern)))
        else:
            collected.append(item)
    unique: list[Path] = []
    for path in collected:
        if path not in unique:
            unique.append(path)
    if not unique:
        raise ValueError(f"No audio files found among: {', '.join(str(item) for item in inputs)}")
    return unique


def enhance_audio_files(
    run_reference: str | Path | None = None,
    *,
    checkpoint: str | Path | None = None,
    inputs: list[Path],
    output_dir: Path,
    sampler: Literal["SB_SDE_Solver", "SB_ODE_Solver", "OTCFM_ODE_Solver", "AUTO"] | None = None,
    num_steps: int | None = None,
    skip_type: Literal["time_uniform", "time_quadratic"] | None = None,
    t_min: float | None = None,
    seed: int = 1234,
    device: DeviceName = "auto",
    num_workers: int = 0,
    overwrite: bool = False,
    progress: bool = True,
) -> tuple[Path, int]:
    """Enhance individual audio files with a trained generative CoF run.

    Args:
        run_reference: Run name or directory providing the model.
        checkpoint: Self-describing model file used directly, without a run
            directory; mutually exclusive with ``run_reference``.
        inputs: Audio files and/or directories to enhance.
        output_dir: Directory receiving the enhanced WAVs and
            ``enhance.json`` manifest.
        sampler: Solver override; ``None`` or ``"AUTO"`` resolves the
            configured or path-default solver.
        num_steps: Sampling step count (NFE) override.
        skip_type: Time-skip schedule override.
        t_min: Lower network-evaluation time override.
        seed: Base seed; each file derives a deterministic per-file seed.
        device: Device selection understood by ``resolve_device``.
        num_workers: Dataloader worker processes for audio loading.
        overwrite: Clear and rewrite a conflicting output directory.
        progress: Whether to display a progress bar.

    Returns:
        The output directory and the number of enhanced files.

    Raises:
        ValueError: If no input resolves to an audio file or the run or
            sampling protocol cannot be resolved.
    """
    files = _collect_inputs([Path(item).expanduser() for item in inputs])
    output = output_dir.expanduser().resolve()
    if overwrite and any(path.parent.resolve() == output or path.resolve().parent == output for path in files):
        raise ValueError("With --overwrite, input and output paths must not overlap")
    run = resolve_model_reference(run_reference, checkpoint)
    protocol = resolve_sampling_protocol(
        run.config,
        solver=sampler,
        num_steps=num_steps,
        skip_type=skip_type,
    )
    _warn_off_rollout_solver(run, protocol.solver)
    resolved_t_min = resolve_inference_t_min(run.config, t_min)
    target_device = resolve_device(device)
    sample_rate = int(run.config.get("data.sample_rate", 16_000))
    outputs = [path.stem + ".wav" for path in files]
    if len(set(outputs)) != len(outputs):
        raise ValueError("Input files share a filename stem; output names would collide")
    bridge = GenerativeModel4SE.from_checkpoint(run.model_path, config=run.config, device=target_device)
    time_grid = bridge.sampling_time_grid(
        num_steps=protocol.num_steps, solver=protocol.solver, skip_type=protocol.skip_type, t_min=resolved_t_min
    )
    signature = {
        "artifact_type": "generative_enhancement",
        "source_type": run.source_type,
        "model_path": str(run.model_path),
        "run_name": run.name,
        "run_path": str(run.path),
        "run_id": run.config.get("run.run_id"),
        "model_sha256": run.model_sha256,
        "sample_rate": sample_rate,
        "device": str(target_device),
        "sampler": protocol.solver,
        "num_steps": protocol.num_steps,
        "skip_type": protocol.skip_type,
        "t_min": resolved_t_min,
        "time_grid": time_grid.tolist(),
        "terminal_time": time_grid[-1].item(),
        "seed": seed,
        "inputs": [str(path) for path in files],
    }
    existing = prepare_artifact_directory(
        output, signature, overwrite=overwrite, expected_files=outputs, manifest_name="enhance.json"
    )
    pending = [path for path, name in zip(files, outputs, strict=True) if name not in existing]
    if pending:
        for name, audio in _audio_iterator(
            pending,
            sample_rate=sample_rate,
            num_workers=num_workers,
            progress=progress,
            description="CoF enhancement",
        ):
            _set_file_seed(seed, name)
            enhanced, _, _ = bridge.enhance(
                audio,
                num_steps=protocol.num_steps,
                solver=protocol.solver,
                skip_type=protocol.skip_type,
                t_min=resolved_t_min,
                time_grid=time_grid,
            )
            with atomic_output_path(output / (Path(name).stem + ".wav")) as temporary:
                torchaudio.save(temporary, enhanced.float().reshape(1, -1), sample_rate)
    finish_manifest(output, signature, outputs, manifest_name="enhance.json")
    return output, len(files)
