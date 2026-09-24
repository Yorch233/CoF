"""Evaluate speech-enhancement results and write per-file plus summary metrics.

Three entry points share one calculation core: run-linked evaluation of a
manifested result directory, direct directory evaluation through the same
manifest, and third-party evaluation of explicit clean/noisy/enhanced
directories. Registered MetricSuite components define their own output names
and are shared with sampled validation.

Author: Qing Yao
Date: 2026/9/23
"""

from __future__ import annotations

import concurrent.futures
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import pandas as pd

from cof.evaluation import MetricSuite, supported_metrics
from cof.pipeline.provenance import (
    file_sha256,
    paired_wavs,
    read_json,
    resolve_dataset,
    resolve_model_reference,
    result_directory,
    write_json,
)
from cof.utils.atomic import atomic_output_path
from cof.utils.progress import build_workflow_progress

DEFAULT_METRICS = ("pesq", "estoi", "si_sdr")


def _load_audio(path: Path, sample_rate: int) -> np.ndarray:
    """Load one waveform as mono audio resampled to the target rate."""
    audio, _ = librosa.load(path, sr=sample_rate, mono=True)
    return audio


def _evaluate_file(
    clean_path: Path,
    noisy_path: Path,
    enhanced_path: Path,
    suite: MetricSuite,
    sample_rate: int,
) -> dict[str, Any]:
    """Score one clean/noisy/enhanced triple for the selected metrics.

    Args:
        clean_path: Reference WAV path.
        noisy_path: Degraded WAV path (loaded for symmetry with the pair
            contract; not scored by the intrusive metrics).
        enhanced_path: Enhanced WAV path under evaluation.
        suite: The shared configured metric components.
        sample_rate: Target sample rate all three files are resampled to.

    Returns:
        A row mapping ``filename`` to one float per requested output.
    """
    clean = _load_audio(clean_path, sample_rate)
    noisy = _load_audio(noisy_path, sample_rate)
    enhanced = _load_audio(enhanced_path, sample_rate)
    length = min(clean.size, noisy.size, enhanced.size)
    clean, noisy, enhanced = clean[:length], noisy[:length], enhanced[:length]
    values: dict[str, Any] = {"filename": clean_path.name}
    values.update(suite.calculate(clean, enhanced))
    return values


def _validate_metric_cache(
    csv_path: Path,
    metadata: dict[str, Any],
    filenames: list[str],
    output_names: tuple[str, ...],
) -> None:
    """Reject incomplete or inconsistent CSV/JSON pairs before reuse.

    Args:
        csv_path: Persisted per-file scores.
        metadata: Parsed metrics JSON, optionally including a CSV hash.
        filenames: Expected input filenames.
        output_names: Columns declared by the selected metric components.

    Raises:
        ValueError: If the files disagree or the CSV is malformed.
    """
    try:
        recorded_hash = metadata.get("csv_sha256")
        if recorded_hash is not None and recorded_hash != file_sha256(csv_path):
            raise ValueError("CSV hash does not match metrics.json")
        frame = pd.read_csv(csv_path, dtype={"filename": str}, keep_default_na=False)
        if set(frame.columns) != {"filename", *output_names}:
            raise ValueError("Unexpected metric columns")
        if not filenames or sorted(frame["filename"].tolist()) != sorted(filenames):
            raise ValueError("Metric filenames do not match the requested files")
        if metadata.get("num_files") != len(frame):
            raise ValueError("Metric file count disagrees with the CSV")
        for name in output_names:
            values = frame[name].to_numpy(dtype=np.float64)
            if not np.isfinite(values).all():
                raise ValueError("Metric scores must be finite")
            recorded = metadata["summary"][name]
            for key, value in (("mean", values.mean()), ("std", values.std())):
                if not np.isclose(float(recorded[key]), value, rtol=1e-6, atol=1e-8):
                    raise ValueError("Metric summary disagrees with the CSV")
    except (ValueError, KeyError, TypeError) as error:
        raise ValueError(f"Invalid metric artifacts: {error}; use --overwrite") from error


def _calculate_metrics(
    pairs: list[tuple[Path, Path]],
    output: Path,
    metadata: dict[str, Any],
    *,
    sample_rate: int,
    metrics: tuple[str, ...],
    max_workers: int,
    overwrite: bool,
) -> tuple[Path, Path]:
    """Calculate metrics over all pairs and write the CSV and JSON artifacts.

    Args:
        pairs: Clean/noisy WAV path pairs; enhanced audio is read from
            ``output`` using the noisy filename.
        output: Result directory receiving ``metrics.csv`` and
            ``metrics.json``.
        metadata: Provenance fields recorded in the JSON summary.
        sample_rate: Sample rate used for loading and scoring.
        metrics: Metric names to compute; unknown names are rejected.
        max_workers: Thread count for per-file scoring; zero uses the
            executor default.
        overwrite: Recompute even when complete metric artifacts exist.

    Returns:
        The written CSV and JSON paths, short-circuiting to the existing
        pair when both already exist and ``overwrite`` is false.

    Raises:
        ValueError: If a metric name is unsupported or only one of the two
            metric artifacts exists.
    """
    invalid = [metric for metric in metrics if metric not in supported_metrics()]
    if invalid:
        raise ValueError(f"Unsupported metrics: {invalid}; choose from {supported_metrics()}")
    selected_metrics = tuple(dict.fromkeys(metrics or DEFAULT_METRICS))
    csv_path = output / "metrics.csv"
    json_path = output / "metrics.json"
    if (csv_path.exists() or json_path.exists()) and not overwrite:
        if csv_path.is_file() and json_path.is_file():
            current = read_json(json_path)
            expected = {**metadata, "sample_rate": sample_rate, "metrics": list(selected_metrics)}
            conflicts = [key for key, value in expected.items() if current.get(key) != value]
            if conflicts:
                raise ValueError(f"Existing metric artifacts conflict on {conflicts}; use --overwrite")
            _validate_metric_cache(
                csv_path,
                current,
                [noisy.name for _, noisy in pairs],
                MetricSuite(selected_metrics, sample_rate).output_names,
            )
            return csv_path, json_path
        raise ValueError(f"Partial metric artifacts exist in {output}; use --overwrite")

    suite = MetricSuite(selected_metrics, sample_rate)
    workers = None if max_workers == 0 else max_workers
    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                _evaluate_file,
                clean,
                noisy,
                output / noisy.name,
                suite,
                sample_rate,
            )
            for clean, noisy in pairs
        ]
        with build_workflow_progress() as progress:
            task = progress.add_task("Calculating metrics", total=len(futures))
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
                progress.advance(task)
    results.sort(key=lambda item: str(item["filename"]))
    frame = pd.DataFrame(results)
    summary = {
        column: {
            "mean": float(np.nanmean(frame[column].to_numpy(dtype=np.float64))),
            "std": float(np.nanstd(frame[column].to_numpy(dtype=np.float64))),
        }
        for column in frame.columns
        if column != "filename"
    }
    # Stage both files before publishing CSV then JSON. If interrupted between
    # replacements, the JSON's hash prevents reuse of a mixed-generation pair.
    with atomic_output_path(json_path) as temporary_json, atomic_output_path(csv_path) as temporary_csv:
        frame.to_csv(temporary_csv, index=False)
        write_json(
            temporary_json,
            {
                "artifact_type": "metrics",
                **metadata,
                "sample_rate": sample_rate,
                "metrics": list(selected_metrics),
                "num_files": len(results),
                "summary": summary,
                "csv_sha256": file_sha256(temporary_csv),
            },
        )
    return csv_path, json_path


def evaluate_results(
    run_reference: str | Path | None,
    variant: str,
    *,
    dataset_id: str | None = None,
    split: str = "test",
    metrics: tuple[str, ...] = DEFAULT_METRICS,
    max_workers: int = 0,
    overwrite: bool = False,
    result_dir: Path | None = None,
    checkpoint: str | Path | None = None,
) -> tuple[Path, Path]:
    """Evaluate a canonical run result directory and save per-file and summary metrics.

    Args:
        run_reference: Run name or directory owning the result.
        variant: Sampling-protocol variant directory name.
        dataset_id: Explicit dataset ID overriding the configured selection.
        split: Dataset split to score.
        metrics: Metric names to compute.
        max_workers: Thread count for per-file scoring; zero uses the
            executor default.
        overwrite: Recompute even when complete metric artifacts exist.
        result_dir: Explicit result directory overriding the canonical one.
        checkpoint: Model file source, mutually exclusive with ``run_reference``.

    Returns:
        The written ``metrics.csv`` and ``metrics.json`` paths.

    Raises:
        ValueError: If the run or dataset cannot be resolved, or the result
            fails provenance validation against its inference manifest.
    """
    run = resolve_model_reference(run_reference, checkpoint)
    selected_dataset, dataset_root = resolve_dataset(run.config, dataset_id)
    pairs = paired_wavs(dataset_root, split)
    output = result_dir.expanduser().resolve() if result_dir is not None else result_directory(run, variant)
    inference_manifest_path = output / "inference.json"
    if not inference_manifest_path.is_file():
        raise ValueError(f"Inference results are missing inference.json: {output}")
    inference_manifest = read_json(inference_manifest_path)
    if inference_manifest.get("status", "complete") != "complete":
        raise ValueError("Inference results are incomplete; resume inference before evaluating")
    if inference_manifest.get("source_type", run.source_type) != run.source_type:
        raise ValueError("Inference manifest conflicts on source_type")
    expected_manifest = {
        "dataset_id": selected_dataset,
        "split": split,
        "run_name": run.name,
        "model_sha256": run.model_sha256,
    }
    # IMPORTANT: Provenance gate — the result is only scored when the manifest
    # it was produced under agrees with the run, dataset, split, and exact
    # model hash being attributed; any mismatch is refused rather than scored.
    mismatches = [key for key, value in expected_manifest.items() if inference_manifest.get(key) != value]
    if mismatches:
        raise ValueError(f"Inference manifest conflicts on {', '.join(mismatches)}")
    pairs_by_name = {noisy.name: (clean, noisy) for clean, noisy in pairs}
    manifest_files = inference_manifest.get("files")
    if manifest_files is None:
        expected_files = list(pairs_by_name)
    elif not isinstance(manifest_files, list) or not all(isinstance(name, str) for name in manifest_files):
        raise ValueError(f"Inference manifest has an invalid files list: {inference_manifest_path}")
    elif len(manifest_files) != len(set(manifest_files)):
        raise ValueError(f"Inference manifest contains duplicate filenames: {inference_manifest_path}")
    else:
        expected_files = manifest_files
    unknown = sorted(set(expected_files) - pairs_by_name.keys())
    if unknown:
        raise ValueError(f"Inference manifest contains files outside the dataset split: {unknown[:5]}")
    pairs = [pairs_by_name[name] for name in expected_files]
    available = {path.name for path in output.glob("*.wav")}
    if not expected_files or set(expected_files) != available:
        raise ValueError(f"Inference results are incomplete or contain unexpected WAV files: {output}")

    sample_rate = int(inference_manifest.get("sample_rate", run.config.get("data.sample_rate", 16_000)))
    return _calculate_metrics(
        pairs,
        output,
        {
            "source_type": run.source_type,
            "run_name": run.name,
            "run_id": run.config.get("run.run_id"),
            "model_sha256": run.model_sha256,
            "dataset_id": selected_dataset,
            "split": split,
            "variant": variant,
        },
        sample_rate=sample_rate,
        metrics=metrics,
        max_workers=max_workers,
        overwrite=overwrite,
    )


def evaluate_directory(
    directory: Path,
    *,
    metrics: tuple[str, ...] = DEFAULT_METRICS,
    max_workers: int = 0,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    """Evaluate an inference result directory using its provenance manifest.

    Args:
        directory: Result directory containing WAVs and ``inference.json``.
        metrics: Metric names to compute.
        max_workers: Thread count for per-file scoring; zero uses the
            executor default.
        overwrite: Recompute even when complete metric artifacts exist.

    Returns:
        The written ``metrics.csv`` and ``metrics.json`` paths.

    Raises:
        ValueError: If the manifest is missing or lacks run, dataset, or
            split provenance.
    """
    output = directory.expanduser().resolve()
    manifest_path = output / "inference.json"
    if not manifest_path.is_file():
        raise ValueError(f"Result directory is missing inference.json: {output}")
    manifest = read_json(manifest_path)
    run_reference = manifest.get("run_path") or manifest.get("run_name")
    source_type = manifest.get("source_type")
    if source_type is None:
        # Legacy manifests stored a bare checkpoint in run_path without a source tag.
        source_type = (
            "checkpoint" if isinstance(run_reference, str) and run_reference.endswith(".safetensors") else "run"
        )
    if source_type not in {"run", "checkpoint"}:
        raise ValueError(f"Unsupported inference model source: {source_type!r}")
    checkpoint = manifest.get("model_path") or run_reference if source_type == "checkpoint" else None
    dataset_id = manifest.get("dataset_id")
    split = manifest.get("split")
    if not isinstance(run_reference, str) or not isinstance(dataset_id, str) or not isinstance(split, str):
        raise ValueError(f"Inference manifest lacks run, dataset, or split provenance: {manifest_path}")
    return evaluate_results(
        run_reference if source_type == "run" else None,
        output.name,
        dataset_id=dataset_id,
        split=split,
        metrics=metrics,
        max_workers=max_workers,
        overwrite=overwrite,
        result_dir=output,
        checkpoint=checkpoint,
    )


def evaluate_external(
    clean_dir: Path,
    noisy_dir: Path,
    enhanced_dir: Path,
    *,
    metrics: tuple[str, ...] = DEFAULT_METRICS,
    sample_rate: int = 16_000,
    max_workers: int = 0,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    """Evaluate third-party enhanced WAVs against explicit clean and noisy directories.

    Args:
        clean_dir: Directory of reference WAVs.
        noisy_dir: Directory of degraded WAVs sharing the clean filename set.
        enhanced_dir: Directory of enhanced WAVs sharing the same filename
            set; also receives the metric artifacts.
        metrics: Metric names to compute.
        sample_rate: Sample rate all files are resampled to.
        max_workers: Thread count for per-file scoring; zero uses the
            executor default.
        overwrite: Recompute even when complete metric artifacts exist.

    Returns:
        The written ``metrics.csv`` and ``metrics.json`` paths.

    Raises:
        ValueError: If the three directories do not share one identical
            non-empty WAV filename set.
    """
    clean_dir = clean_dir.expanduser().resolve()
    noisy_dir = noisy_dir.expanduser().resolve()
    enhanced_dir = enhanced_dir.expanduser().resolve()
    clean = {path.name: path for path in clean_dir.glob("*.wav")}
    noisy = {path.name: path for path in noisy_dir.glob("*.wav")}
    enhanced = {path.name: path for path in enhanced_dir.glob("*.wav")}
    if not clean or clean.keys() != noisy.keys() or clean.keys() != enhanced.keys():
        raise ValueError("Clean, noisy, and enhanced directories must contain the same non-empty WAV filename set")
    pairs = [(clean[name], noisy[name]) for name in sorted(clean)]
    return _calculate_metrics(
        pairs,
        enhanced_dir,
        {
            "source_type": "external",
            "clean_dir": str(clean_dir),
            "noisy_dir": str(noisy_dir),
            "enhanced_dir": str(enhanced_dir),
        },
        sample_rate=sample_rate,
        metrics=metrics,
        max_workers=max_workers,
        overwrite=overwrite,
    )
