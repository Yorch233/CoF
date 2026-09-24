"""Resolve runs, datasets, result directories, and provenance artifacts.

Every artifact consumed by the inference and metric workflows is resolved
here rather than guessed by callers: a run resolves to its directory plus the
exact model file it exports (hashed for provenance), a dataset resolves
through the host registry with a run-time snapshot fallback, and a result
directory is manifested so a re-run with a conflicting signature fails
instead of mixing outputs.

Author: Qing Yao
Date: 2026/9/23
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from cof.config.manager import Config, read_config_from_yaml, read_yml
from cof.data.pairs import paired_wavs as paired_wavs
from cof.utils.atomic import atomic_output_path
from cof.utils.paths import PROJECT_ROOT, RESULTS_DIR, RUNS_DIR, USER_CONFIG_PATH


@dataclass(frozen=True)
class RunArtifact:
    """Resolved trained-model run and its persisted configuration."""

    path: Path
    config: Config
    model_path: Path
    model_sha256: str
    source_type: Literal["run", "checkpoint"] = "run"

    @property
    def name(self) -> str:
        """Return the persisted run name, falling back to the directory name."""
        return str(self.config.get("run.run_name") or self.path.name)


def _project_path(value: str | Path) -> Path:
    """Resolve a configured path, making relative paths project-root anchored."""
    path = Path(value).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def _local_settings() -> dict[str, Any]:
    """Return the host-local settings, or an empty mapping when unconfigured."""
    if not USER_CONFIG_PATH.is_file():
        return {}
    return dict(read_yml(USER_CONFIG_PATH))


def file_sha256(path: Path) -> str:
    """Calculate the SHA-256 digest of a file without loading it all into memory.

    Args:
        path: File to hash.

    Returns:
        The lowercase hexadecimal digest.
    """
    digest = hashlib.sha256(usedforsecurity=False)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metric_model_path(path: Path, pattern: str, *, mode: Literal["min", "max"]) -> Path | None:
    """Resolve a metric-valued model filename when its config field is unavailable.

    Args:
        path: Run directory scanned with the glob pattern.
        pattern: Filename pattern embedding the score, e.g.
            ``model_valid_pesq=*.safetensors``.
        mode: Whether the best score is the minimum or maximum.

    Returns:
        The path of the best-scoring model file, or ``None`` when no
        parseable candidates exist.
    """
    scored: list[tuple[float, Path]] = []
    for candidate in path.glob(pattern):
        try:
            score = float(candidate.stem.rsplit("=", maxsplit=1)[1])
        except (IndexError, ValueError):
            continue
        scored.append((score, candidate))
    if not scored:
        return None
    select = min if mode == "min" else max
    return select(scored, key=lambda item: item[0])[1]


def resolve_model_path(path: Path, config: Config) -> Path:
    """Resolve the default model artifact, preferring validation PESQ for generative runs.

    The candidate order mirrors the deployment contract: the recorded default
    test model, the best validation PESQ export, ``model_last.safetensors``,
    ``model.safetensors``, then the best validation loss export.

    Args:
        path: Run directory to search for exported weights.
        config: Persisted run configuration naming recorded exports.

    Returns:
        The first existing candidate model file.

    Raises:
        ValueError: If no usable safetensors model exists in the run
            directory.
    """
    candidates = []
    configured = config.get("weights.default_test_model")
    if configured is not None:
        candidates.append(path / str(configured))
    pesq_model = _metric_model_path(path, "model_valid_pesq=*.safetensors", mode="max")
    if pesq_model is not None:
        candidates.append(pesq_model)
    candidates.extend(
        [
            path / "model_last.safetensors",
            path / "model.safetensors",
        ]
    )
    configured_loss = config.get("metrics.best_valid_loss_model")
    if configured_loss is not None:
        candidates.append(path / str(configured_loss))
    loss_model = _metric_model_path(path, "model_valid_loss=*.safetensors", mode="min")
    if loss_model is not None:
        candidates.append(loss_model)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise ValueError(f"Run is missing a usable safetensors model: {path}")


def resolve_runs_root() -> Path:
    """Resolve the configured project runs directory.

    Returns:
        The resolved run root directory.
    """
    settings = _local_settings()
    return _project_path(settings.get("run_dir", RUNS_DIR))


def _config_from_checkpoint(model_path: Path) -> Config:
    """Build a minimal run configuration from a self-describing checkpoint.

    Args:
        model_path: Safetensors model export carrying constructor metadata.

    Returns:
        Configuration with the model, formulation (including the recorded
        sampling protocol), and a run identity derived from the file stem.

    Raises:
        ValueError: If the file carries no construction metadata.
    """
    from cof.model import GenerativeModel4SE

    metadata = GenerativeModel4SE.checkpoint_metadata(model_path)
    if metadata is None:
        raise ValueError(
            f"Checkpoint carries no construction metadata and no run configuration was provided: {model_path}"
        )
    return Config(
        {
            "model": {"backbone": metadata.get("backbone"), "backbone_kwargs": metadata.get("backbone_kwargs") or {}},
            "formulation": {
                "name": metadata.get("formulation"),
                "kwargs": metadata.get("formulation_kwargs") or {},
                "sampling": {
                    "solver": metadata.get("sampling_solver"),
                    "num_steps": metadata.get("sampling_num_steps"),
                    "skip_type": metadata.get("sampling_skip_type"),
                },
            },
            "run": {"run_name": model_path.stem, "run_path": str(model_path), "run_id": None},
        }
    )


def resolve_run(run: str | Path | None) -> RunArtifact:
    """Resolve a local trained-model run.

    Args:
        run: Run name, run directory, or ``None``; a name is resolved against
            the configured run root.

    Returns:
        The resolved run artifact with its configuration and hashed model
        path.

    Raises:
        ValueError: If the run is missing, lacks ``config.yml``, or exports
            no usable model.
    """
    if run is None:
        raise ValueError("A local run name or directory is required")
    candidate = Path(run).expanduser()
    if not candidate.is_dir():
        candidate = resolve_runs_root() / str(run)
    path = candidate.resolve()
    if not path.is_dir():
        raise ValueError(f"Run directory does not exist: {path}")
    config_path = path / "config.yml"
    if not config_path.is_file():
        raise ValueError(f"Run is missing config.yml: {path}")
    config = read_config_from_yaml(config_path)
    model_path = resolve_model_path(path, config)
    return RunArtifact(path, config, model_path, file_sha256(model_path))


def resolve_model_reference(
    run: str | Path | None = None,
    checkpoint: str | Path | None = None,
) -> RunArtifact:
    """Resolve exactly one of a run reference or a model checkpoint file.

    Args:
        run: Run name or run directory resolved against the run root.
        checkpoint: Self-describing safetensors model file used directly,
            without a run directory.

    Returns:
        The resolved run artifact; a checkpoint yields a minimal
        configuration synthesized from its embedded constructor metadata,
        with the file stem as the run name.

    Raises:
        ValueError: If not exactly one reference is given, the checkpoint
            file is missing, or it carries no construction metadata.
    """
    if (run is None) == (checkpoint is None):
        raise ValueError("Pass exactly one of a run reference or a model checkpoint")
    if checkpoint is not None:
        path = Path(checkpoint).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"Model checkpoint does not exist: {path}")
        config = _config_from_checkpoint(path)
        return RunArtifact(path, config, path, file_sha256(path), source_type="checkpoint")
    return resolve_run(run)


def resolve_dataset(run_config: Config, dataset_id: str | None = None) -> tuple[str, Path]:
    """Resolve a registered dataset, preferring the current project registry over its run snapshot.

    Args:
        run_config: Persisted run configuration carrying the dataset
            snapshot taken at training time.
        dataset_id: Explicit dataset ID overriding the configured selection.

    Returns:
        The selected dataset ID and its resolved root directory.

    Raises:
        ValueError: If the selected ID is registered in neither the current
            registry nor the run snapshot.
    """
    settings = _local_settings()
    current = settings.get("datasets", {})
    snapshot = Config.unwrap(run_config.get("registry.datasets") or {})
    current = current if isinstance(current, dict) else {}
    snapshot = snapshot if isinstance(snapshot, dict) else {}
    selected = dataset_id or settings.get("dataset") or run_config.get("registry.dataset")
    datasets = current if isinstance(selected, str) and selected in current else snapshot
    if not isinstance(selected, str) or selected not in datasets:
        available = ", ".join(dict.fromkeys([*current, *snapshot])) or "none"
        raise ValueError(f"Dataset ID {selected!r} is not registered; available IDs: {available}")
    return selected, _project_path(str(datasets[selected]))


def resolve_results_root(run_config: Config) -> Path:
    """Resolve the configured project results directory.

    Args:
        run_config: Run configuration consulted when the host settings do
            not define ``results_dir``.

    Returns:
        The resolved results root directory.
    """
    settings = _local_settings()
    return _project_path(settings.get("results_dir", run_config.get("paths.results_dir", RESULTS_DIR)))


def result_directory(run: RunArtifact, variant: str) -> Path:
    """Return the canonical result directory for a run and inference variant.

    Args:
        run: Resolved run artifact.
        variant: Sampling-protocol variant directory name, e.g. ``SB_SDE_Solver_N=4``.

    Returns:
        ``<results_root>/<run_name>/<variant>``.
    """
    return resolve_results_root(run.config) / run.name / variant


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON mapping from disk.

    Args:
        path: JSON file to read.

    Returns:
        The parsed mapping.

    Raises:
        ValueError: If the file does not contain a JSON object.
    """
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    """Write a stable, human-readable JSON mapping.

    Args:
        path: Destination file; parent directories are created.
        value: Mapping to serialize with sorted keys and two-space indent.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_output_path(path) as temporary, temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def prepare_artifact_directory(
    output: Path,
    signature: dict[str, Any],
    *,
    overwrite: bool,
    expected_files: list[str],
    manifest_name: str = "inference.json",
) -> set[str]:
    """Validate a resumable artifact directory and return already generated WAV names.

    The manifest is the provenance guard: an existing manifest whose
    signature disagrees with the requested protocol raises instead of mixing
    outputs, and WAV files without a manifest are rejected outright.

    Args:
        output: Result directory to validate or create.
        signature: Manifest fields that identify the requested artifact.
        overwrite: When true, clear existing WAVs, manifests, and metric
            files before validating.
        expected_files: Complete ordered output filename list, recorded before any WAV is written.
        manifest_name: Manifest filename identifying the artifact type.

    Returns:
        Filenames already present in the directory, so a resumed run skips
        them.

    Raises:
        ValueError: If the recorded manifest conflicts on any signature field,
            or WAV files exist without a manifest and ``overwrite`` is false.
    """
    if not expected_files or len(expected_files) != len(set(expected_files)):
        raise ValueError("Expected files must be non-empty and unique")
    manifest_path = output / manifest_name
    if output.exists() and overwrite:
        for path in output.glob("*.wav"):
            path.unlink()
        for path in (manifest_path, output / "metrics.csv", output / "metrics.json"):
            path.unlink(missing_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    existing = {path.name for path in output.glob("*.wav")}
    if manifest_path.is_file():
        current = read_json(manifest_path)
        mismatches = [key for key, value in signature.items() if current.get(key) != value]
        if current.get("files") != expected_files:
            mismatches.append("files")
        if mismatches:
            names = ", ".join(mismatches)
            raise ValueError(f"Existing {manifest_name} manifest conflicts on {names}; use --overwrite")
    elif existing:
        raise ValueError(f"Result directory contains WAV files without {manifest_name}: {output}")
    unexpected = existing - set(expected_files)
    if unexpected:
        raise ValueError(f"Result directory contains unexpected WAV files: {sorted(unexpected)}")
    if existing != set(expected_files) or not manifest_path.is_file():
        # Repaired outputs invalidate any scores computed from the earlier files.
        for path in (output / "metrics.csv", output / "metrics.json"):
            path.unlink(missing_ok=True)
        write_json(
            manifest_path,
            {**signature, "status": "in_progress", "files": expected_files, "num_files": len(expected_files)},
        )
    return existing


def finish_manifest(
    output: Path,
    signature: dict[str, Any],
    expected_files: list[str],
    *,
    manifest_name: str = "inference.json",
) -> Path:
    """Write a completed artifact manifest and verify every expected result exists.

    Args:
        output: Result directory receiving the manifest.
        signature: Provenance fields identifying the artifact.
        expected_files: WAV filenames the protocol must have produced.
        manifest_name: Manifest filename identifying the artifact type.

    Returns:
        The written manifest path.

    Raises:
        RuntimeError: If any expected file is missing from the directory.
    """
    available = {path.name for path in output.glob("*.wav") if path.is_file()}
    if available != set(expected_files):
        raise RuntimeError("Inference output files do not match the expected filename set")
    manifest = {
        **signature,
        "status": "complete",
        "generated_at": datetime.now(UTC).isoformat(),
        "num_files": len(expected_files),
        "files": expected_files,
    }
    path = output / manifest_name
    write_json(path, manifest)
    return path
