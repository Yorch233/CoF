"""Register paired training datasets in the project-level registry.

The registry stores only dataset IDs and root paths in the host-local
``.config/cof.yml``: it never copies, modifies, or deletes source data, and
every mutation returns the updated registry so the CLI stays the single
writer of the file.

Author: Qing Yao
Date: 2026/9/23
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from cof.config.manager import BaseConfiguer
from cof.utils.paths import DEFAULT_CONFIG_PATH, PROJECT_ROOT, USER_CONFIG_PATH

DATASET_SPLITS = ("train", "valid", "test")
DATASET_SIGNALS = ("clean", "noisy")
_DATASET_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True)
class DatasetRegistryState:
    """Registered dataset paths and the ID selected for training."""

    datasets: dict[str, str]
    selected_id: str | None


def resolve_dataset_path(path: Path) -> Path:
    """Resolve a dataset path relative to the project root.

    Args:
        path: Absolute or project-relative path.

    Returns:
        The resolved absolute path.
    """
    expanded = path.expanduser()
    return (expanded if expanded.is_absolute() else PROJECT_ROOT / expanded).resolve()


def validate_dataset_id(dataset_id: str) -> str:
    """Validate and normalize a user-facing dataset ID.

    Args:
        dataset_id: Raw dataset ID.

    Returns:
        The stripped ID.

    Raises:
        ValueError: If the ID uses characters outside the allowed set.
    """
    normalized = dataset_id.strip()
    if not _DATASET_ID_PATTERN.fullmatch(normalized):
        raise ValueError(
            "Dataset ID must start with a letter or number and contain only letters, numbers, '.', '_', or '-'"
        )
    return normalized


def validate_dataset_root(dataset_root: Path) -> Path:
    """Validate and resolve a paired dataset directory.

    Args:
        dataset_root: Dataset root to validate.

    Returns:
        The resolved root path.

    Raises:
        ValueError: If the root does not exist or lacks any required
            ``<split>/<signal>`` directory.
    """
    resolved = resolve_dataset_path(dataset_root)
    if not resolved.is_dir():
        raise ValueError(f"Dataset directory does not exist: {resolved}")
    missing = [resolved / split / signal for split in DATASET_SPLITS for signal in DATASET_SIGNALS]
    missing = [path for path in missing if not path.is_dir()]
    if missing:
        relative = ", ".join(str(path.relative_to(resolved)) for path in missing)
        raise ValueError(f"Dataset is missing required directories: {relative}")
    return resolved


def normalize_dataset_registry(value: Any) -> dict[str, str]:
    """Normalize a YAML dataset mapping without touching its paths.

    Args:
        value: Raw ``datasets`` mapping from YAML.

    Returns:
        The mapping with validated IDs and string paths.

    Raises:
        TypeError: If the value is not a mapping or a path is not a string.
        ValueError: If an ID fails validation.
    """
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError("The datasets configuration must be a mapping of ID to path")
    normalized: dict[str, str] = {}
    for dataset_id, path in value.items():
        normalized_id = validate_dataset_id(str(dataset_id))
        if not isinstance(path, (str, os.PathLike)):
            raise TypeError(f"Dataset path for {normalized_id!r} must be a string")
        normalized[normalized_id] = str(path)
    return normalized


def add_dataset_entry(datasets: dict[str, str], dataset_id: str, path: Path) -> dict[str, str]:
    """Return a registry containing one newly validated dataset.

    Args:
        datasets: Existing ID-to-path registry.
        dataset_id: ID of the dataset to add.
        path: Root directory of the paired dataset.

    Returns:
        The updated registry.

    Raises:
        ValueError: If the ID is already registered or the root is invalid.
    """
    normalized_id = validate_dataset_id(dataset_id)
    if normalized_id in datasets:
        raise ValueError(f"Dataset ID {normalized_id!r} is already registered")
    updated = dict(datasets)
    updated[normalized_id] = str(validate_dataset_root(path))
    return updated


def edit_dataset_entry(
    datasets: dict[str, str],
    dataset_id: str,
    *,
    new_id: str | None = None,
    path: Path | None = None,
) -> dict[str, str]:
    """Return a registry with one dataset ID and/or path updated.

    Args:
        datasets: Existing ID-to-path registry.
        dataset_id: ID of the dataset to edit.
        new_id: Replacement ID, or ``None`` to keep the current one.
        path: Replacement root, or ``None`` to keep the current one.

    Returns:
        The updated registry with the entry's position preserved.

    Raises:
        KeyError: If the dataset ID is not registered.
        ValueError: If the replacement ID is invalid or already registered.
    """
    if dataset_id not in datasets:
        raise KeyError(f"Dataset ID {dataset_id!r} is not registered")
    replacement_id = validate_dataset_id(new_id if new_id is not None else dataset_id)
    if replacement_id != dataset_id and replacement_id in datasets:
        raise ValueError(f"Dataset ID {replacement_id!r} is already registered")
    replacement_path = str(validate_dataset_root(path if path is not None else Path(datasets[dataset_id])))
    return {
        (replacement_id if current_id == dataset_id else current_id): (
            replacement_path if current_id == dataset_id else current_path
        )
        for current_id, current_path in datasets.items()
    }


def remove_dataset_entry(datasets: dict[str, str], dataset_id: str) -> dict[str, str]:
    """Return a registry without the requested dataset ID.

    Args:
        datasets: Existing ID-to-path registry.
        dataset_id: ID of the dataset to remove.

    Returns:
        The updated registry.

    Raises:
        KeyError: If the dataset ID is not registered.
    """
    if dataset_id not in datasets:
        raise KeyError(f"Dataset ID {dataset_id!r} is not registered")
    return {current_id: path for current_id, path in datasets.items() if current_id != dataset_id}


def _read_local_config(config_path: Path) -> dict[str, Any]:
    """Read the host-local configuration, tolerating a missing file.

    Args:
        config_path: Path to the local YAML settings.

    Returns:
        The top-level mapping, or an empty mapping when absent.
    """
    if not config_path.is_file():
        return {}
    with config_path.open(encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    if not isinstance(raw, dict):
        raise TypeError(f"Configuration root in {config_path} must be a mapping")
    return dict(raw)


def load_dataset_registry(config_path: Path = USER_CONFIG_PATH) -> DatasetRegistryState:
    """Load registered datasets from the project-level user configuration.

    Args:
        config_path: Path to the local YAML settings.

    Returns:
        The registry state with the selected ID, dropping a selection that
        is not registered.
    """
    local = _read_local_config(config_path)
    datasets = normalize_dataset_registry(local.get("datasets"))
    selected = local.get("dataset")
    selected_id = str(selected) if selected is not None else None
    if selected_id not in datasets:
        selected_id = None
    return DatasetRegistryState(datasets, selected_id)


def save_dataset_registry(
    datasets: dict[str, str],
    selected_id: str | None,
    config_path: Path = USER_CONFIG_PATH,
) -> DatasetRegistryState:
    """Persist a registry while preserving all other local configuration fields.

    Args:
        datasets: New ID-to-path registry.
        selected_id: Dataset ID to select, or ``None`` to clear the
            selection.
        config_path: Destination settings file.

    Returns:
        The persisted registry state.

    Raises:
        ValueError: If the selected ID is not registered.
    """
    normalized = normalize_dataset_registry(datasets)
    if selected_id is not None and selected_id not in normalized:
        raise ValueError(f"Selected dataset ID {selected_id!r} is not registered")
    destination = config_path.expanduser().resolve()
    local = _read_local_config(destination)
    local.setdefault("inherit", os.path.relpath(DEFAULT_CONFIG_PATH.resolve(), destination.parent))
    local["datasets"] = normalized
    local["dataset"] = selected_id
    destination.parent.mkdir(parents=True, exist_ok=True)
    BaseConfiguer.dump(local, destination)
    return DatasetRegistryState(normalized, selected_id)
