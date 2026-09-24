"""Inspect training runs and their inference results from the command line.

Read-only views over the run and result roots — list the recorded runs,
summarize one run's persisted configuration and exported weights, and list a
run's manifested inference variants together with their metric summaries —
plus the sanctioned export of a run's resolved model with a provenance
record.

Author: Qing Yao
Date: 2026/9/25
"""

from __future__ import annotations

import json
import re
import shutil
from datetime import UTC, datetime
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any

import typer
from rich.table import Table

from cof.cli.tui import get_console
from cof.cli.tui.theme import ACCENT, TEXT
from cof.config.manager import BaseConfiguer, Config, read_config_from_yaml, read_yml
from cof.pipeline.provenance import resolve_results_root, resolve_run, resolve_runs_root

app = typer.Typer(help="Inspect training runs and their results.")

CORE_METRICS = ("PESQ", "ESTOI", "SI_SDR")


def _value(config: Config, key: str) -> str:
    """Return a display string for a dotted configuration key."""
    raw = config.get(key)
    if isinstance(raw, float):
        return f"{raw:.4f}"
    return "—" if raw is None else str(raw)


def _load_run(run: str) -> tuple[Path, Config]:
    """Resolve a run reference to its directory and persisted configuration.

    Args:
        run: Run name below the configured run root, or a run directory path.

    Returns:
        The resolved run directory and its configuration.

    Raises:
        typer.BadParameter: If the reference resolves to no directory that
            carries a ``config.yml``.
    """
    candidate = Path(run).expanduser()
    if not candidate.is_dir():
        candidate = resolve_runs_root() / run
    path = candidate.resolve()
    config_path = path / "config.yml"
    if not path.is_dir() or not config_path.is_file():
        raise typer.BadParameter(f"Not a run directory with a config.yml: {run}")
    return path, read_config_from_yaml(config_path)


def _read_json(path: Path) -> dict[str, Any] | None:
    """Read a JSON manifest, returning ``None`` when absent or unreadable."""
    if not path.is_file():
        return None
    try:
        loaded = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return loaded if isinstance(loaded, dict) else None


def _run_name(path: Path, config: Config) -> str:
    """Return the persisted run name, falling back to the directory name."""
    return str(config.get("run.run_name") or path.name)


def _created(path: Path) -> str:
    """Return the short creation timestamp recorded on the run config."""
    stamp = datetime.fromtimestamp(path.stat().st_mtime)
    return stamp.strftime("%Y-%m-%d %H:%M")


def _summarize_metrics(summary: Any) -> dict[str, str]:
    """Format a metrics ``summary`` mapping as ``mean ± std`` strings."""
    formatted: dict[str, str] = {}
    if not isinstance(summary, dict):
        return formatted
    for metric, values in summary.items():
        if not isinstance(values, dict) or not isinstance(values.get("mean"), (int, float)):
            continue
        std = values.get("std")
        spread = f" ± {std:.3f}" if isinstance(std, (int, float)) else ""
        formatted[str(metric)] = f"{values['mean']:.3f}{spread}"
    return formatted


@app.command("list")
def list_runs() -> None:
    """List the training runs recorded below the configured run root."""
    root = resolve_runs_root()
    config_paths = sorted(root.glob("*/config.yml")) if root.is_dir() else []
    if not config_paths:
        typer.echo(f"No runs found below {root}.")
        return
    table = Table(show_header=True, header_style=f"bold {ACCENT}", box=None)
    table.add_column("Run", style=f"bold {TEXT}")
    table.add_column("Stage")
    table.add_column("Formulation")
    table.add_column("Backbone")
    table.add_column("Dataset")
    table.add_column("Created")
    for config_path in config_paths:
        config = read_config_from_yaml(config_path)
        table.add_row(
            config_path.parent.name,
            _value(config, "stage"),
            _value(config, "formulation.name"),
            _value(config, "model.backbone"),
            _value(config, "registry.dataset"),
            _created(config_path),
        )
    get_console().print(table)


@app.command()
def show(run: str = typer.Argument(..., help="Run name below the run root, or a run directory path.")) -> None:
    """Summarize one run's configuration, provenance, and exported weights."""
    path, config = _load_run(run)
    sampling = " · ".join(
        part
        for part in (
            _value(config, "formulation.sampling.solver"),
            f"N={_value(config, 'formulation.sampling.num_steps')}",
            _value(config, "formulation.sampling.skip_type"),
        )
        if part != "—"
    )
    rows: list[tuple[str, str]] = [
        ("Run", _run_name(path, config)),
        ("Path", str(path)),
        ("Stage", _value(config, "stage")),
        ("Formulation", _value(config, "formulation.name")),
        ("Backbone", _value(config, "model.backbone")),
        ("Dataset", _value(config, "registry.dataset")),
        ("Sampling", sampling or "—"),
        ("Best PESQ", _value(config, "metrics.best_pesq")),
        ("Best valid loss", _value(config, "metrics.best_valid_loss")),
    ]
    pretrain_run = config.get("run.pretrain_run")
    if pretrain_run is not None:
        init_model = config.get("run.init_model_path")
        init_sha = config.get("run.init_model_sha256")
        rows.extend(
            [
                ("Pretrain run", str(pretrain_run)),
                ("Init model", Path(str(init_model)).name if init_model else "—"),
                ("Init SHA-256", str(init_sha)[:12] if init_sha else "—"),
            ]
        )
    exports = sorted(item.name for item in path.glob("*.safetensors"))
    rows.append(("Exports", ", ".join(exports) if exports else "—"))
    rows.append(("Default test model", _value(config, "weights.default_test_model")))
    rows.append(("Last model weights", _value(config, "weights.last_model_weights")))
    rows.append(("Resumable", "checkpoints/last.ckpt" if (path / "checkpoints" / "last.ckpt").is_file() else "—"))
    table = Table(show_header=False, box=None)
    table.add_column("Field", style=f"bold {ACCENT}")
    table.add_column("Value", style=TEXT, overflow="fold")
    for field, value in rows:
        table.add_row(field, value)
    get_console().print(table)


@app.command()
def results(run: str = typer.Argument(..., help="Run name below the run root, or a run directory path.")) -> None:
    """List a run's manifested inference variants and their metric summaries."""
    path, config = _load_run(run)
    name = _run_name(path, config)
    results_root = resolve_results_root(config)
    run_results = results_root / name
    if not run_results.is_dir():
        typer.echo(f"No results found for run {name!r} below {results_root}.")
        return
    variants = sorted(item for item in run_results.iterdir() if item.is_dir() and (item / "inference.json").is_file())
    if not variants:
        typer.echo(f"No manifested inference results found for run {name!r}.")
        return
    manifests = {item.name: _read_json(item / "inference.json") or {} for item in variants}
    evaluated = {
        item.name: _summarize_metrics((_read_json(item / "metrics.json") or {}).get("summary")) for item in variants
    }
    ordered: list[str] = [metric for metric in CORE_METRICS if any(metric in row for row in evaluated.values())]
    ordered.extend(
        metric for metric in dict.fromkeys(key for row in evaluated.values() for key in row) if metric not in ordered
    )
    table = Table(show_header=True, header_style=f"bold {ACCENT}", box=None)
    table.add_column("Variant", style=f"bold {TEXT}")
    table.add_column("Split", justify="center")
    table.add_column("Sampler", justify="center")
    table.add_column("NFE", justify="right")
    table.add_column("Skip type")
    table.add_column("Seed", justify="right")
    table.add_column("Files", justify="right")
    for metric in ordered:
        table.add_column(metric, justify="right")
    for variant in variants:
        manifest = manifests[variant.name]
        row = evaluated[variant.name]
        values = [
            variant.name,
            str(manifest.get("split", "—")),
            str(manifest.get("sampler", "—")),
            str(manifest.get("num_steps", "—")),
            str(manifest.get("skip_type", "—")),
            str(manifest.get("seed", "—")),
            str(manifest.get("num_files", "—")),
        ]
        values.extend(row.get(metric, "—") for metric in ordered)
        table.add_row(*values)
    get_console().print(table)
    unevaluated = [variant.name for variant in variants if not evaluated[variant.name]]
    if unevaluated:
        typer.echo(f"Not evaluated: {', '.join(unevaluated)}")


def _export_model_name(artifact: Any) -> str:
    """Derive the exported weight name: ``<dataset>_<formulation>[_cof].safetensors``.

    The name encodes the run's registered dataset, formulation, and — for the
    CoF stage — a ``_cof`` suffix, so weights from different deployments stay
    distinguishable once they leave their run directories.
    """
    dataset = str(artifact.config.get("registry.dataset") or "dataset")
    formulation = str(artifact.config.get("formulation.name") or "model")
    stem = f"{dataset}_{formulation}".lower()
    if artifact.config.get("stage") == "post_training":
        stem += "_cof"
    return re.sub(r"[^a-z0-9_.-]+", "-", stem) + ".safetensors"


@app.command()
def export(
    run: str = typer.Argument(..., help="Run name below the run root, or a run directory path."),
    output_dir: Path = typer.Option(
        ...,
        "--output-dir",
        "--output",
        file_okay=False,
        help="Directory receiving the model export and its provenance record.",
    ),
    overwrite: bool = typer.Option(False, help="Replace an existing export in the output directory."),
) -> None:
    """Export the run's resolved default model, config, and provenance record.

    The model file is the one inference would resolve (recorded default test
    model first), renamed to ``<dataset>_<formulation>[_cof].safetensors`` —
    for example ``voicebank_sbve_cof.safetensors``. The run's ``config.yml``
    is copied with ``weights.default_test_model`` rewritten accordingly, so
    the export itself resolves as a run reference. Copying a self-describing
    checkpoint preserves its embedded metadata, but only this command also
    writes the export provenance record.
    """
    try:
        artifact = resolve_run(run)
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error
    output = output_dir.expanduser().resolve()
    model_destination = output / _export_model_name(artifact)
    config_destination = output / "config.yml"
    manifest_path = output / "export.json"
    if not overwrite and any(path.exists() for path in (model_destination, config_destination, manifest_path)):
        raise typer.BadParameter(f"An export already exists below {output}; pass --overwrite to replace it")
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(artifact.model_path, model_destination)
    raw_config = read_yml(artifact.path / "config.yml")
    weights = raw_config.setdefault("weights", {})
    weights["default_test_model"] = model_destination.name
    BaseConfiguer.dump(raw_config, config_destination)
    record = {
        "artifact_type": "model_export",
        "run_name": artifact.name,
        "run_path": str(artifact.path),
        "run_id": artifact.config.get("run.run_id"),
        "model_file": artifact.model_path.name,
        "exported_file": model_destination.name,
        "model_sha256": artifact.model_sha256,
        "exported_at": datetime.now(UTC).isoformat(),
        "cof_version": package_version("cof"),
    }
    manifest_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    typer.echo(f"Exported {model_destination.name} (sha256 {artifact.model_sha256[:12]}) to {output}")
