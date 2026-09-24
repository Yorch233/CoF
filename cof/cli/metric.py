"""Evaluate metrics for manifested inference results or third-party WAV sets.

Resolve exactly one input mode per invocation (``--dir``, ``--run`` with an
optional ``--result``, or the ``--clean``/``--noisy``/``--enhanced`` triplet),
parse and validate the requested metric names, then delegate to the workflow
evaluation layer, which refuses to score a result whose manifest disagrees
with the run it is attributed to.

Author: Qing Yao
Date: 2026/9/23
"""

from __future__ import annotations

import sys
from pathlib import Path

import typer

from cof.cli.tui import rich_select
from cof.evaluation import supported_metrics
from cof.pipeline.evaluation import DEFAULT_METRICS, evaluate_directory, evaluate_external
from cof.pipeline.provenance import resolve_results_root, resolve_run


def _parse_metrics(values: list[str] | None) -> tuple[str, ...]:
    """Normalize repeated or comma-separated metric names into a validated tuple.

    Args:
        values: Raw ``--metrics`` entries; each may hold comma-separated
            names, matched case-insensitively with ``-`` normalized to ``_``.

    Returns:
        ``DEFAULT_METRICS`` when nothing was requested, otherwise the
        de-duplicated supported metric names in request order.

    Raises:
        typer.BadParameter: If any requested name is missing from
            ``supported_metrics()``.
    """
    if not values:
        return DEFAULT_METRICS
    metrics = tuple(
        metric.strip().lower().replace("-", "_") for value in values for metric in value.split(",") if metric.strip()
    )
    invalid = [metric for metric in metrics if metric not in supported_metrics()]
    if invalid:
        raise typer.BadParameter(f"Unsupported metrics: {invalid}; choose from {supported_metrics()}")
    return tuple(dict.fromkeys(metrics))


def _run_result_directory(run_reference: str, result: str | None) -> Path:
    """Resolve the manifested result directory to evaluate for a run.

    Args:
        run_reference: Run name or path resolved through the artifact layer.
        result: Result directory name under the run's results root, such as
            ``SB_SDE_Solver_N=4``; omitted to auto-select.

    Returns:
        The selected result directory containing ``inference.json``.

    Raises:
        ValueError: If the run has no manifested results, the requested
            result is unavailable, or several exist while stdin is not a TTY.
    """
    run = resolve_run(run_reference)
    root = resolve_results_root(run.config) / run.name
    candidates = sorted(path for path in root.iterdir() if path.is_dir() and (path / "inference.json").is_file())
    if not candidates:
        raise ValueError(f"Run has no manifested inference results: {root}")
    if result is not None:
        selected = root / result
        if selected not in candidates:
            available = ", ".join(path.name for path in candidates)
            raise ValueError(f"Result {result!r} is not available; choose from: {available}")
        return selected
    if len(candidates) == 1:
        return candidates[0]
    if not sys.stdin.isatty():
        # NOTE: Never prompt when non-interactive; require the caller to name the result.
        available = ", ".join(path.name for path in candidates)
        raise ValueError(f"Multiple results are available; pass --result with one of: {available}")
    selected = rich_select("Which inference result should be evaluated", [path.name for path in candidates])
    return candidates[selected]


def calculate(
    directory: Path | None = typer.Option(
        None,
        "--dir",
        exists=True,
        file_okay=False,
        help="Manifested inference result directory.",
    ),
    run: str | None = typer.Option(None, help="Run name or path whose result should be selected."),
    result: str | None = typer.Option(None, help="Result directory name, such as SB_SDE_Solver_N=4."),
    clean: Path | None = typer.Option(None, exists=True, file_okay=False, help="Third-party clean WAV directory."),
    noisy: Path | None = typer.Option(None, exists=True, file_okay=False, help="Third-party noisy WAV directory."),
    enhanced: Path | None = typer.Option(
        None,
        exists=True,
        file_okay=False,
        help="Third-party enhanced WAV directory and metric output location.",
    ),
    metrics: list[str] | None = typer.Option(
        None,
        "--metrics",
        "--metric",
        help="Metric name or comma-separated names; repeatable. Defaults to PESQ, ESTOI, and SI-SDR.",
    ),
    sample_rate: int = typer.Option(16_000, "--sample-rate", min=1, help="Third-party audio sample rate."),
    max_workers: int = typer.Option(0, "--max-workers", min=0),
    overwrite: bool = typer.Option(False, help="Recalculate existing metric artifacts."),
) -> None:
    """Calculate metrics for a result directory, a run result, or third-party WAV directories.

    Exactly one input mode is accepted per invocation: ``--dir``, ``--run``
    (optionally with ``--result``), or all three of ``--clean``/``--noisy``/
    ``--enhanced`` together with identical WAV filename sets.

    Args:
        directory: Input mode ``--dir``; manifested inference result
            directory.
        run: Input mode ``--run``; run name or path whose result is selected.
        result: Result directory name under the run, such as ``SB_SDE_Solver_N=4``;
            requires ``--run``.
        clean: Third-party mode; clean WAV directory.
        noisy: Third-party mode; noisy WAV directory.
        enhanced: Third-party mode; enhanced WAV directory, which also
            receives the metric artifacts.
        metrics: Metric names (repeatable or comma-separated); defaults to
            the intrusive core metrics PESQ, ESTOI, and SI-SDR.
        sample_rate: Execution; sample rate used for third-party audio.
        max_workers: Execution; parallel worker count.
        overwrite: Execution; recalculate existing metric artifacts.

    Raises:
        typer.BadParameter: If the input modes are mixed or incomplete, the
            metric names are unsupported, or evaluation fails.
    """
    selected = _parse_metrics(metrics)
    try:
        external_paths = (clean, noisy, enhanced)
        external_mode = any(path is not None for path in external_paths)
        modes = int(directory is not None) + int(run is not None) + int(external_mode)
        if modes != 1:
            raise ValueError("Choose exactly one input mode: --dir, --run, or --clean/--noisy/--enhanced")
        if result is not None and run is None:
            raise ValueError("--result requires --run")
        if external_mode and not all(path is not None for path in external_paths):
            raise ValueError("Third-party evaluation requires --clean, --noisy, and --enhanced together")
        if run is not None:
            directory = _run_result_directory(run, result)
        if directory is not None:
            csv_path, json_path = evaluate_directory(
                directory,
                metrics=selected,
                max_workers=max_workers,
                overwrite=overwrite,
            )
        else:
            assert clean is not None
            assert noisy is not None
            assert enhanced is not None
            csv_path, json_path = evaluate_external(
                clean,
                noisy,
                enhanced,
                metrics=selected,
                sample_rate=sample_rate,
                max_workers=max_workers,
                overwrite=overwrite,
            )
    except (OSError, RuntimeError, ValueError) as error:
        raise typer.BadParameter(str(error)) from error
    typer.echo(f"Per-file metrics: {csv_path}")
    typer.echo(f"Metric summary: {json_path}")
