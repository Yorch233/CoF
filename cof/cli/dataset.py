"""Manage the paired-dataset registry and synthesize paired datasets.

Wrap the registry operations (list, add, edit, delete) and the paired-corpus
synthesis command. Registrations stay host-local and reference the source
data in place; created datasets keep their ``create_configuraton.json``
provenance record alongside the exported splits.

Author: Qing Yao
Date: 2026/9/23
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.table import Table

from cof.cli.tui import get_console
from cof.cli.tui.theme import ACCENT, TEXT
from cof.data.create_dataset import CleanDataset, NoiseDataset, create_dataset
from cof.data.dataset_registry import (
    add_dataset_entry,
    edit_dataset_entry,
    load_dataset_registry,
    remove_dataset_entry,
    save_dataset_registry,
)
from cof.utils.paths import USER_CONFIG_PATH

app = typer.Typer(help="Create and inspect paired speech datasets.")


def _registry_error(error: KeyError | TypeError | ValueError) -> typer.BadParameter:
    """Convert a registry validation error into a Typer parameter error."""
    # NOTE: KeyError carries its message (already quoted) in args[0]; reuse it verbatim
    # instead of the bare "KEY" repr that str(KeyError) would produce.
    message = error.args[0] if isinstance(error, KeyError) else str(error)
    return typer.BadParameter(str(message))


@app.command("list")
def list_registered() -> None:
    """List datasets registered in the project configuration."""
    try:
        state = load_dataset_registry(USER_CONFIG_PATH)
    except (TypeError, ValueError) as error:
        raise _registry_error(error) from error
    if not state.datasets:
        typer.echo("No datasets registered.")
        return
    table = Table(show_header=True, header_style=f"bold {ACCENT}", box=None)
    table.add_column("ID", style=f"bold {TEXT}")
    table.add_column("Path", style=TEXT, overflow="fold")
    table.add_column("Training", justify="center")
    for dataset_id, path in state.datasets.items():
        table.add_row(dataset_id, path, "●" if dataset_id == state.selected_id else "")
    get_console().print(table)


@app.command("add")
def add_registered(
    dataset_id: str = typer.Option(..., "--id", help="Unique dataset ID used by training commands."),
    path: Path = typer.Option(..., "--path", file_okay=False, help="Paired dataset root directory."),
    select: bool = typer.Option(False, "--select", help="Select this ID as the default training dataset."),
) -> None:
    """Register a paired dataset by ID and path.

    Args:
        dataset_id: Unique ID used by training commands.
        path: Paired dataset root directory.
        select: Also make this ID the default training dataset.
    """
    try:
        state = load_dataset_registry(USER_CONFIG_PATH)
        datasets = add_dataset_entry(state.datasets, dataset_id, path)
        selected_id = dataset_id.strip() if select or state.selected_id is None else state.selected_id
        save_dataset_registry(datasets, selected_id, USER_CONFIG_PATH)
    except (KeyError, TypeError, ValueError) as error:
        raise _registry_error(error) from error
    typer.echo(f"Registered dataset {dataset_id.strip()!r}.")


@app.command("edit")
def edit_registered(
    dataset_id: str = typer.Option(..., "--id", help="Existing registered dataset ID."),
    new_id: str | None = typer.Option(None, "--new-id", help="Replacement dataset ID."),
    path: Path | None = typer.Option(None, "--path", file_okay=False, help="Replacement dataset root."),
    select: bool = typer.Option(False, "--select", help="Select the edited ID for training."),
) -> None:
    """Edit a registered dataset ID and/or path.

    Args:
        dataset_id: Existing registered dataset ID.
        new_id: Replacement dataset ID.
        path: Replacement dataset root.
        select: Move the training selection to the edited ID.

    Raises:
        typer.BadParameter: If neither a replacement nor a re-selection was
            requested.
    """
    if new_id is None and path is None and not select:
        raise typer.BadParameter("Provide --new-id, --path, and/or --select")
    try:
        state = load_dataset_registry(USER_CONFIG_PATH)
        datasets = edit_dataset_entry(state.datasets, dataset_id, new_id=new_id, path=path)
        replacement_id = (new_id or dataset_id).strip()
        selected_id = state.selected_id
        if selected_id == dataset_id or select:
            selected_id = replacement_id
        save_dataset_registry(datasets, selected_id, USER_CONFIG_PATH)
    except (KeyError, TypeError, ValueError) as error:
        raise _registry_error(error) from error
    typer.echo(f"Updated dataset {dataset_id!r} as {replacement_id!r}.")


@app.command("delete")
def delete_registered(
    dataset_id: str = typer.Option(..., "--id", help="Registered dataset ID to remove."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Delete without an interactive confirmation."),
) -> None:
    """Remove a dataset registration without deleting its files.

    Args:
        dataset_id: Registered dataset ID to remove.
        yes: Remove the registration without an interactive confirmation.
    """
    try:
        state = load_dataset_registry(USER_CONFIG_PATH)
        if dataset_id not in state.datasets:
            raise KeyError(f"Dataset ID {dataset_id!r} is not registered")
        if not yes and not typer.confirm(f"Remove dataset registration {dataset_id!r}?"):
            raise typer.Abort
        datasets = remove_dataset_entry(state.datasets, dataset_id)
        selected_id = state.selected_id
        if selected_id == dataset_id:
            selected_id = next(iter(datasets), None)
        save_dataset_registry(datasets, selected_id, USER_CONFIG_PATH)
    except (KeyError, TypeError, ValueError) as error:
        raise _registry_error(error) from error
    typer.echo(f"Removed dataset registration {dataset_id!r}; files were not deleted.")


@app.command()
def create(
    task: list[str] = typer.Option(..., "--task", help="Repeat for enhancement and/or dereverberation."),
    clean: tuple[CleanDataset, Path] = typer.Option(..., "--clean", help="Clean dataset TYPE and PATH."),
    noise: tuple[NoiseDataset, Path] = typer.Option(..., "--noise", help="Noise dataset TYPE and PATH."),
    output_dir: Path = typer.Option(..., "--output_dir", "--output-dir", help="Dataset output directory."),
    sample_rate: int = typer.Option(16_000, min=1),
    snr_min: float = typer.Option(-6.0),
    snr_max: float = typer.Option(14.0),
    t60_min: float = typer.Option(0.4, min=0.01),
    t60_max: float = typer.Option(1.0, min=0.01),
    seed: int = typer.Option(100),
    overwrite: bool = typer.Option(False, help="Replace a non-empty output directory."),
) -> None:
    """Create paired clean/noisy splits and export synthesis statistics.

    Args:
        task: Task selection; repeat ``--task`` for enhancement and/or
            dereverberation.
        clean: Clean corpus type and path.
        noise: Noise corpus type and path.
        output_dir: Output directory receiving the split WAVs and the
            synthesis provenance record.
        sample_rate: Synthesis parameter; target sample rate in Hz.
        snr_min: Synthesis parameter; minimum SNR in dB.
        snr_max: Synthesis parameter; maximum SNR in dB.
        t60_min: Synthesis parameter; minimum reverberation time in seconds.
        t60_max: Synthesis parameter; maximum reverberation time in seconds.
        seed: Synthesis parameter; RNG seed for the degradation process.
        overwrite: Output control; replace a non-empty output directory.

    Raises:
        typer.BadParameter: If the output directory exists without
            ``--overwrite`` or the synthesis parameters are invalid.
    """
    try:
        configuration = create_dataset(
            tasks=task,
            clean_dataset=clean[0],
            clean_inputs=[clean[1]],
            noise_dataset=noise[0],
            noise_inputs=[noise[1]],
            output_dir=output_dir,
            sample_rate=sample_rate,
            snr_range_db=(snr_min, snr_max),
            t60_range_s=(t60_min, t60_max),
            seed=seed,
            overwrite=overwrite,
        )
    except (FileExistsError, ValueError) as error:
        raise typer.BadParameter(str(error)) from error

    summary = configuration["summary"]
    typer.echo(f"Created {summary['num_files']} pairs in {output_dir.expanduser().resolve()}")
    typer.echo(f"Configuration: {output_dir.expanduser().resolve() / 'create_configuraton.json'}")


@app.command()
def inspect(directory: Path = typer.Argument(..., exists=True, file_okay=False)) -> None:
    """Report paired WAV counts in an exported dataset."""
    for split in ("train", "valid", "test"):
        clean_count = len(list((directory / split / "clean").glob("*.wav")))
        noisy_count = len(list((directory / split / "noisy").glob("*.wav")))
        typer.echo(f"{split}: clean={clean_count}, noisy={noisy_count}")
