"""Assemble the root ``cof`` Typer application.

Register every command module on a single themed Typer instance so the ``cof``
executable exposes configuration, run inspection, dataset, training,
inference, single-file enhancement, metric, and version commands through one
consistent help page.

Author: Qing Yao
Date: 2026/9/25
"""

from __future__ import annotations

from importlib.metadata import version as package_version

import typer

from cof.cli import config, dataset, enhance, inference, metric, run, train
from cof.cli.tui import ThemedTyperGroup

app = typer.Typer(
    name="cof",
    help="Corrective Forcing Schrodinger Bridge command line interface.",
    cls=ThemedTyperGroup,
)
app.command("config")(config.configure)
app.add_typer(run.app, name="run")
app.command("inference")(inference.inference)
app.command("enhance")(enhance.enhance)
app.command("train")(train.train)
app.add_typer(dataset.app, name="dataset")
app.command("metric")(metric.calculate)


@app.command()
def version() -> None:
    """Print the installed CoF package version."""
    typer.echo(f"cof {package_version('cof')}")


@app.callback()
def main() -> None:
    """Run CoF training, inference, evaluation, and demos."""


if __name__ == "__main__":
    app()
