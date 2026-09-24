"""End-to-end tests for the flattened CoF CLI deployment sequence.

One module-scoped fixture drives the real command surface — ``cof dataset``,
``cof train --training-stage pretrain`` (with resume), ``cof inference``,
``cof metric`` — through ``typer.testing.CliRunner`` against a tiny synthetic
dataset and a shrunken NCSN++ backbone on CPU. The CoF post-training stage is
withheld during peer review; the suite asserts that the CLI refuses it with the
release notice.  Host state is
isolated by backing up ``.config/cof.yml`` and redirecting the dataset registry,
run root, and result root into a temporary directory for the duration of the
module; every artifact is then asserted against the acceptance contract from
``AGENTS.md``.

Author: Qing Yao
Date: 2026/9/23
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest
import torch
import torchaudio
from typer.testing import CliRunner

from cof.cli.app import app
from cof.utils.paths import USER_CONFIG_PATH

runner = CliRunner()

SAMPLE_RATE = 16_000
DATASET_ID = "e2e"
PRETRAIN_RUN = "e2e_pretrain"
POSTTRAIN_RUN = "e2e_posttrain"
VARIANT = "SB_SDE_Solver_N=4"
# NOTE: A miniature NCSN++ (width 8, four levels so the bottleneck attention
# runs at 32x32, no per-level attention) keeps one training step in the
# millisecond range on CPU while exercising the real model code.
TINY_BACKBONE = json.dumps(
    {"nf": 8, "ch_mult": [1, 1, 1, 1], "num_res_blocks": 1, "attn_resolutions": [], "image_size": 256}
)
UTTERANCES = ("utt1.wav", "utt2.wav")
SPLITS = ("train", "valid", "test")


def _write_pair(dataset_root: Path, split: str, name: str) -> None:
    """Write one clean/noisy sine-wave pair into a dataset split."""
    time = torch.linspace(0.0, 1.0, SAMPLE_RATE)
    clean = 0.1 * torch.sin(2 * torch.pi * 220.0 * time)
    noisy = clean + 0.02 * torch.randn_like(clean)
    for signal, kind in ((clean, "clean"), (noisy, "noisy")):
        directory = dataset_root / split / kind
        directory.mkdir(parents=True, exist_ok=True)
        torchaudio.save(directory / name, signal.unsqueeze(0), SAMPLE_RATE)


@pytest.fixture(scope="module", params=["SB-VE", "OT-CFM"])
def workspace(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Create the synthetic dataset and redirect host state into a temporary root.

    Args:
        tmp_path_factory: Pytest factory for the module-scoped temporary
            directory.

    Returns:
        Mapping with the temporary root, dataset, run, and result
        directories; the original host configuration is restored on teardown.
    """
    root = tmp_path_factory.mktemp("cof-e2e")
    dataset_root = root / "dataset"
    for split in SPLITS:
        for name in UTTERANCES:
            _write_pair(dataset_root, split, name)

    original = USER_CONFIG_PATH.read_bytes() if USER_CONFIG_PATH.exists() else None
    USER_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    USER_CONFIG_PATH.write_text(
        "\n".join(
            [
                f"dataset: {DATASET_ID}",
                "datasets:",
                f"  {DATASET_ID}: {dataset_root}",
                f"run_dir: {root / 'runs'}",
                f"results_dir: {root / 'results'}",
                "logger: none",
                "num_workers: 0",
                "precision: 32-true",
                "accelerator: cpu",
                "devices: 1",
                "ncsnpp_operator_backend: pytorch_native",
                "valid_metric_samples: 1",
                "save_state_steps: 1",
                "checkpoints_total_limit: 1",
                "",
            ]
        ),
        encoding="utf-8",
    )
    yield {
        "root": root,
        "dataset": dataset_root,
        "runs": root / "runs",
        "results": root / "results",
        "formulation": request.param,
    }
    if original is None:
        USER_CONFIG_PATH.unlink(missing_ok=True)
    else:
        USER_CONFIG_PATH.write_bytes(original)


def _invoke(arguments: list[str]) -> None:
    """Run one CLI invocation and fail with its output on a nonzero exit."""
    result = runner.invoke(app, arguments, env={"COLUMNS": "200"})
    assert result.exit_code == 0, f"cof {' '.join(arguments)} failed:\n{result.output}"


@pytest.fixture(scope="module")
def deployment(workspace: dict[str, Path]) -> dict[str, Any]:
    """Run the full deployment sequence once and hand the artifact paths to the tests.

    Args:
        workspace: Temporary dataset, run, and result directories.

    Returns:
        Mapping with the pretraining, post-training, and result variant
        directories.
    """
    runs = workspace["runs"]

    _invoke(["dataset", "list"])

    _invoke(
        [
            "train",
            "--training-stage",
            "pretrain",
            "--formulation",
            workspace["formulation"],
            "--dataset",
            DATASET_ID,
            "--run-name",
            PRETRAIN_RUN,
            "--max-epoch",
            "1",
            "--batch-size",
            "2",
            "--backbone-kwargs",
            TINY_BACKBONE,
            "--yes",
        ]
    )

    refused = runner.invoke(
        app,
        [
            "train",
            "--training-stage",
            "posttrain",
            "--post-training-method",
            "cof",
            "--pretrain-run",
            str(runs / PRETRAIN_RUN),
            "--dataset",
            DATASET_ID,
            "--run-name",
            POSTTRAIN_RUN,
            "--yes",
        ],
        env={"COLUMNS": "200"},
    )
    assert refused.exit_code != 0
    assert "withheld during peer review" in refused.output
    assert not (runs / POSTTRAIN_RUN).exists()

    _invoke(
        [
            "train",
            "--training-stage",
            "pretrain",
            "--resume",
            "--checkpoint-path",
            str(runs / PRETRAIN_RUN / "checkpoints" / "last.ckpt"),
            "--max-epoch",
            "2",
            "--yes",
        ]
    )

    solver = "SB_SDE_Solver" if workspace["formulation"] == "SB-VE" else "OTCFM_ODE_Solver"
    variant = f"{solver}_N=4"
    _invoke(
        [
            "inference",
            "--run",
            PRETRAIN_RUN,
            "--dataset",
            DATASET_ID,
            "--split",
            "test",
            "--sampler",
            solver,
            "--num-steps",
            "4",
            "--skip-type",
            "time_uniform",
            "--device",
            "cpu",
            "--no-progress",
            "--seed",
            "1234",
        ]
    )

    _invoke(["metric", "--run", PRETRAIN_RUN, "--result", variant])

    return {
        "pretrain_dir": runs / PRETRAIN_RUN,
        "variant_dir": workspace["results"] / PRETRAIN_RUN / variant,
        "formulation": workspace["formulation"].replace("-", ""),
        "solver": solver,
    }


def _read_yaml(path: Path) -> dict[str, Any]:
    """Read one YAML mapping and assert it parses to a dictionary."""
    import yaml

    values = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(values, dict)
    return values


def test_dataset_list_shows_registered_workspace(deployment: dict[str, Any], workspace: dict[str, Path]) -> None:
    result = runner.invoke(app, ["dataset", "list"], env={"COLUMNS": "200"})
    assert result.exit_code == 0
    assert DATASET_ID in result.stdout
    assert str(workspace["dataset"]) in result.stdout


def test_pretrain_run_records_stage_and_tiny_backbone(deployment: dict[str, Any]) -> None:
    config = _read_yaml(deployment["pretrain_dir"] / "config.yml")
    assert config["stage"] == "pretrain"
    assert config["model"]["backbone"] == "ncsnpp_base"
    assert config["model"]["backbone_kwargs"]["nf"] == 8
    assert config["formulation"]["name"] == deployment["formulation"]


def test_posttrain_stage_is_refused_with_notice(deployment: dict[str, Any], workspace: dict[str, Path]) -> None:
    """The withheld post-training stage explains itself and creates no run."""
    result = runner.invoke(
        app,
        ["train", "--training-stage", "posttrain", "--pretrain-run", str(deployment["pretrain_dir"]), "--yes"],
        env={"COLUMNS": "200"},
    )
    assert result.exit_code != 0
    assert "withheld during peer review" in result.output
    assert "released upon" in result.output
    assert not (workspace["runs"] / POSTTRAIN_RUN).exists()


def test_training_runs_contain_acceptance_artifacts(deployment: dict[str, Any]) -> None:
    run_dir = deployment["pretrain_dir"]
    assert (run_dir / "config.yml").is_file()
    assert (run_dir / "checkpoints" / "last.ckpt").is_file()
    exports = list(run_dir.glob("model_*.safetensors")) + [run_dir / "model_last.safetensors"]
    assert any(path.is_file() for path in exports), f"No exported weights in {run_dir}"


def test_inference_result_is_manifested(deployment: dict[str, Any]) -> None:
    variant = deployment["variant_dir"]
    manifest = json.loads((variant / "inference.json").read_text(encoding="utf-8"))
    assert manifest["artifact_type"] == "generative_inference"
    assert manifest["run_name"] == PRETRAIN_RUN
    assert manifest["dataset_id"] == DATASET_ID
    assert manifest["sampler"] == deployment["solver"]
    assert manifest["num_steps"] == 4
    assert manifest["files"] == sorted(UTTERANCES)
    assert {path.name for path in variant.glob("*.wav")} == set(UTTERANCES)


def test_metric_evaluation_writes_core_protocol(deployment: dict[str, Any]) -> None:
    variant = deployment["variant_dir"]
    summary = json.loads((variant / "metrics.json").read_text(encoding="utf-8"))
    assert summary["artifact_type"] == "metrics"
    assert summary["metrics"] == ["pesq", "estoi", "si_sdr"]
    assert summary["num_files"] == len(UTTERANCES)
    assert set(summary["summary"]) == {"PESQ", "ESTOI", "SI_SDR"}
    for statistics in summary["summary"].values():
        assert set(statistics) == {"mean", "std"}

    with (variant / "metrics.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["filename"] for row in rows] == sorted(UTTERANCES)
    for row in rows:
        for column in ("PESQ", "ESTOI", "SI_SDR"):
            assert row[column] != ""


def test_removed_metric_names_are_rejected(deployment: dict[str, Any]) -> None:
    arguments = ["metric", "--run", PRETRAIN_RUN, "--result", VARIANT, "--metrics", "utmos"]
    result = runner.invoke(app, arguments, env={"COLUMNS": "200"})
    assert result.exit_code != 0
    assert "utmos" in result.output


def test_deployed_model_enhances_audio(deployment: dict[str, Any], workspace: dict[str, Path]) -> None:
    """The inference WAVs are finite, audible-length waveforms rather than silence."""
    for name in UTTERANCES:
        enhanced, sample_rate = torchaudio.load(deployment["variant_dir"] / name)
        assert sample_rate == SAMPLE_RATE
        assert enhanced.numel() > 0
        assert torch.isfinite(enhanced).all()


def test_resume_preserves_full_state(deployment):
    checkpoint = torch.load(
        deployment["pretrain_dir"] / "checkpoints" / "last.ckpt", weights_only=False, map_location="cpu"
    )
    assert checkpoint["global_step"] == 2
    assert checkpoint["optimizer_states"]
    assert checkpoint["method"]["name"] == "base"
    ema = next(state for name, state in checkpoint["callbacks"].items() if "EmaCallback" in name)
    assert ema["num_updates"] == 2
    assert ema["shadow_parameters"]


def test_run_cli_reports_runs_and_results(deployment: dict[str, Any]) -> None:
    """The run inspection commands list runs and report evaluated results."""
    listing = runner.invoke(app, ["run", "list"], env={"COLUMNS": "200"})
    assert listing.exit_code == 0
    assert PRETRAIN_RUN in listing.output
    assert POSTTRAIN_RUN not in listing.output

    shown = runner.invoke(app, ["run", "show", PRETRAIN_RUN], env={"COLUMNS": "200"})
    assert shown.exit_code == 0
    assert "pretrain" in shown.output
    assert deployment["formulation"] in shown.output
    assert "Default test model" in shown.output

    reported = runner.invoke(app, ["run", "results", PRETRAIN_RUN], env={"COLUMNS": "200"})
    assert reported.exit_code == 0
    assert deployment["variant_dir"].name in reported.output
    assert "PESQ" in reported.output
    assert "Not evaluated" not in reported.output


def test_run_export_carries_provenance(deployment: dict[str, Any], tmp_path: Path) -> None:
    """The export command copies the resolved model, config, and provenance."""
    output = tmp_path / "export"
    _invoke(["run", "export", PRETRAIN_RUN, "--output-dir", str(output)])
    expected_model = f"e2e_{deployment['formulation'].lower()}.safetensors"
    exported = sorted(path.name for path in output.iterdir())
    assert exported == sorted(["config.yml", "export.json", expected_model])
    record = json.loads((output / "export.json").read_text(encoding="utf-8"))
    assert record["artifact_type"] == "model_export"
    assert record["run_name"] == PRETRAIN_RUN
    assert record["exported_file"] == expected_model
    assert record["model_sha256"]
    exported_config = _read_yaml(output / "config.yml")
    assert exported_config["weights"]["default_test_model"] == expected_model
    duplicate = runner.invoke(app, ["run", "export", PRETRAIN_RUN, "--output-dir", str(output)])
    assert duplicate.exit_code != 0


def test_enhance_individual_files(deployment: dict[str, Any], workspace: dict[str, Path], tmp_path: Path) -> None:
    """The enhance command enhances loose files with a manifested output directory."""
    noisy = workspace["dataset"] / "test" / "noisy" / UTTERANCES[0]
    output = tmp_path / "enhanced"
    _invoke(
        [
            "enhance",
            "--run",
            PRETRAIN_RUN,
            "--input",
            str(noisy),
            "--output-dir",
            str(output),
            "--device",
            "cpu",
            "--no-progress",
        ]
    )
    enhanced, sample_rate = torchaudio.load(output / UTTERANCES[0])
    assert sample_rate == SAMPLE_RATE
    assert enhanced.numel() > 0
    assert torch.isfinite(enhanced).all()
    manifest = json.loads((output / "enhance.json").read_text(encoding="utf-8"))
    assert manifest["artifact_type"] == "generative_enhancement"
    assert manifest["num_files"] == 1
    assert manifest["files"] == [UTTERANCES[0]]


def test_bare_checkpoint_resolves_as_run(
    deployment: dict[str, Any], workspace: dict[str, Path], tmp_path: Path
) -> None:
    """A self-describing checkpoint file works through --ckpt without a run directory."""
    checkpoint = tmp_path / "bare_model.safetensors"
    checkpoint.write_bytes((deployment["pretrain_dir"] / "model_last.safetensors").read_bytes())
    noisy = workspace["dataset"] / "test" / "noisy" / UTTERANCES[0]
    enhanced_dir = tmp_path / "enhanced"
    _invoke(
        [
            "enhance",
            "--ckpt",
            str(checkpoint),
            "--input",
            str(noisy),
            "--output-dir",
            str(enhanced_dir),
            "--device",
            "cpu",
            "--no-progress",
        ]
    )
    enhanced, sample_rate = torchaudio.load(enhanced_dir / UTTERANCES[0])
    assert sample_rate == SAMPLE_RATE
    assert torch.isfinite(enhanced).all()
    manifest = json.loads((enhanced_dir / "enhance.json").read_text(encoding="utf-8"))
    assert manifest["run_name"] == "bare_model"
    assert manifest["model_sha256"]

    _invoke(
        [
            "inference",
            "--ckpt",
            str(checkpoint),
            "--dataset",
            DATASET_ID,
            "--split",
            "test",
            "--device",
            "cpu",
            "--no-progress",
        ]
    )
    variant_dir = workspace["results"] / "bare_model" / f"{deployment['solver']}_N=4"
    assert (variant_dir / "inference.json").is_file()
    for name in UTTERANCES:
        assert (variant_dir / name).is_file()
    inference_manifest = json.loads((variant_dir / "inference.json").read_text(encoding="utf-8"))
    assert inference_manifest["run_name"] == "bare_model"
    assert inference_manifest["model_sha256"] == manifest["model_sha256"]
    assert inference_manifest["source_type"] == "checkpoint"
    assert inference_manifest["status"] == "complete"
    _invoke(["metric", "--dir", str(variant_dir)])
    metrics = json.loads((variant_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["source_type"] == "checkpoint"
    assert metrics["model_sha256"] == inference_manifest["model_sha256"]

    both = runner.invoke(
        app,
        [
            "enhance",
            "--run",
            PRETRAIN_RUN,
            "--ckpt",
            str(checkpoint),
            "--input",
            str(noisy),
            "--output-dir",
            str(tmp_path / "invalid"),
        ],
        env={"COLUMNS": "200"},
    )
    assert both.exit_code != 0


def test_version_reports_package() -> None:
    """The version command prints the installed package version."""
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.output.startswith("cof ")
    assert result.output.strip().split()[-1].count(".") == 2
