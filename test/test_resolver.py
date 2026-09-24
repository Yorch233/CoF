"""Configuration-resolution contract: merge order, stage validation, locking.

Author: Qing Yao
Date: 2026/9/23
"""

from pathlib import Path

import pytest
import yaml

from cof.config.resolver import (
    TrainingStage,
    normalize_explicit_overrides,
    read_host_settings,
    resolve_training_config,
    validate_resume,
    validate_stage_args,
)

DATASET_REGISTRY = {"dataset": "voicebank", "datasets": {"voicebank": "~/datasets/voicebank"}}


def write_host(tmp_path: Path, values: dict) -> Path:
    values = {**DATASET_REGISTRY, **values}
    host_path = tmp_path / "cof.yml"
    host_path.write_text(yaml.safe_dump(values), encoding="utf-8")
    return host_path


def make_pretrain_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "runs" / "cof_generative_pre"
    run_dir.mkdir(parents=True)
    (run_dir / "config.yml").write_text(
        yaml.safe_dump(
            {
                "run_name": run_dir.name,
                "stage": "pretrain",
                "model": {"backbone": "ncsnpp_base", "backbone_kwargs": {}},
                "formulation": {
                    "name": "SBVE",
                    "kwargs": {},
                    "training_target": "data",
                    "sampling": {"solver": "SB_SDE_Solver", "num_steps": 4, "skip_type": "time_uniform"},
                },
                "registry": {"dataset": "voicebank", "datasets": {"voicebank": "~/datasets/voicebank"}},
                "metrics": {"best_pesq": 3.21, "last_model_step": 6755},
                "init_model_path": "~/weights/old.safetensors",
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "model_last.safetensors").touch()
    return run_dir


def test_host_settings_drop_inherit(tmp_path: Path):
    host_path = tmp_path / "cof.yml"
    host_path.write_text(yaml.safe_dump({"inherit": "config/default.yml", "logger": "none"}), encoding="utf-8")

    settings = read_host_settings(host_path)

    assert settings == {"logger": "none"}


def test_normalize_overrides_filters_empty_and_parses_json():
    normalized = normalize_explicit_overrides(
        {"dataset": "voicebank", "comment": "", "run_name": None, "backbone_kwargs": '{"nf": 8}'}
    )

    assert normalized == {"dataset": "voicebank", "backbone_kwargs": {"nf": 8}}


def test_pretrain_merge_order_preset_host_cli(tmp_path: Path):
    """CLI beats host beats preset for the same key."""
    host_path = write_host(tmp_path, {"batch_size": 4, "ema": False})

    config, overrides = resolve_training_config(
        training_stage=TrainingStage.PRETRAIN,
        formulation="SB-VE",
        pretrain_run=None,
        host_config_path=host_path,
        batch_size=16,
        num_epoch=3,
    )

    assert config.get("optimization.batch_size") == 16  # CLI override wins
    assert config.get("optimization.ema") is False  # host setting wins over the preset default (True)
    assert config.get("pretrain.num_epoch") == 3
    assert config.get("stage") == "pretrain"

    assert overrides == {"optimization.batch_size": 16, "pretrain.num_epoch": 3}


def test_posttrain_inherits_identity_and_strips_run_bookkeeping(tmp_path: Path):
    pretrain_dir = make_pretrain_run(tmp_path)
    host_path = write_host(tmp_path, {})

    config, _ = resolve_training_config(
        training_stage=TrainingStage.POSTTRAIN,
        formulation=None,
        pretrain_run=pretrain_dir,
        host_config_path=host_path,
        max_steps=8,
    )

    assert config.get("stage") == "post_training"

    assert config.get("model.backbone") == "ncsnpp_base"
    assert config.get("formulation.sampling.num_steps") == 4
    assert config.get("metrics.best_pesq") is None  # run bookkeeping must not leak
    assert config.get("metrics.last_model_step") is None  # stale end-of-pretrain records must not leak

    assert Path(config.get("run.init_model_path")).is_file()
    assert Path(config.get("run.pretrain_run")) == pretrain_dir.resolve()


def test_posttrain_rejects_locked_flags(tmp_path: Path):
    pretrain_dir = make_pretrain_run(tmp_path)
    host_path = write_host(tmp_path, {})

    with pytest.raises(ValueError, match="--sampling-num-steps"):
        resolve_training_config(
            training_stage=TrainingStage.POSTTRAIN,
            formulation=None,
            pretrain_run=pretrain_dir,
            host_config_path=host_path,
            sampling_num_steps=8,
        )


def test_posttrain_selects_its_own_rollout_solver(tmp_path: Path):
    pretrain_dir = make_pretrain_run(tmp_path)
    host_path = write_host(tmp_path, {})

    config, overrides = resolve_training_config(
        training_stage=TrainingStage.POSTTRAIN,
        formulation=None,
        pretrain_run=pretrain_dir,
        host_config_path=host_path,
        sampling_solver="SB_ODE_Solver",
    )

    assert config.get("formulation.sampling.solver") == "SB_ODE_Solver"
    assert overrides == {"formulation.sampling.solver": "SB_ODE_Solver"}


def test_pretrain_sampling_flags_reach_the_protocol(tmp_path: Path):
    host_path = write_host(tmp_path, {})

    config, _ = resolve_training_config(
        training_stage=TrainingStage.PRETRAIN,
        formulation="SB-VE",
        pretrain_run=None,
        host_config_path=host_path,
        sampling_num_steps=8,
        sampling_solver="SB_ODE_Solver",
    )

    assert config.get("formulation.sampling.num_steps") == 8
    assert config.get("formulation.sampling.solver") == "SB_ODE_Solver"
    assert config.get("sampling_num_steps") is None


def test_posttrain_requires_a_pretrain_run_directory(tmp_path: Path):
    host_path = write_host(tmp_path, {})

    with pytest.raises(ValueError, match="--pretrain-run"):
        resolve_training_config(
            training_stage=TrainingStage.POSTTRAIN,
            formulation=None,
            pretrain_run=tmp_path / "missing",
            host_config_path=host_path,
        )


def test_posttrain_rejects_a_posttrain_source_run(tmp_path: Path):
    pretrain_dir = make_pretrain_run(tmp_path)
    config = yaml.safe_load((pretrain_dir / "config.yml").read_text())
    config["stage"] = "post_training"
    (pretrain_dir / "config.yml").write_text(yaml.safe_dump(config), encoding="utf-8")
    host_path = write_host(tmp_path, {})

    with pytest.raises(ValueError, match="not a pretraining run"):
        resolve_training_config(
            training_stage=TrainingStage.POSTTRAIN,
            formulation=None,
            pretrain_run=pretrain_dir,
            host_config_path=host_path,
        )


def test_resolve_requires_a_registered_dataset(tmp_path: Path):
    host_path = write_host(tmp_path, {"dataset": "nowhere", "datasets": {"other": "~/datasets/other"}})

    with pytest.raises(ValueError, match="nowhere"):
        resolve_training_config(
            training_stage=TrainingStage.PRETRAIN,
            formulation="SB-VE",
            pretrain_run=None,
            host_config_path=host_path,
        )


def test_stage_arg_contract():
    with pytest.raises(ValueError, match="--formulation is required"):
        validate_stage_args(TrainingStage.PRETRAIN, None, None, None)
    validate_stage_args(TrainingStage.PRETRAIN, "SB-VE", "cof", None)
    with pytest.raises(ValueError, match="--post-training-method is not allowed"):
        validate_stage_args(TrainingStage.PRETRAIN, "SB-VE", "other", None)
    with pytest.raises(ValueError, match="--pretrain-run is not allowed"):
        validate_stage_args(TrainingStage.PRETRAIN, "SB-VE", None, Path("runs/x"))
    with pytest.raises(ValueError, match="--formulation is not allowed"):
        validate_stage_args(TrainingStage.POSTTRAIN, "SB-VE", "cof", Path("runs/x"))
    with pytest.raises(ValueError, match="--post-training-method is required"):
        validate_stage_args(TrainingStage.POSTTRAIN, None, None, Path("runs/x"))
    with pytest.raises(ValueError, match="--pretrain-run is required"):
        validate_stage_args(TrainingStage.POSTTRAIN, None, "cof", None)
    validate_stage_args(TrainingStage.PRETRAIN, "OT-CFM", None, None)
    validate_stage_args(TrainingStage.POSTTRAIN, None, "cof", Path("runs/x"))


def test_resume_contract():
    with pytest.raises(ValueError, match="--checkpoint-path is required"):
        validate_resume(resume=True, checkpoint_path=None)
    with pytest.raises(ValueError, match="--checkpoint-path requires --resume"):
        validate_resume(resume=False, checkpoint_path=Path("runs/x/checkpoints/last.ckpt"))
    validate_resume(resume=True, checkpoint_path=Path("runs/x/checkpoints/last.ckpt"))
    validate_resume(resume=False, checkpoint_path=None)


@pytest.mark.parametrize("selection", ["run", "directory", "last.ckpt", "step=1000.ckpt"])
def test_resume_resolves_exact_file(tmp_path, selection):
    from cof.config.manager import read_config_from_yaml
    from cof.training.runner import prepare_generative_run

    run = make_pretrain_run(tmp_path)
    checkpoints = run / "checkpoints"
    checkpoints.mkdir()
    for name in ("last.ckpt", "step=1000.ckpt"):
        (checkpoints / name).touch()
    selected = run if selection == "run" else checkpoints if selection == "directory" else checkpoints / selection
    expected = checkpoints / (selection if selection.endswith(".ckpt") else "last.ckpt")
    resolved = prepare_generative_run(read_config_from_yaml(run / "config.yml"), checkpoint_path=selected)
    assert resolved.checkpoint_path == expected
    assert resolved.run_path == run


def test_resume_explicit_checkpoint_without_last(tmp_path):
    from cof.training.runtime import resolve_resume_checkpoint

    run = make_pretrain_run(tmp_path)
    checkpoints = run / "checkpoints"
    checkpoints.mkdir()
    explicit = checkpoints / "step=1000.ckpt"
    explicit.touch()
    assert resolve_resume_checkpoint(explicit) == explicit
    for selected in (run, checkpoints, checkpoints / "step=2000.ckpt"):
        with pytest.raises(ValueError, match="does not exist"):
            resolve_resume_checkpoint(selected)
