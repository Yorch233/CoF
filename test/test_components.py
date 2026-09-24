"""Composition, checkpoint and extension behavior independent of production backbones."""

from pathlib import Path

import lightning as lightning
import pytest
import torch
from torch import nn

from cof.backbone.registry import BackboneRegister
from cof.config.manager import Config
from cof.config.schemas import OptimizationConfig
from cof.data.stft import StftTransform
from cof.evaluation import MetricRegister, MetricSuite, supported_metrics
from cof.formulation.base import ODE, SDE
from cof.formulation.ot_cfm.definition import OTCFMODE
from cof.formulation.registry import PATH_TYPES, register_formulation
from cof.formulation.sb_ve.definition import SBVESDE, SBVEFormulation
from cof.method.base import StepContext, StepResult, TrainingMethod
from cof.method.pretrain.base import BasePretraining
from cof.method.registry import PretrainingRegister
from cof.model import GenerativeModel4SE
from cof.pipeline.build import build_generative_pipeline
from cof.pipeline.posttrain import PosttrainPipeline
from cof.pipeline.pretrain import PretrainPipeline
from cof.training.callbacks.sample_metrics import GenerativeSampleMetrics


class TinyBackbone(nn.Module):
    """Minimal complex-compatible trainable predictor for contract tests."""

    def __init__(self, input_channels=4):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.5))

    def forward(self, state, time):
        return state[:, :1] * self.scale


@pytest.fixture(autouse=True)
def register_tiny(monkeypatch):
    monkeypatch.setitem(BackboneRegister._dict, "test_tiny", TinyBackbone)


def make_model(formulation="SBVE", **kwargs):
    return GenerativeModel4SE(formulation, backbone="test_tiny", **kwargs)


@pytest.mark.parametrize("formulation", ["SBVE", "OTCFM"])
def test_checkpoint_roundtrip_and_identity(tmp_path, formulation):
    model = make_model(formulation)
    checkpoint = model.save_checkpoint(tmp_path / "model.safetensors")
    restored = GenerativeModel4SE.from_checkpoint(checkpoint)
    assert restored.model_config() == model.model_config()
    assert not restored.training
    noisy = torch.randn(1, 1, 256, 64, dtype=torch.complex64)
    torch.manual_seed(23)
    expected = model.sample(noisy, num_steps=3)
    torch.manual_seed(23)
    actual = restored.sample(noisy, num_steps=3)
    for original, loaded in zip(expected, actual, strict=True):
        torch.testing.assert_close(original, loaded, rtol=0, atol=0)
    different = make_model(formulation, transform_kwargs={"spec_factor": 0.8})
    with pytest.warns(UserWarning, match="keeping the checkpoint values"):
        different.load_checkpoint(checkpoint)  # checkpoint 为准，加载继续


def test_checkpoint_priority_ladder(tmp_path):
    """Self-describing checkpoints win over config; legacy files need config."""
    model = make_model()
    checkpoint = model.save_checkpoint(tmp_path / "v1.safetensors")
    legacy = tmp_path / "legacy.safetensors"
    import safetensors.torch as st

    st.save_model(model, legacy)  # 同张量、无 metadata

    matching = {
        "formulation": {"name": "SBVE", "kwargs": {}},
        "model": {"backbone": "test_tiny", "backbone_kwargs": {}},
        "data": {"spec_factor": 0.33},
    }
    restored = GenerativeModel4SE.from_checkpoint(legacy, config=matching)
    assert restored.model_config() == model.model_config()

    conflicting = {
        "formulation": {"name": "SBVE", "kwargs": {}},
        "model": {"backbone": "test_tiny", "backbone_kwargs": {}},
        "data": {"spec_factor": 0.8},
    }
    with pytest.warns(UserWarning, match="keeping the checkpoint values"):
        restored = GenerativeModel4SE.from_checkpoint(checkpoint, config=conflicting)
    assert restored.transform.to_config()["spec_factor"] == 0.33  # checkpoint 值胜出

    with pytest.raises(ValueError, match="no run configuration was provided"):
        GenerativeModel4SE.from_checkpoint(legacy)


def test_explicit_dynamics_and_sb_ode_endpoint():
    assert issubclass(SBVESDE, SDE)
    assert issubclass(OTCFMODE, ODE)
    model = make_model(sampling_solver="SB_ODE_Solver")
    noisy = torch.randn(1, 1, 256, 64, dtype=torch.complex64)
    final, trajectory, _ = model.sample(noisy, num_steps=4)
    assert torch.isfinite(trajectory).all()
    assert torch.isfinite(final).all()
    assert model.sampling_evaluations == 4
    assert model.training


def test_transform_instances_do_not_leak_configuration():
    first = StftTransform(n_fft=510, spec_factor=0.33)
    audio = torch.randn(1, 4096)
    before = first.stft(audio)
    other = StftTransform(n_fft=254, hop_length=64, spec_factor=0.7)
    assert other.stft(audio).shape != before.shape
    torch.testing.assert_close(first.stft(audio), before, rtol=0, atol=0)
    torch.testing.assert_close(first.istft(before, length=4096), audio, atol=1e-5, rtol=1e-5)


def test_formulation_and_method_registry_extensions(monkeypatch):
    class CustomBridge(SBVEFormulation):
        path_name = "TESTBRIDGE"

    monkeypatch.setitem(PATH_TYPES, "PLACEHOLDER", CustomBridge)
    # Keep registration cleanup local to this test.
    with monkeypatch.context() as patch:
        patch.setattr("cof.formulation.registry.PATH_TYPES", dict(PATH_TYPES))
        register_formulation("test_bridge")(CustomBridge)
        config = Config(
            {
                "stage": "pretrain",
                "model": {"backbone": "test_tiny"},
                "formulation": {"name": "test_bridge"},
                "pretrain": {"method": "base"},
            }
        )
        pipeline = build_generative_pipeline(config)
        assert isinstance(pipeline.model.formulation, CustomBridge)
        assert isinstance(pipeline.method, BasePretraining)

    class CustomMethod(TrainingMethod):
        name = "test_custom"
        stage = "pretrain"

        @classmethod
        def from_config(cls, model, config):
            return cls(model)

        def step(self, batch, context):
            return StepResult(self.model.generator.scale.square())

    monkeypatch.setitem(PretrainingRegister._dict, "test_custom", CustomMethod)
    config = Config(
        {
            "stage": "pretrain",
            "model": {"backbone": "test_tiny"},
            "formulation": {"name": "SBVE"},
            "pretrain": {"method": "test_custom"},
        }
    )
    pipeline = build_generative_pipeline(config)
    assert isinstance(pipeline.method, CustomMethod)
    assert issubclass(PretrainPipeline, lightning.LightningModule)
    assert issubclass(PosttrainPipeline, lightning.LightningModule)
    result = pipeline.method.step((torch.zeros(1), torch.zeros(1)), StepContext())
    result.loss.backward()
    assert pipeline.model.generator.scale.grad is not None


def test_injected_trainable_loss_is_owned_once_and_checkpointed():
    class ScaledLoss(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(2.0))

        def forward(self, prediction, state, time, clean, noisy):
            return {"loss": self.weight * (prediction - clean).abs().square().mean()}

    model = make_model()
    loss = ScaledLoss()
    pipeline = PretrainPipeline(model, BasePretraining(model, loss_fn=loss), OptimizationConfig())
    assert "method_modules.loss.weight" in pipeline.state_dict()
    assert len(list(pipeline.parameters())) == 2
    clean = torch.randn(1, 1, 256, 64, dtype=torch.complex64)
    result = pipeline.method.step((clean, clean + 0.1), StepContext())
    result.loss.backward()
    assert loss.weight.grad is not None
    optimizer = pipeline.configure_optimizers()
    assert any(parameter is loss.weight for group in optimizer.param_groups for parameter in group["params"])
    checkpoint = {}
    pipeline.on_save_checkpoint(checkpoint)
    pipeline.on_load_checkpoint(checkpoint)
    checkpoint["method"]["name"] = "different_method"
    with pytest.raises(ValueError, match="method/stage"):
        pipeline.on_load_checkpoint(checkpoint)


def test_metric_extension_reaches_validation(monkeypatch):
    class EnergyMetric:
        output_names = ("ENERGY",)
        higher_is_better = False
        requires_reference = False

        def __init__(self, sample_rate):
            self.sample_rate = sample_rate

        def calculate(self, ref_wav, deg_wav, sample_rate):
            return {"ENERGY": float(torch.as_tensor(deg_wav).square().mean())}

    monkeypatch.setitem(MetricRegister._dict, "test_energy", EnergyMetric)
    assert "test_energy" in supported_metrics()
    suite = MetricSuite(["test_energy"])
    assert suite.calculate(None, torch.ones(8)) == {"ENERGY": 1.0}
    assert suite.modes["ENERGY"] == "min"
    callback = GenerativeSampleMetrics(Config({"stage": "pretrain", "evaluation": {"metrics": ["test_energy"]}}))
    assert callback.metric_names == ("pretrain/valid/ENERGY_per_epoch",)


def test_metric_suite_rejects_unknown_components():
    with pytest.raises(KeyError, match="Unregistered"):
        MetricSuite(["si_sdr", "nonexistent"])


def test_resume_does_not_reload_initial_weights():
    config = Config(
        {
            "stage": "pretrain",
            "model": {"backbone": "test_tiny"},
            "formulation": {"name": "SBVE"},
            "pretrain": {"method": "base"},
            "run": {"init_model_path": "missing.safetensors"},
        }
    )
    pipeline = build_generative_pipeline(config, load_initial_weights=False)
    assert isinstance(pipeline, PretrainPipeline)


def test_metric_artifact_reuse_rejects_changed_selection(tmp_path):
    from cof.pipeline.evaluation import _calculate_metrics
    from cof.pipeline.provenance import write_json

    (tmp_path / "metrics.csv").write_text("filename,SI_SDR\n")
    write_json(tmp_path / "metrics.json", {"sample_rate": 16000, "metrics": ["si_sdr"]})
    with pytest.raises(ValueError, match="conflict"):
        _calculate_metrics([], tmp_path, {}, sample_rate=16000, metrics=("pesq",), max_workers=1, overwrite=False)


def test_registered_backbone_passes_training_preflight():
    from cof.training.preflight import prepare_training_launch

    config = Config({"model": {"backbone": "test_tiny"}})
    prepare_training_launch(config, assume_yes=True)
    assert config.get("host.ncsnpp_operator_backend") == "not_applicable"


def test_ema_updates_only_after_optimizer_steps():
    from types import SimpleNamespace

    from cof.training.callbacks.ema import EmaCallback

    model = make_model()
    pipeline = PretrainPipeline(model, BasePretraining(model))
    ema = EmaCallback(0.5)
    trainer = SimpleNamespace(global_step=0)
    ema.on_fit_start(trainer, pipeline)
    ema.on_train_batch_end(trainer, pipeline, None, None, 0)
    assert ema.num_updates == 0
    trainer.global_step = 1
    ema.on_train_batch_end(trainer, pipeline, None, None, 1)
    ema.on_train_batch_end(trainer, pipeline, None, None, 2)
    assert ema.num_updates == 1
    restored = EmaCallback()
    restored.load_state_dict(ema.state_dict())
    assert restored.last_optimizer_step == 1


def test_mathematical_components_have_no_orchestration_dependencies():
    import ast

    import cof

    root = Path(cof.__file__).parent
    files = list((root / "formulation").rglob("*.py")) + [root / "model.py"]
    for source in files:
        for node in ast.walk(ast.parse(source.read_text())):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                assert not module.startswith(("cof.method", "cof.pipeline", "cof.training", "lightning")), (
                    source,
                    module,
                )
