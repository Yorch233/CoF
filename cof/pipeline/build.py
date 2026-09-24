"""Single configuration-to-model/method/pipeline assembly entry point."""

from __future__ import annotations

from cof.config.manager import Config
from cof.config.schemas import OptimizationConfig
from cof.data.stft import StftTransform
from cof.method.registry import PostTrainingRegister, PretrainingRegister, load_builtin_methods
from cof.model import GenerativeModel4SE
from cof.pipeline.base import BaseTrainingPipeline
from cof.pipeline.posttrain import PosttrainPipeline
from cof.pipeline.pretrain import PretrainPipeline
from cof.utils.sampling import resolve_sampling_protocol


def build_generative_pipeline(config: Config, *, load_initial_weights: bool = True) -> BaseTrainingPipeline:
    """Construct the named model and inject a registered stage-specific algorithm.

    Model internals are assembled by GenerativeModel4SE. Method factories own
    their configuration extraction, so new algorithms do not change this function.
    """
    load_builtin_methods()
    stage = str(config.get("stage", "pretrain"))
    if stage not in {"pretrain", "post_training"}:
        raise ValueError(f"Unknown training stage: {stage}")
    protocol = resolve_sampling_protocol(config)
    model = GenerativeModel4SE(
        formulation=protocol.formulation,
        backbone=config.get("model.backbone", "ncsnpp_base"),
        backbone_kwargs=Config.unwrap(config.get("model.backbone_kwargs") or {}),
        formulation_kwargs=Config.unwrap(config.get("formulation.kwargs") or {}),
        transform_kwargs=StftTransform.from_config(config).to_config(),
        sampling_solver=protocol.solver,
        sampling_num_steps=protocol.num_steps,
        sampling_skip_type=protocol.skip_type,
        gradient_checkpointing=bool(config.get("optimization.gradient_checkpointing", False)),
    )
    if config.get("formulation.training_target", model.training_target) != model.training_target:
        raise ValueError("Configured training target conflicts with the formulation")
    initial_path = config.get("run.init_model_path")
    if initial_path and load_initial_weights:
        model.load_checkpoint(initial_path)
    scope = "pretrain" if stage == "pretrain" else "posttrain"
    registry = PretrainingRegister if stage == "pretrain" else PostTrainingRegister
    method_name = str(config.get(f"{scope}.method", "base" if stage == "pretrain" else "cof"))
    method = registry.fetch(method_name).from_config(model, config)
    if "ema" in method.required_references() and not config.get("optimization.ema", True):
        raise ValueError("The selected method requires EMA references but EMA is disabled")
    optimization = OptimizationConfig(
        learning_rate=float(config.get("optimization.learning_rate", 1e-4)),
        optimizer=config.get("optimization.optimizer", "Adam"),
        warmup_steps=int(config.get("optimization.learning_rate_warmup_steps", 0)),
        gradient_clip=float(config.get("optimization.gradient_clip_val", 0.0)),
    )
    pipeline_type = PretrainPipeline if stage == "pretrain" else PosttrainPipeline
    return pipeline_type(model, method, optimization)
