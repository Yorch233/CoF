"""Sample validation utterances and log speech-quality metrics during training.

The evaluation is monitoring only: a fixed, seeded subset of a split is
sampled through the model's own sampling protocol and scored with the
intrusive metrics, giving comparable curves across epochs without touching
the deployed result pipeline.

Author: Qing Yao
Date: 2026/9/23
"""

from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torchaudio
from lightning.pytorch.callbacks import Callback
from torch import Tensor

from cof.config.manager import Config
from cof.evaluation import MetricSuite
from cof.pipeline.base import training_stage_metric_name
from cof.pipeline.provenance import paired_wavs, resolve_dataset
from cof.utils.progress import track_workflow_items
from cof.utils.sampling import resolve_sampling_protocol

if TYPE_CHECKING:
    from lightning.pytorch import LightningModule, Trainer

    from cof.model import GenerativeModel4SE


def _load_audio(path: Path, sample_rate: int) -> Tensor:
    """Load one waveform and resample it when necessary.

    Args:
        path: WAV file to load.
        sample_rate: Target sample rate.

    Returns:
        The waveform tensor at the target rate.
    """
    audio, source_rate = torchaudio.load(path)
    if source_rate != sample_rate:
        audio = torchaudio.functional.resample(audio, source_rate, sample_rate)
    return audio


def evaluate_sampled_split(
    model: GenerativeModel4SE,
    config: Config,
    split: str,
    *,
    max_samples: int,
    num_steps: int,
    solver: str | None,
    skip_type: str,
    seed: int,
    progress: bool = True,
    repeats: int = 1,
) -> dict[str, float]:
    """Sample a stable subset and average PESQ and SI-SDR over seeded trajectories.

    Args:
        model: Generative model whose ``enhance`` produces the estimates.
        config: Run configuration carrying the dataset selection and sample
            rate.
        split: Dataset split to sample from.
        max_samples: Upper bound on the number of utterances sampled.
        num_steps: Sampling step count (NFE) passed to ``enhance``.
        solver: Sampling solver name.
        skip_type: Time-skip schedule name.
        seed: Base seed; each file derives a deterministic per-file seed from
            its name, and repeats stride the derived seed so trajectories
            differ without reordering.
        progress: Whether to display a progress bar.
        repeats: Number of seeded sampling trajectories per utterance.

    Returns:
        Mean score per metric output over all scored trajectories.

    Raises:
        ValueError: If ``repeats`` is below one.
        FloatingPointError: If an enhanced waveform or a metric value is
            non-finite.
    """
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    _, dataset_root = resolve_dataset(config, config.get("registry.dataset"))
    all_pairs = paired_wavs(dataset_root, split)
    pairs = random.Random(seed).sample(all_pairs, k=min(max_samples, len(all_pairs)))
    sample_rate = int(config.get("data.sample_rate", 16_000))
    suite = MetricSuite(tuple(config.get("evaluation.metrics", ["pesq", "si_sdr"])), sample_rate)
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for clean_path, noisy_path in track_workflow_items(pairs, f"Sampling {split} metrics", enabled=progress):
        clean = _load_audio(clean_path, sample_rate)
        noisy = _load_audio(noisy_path, sample_rate)
        # NOTE: The per-file base seed is derived from the filename digest, so
        # trajectories stay deterministic and order-independent for a given
        # seed; repeats stride the base seed to vary them.
        digest = hashlib.sha256(noisy_path.name.encode(), usedforsecurity=False).digest()
        base_seed = (seed + int.from_bytes(digest[:4], "big")) % (2**31)
        for repeat in range(repeats):
            sample_seed = (base_seed + repeat * 1_000_003) % (2**31)
            model_device = getattr(model, "model_device", torch.device("cpu"))
            cuda_devices = []
            if model_device.type == "cuda":
                cuda_devices = [model_device.index if model_device.index is not None else torch.cuda.current_device()]
            with torch.random.fork_rng(devices=cuda_devices):
                torch.manual_seed(sample_seed)
                if cuda_devices:
                    torch.cuda.manual_seed_all(sample_seed)
                enhanced, _, _ = model.enhance(
                    noisy,
                    num_steps=num_steps,
                    solver=solver,
                    skip_type=skip_type,
                )
            if not torch.isfinite(enhanced).all():
                raise FloatingPointError(
                    f"Non-finite enhanced waveform for {noisy_path.name} "
                    f"(solver={solver}, num_steps={num_steps}, repeat={repeat})"
                )
            reference = clean.detach().float().cpu().reshape(-1).numpy()
            estimate = enhanced.detach().float().cpu().reshape(-1).numpy()
            minimum = min(reference.size, estimate.size)
            reference, estimate = reference[:minimum], estimate[:minimum]
            values = suite.calculate(reference, estimate)
            for name, value in values.items():
                totals[name] = totals.get(name, 0.0) + value
                counts[name] = counts.get(name, 0) + 1

    return {name: value / counts[name] for name, value in totals.items()}


def _post_training_validation_line(trainer: Trainer, scores: dict[str, float]) -> str:
    """Format the summary line printed after every post-training validation run.

    Args:
        trainer: Lightning trainer that ran validation.
        scores: Mean metric scores from ``evaluate_sampled_split``.

    Returns:
        The human-readable validation summary line.
    """
    summary: list[str] = []
    loss = trainer.callback_metrics.get(f"{training_stage_metric_name('post_training')}/valid/loss")
    if loss is not None:
        summary.append(f"loss={float(loss):.4f}")
    for name, value in scores.items():
        summary.append(f"{name}={value:.4f}")
    return f"Post-training validation @ step {trainer.global_step}: " + ", ".join(summary)


class GenerativeSampleMetrics(Callback):
    """Periodically log validation speech metrics for monitoring only.

    Metrics are namespaced ``{stage}/valid/{name}`` so the stage groups first
    in the wandb panel tree.
    """

    def __init__(
        self,
        config: Config,
        *,
        valid_samples: int = 50,
        num_steps: int = 4,
        solver: str | None = None,
        skip_type: str = "time_uniform",
        seed: int = 1234,
    ) -> None:
        """Configure validation sample size and sampling protocol.

        Args:
            config: Run configuration carrying the dataset, sample rate, and
                training stage used to namespace the logged metrics.
            valid_samples: Number of validation utterances to sample.
            num_steps: Sampling step count (NFE).
            solver: Sampling solver name; ``None`` resolves the configured or
                path-default solver.
            skip_type: Time-skip schedule name.
            seed: Base seed for the per-file seed derivation.

        Raises:
            ValueError: If the sample count or step count is not positive.
        """
        super().__init__()
        if valid_samples < 1 or num_steps < 1:
            raise ValueError("Sample count and num_steps must be positive")
        self.config = config
        stage_namespace = training_stage_metric_name(str(config.get("stage") or "pretrain"))
        suite = MetricSuite(
            tuple(config.get("evaluation.metrics", ["pesq", "si_sdr"])), int(config.get("data.sample_rate", 16000))
        )
        self.output_names = suite.output_names
        self.metric_names = tuple(f"{stage_namespace}/valid/{name}_per_epoch" for name in self.output_names)
        self.valid_samples = valid_samples
        self.num_steps = num_steps
        self.solver = solver
        if self.solver is None:
            try:
                self.solver = str(resolve_sampling_protocol(config).solver)
            except ValueError:
                self.solver = None
        self.skip_type = skip_type
        self.seed = seed

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Evaluate on rank zero, broadcast scores, and expose them to later callbacks.

        Args:
            trainer: Lightning trainer driving validation.
            pl_module: Validated module receiving the logged metrics.
        """
        if trainer.sanity_checking:
            return
        scores = torch.zeros(len(self.metric_names), device=pl_module.device, dtype=torch.float64)
        if trainer.is_global_zero:
            model = getattr(pl_module, "model", pl_module)
            evaluate_kwargs = {
                "max_samples": self.valid_samples,
                "num_steps": self.num_steps,
                "solver": self.solver,
                "skip_type": self.skip_type,
                "seed": self.seed,
            }
            valid = evaluate_sampled_split(
                model,
                self.config,
                "valid",
                **evaluate_kwargs,
            )
            if str(self.config.get("stage")) == "post_training":
                print(_post_training_validation_line(trainer, valid))
            scores = torch.tensor(
                [valid[name] for name in self.output_names], device=pl_module.device, dtype=torch.float64
            )
        scores = trainer.strategy.broadcast(scores, src=0)
        logged = dict(zip(self.metric_names, scores, strict=True))
        for name, score in logged.items():
            pl_module.log(name, score, on_step=False, on_epoch=True, prog_bar=True)
        trainer.callback_metrics.update(logged)
