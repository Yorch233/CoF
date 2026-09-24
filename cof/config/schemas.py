"""Validated component-local configuration records used during assembly."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class OptimizationConfig:
    """Optimizer configuration shared by training pipelines, without method knobs."""

    learning_rate: float = 1e-4
    optimizer: str = "Adam"
    warmup_steps: int = 0
    gradient_clip: float = 0.0

    def __post_init__(self) -> None:
        """Reject invalid optimization settings before allocating trainer state."""
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if self.optimizer not in {"Adam", "AdamW"}:
            raise ValueError("optimizer must be Adam or AdamW")
        if self.warmup_steps < 0 or not math.isfinite(self.gradient_clip) or self.gradient_clip < 0:
            raise ValueError("warmup_steps and gradient_clip must be non-negative")


@dataclass(frozen=True)
class DrcConfig:
    """Dynamic rollout-depth distribution for DRC training-state construction."""

    n_max: int = 16
    fixed_n: int | None = None

    def __post_init__(self) -> None:
        """Validate dynamic and optional fixed rollout budgets."""
        if self.n_max < 1 or self.fixed_n is not None and not 1 <= self.fixed_n <= self.n_max:
            raise ValueError("DRC requires n_max >= 1 and fixed_n in [1, n_max]")


@dataclass(frozen=True)
class CtcConfig:
    """Counterfactual transition pairing and frozen reference selection."""

    weight: float = 0.1
    steps: int = 1
    factual_model: str = "ema"
    target_model: str = "ema"
    shared_noise: bool = True

    def __post_init__(self) -> None:
        """Reject invalid weights, substeps or reference names."""
        if not math.isfinite(self.weight) or self.weight < 0 or self.steps < 1:
            raise ValueError("CTC requires finite weight >= 0 and steps >= 1")
        if self.factual_model not in {"online", "ema"} or self.target_model not in {"online", "ema"}:
            raise ValueError("CTC references must be online or ema")


@dataclass(frozen=True)
class EndpointLossConfig:
    """Clean-domain CoF discrepancy weights and spectral feature transform."""

    magnitude_weight: float = 0.7
    complex_weight: float = 0.3
    si_sdr_weight: float = 0.01
    compression: str = "power"

    def __post_init__(self) -> None:
        """Validate a finite, nonzero objective and supported compression."""
        weights = (self.magnitude_weight, self.complex_weight, self.si_sdr_weight)
        if any(not math.isfinite(value) or value < 0 for value in weights) or sum(weights) == 0:
            raise ValueError("Endpoint-loss weights must be finite, non-negative and nonzero in total")
        if self.compression not in {"power", "none"}:
            raise ValueError("compression must be power or none")
