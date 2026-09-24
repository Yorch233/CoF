"""SB-VE stochastic and probability-flow trajectory sampling."""

from __future__ import annotations

import torch

from cof.formulation.base import Formulation, PredictionFunction, Sampler


class SBVESampler(Sampler):
    """Integrate the SB-VE kernel selected by its solver protocol."""


class SB_SDE_Solver(SBVESampler):
    """Stochastic SB-VE solver; the formulation's canonical sampler."""

    def __init__(
        self,
        formulation: Formulation,
        predictor: PredictionFunction | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        """Bind the stochastic kernel under its explicit solver name."""
        super().__init__(formulation, predictor, device, solver="SB_SDE_Solver")


class SB_ODE_Solver(SBVESampler):
    """Probability-flow SB-VE solver integrating the same bridge deterministically."""

    def __init__(
        self,
        formulation: Formulation,
        predictor: PredictionFunction | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        """Bind the deterministic kernel under its explicit solver name."""
        super().__init__(formulation, predictor, device, solver="SB_ODE_Solver")
