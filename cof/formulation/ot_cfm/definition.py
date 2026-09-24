"""OT-CFM ordinary dynamics, native vector targets and supervised loss."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, ClassVar

import torch
from torch import Tensor

from cof.data.stft import StftTransform
from cof.formulation.base import ODE, Formulation, unsqueeze_xdim


class OTCFMODE(ODE):
    """FlowSE's noisy-conditioned optimal-transport CFM path.

    The network output for this path is a conditional vector field.  With
    reverse bridge time ``t`` (noisy endpoint at one and clean endpoint at
    zero),
    the conditional Gaussian path is

    ``mu_t = (1 - t) * clean + t * noisy`` and
    ``sigma_t = (1 - t) * sigma_min + t * sigma_max``.

    This is the time convention used by the official FlowSE implementation and
    by the SB-VE path in this project. FlowSE uses ``sigma_min=0`` and samples
    training times from ``[t_min, 1]``.

    Sampling is deterministic and runs through
    :class:`~cof.formulation.ot_cfm.sampling.OTCFM_ODE_Solver`; no stochastic
    solver is accepted.
    """

    path_name: ClassVar[str] = "OTCFM"
    default_solver: ClassVar[str] = "OTCFM_ODE_Solver"
    allowed_solvers: ClassVar[tuple[str, ...]] = ("OTCFM_ODE_Solver",)

    def __init__(
        self,
        sigma_max: float = 0.487,
        sigma_min: float = 0.0,
        t_min: float = 0.03,
        training_target: str = "vector",
        loss_weight_type: str = "constant",
        device: str | torch.device = "cpu",
    ) -> None:
        """Initialize the OT-CFM path with its FlowSE parameters."""
        super().__init__(training_target=training_target, loss_weight_type=loss_weight_type, device=device)
        if training_target != "vector":
            raise ValueError("OT-CFM requires training_target='vector'")
        if (
            not all(math.isfinite(value) for value in (sigma_min, sigma_max))
            or sigma_min < 0.0
            or sigma_max <= sigma_min
            or sigma_max <= 0.0
        ):
            raise ValueError("OT-CFM requires sigma_max > sigma_min >= 0")
        if not 0.0 <= t_min < 1.0:
            raise ValueError("OT-CFM requires 0 <= t_min < 1")
        self.time_direction = "reverse"
        self.start_time = 1.0
        self.end_time = 0.0
        self.training_time_start = float(t_min)
        self.training_time_end = 1.0
        self.sigma_max = float(sigma_max)
        self.sigma_min = float(sigma_min)
        self.t_min = float(t_min)

    def marginal_alpha(self, t: torch.Tensor) -> torch.Tensor:
        """Return alpha(t) for the OT path: constant 1.

        Args:
            t: Time steps.

        Returns:
            torch.Tensor: Alpha values (all ones).
        """
        return torch.ones_like(t)

    def marginal_sigma_square(self, t: torch.Tensor) -> torch.Tensor:
        """Return sigma^2(t): the squared linear sigma interpolation.

        Args:
            t: Time steps.

        Returns:
            torch.Tensor: Sigma squared values.
        """
        sigma = self.sigma_min + t * (self.sigma_max - self.sigma_min)
        return sigma.square()

    def bridge_mean(self, clean: torch.Tensor, t: torch.Tensor, noisy: torch.Tensor) -> torch.Tensor:
        """Return the linear OT interpolation mean between the endpoints.

        Args:
            clean: Clean endpoint ``x0``.
            t: Bridge time (1 at the noisy endpoint, 0 at the clean endpoint).
            noisy: Noisy endpoint ``x1``.

        Returns:
            torch.Tensor: The conditional path mean ``mu_t``.
        """
        weight = unsqueeze_xdim(t, xdim=clean.shape[1:])
        return (1.0 - weight) * clean + weight * noisy

    def q_sample(self, t: torch.Tensor, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        """Sample from the conditional Gaussian ``q(x_t | x_0, x_1)``.

        Args:
            t: Time steps.
            x0: Clean endpoint tensor.
            x1: Noisy endpoint tensor.

        Returns:
            torch.Tensor: Sampled intermediate states ``x_t``.
        """
        mean = self.bridge_mean(x0, t, x1)
        sigma = unsqueeze_xdim(self.marginal_sigma(t), xdim=x0.shape[1:])
        return mean + sigma * torch.randn_like(mean)

    def prior_sample(self, condition: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        """Sample the noisy endpoint: the condition plus ``sigma_max`` noise.

        Args:
            condition: The observed noisy input the trajectory is conditioned on.
            noise: Optional standardized noise; defaults to fresh Gaussian
                noise with the condition's shape.

        Returns:
            torch.Tensor: The noisy endpoint sample.
        """
        sigma = condition.new_tensor(self.sigma_max)
        return condition + sigma * (torch.randn_like(condition) if noise is None else noise)

    def sampling_time_grid(
        self,
        num_step: int,
        *,
        t_start: float,
        t_end: float,
        device: torch.device,
    ) -> torch.Tensor | None:
        """Return FlowSE's uniform-call grid with a final ``t_min`` step."""
        if not math.isclose(t_start, self.start_time) or not math.isclose(t_end, self.end_time):
            return None
        if self.t_min == 0.0:
            return torch.linspace(1.0, 0.0, num_step + 1, device=device)
        call_times = torch.linspace(1.0, self.t_min, num_step, device=device)
        return torch.cat((call_times, call_times.new_zeros(1)))

    def vector_field_target(
        self,
        xt: torch.Tensor,
        t: torch.Tensor,
        clean: torch.Tensor,
        noisy: torch.Tensor,
    ) -> torch.Tensor:
        """Return the analytic FlowSE conditional vector-field target."""
        mean = self.bridge_mean(clean, t, noisy)
        sigma = self.marginal_sigma(t)
        sigma_dot = t.new_tensor(self.sigma_max - self.sigma_min)
        epsilon = torch.finfo(sigma.dtype).eps
        ratio = torch.where(sigma > epsilon, sigma_dot / sigma, torch.zeros_like(sigma))
        return noisy - clean + unsqueeze_xdim(ratio, xdim=clean.shape[1:]) * (xt - mean)

    def compute_label(
        self,
        xt: torch.Tensor,
        t: torch.Tensor,
        x0: torch.Tensor,
        x1: torch.Tensor,
    ) -> torch.Tensor:
        """Use the conditional vector field as the supervised target."""
        return self.vector_field_target(xt, t, clean=x0, noisy=x1)

    def compute_pred_x0(
        self,
        xt: torch.Tensor,
        t: torch.Tensor,
        x1: torch.Tensor | None = None,
        net_out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Recover the clean endpoint from a FlowSE vector-field prediction.

        The algebraically stable form avoids dividing by ``sigma_t`` near the
        clean endpoint. For FlowSE's default ``sigma_min=0`` it reduces to
        ``xt - t * net_out`` exactly.
        """
        if x1 is None:
            raise ValueError("OT-CFM clean-endpoint reconstruction requires x1")
        if net_out is None:
            raise ValueError("OT-CFM clean-endpoint reconstruction requires net_out")
        sigma = unsqueeze_xdim(self.marginal_sigma(t), xdim=xt.shape[1:])
        sigma_delta = xt.new_tensor(self.sigma_max - self.sigma_min)
        return (sigma_delta * xt + self.sigma_min * x1 - sigma * net_out) / self.sigma_max

    def vector_field_from_data_prediction(
        self,
        xt: torch.Tensor,
        t: torch.Tensor,
        x1: torch.Tensor,
        prediction: torch.Tensor,
    ) -> torch.Tensor:
        """Convert a clean-endpoint prediction to the FlowSE vector field."""
        return self.vector_field_target(xt, t, clean=prediction, noisy=x1)

    def first_order_step(
        self,
        xt: torch.Tensor,
        prediction: torch.Tensor,
        t: torch.Tensor,
        t_next: torch.Tensor,
        condition: torch.Tensor,
        *,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Advance one FlowSE Euler step along the predicted vector field.

        Args:
            xt: Current bridge state.
            prediction: Path-native prediction at ``xt`` (already the vector
                field for OT-CFM).
            t: Current time step.
            t_next: Target time step.
            condition: Noisy endpoint the trajectory is conditioned on.
            noise: Unused; accepted for signature compatibility.

        Returns:
            torch.Tensor: The advanced bridge state.
        """
        del noise
        # NOTE: OTCFM networks already predict the vector field; no endpoint
        # conversion is applied at sampling time.
        drift = prediction
        return xt + unsqueeze_xdim(t_next - t, xdim=xt.shape[1:]) * drift

    def sde_step(
        self,
        xt: torch.Tensor,
        prediction: torch.Tensor,
        t: torch.Tensor,
        t_next: torch.Tensor,
        condition: torch.Tensor,
        *,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Reject stochastic sampling: OT-CFM is deterministic only."""
        del xt, prediction, t, t_next, condition, noise
        raise ValueError("OT-CFM supports deterministic ODE sampling only")

    def ode_step(
        self,
        xt: torch.Tensor,
        prediction: torch.Tensor,
        t: torch.Tensor,
        t_next: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """Advance one deterministic ODE step."""
        return self.first_order_step(xt, prediction, t, t_next, condition)


class OTCFMFormulation(Formulation):
    """Compose the conditional Gaussian path with explicit OT-CFM ODE dynamics."""

    path_name: ClassVar[str] = "OTCFM"
    default_kwargs: ClassVar[dict[str, Any]] = {"sigma_max": 0.487, "sigma_min": 0.0, "t_min": 0.03}
    preset_name: ClassVar[str] = "OT-CFM"
    prediction_type: ClassVar[str] = "vector"
    default_solver: ClassVar[str] = "OTCFM_ODE_Solver"
    allowed_solvers: ClassVar[tuple[str, ...]] = ("OTCFM_ODE_Solver",)

    def __init__(
        self,
        sigma_max: float = 0.487,
        sigma_min: float = 0.0,
        t_min: float = 0.03,
        device: str | torch.device = "cpu",
        **kwargs: Any,
    ) -> None:
        """Construct the flow with explicit variance and safe network-call bounds.

        The native objective is pure vector-field regression with no auxiliary
        time-domain term: ``time_loss_weight`` is accepted through ``kwargs``
        for configuration compatibility and has no effect on this path.
        """
        super().__init__(OTCFMODE(sigma_max=sigma_max, sigma_min=sigma_min, t_min=t_min, device=device))

    @property
    def inference_t_min(self) -> float:
        """Use the safe final vector-field call before stepping to zero."""
        return self.ode.t_min

    @property
    def ode(self) -> OTCFMODE:
        """Expose the explicitly typed mathematical ODE."""
        return self.dynamics

    @property
    def t_min(self) -> float:
        """Return the lowest time at which the network may be evaluated."""
        return self.ode.t_min

    def transition_from_clean(
        self,
        state: Tensor,
        clean: Tensor,
        time: Tensor,
        next_time: Tensor,
        condition: Tensor,
        *,
        solver: str,
        noise: Tensor | None = None,
    ) -> Tensor:
        """Convert an endpoint into its conditional field before ODE advancement."""
        field = self.ode.vector_field_from_data_prediction(state, time, condition, clean)
        return self.step(state, field, time, next_time, condition, solver=solver)

    def training_loss(
        self,
        prediction: Tensor,
        state: Tensor,
        time: Tensor,
        clean: Tensor,
        noisy: Tensor,
        *,
        transform: StftTransform,
        reduction: str = "sum",
        time_loss_weight: float | None = None,
    ) -> dict[str, Tensor]:
        """Regress the conditional vector field with explicit per-bin reduction."""
        target = self.training_target_at(state, time, clean, noisy)
        loss = self.reduce_loss((prediction - target).abs().square(), time, reduction)
        return {"loss": loss, "prediction_loss": loss}

    def sampling_time_grid(self, num_step: int, *, t_start: float, t_end: float, device: torch.device) -> Tensor | None:
        """Preserve the final safe-time network call followed by the clean endpoint."""
        return self.ode.sampling_time_grid(num_step, t_start=t_start, t_end=t_end, device=device)

    def build_sampler(self, solver: str, predictor: Callable[..., Tensor]) -> Any:
        """Build the supported deterministic flow sampler."""
        from cof.formulation.ot_cfm.sampling import OTCFM_ODE_Solver

        if solver not in self.allowed_solvers:
            raise ValueError(f"OTCFM does not support {solver}")
        return OTCFM_ODE_Solver(self, predictor)
