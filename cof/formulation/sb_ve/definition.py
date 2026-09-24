"""Variance-exploding bridge dynamics, clean prediction and supervised loss."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, ClassVar

import torch
import torch.nn.functional as functional
from torch import Tensor

from cof.data.stft import StftTransform
from cof.formulation.base import SDE, Formulation, unsqueeze_xdim


class SBVESDE(SDE):
    """Explicit SB-VE dynamics, including stochastic and probability-flow updates."""

    def __init__(self, c: float = 0.4, k: float = 2.6, device: str | torch.device = "cpu") -> None:
        """Initialize positive VE coefficients and fixed clean-data prediction."""
        if not math.isfinite(c) or c <= 0 or not math.isfinite(k) or k <= 1:
            raise ValueError("SB-VE requires finite c > 0 and k > 1")
        super().__init__(training_target="data", device=device)
        self.register_buffer("c", torch.tensor(c, device=device), persistent=False)
        self.register_buffer("k", torch.tensor(k, device=device), persistent=False)

    def drift(self, state: Tensor, time: Tensor) -> Tensor:
        """Return the zero forward drift of the underlying VE process."""
        return torch.zeros_like(state)

    def diffusion(self, time: Tensor) -> Tensor:
        """Return sqrt(c) * k**time for the underlying VE process."""
        return self.c.sqrt() * self.k.pow(time)

    def marginal_alpha(self, time: Tensor) -> Tensor:
        """Return the unit signal coefficient of the VE process."""
        return torch.ones_like(time)

    def marginal_sigma_square(self, time: Tensor) -> Tensor:
        """Return c * (k**(2*time) - 1) / (2*log(k))."""
        return self.c * (self.k ** (2 * time) - 1) / (2 * torch.log(self.k))

    def compute_label(self, xt: Tensor, t: Tensor, x0: Tensor, x1: Tensor) -> Tensor:
        """Return the supervised clean endpoint."""
        return x0

    def compute_pred_x0(self, xt: Tensor, t: Tensor, x1: Tensor, net_out: Tensor) -> Tensor:
        """Return the native data prediction without an additional model call."""
        return net_out

    def terminal_time(self, t: torch.Tensor) -> torch.Tensor:
        """Return the terminal time, often normalized to 1.

        Args:
            t: Time steps.

        Returns:
            torch.Tensor: Tensor of ones with the same shape as ``t``.
        """
        return torch.ones_like(t, device=self.device)

    def marginal_alpha_bar(self, t: torch.Tensor) -> torch.Tensor:
        """Return the marginal alpha_bar coefficient at time ``t``.

        Often used in the context of bridging between initial and final states.

        Args:
            t: Time steps.

        Returns:
            torch.Tensor: Alpha_bar values at time ``t``.
        """
        return self.marginal_alpha(t) / self.marginal_alpha(self.terminal_time(t))

    def marginal_sigma_bar(self, t: torch.Tensor) -> torch.Tensor:
        """Return the marginal sigma_bar (standard deviation) at time ``t``.

        Args:
            t: Time steps.

        Returns:
            torch.Tensor: Sigma_bar values at time ``t``.
        """
        return torch.sqrt(self.marginal_sigma_bar_square(t))

    def marginal_sigma_bar_square(self, t: torch.Tensor) -> torch.Tensor:
        """Return the marginal sigma_bar squared (variance) at time ``t``.

        Args:
            t: Time steps.

        Returns:
            torch.Tensor: Sigma_bar squared values at time ``t``.
        """
        return self.marginal_sigma_square(self.terminal_time(t)) - self.marginal_sigma_square(t)

    def bridge_mean(self, clean: torch.Tensor, t: torch.Tensor, noisy: torch.Tensor) -> torch.Tensor:
        """Return the analytic bridge marginal mean for an SB path.

        Args:
            clean: Clean endpoint ``x0``.
            t: Bridge time.
            noisy: Noisy endpoint ``x1``.

        Returns:
            torch.Tensor: The marginal mean of ``q(x_t | x0, x1)``.
        """
        _, *xdim = clean.shape
        terminal_variance = self.marginal_sigma_square(self.terminal_time(t))
        weight_clean = self.marginal_alpha(t) * self.marginal_sigma_bar_square(t) / terminal_variance
        weight_noisy = self.marginal_alpha_bar(t) * self.marginal_sigma_square(t) / terminal_variance
        return unsqueeze_xdim(weight_clean, xdim=xdim) * clean + unsqueeze_xdim(weight_noisy, xdim=xdim) * noisy

    def q_sample(self, t: torch.Tensor, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        """Sample from the conditional distribution ``q(x_t | x_0, x_1)``.

        Generates intermediate noisy states given the initial (x0) and final
        (x1) states.

        Args:
            t: Time steps.
            x0: Initial condition tensor (e.g., clean data).
            x1: Terminal condition tensor (e.g., noisy/observed data).

        Returns:
            torch.Tensor: Sampled ``x_t`` values (intermediate noisy states).
        """
        _, *xdim = x0.shape
        # Calculate weights and variance for the sampling distribution
        w_x0 = (
            self.marginal_alpha(t)
            * self.marginal_sigma_bar_square(t)
            / self.marginal_sigma_square(self.terminal_time(t))
        )
        w_x1 = (
            self.marginal_alpha_bar(t)
            * self.marginal_sigma_square(t)
            / self.marginal_sigma_square(self.terminal_time(t))
        )
        var = (
            self.marginal_alpha(t) ** 2
            * self.marginal_sigma_bar_square(t)
            * self.marginal_sigma_square(t)
            / self.marginal_sigma_square(self.terminal_time(t))
        )
        # Expand dimensions for broadcasting
        w_x0, w_x1, var = (unsqueeze_xdim(value, xdim=xdim) for value in (w_x0, w_x1, var))
        # Compute mean and sample
        mean = w_x1 * x1 + w_x0 * x0
        return mean + var.sqrt() * torch.randn_like(mean)

    def first_order_sde_sampling(
        self,
        xt: torch.Tensor,
        x0: torch.Tensor,
        t: torch.Tensor,
        t_prev: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply the conditional Gaussian SB transition for SDE sampling.

        Computes the previous state ``x_{t_prev}`` given the current state
        ``xt``, the initial state ``x0``, and the time steps.

        Args:
            xt: Current state tensor at time ``t``.
            x0: Initial condition tensor (predicted or known).
            t: Current time step.
            t_prev: Previous time step.
            noise: Optional standardized transition noise. Supplying it
                enables coupled or antithetic transitions; the default
                uses independent Gaussian transition noise.

        Returns:
            torch.Tensor: Estimated state ``x_{t_prev}``.

        Raises:
            ValueError: If a supplied ``noise`` tensor does not match the
                shape of the sampled state.
        """
        _, *xdim = x0.shape
        # Calculate weights and variance for the transition
        w_xt = (
            self.marginal_alpha(t_prev)
            * self.marginal_sigma_square(t_prev)
            / (self.marginal_alpha(t) * self.marginal_sigma_square(t))
        )
        w_x0 = self.marginal_alpha(t_prev) * (1 - self.marginal_sigma_square(t_prev) / self.marginal_sigma_square(t))
        var = (
            self.marginal_alpha(t_prev) ** 2
            * self.marginal_sigma_square(t_prev)
            * (1 - self.marginal_sigma_square(t_prev) / self.marginal_sigma_square(t))
        )
        # Expand dimensions for broadcasting
        w_x0, w_xt, var = (unsqueeze_xdim(value, xdim=xdim) for value in (w_x0, w_xt, var))
        # Compute deterministic part of the update
        x_prev = w_xt * xt + w_x0 * x0
        # Add stochastic (diffusion) noise only away from the clean endpoint.
        # ``t_prev`` may contain one independently sampled time per example.
        nonterminal = unsqueeze_xdim(t_prev > 0, xdim=xdim)
        standardized_noise = torch.randn_like(x_prev) if noise is None else noise
        if standardized_noise.shape != x_prev.shape:
            raise ValueError("noise must have the same shape as the sampled state")
        diffusion = var.clamp_min(0).sqrt() * standardized_noise
        return x_prev + nonterminal * diffusion

    def first_order_ode_sampling(
        self,
        x1: torch.Tensor,
        xt: torch.Tensor,
        x0: torch.Tensor,
        t: torch.Tensor,
        t_prev: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the first-order SB-ODE update from time ``t`` to ``t_prev``.

        The coefficients follow Table 2 of Jukic et al., "Schrodinger
        Bridge for Generative Speech Enhancement". At the terminal time the
        individual ``xt`` and ``x1`` coefficients are singular even though
        ``xt == x1`` and their sum has a finite limit, so that first step is
        evaluated in its analytic limiting form.

        Args:
            x1: Terminal (noisy) condition tensor.
            xt: Current bridge state.
            x0: Clean endpoint prediction driving the step.
            t: Current time step.
            t_prev: Previous (target) time step.

        Returns:
            torch.Tensor: Estimated state ``x_{t_prev}``.
        """
        _, *xdim = x0.shape
        terminal_time = self.terminal_time(t)
        alpha_prev = self.marginal_alpha(t_prev)
        alpha_terminal = self.marginal_alpha(terminal_time)
        sigma_terminal_square = self.marginal_sigma_square(terminal_time)

        terminal = t == terminal_time
        terminal_w_x0 = alpha_prev * self.marginal_sigma_bar_square(t_prev) / sigma_terminal_square
        terminal_w_x1 = alpha_prev * self.marginal_sigma_square(t_prev) / (alpha_terminal * sigma_terminal_square)

        if torch.all(terminal):
            terminal_w_x0, terminal_w_x1 = (
                unsqueeze_xdim(value, xdim=xdim) for value in (terminal_w_x0, terminal_w_x1)
            )
            return terminal_w_x0 * x0 + terminal_w_x1 * x1

        sigma = self.marginal_sigma(t)
        sigma_prev = self.marginal_sigma(t_prev)
        sigma_bar = self.marginal_sigma_bar(t)
        sigma_bar_prev = self.marginal_sigma_bar(t_prev)

        safe_sigma_bar = torch.where(terminal, torch.ones_like(sigma_bar), sigma_bar)
        w_xt = alpha_prev * sigma_prev * sigma_bar_prev / (self.marginal_alpha(t) * sigma * safe_sigma_bar)
        w_x0 = (
            alpha_prev
            / sigma_terminal_square
            * (self.marginal_sigma_bar_square(t_prev) - sigma_bar * sigma_prev * sigma_bar_prev / sigma)
        )
        w_x1 = (
            alpha_prev
            / (alpha_terminal * sigma_terminal_square)
            * (self.marginal_sigma_square(t_prev) - sigma * sigma_prev * sigma_bar_prev / safe_sigma_bar)
        )
        w_xt = torch.where(terminal, torch.zeros_like(w_xt), w_xt)
        w_x0 = torch.where(terminal, terminal_w_x0, w_x0)
        w_x1 = torch.where(terminal, terminal_w_x1, w_x1)
        w_x0, w_xt, w_x1 = (unsqueeze_xdim(value, xdim=xdim) for value in (w_x0, w_xt, w_x1))
        return w_xt * xt + w_x0 * x0 + w_x1 * x1

    def score_from_data_prediction(
        self,
        xt: torch.Tensor,
        t: torch.Tensor,
        x1: torch.Tensor,
        prediction: torch.Tensor,
    ) -> torch.Tensor:
        """Convert a clean-data prediction into the conditional bridge score.

        Args:
            xt: Bridge state the score is evaluated at.
            t: Bridge time of the state.
            x1: Noisy endpoint ``x1``.
            prediction: Clean-data prediction to convert.

        Returns:
            torch.Tensor: The conditional bridge score.
        """
        _, *xdim = xt.shape
        w_x0 = (
            self.marginal_alpha(t)
            * self.marginal_sigma_bar_square(t)
            / self.marginal_sigma_square(self.terminal_time(t))
        )
        w_x1 = (
            self.marginal_alpha_bar(t)
            * self.marginal_sigma_square(t)
            / self.marginal_sigma_square(self.terminal_time(t))
        )
        variance = (
            self.marginal_alpha(t) ** 2
            * self.marginal_sigma_bar_square(t)
            * self.marginal_sigma_square(t)
            / self.marginal_sigma_square(self.terminal_time(t))
        )
        mean = unsqueeze_xdim(w_x0, xdim=xdim) * prediction + unsqueeze_xdim(w_x1, xdim=xdim) * x1
        return (mean - xt) / unsqueeze_xdim(variance.clamp_min(torch.finfo(variance.dtype).eps), xdim=xdim)

    def sde_step(
        self,
        state: Tensor,
        prediction: Tensor,
        time: Tensor,
        next_time: Tensor,
        condition: Tensor,
        *,
        noise: Tensor | None = None,
    ) -> Tensor:
        """Advance the SB reverse stochastic kernel with explicit optional noise."""
        return self.first_order_sde_sampling(state, prediction, time, next_time, noise=noise)

    def ode_step(self, state: Tensor, prediction: Tensor, time: Tensor, next_time: Tensor, condition: Tensor) -> Tensor:
        """Advance the SB probability-flow update including its terminal limit."""
        return self.first_order_ode_sampling(condition, state, prediction, time, next_time)


class SBVEFormulation(Formulation):
    """Compose SB-VE dynamics, data prediction, sampling and the base objective."""

    path_name: ClassVar[str] = "SBVE"
    preset_name: ClassVar[str] = "SB-VE"
    prediction_type: ClassVar[str] = "data"
    default_solver: ClassVar[str] = "SB_SDE_Solver"
    allowed_solvers: ClassVar[tuple[str, ...]] = ("SB_SDE_Solver", "SB_ODE_Solver")
    STOCHASTIC_SOLVERS: ClassVar[frozenset[str]] = frozenset({"SB_SDE_Solver"})

    def __init__(self, c: float = 0.4, k: float = 2.6, device: str | torch.device = "cpu", **kwargs: Any) -> None:
        """Construct the named stochastic dynamics once.

        Args:
            c: SB-VE diffusion schedule coefficient.
            k: SB-VE diffusion schedule exponent.
            device: Mathematical-buffer device.
            **kwargs: ``time_loss_weight`` weights the auxiliary waveform term
                of ``training_loss`` (default ``1e-3``); other keys are
                ignored.
        """
        super().__init__(SBVESDE(c=c, k=k, device=device))
        self.time_loss_weight = float(kwargs.get("time_loss_weight", 1e-3))

    @property
    def sde(self) -> SBVESDE:
        """Expose the explicitly typed mathematical SDE."""
        return self.dynamics

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
        """Feed the supplied endpoint into the selected SB transition."""
        return self.step(state, clean, time, next_time, condition, solver=solver, noise=noise)

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
        """Compute dataset-domain complex regression plus waveform L1 per batch."""
        weight = self.time_loss_weight if time_loss_weight is None else float(time_loss_weight)
        target = self.training_target_at(state, time, clean, noisy)
        prediction_loss = self.reduce_loss((prediction - target).abs().square(), time, reduction)
        clean_prediction = self.to_clean_prediction(prediction, state, time, noisy)
        predicted_audio = transform.istft(clean_prediction.squeeze(1)).squeeze(1)
        clean_audio = transform.istft(clean.squeeze(1)).squeeze(1)
        audio_loss = functional.l1_loss(predicted_audio, clean_audio, reduction="sum") / clean.shape[0]
        return {
            "loss": prediction_loss + weight * audio_loss,
            "prediction_loss": prediction_loss,
            "time_loss": audio_loss,
        }

    def build_sampler(self, solver: str, predictor: Callable[..., Tensor]) -> Any:
        """Build the requested SB solver without importing method or pipeline code."""
        from cof.formulation.sb_ve.sampling import SB_ODE_Solver, SB_SDE_Solver

        if solver not in self.allowed_solvers:
            raise ValueError(f"SBVE does not support {solver}")
        classes = {"SB_SDE_Solver": SB_SDE_Solver, "SB_ODE_Solver": SB_ODE_Solver}
        return classes[solver](self, predictor)
