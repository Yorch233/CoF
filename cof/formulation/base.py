"""Explicit stochastic/ordinary dynamics and the formulation composition contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Any, ClassVar

import torch
from torch import Tensor, nn

from cof.data.stft import StftTransform


def unsqueeze_xdim(value: Tensor, *, xdim: tuple[int, ...] | torch.Size) -> Tensor:
    """Broadcast a batch/time coefficient over every non-batch state dimension."""
    return value[(...,) + (None,) * len(xdim)]


class Dynamics(nn.Module, ABC):
    """Mathematical dynamics with device-aware, non-checkpointed fixed coefficients."""

    def __init__(
        self, training_target: str, loss_weight_type: str = "constant", device: str | torch.device = "cpu"
    ) -> None:
        """Set native prediction semantics and the reverse integration interval."""
        super().__init__()
        if loss_weight_type != "constant":
            raise ValueError("Only constant loss weighting is supported")
        self.register_buffer("_device_reference", torch.empty(0, device=device), persistent=False)
        self.training_target = training_target
        self.time_direction = "reverse"
        self.start_time = 1.0
        self.end_time = 0.0
        self.training_time_start = 1e-4
        self.training_time_end = 1.0

    @abstractmethod
    def marginal_sigma_square(self, time: Tensor) -> Tensor:
        """Return the variance coefficient at each batch time."""

    @abstractmethod
    def q_sample(self, t: Tensor, x0: Tensor, x1: Tensor) -> Tensor:
        """Sample an analytic path state between clean and noisy endpoints."""

    @abstractmethod
    def compute_label(self, xt: Tensor, t: Tensor, x0: Tensor, x1: Tensor) -> Tensor:
        """Return the supervised target in the native prediction domain."""

    @abstractmethod
    def compute_pred_x0(self, xt: Tensor, t: Tensor, x1: Tensor, net_out: Tensor) -> Tensor:
        """Recover the clean endpoint from the native network prediction."""

    @property
    def device(self) -> torch.device:
        """Return the device shared by all registered mathematical buffers."""
        return self._device_reference.device

    @property
    def is_forward(self) -> bool:
        """Return whether the integration interval uses increasing time."""
        return self.start_time < self.end_time

    def prior_sample(self, condition: Tensor, noise: Tensor | None = None) -> Tensor:
        """Return the observed endpoint for dynamics with a deterministic prior."""
        del noise
        return condition

    def marginal_sigma(self, time: Tensor) -> Tensor:
        """Return the standard deviation from the concrete marginal variance."""
        return self.marginal_sigma_square(time).sqrt()

    def compute_weight(self, time: Tensor) -> Tensor:
        """Return the constant weighting used by the current formulations."""
        return torch.ones_like(time)


class SDE(Dynamics, ABC):
    """Stochastic dynamics with explicit diffusion and conditional transitions."""

    @abstractmethod
    def drift(self, state: Tensor, time: Tensor) -> Tensor:
        """Return the underlying forward-process drift."""

    @abstractmethod
    def diffusion(self, time: Tensor) -> Tensor:
        """Return the forward-process diffusion coefficient at each time."""

    @abstractmethod
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
        """Advance the stochastic process using an optional explicit noise draw."""


class ODE(Dynamics, ABC):
    """Ordinary dynamics with explicit vector fields and deterministic transitions."""

    @abstractmethod
    def ode_step(self, state: Tensor, prediction: Tensor, time: Tensor, next_time: Tensor, condition: Tensor) -> Tensor:
        """Advance the deterministic process using its native prediction."""


class Formulation(nn.Module, ABC):
    """Compose dynamics, prediction semantics, training loss and solver factories.

    Concrete definitions own the mathematics. Training methods depend on this
    interface instead of recognizing individual formulation names.
    """

    #: Solver names that integrate a stochastic kernel and therefore consume noise.
    STOCHASTIC_SOLVERS: ClassVar[frozenset[str]] = frozenset()

    default_kwargs: ClassVar[dict[str, Any]] = {}
    path_name: ClassVar[str]
    preset_name: ClassVar[str]
    prediction_type: ClassVar[str]
    default_solver: ClassVar[str]
    allowed_solvers: ClassVar[tuple[str, ...]]
    capabilities: ClassVar[frozenset[str]] = frozenset({"supervised", "clean_prediction", "clean_transition"})

    def __init__(self, dynamics: Dynamics) -> None:
        """Register one mathematical object without duplicating its buffers."""
        super().__init__()
        self.dynamics = dynamics

    @property
    def inference_t_min(self) -> float:
        """Return the default sampler lower-time argument for this formulation."""
        return self.end_time

    @property
    def training_target(self) -> str:
        """Return the native network target defined by the dynamics."""
        return self.dynamics.training_target

    @property
    def device(self) -> torch.device:
        """Return the current mathematical-buffer device."""
        return self.dynamics.device

    @property
    def start_time(self) -> float:
        """Return the noisy integration endpoint."""
        return self.dynamics.start_time

    @property
    def end_time(self) -> float:
        """Return the clean integration endpoint."""
        return self.dynamics.end_time

    @property
    def training_time_start(self) -> float:
        """Return the lowest safe trainable network-call time."""
        return self.dynamics.training_time_start

    @property
    def training_time_end(self) -> float:
        """Return the highest permitted training time."""
        return self.dynamics.training_time_end

    @property
    def is_forward(self) -> bool:
        """Return the integration direction exposed to samplers."""
        return self.dynamics.is_forward

    def model_time(self, time: Tensor) -> Tensor:
        """Map integration time to the backbone's time conditioning."""
        return time

    def model_output(self, output: Tensor) -> Tensor:
        """Map backbone output to the native prediction domain."""
        return output

    def prior_sample(self, condition: Tensor, noise: Tensor | None = None) -> Tensor:
        """Sample the formulation's noisy endpoint."""
        return self.dynamics.prior_sample(condition, noise)

    def sample_training_state(self, clean: Tensor, noisy: Tensor, time: Tensor) -> Tensor:
        """Sample an analytic conditional training state without model rollout."""
        return self.dynamics.q_sample(t=time, x0=clean, x1=noisy)

    def training_target_at(self, state: Tensor, time: Tensor, clean: Tensor, noisy: Tensor) -> Tensor:
        """Construct the supervised target in the native prediction domain."""
        return self.dynamics.compute_label(xt=state, t=time, x0=clean, x1=noisy)

    def to_clean_prediction(self, prediction: Tensor, state: Tensor, time: Tensor, condition: Tensor) -> Tensor:
        """Convert a native prediction to a differentiable clean endpoint."""
        return self.dynamics.compute_pred_x0(xt=state, t=time, x1=condition, net_out=prediction)

    def compute_weight(self, time: Tensor) -> Tensor:
        """Return per-example objective weights without exposing a training host."""
        return self.dynamics.compute_weight(time)

    def reduce_loss(self, error: Tensor, time: Tensor, reduction: str) -> Tensor:
        """Apply the configured bin reduction, time weighting and batch mean."""
        if reduction not in {"mean", "sum"}:
            raise ValueError("reduction must be 'mean' or 'sum'")
        flattened = error.reshape(error.shape[0], -1)
        per_sample = flattened.mean(-1) if reduction == "mean" else 0.5 * flattened.sum(-1)
        return (per_sample * self.compute_weight(time)).mean()

    def step(
        self,
        state: Tensor,
        prediction: Tensor,
        time: Tensor,
        next_time: Tensor,
        condition: Tensor,
        *,
        solver: str,
        noise: Tensor | None = None,
    ) -> Tensor:
        """Dispatch a validated native-prediction transition to explicit dynamics."""
        if solver not in self.allowed_solvers:
            raise ValueError(f"{self.path_name} does not support {solver}")
        if solver in self.STOCHASTIC_SOLVERS:
            return self.dynamics.sde_step(state, prediction, time, next_time, condition, noise=noise)
        return self.dynamics.ode_step(state, prediction, time, next_time, condition)

    @abstractmethod
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
        """Advance a counterfactual transition driven by a supplied clean endpoint."""

    @abstractmethod
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
        """Compute the formulation's native supervised objective and its terms."""

    @abstractmethod
    def build_sampler(self, solver: str, predictor: Callable[..., Tensor]) -> Any:
        """Build a supported solver with an explicit prediction callable."""

    def sampling_time_grid(self, num_step: int, *, t_start: float, t_end: float, device: torch.device) -> Tensor | None:
        """Return an optional formulation-specific grid before generic spacing."""
        return None


PredictionFunction = Callable[[Tensor, Tensor | float], Tensor]
TimeGrid = Tensor | Sequence[float]


class Sampler:
    """Base numerical solver with shared timestep construction and step evaluation."""

    #: ``path_name`` a probability path must declare for this solver to accept it.
    expected_path_name: str | None = None

    def __init__(
        self,
        formulation: Formulation,
        predictor: PredictionFunction | None = None,
        device: str | torch.device | None = None,
        *,
        solver: str,
    ) -> None:
        """Bind the solver to one probability path and prediction function.

        Args:
            formulation: The probability path the solver samples.
            predictor: Callable mapping ``(state, timestep)`` to a prediction;
                may be supplied later for solvers that are constructed before
                their model.
            solver: Validated stochastic or ordinary integration algorithm.
            device: Device the sampling loop runs on; defaults to the path's
                own device.
        """
        self.formulation = formulation
        self.predictor = predictor
        self.device = torch.device(device if device is not None else formulation.device)
        self.nfe = 0
        self.solver = solver

    def integration_bounds(
        self,
        t_max: float | None,
        t_min: float | None,
        skip_type: str,
    ) -> tuple[float, float]:
        """Return the integration bounds of one sampling call.

        Args:
            t_max (float | None): Upper bound override; ``None`` uses the path start.
            t_min (float | None): Lower bound requested by the caller; ``None``
                leaves the endpoint to the path. Paths that treat their own
                lower time as a network-call bound override this method.
            skip_type (str): Time-grid spacing of the call.

        Returns:
            tuple[float, float]: The ``(start, end)`` integration bounds.
        """
        del skip_type
        start = float(self.formulation.start_time) if t_max is None else float(t_max)
        if t_min is None:
            return start, float(self.formulation.end_time)
        requested = float(t_min)
        if requested > 0.0 and self.formulation.is_forward:
            raise ValueError("t_min override is only supported for reverse-time paths")
        return start, max(float(self.formulation.end_time), requested)

    def get_time_steps(
        self,
        skip_type: str = "time_uniform",
        t_start: float = 1.0,
        t_end: float = 0.0,
        num_step: int = 20,
        time_grid: TimeGrid | None = None,
    ) -> Tensor:
        """Construct a monotone timestep schedule in the requested direction.

        A caller-supplied ``time_grid`` takes precedence; otherwise a path
        that exposes ``sampling_time_grid`` may replace the uniform grid, and
        only then are the built-in spacings used.

        Args:
            skip_type: Time-grid spacing, ``"time_uniform"`` or
                ``"time_quadratic"``.
            t_start: Schedule start (usually the larger time).
            t_end: Schedule end (usually the smaller time).
            num_step: Number of intervals; the schedule holds ``num_step + 1``
                values.
            time_grid: Explicit schedule overriding ``skip_type``; must match
                the direction and endpoints exactly.

        Returns:
            Tensor: The monotone timestep schedule of ``num_step + 1`` values.

        Raises:
            ValueError: If ``num_step`` is below one, the endpoints coincide,
                the explicit grid is malformed, or ``skip_type`` is unknown.
        """
        if num_step < 1:
            raise ValueError("num_step must be at least 1")
        descending = t_start > t_end
        if t_start == t_end:
            raise ValueError("t_start and t_end must be different")
        if time_grid is not None:
            grid = torch.as_tensor(time_grid, device=self.device, dtype=torch.float32)
            if grid.ndim != 1 or grid.numel() != num_step + 1:
                raise ValueError(f"time_grid must contain exactly {num_step + 1} values")
            if not torch.isfinite(grid).all():
                raise ValueError("time_grid must contain only finite values")
            tolerance = 1e-5
            start_tensor = torch.tensor(t_start, device=self.device)
            end_tensor = torch.tensor(t_end, device=self.device)
            if not torch.isclose(grid[0], start_tensor, atol=tolerance, rtol=0.0):
                raise ValueError(f"time_grid must start at {t_start}")
            if not torch.isclose(grid[-1], end_tensor, atol=tolerance, rtol=0.0):
                raise ValueError(f"time_grid must end at {t_end}")
            monotone = torch.all(grid[:-1] > grid[1:]) if descending else torch.all(grid[:-1] < grid[1:])
            if not monotone:
                direction = "descending" if descending else "ascending"
                raise ValueError(f"time_grid must be strictly {direction}")
            return grid
        custom_grid = getattr(self.formulation, "sampling_time_grid", None)
        if custom_grid is not None and skip_type == "time_uniform":
            path_grid = custom_grid(num_step, t_start=t_start, t_end=t_end, device=self.device)
            if path_grid is not None:
                return path_grid
        if skip_type == "time_uniform":
            return torch.linspace(t_start, t_end, num_step + 1, device=self.device)
        if skip_type == "time_quadratic":
            if t_start < 0.0 or t_end < 0.0:
                raise ValueError("time_quadratic requires non-negative endpoints")
            roots = torch.linspace(t_start**0.5, t_end**0.5, num_step + 1, device=self.device)
            return roots.square()
        raise ValueError(f"Unsupported skip_type {skip_type!r}; choose 'time_uniform' or 'time_quadratic'")

    @staticmethod
    def _stack_trajectory(values: list[Tensor]) -> Tensor:
        """Stack values in traversal order (noisy endpoint first)."""
        return torch.stack([value.detach().to("cpu", non_blocking=True) for value in values], dim=1)

    def _require_predictor(self) -> PredictionFunction:
        """Return the bound prediction function, refusing to sample without one.

        Raises:
            RuntimeError: If no prediction function was supplied.
        """
        if self.predictor is None:
            raise RuntimeError("A prediction function is required for sampling")
        return self.predictor

    def _evaluate(self, state: Tensor, timestep: Tensor) -> Tensor:
        """Evaluate the prediction function once and count the model call."""
        prediction = self._require_predictor()(state, timestep)
        self.nfe += 1
        return prediction

    @torch.no_grad()
    def sampling(
        self,
        x: Tensor,
        num_step: int = 4,
        skip_type: str = "time_uniform",
        t_max: float | None = None,
        t_min: float | None = None,
        time_grid: Tensor | list[float] | None = None,
        condition: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Integrate one explicit predictor call per interval and return CPU traces."""
        state = x.to(self.device)
        condition = state if condition is None else condition.to(self.device)
        start, end = self.integration_bounds(t_max, t_min, skip_type)
        times = self.get_time_steps(skip_type, start, end, num_step, time_grid)
        states, predictions = [state], []
        self.nfe = 0
        for index in range(num_step):
            prediction = self._evaluate(state, times[index])
            state = self.formulation.step(
                state,
                prediction,
                times[index].expand(state.shape[0]),
                times[index + 1].expand(state.shape[0]),
                condition,
                solver=self.solver,
            )
            states.append(state)
            predictions.append(prediction)
        return state, self._stack_trajectory(states), self._stack_trajectory(predictions)
