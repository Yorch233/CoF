"""Complete PyTorch speech-enhancement model with explicit mathematical composition.

The model constructs its named backbone and formulation, performs native or clean
prediction, owns its audio transform, and provides self-describing safetensors
checkpoints. Training methods and Lightning are deliberately external.
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import Any, Self

import torch
from safetensors import safe_open
from safetensors.torch import load_model, save_model
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from cof.backbone.registry import BackboneRegister
from cof.config.manager import Config
from cof.data.stft import StftTransform
from cof.formulation.registry import path_type_for


def _checkpoint_constructor(path: Path) -> dict[str, Any] | None:
    """Return the self-describing constructor mapping, or ``None`` for legacy files."""
    with safe_open(path, framework="pt", device="cpu") as stream:
        metadata = stream.metadata() or {}
    if metadata.get("schema_version") != "1" or "cof_model" not in metadata:
        return None
    return json.loads(metadata["cof_model"])


def _conflict_keys(recorded: dict[str, Any], expected: dict[str, Any]) -> list[str]:
    """Return the identity keys whose recorded values differ from ``expected``."""
    identity = ("formulation", "backbone", "backbone_kwargs", "formulation_kwargs", "transform_kwargs")
    return [key for key in identity if recorded.get(key) != expected[key]]


def _constructor_from_config(config: Config | dict[str, Any]) -> dict[str, Any]:
    """Build the constructor mapping from a run configuration.

    Reads the nested groups (``formulation``/``model``/``data``) and merges
    the formulation registry's default kwargs, mirroring the fields a
    self-describing checkpoint records.
    """
    recorded = config if isinstance(config, Config) else Config(config)
    formulation_name = str(recorded.get("formulation.name"))
    return {
        "formulation": path_type_for(formulation_name).path_name,
        "backbone": recorded.get("model.backbone", "ncsnpp_base"),
        "backbone_kwargs": Config.unwrap(recorded.get("model.backbone_kwargs") or {}),
        "formulation_kwargs": {
            **path_type_for(formulation_name).default_kwargs,
            **Config.unwrap(recorded.get("formulation.kwargs") or {}),
        },
        "transform_kwargs": StftTransform.from_config(recorded).to_config(),
    }


class GenerativeModel4SE(nn.Module):
    """Compose a backbone, mathematical formulation and waveform representation.

    ``forward`` returns the formulation-native prediction. ``predict_clean``
    converts that prediction without detaching it. Sampling uses a fresh solver
    and an explicit conditioning closure, so nested calls share no sampling state.
    """

    def __init__(
        self,
        formulation: str,
        backbone: str = "ncsnpp_base",
        *,
        backbone_kwargs: dict[str, Any] | None = None,
        formulation_kwargs: dict[str, Any] | None = None,
        transform_kwargs: dict[str, Any] | None = None,
        sampling_solver: str = "AUTO",
        sampling_num_steps: int = 4,
        sampling_skip_type: str = "time_uniform",
        gradient_checkpointing: bool = False,
        device: str | torch.device = "cpu",
    ) -> None:
        """Construct named components and validate their default sampling protocol.

        Args:
            formulation: Registered probability formulation name.
            backbone: Registered neural architecture name.
            backbone_kwargs: Architecture-specific constructor arguments.
            formulation_kwargs: Mathematical definition parameters.
            transform_kwargs: Instance-local STFT geometry and warping parameters.
            sampling_solver: Supported solver name, or the formulation default.
            sampling_num_steps: Default number of integration intervals.
            sampling_skip_type: Default time-grid spacing.
            gradient_checkpointing: Recompute supported backbone forwards in backward.
            device: Initial parameter and dynamics-buffer device.

        Raises:
            ValueError: If the sampling protocol is invalid.
        """
        super().__init__()
        self.backbone_name = backbone
        self.backbone_kwargs = dict(backbone_kwargs or {})
        self.formulation_kwargs = {**path_type_for(formulation).default_kwargs, **dict(formulation_kwargs or {})}
        self.transform = StftTransform(**(transform_kwargs or {}))
        self.generator = BackboneRegister.fetch(backbone)(input_channels=4, **self.backbone_kwargs)
        self.formulation = path_type_for(formulation)(**self.formulation_kwargs)
        self.formulation_name = self.formulation.path_name
        self.training_target = self.formulation.training_target
        self.sampling_solver = self.formulation.default_solver if sampling_solver == "AUTO" else sampling_solver
        self.sampling_num_steps = int(sampling_num_steps)
        self.sampling_skip_type = sampling_skip_type
        if self.sampling_solver not in self.formulation.allowed_solvers:
            raise ValueError(f"{self.formulation_name} does not support {self.sampling_solver}")
        if self.sampling_num_steps < 1:
            raise ValueError("sampling_num_steps must be positive")
        if sampling_skip_type not in {"time_uniform", "time_quadratic"}:
            raise ValueError(f"Unsupported grid spacing: {sampling_skip_type}")
        self.gradient_checkpointing = gradient_checkpointing
        self.forward_evaluations = 0
        self.sampling_evaluations = 0
        self.to(device)

    @property
    def model_device(self) -> torch.device:
        """Return the device holding the trainable backbone parameters."""
        return next(self.generator.parameters()).device

    def forward(self, state: Tensor, time: Tensor, condition: list[Tensor] | None = None) -> Tensor:
        """Predict the native target at a state, preserving the training graph.

        Args:
            state: Complex state with batch and spatial-channel dimensions.
            time: One integration time per batch example.
            condition: Conditioning spectra concatenated along the channel axis.

        Returns:
            Data or vector-field prediction as declared by the formulation.
        """
        self.forward_evaluations += 1
        inputs = torch.cat([state, *(condition or [])], dim=1)
        model_time = self.formulation.model_time(time)
        recompute = self.gradient_checkpointing and self.training and torch.is_grad_enabled()
        if recompute and not getattr(self.generator, "handles_gradient_checkpointing", False):
            output = checkpoint(self.generator, inputs, model_time, use_reentrant=False)
        else:
            output = self.generator(inputs, model_time)
        return self.formulation.model_output(output)

    def predict_clean(self, state: Tensor, time: Tensor, condition: Tensor) -> Tensor:
        """Evaluate once and convert the native output to a differentiable endpoint."""
        prediction = self(state, time, [condition])
        return self.formulation.to_clean_prediction(prediction, state, time, condition)

    def sampling_time_grid(
        self,
        *,
        num_steps: int | None = None,
        solver: str | None = None,
        skip_type: str | None = None,
        t_min: float | None = None,
    ) -> Tensor:
        """Resolve the actual integration grid without evaluating the network.

        Args:
            num_steps: Number of integration intervals, defaulting to the model.
            solver: Solver override, defaulting to the model.
            skip_type: Grid spacing override, defaulting to the model.
            t_min: Integration or safe-network-time bound.

        Returns:
            The exact ``num_steps + 1`` time points to pass to ``sample``.

        Raises:
            ValueError: If the lower bound or sampling protocol is invalid.
        """
        if t_min is not None and not 0 <= t_min < 1:
            raise ValueError("t_min must satisfy 0 <= t_min < 1")
        sampler = self.formulation.build_sampler(solver or self.sampling_solver, self)
        spacing = skip_type or self.sampling_skip_type
        start, end = sampler.integration_bounds(self.formulation.start_time, t_min, spacing)
        count = self.sampling_num_steps if num_steps is None else int(num_steps)
        return sampler.get_time_steps(spacing, start, end, count)

    @torch.no_grad()
    def sample(
        self,
        observation: Tensor,
        *,
        num_steps: int | None = None,
        solver: str | None = None,
        skip_type: str | None = None,
        t_min: float | None = None,
        time_grid: Tensor | list[float] | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Generate a spectrum with CPU state/prediction traces and measured NFE.

        Args:
            observation: Noisy conditioning spectrum.
            num_steps: Integration-interval count, defaulting to model configuration.
            solver: Formulation-supported solver override.
            skip_type: Time-grid spacing override.
            t_min: Optional integration or safe-network-time bound.
            time_grid: Explicit schedule with ``num_steps + 1`` entries.

        Returns:
            Final spectrum, state trajectory and native prediction trajectory.
        """
        count = self.sampling_num_steps if num_steps is None else int(num_steps)
        if count < 1:
            raise ValueError("num_steps must be positive")
        if t_min is not None and not 0 <= t_min < 1:
            raise ValueError("t_min must satisfy 0 <= t_min < 1")
        condition = observation.to(self.model_device)

        def predictor(state: Tensor, time: Tensor | float) -> Tensor:
            """Bind this call's condition while expanding scalar integration time."""
            batch_time = torch.as_tensor(time, device=state.device, dtype=torch.float32).expand(state.shape[0])
            return self(state, batch_time, [condition])

        sampler = self.formulation.build_sampler(solver or self.sampling_solver, predictor)
        was_training = self.training
        self.eval()
        try:
            result = sampler.sampling(
                x=self.formulation.prior_sample(condition),
                condition=condition,
                num_step=count,
                skip_type=skip_type or self.sampling_skip_type,
                t_max=self.formulation.start_time,
                t_min=t_min,
                time_grid=time_grid,
            )
        finally:
            self.train(was_training)
        self.sampling_evaluations = sampler.nfe
        return result

    @torch.no_grad()
    def enhance(self, audio: Tensor, **sampling_kwargs: Any) -> tuple[Tensor, Tensor, Tensor]:
        """Normalize a waveform, sample its spectrum and restore its original scale.

        Args:
            audio: Input waveform with a trailing sample dimension.
            **sampling_kwargs: Sampling protocol overrides accepted by ``sample``.

        Returns:
            Enhanced CPU waveform and the spectral state/prediction trajectories.
        """
        observation, invert = self.transform.to_stft(audio, self.model_device)
        enhanced, trajectory, predictions = self.sample(observation, **sampling_kwargs)
        return invert(enhanced), trajectory, predictions

    def model_config(self) -> dict[str, Any]:
        """Return the portable constructor configuration stored with model weights."""
        return {
            "formulation": self.formulation_name,
            "backbone": self.backbone_name,
            "backbone_kwargs": self.backbone_kwargs,
            "formulation_kwargs": self.formulation_kwargs,
            "transform_kwargs": self.transform.to_config(),
            "sampling_solver": self.sampling_solver,
            "sampling_num_steps": self.sampling_num_steps,
            "sampling_skip_type": self.sampling_skip_type,
        }

    def save_checkpoint(self, path: str | Path) -> Path:
        """Atomically save model weights and portable construction metadata.

        Full optimizer/EMA/trainer state belongs to the Lightning checkpoint.
        This method stores exactly the currently selected online or EMA weights.
        """
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        save_model(self, temporary, metadata={"cof_model": json.dumps(self.model_config()), "schema_version": "1"})
        os.replace(temporary, destination)
        return destination

    def load_checkpoint(self, path: str | Path) -> None:
        """Strictly load weights into this already-constructed module.

        Priority: a self-describing checkpoint is authoritative and its
        recorded identity is cross-checked against the constructed model
        (mismatches warn and the checkpoint values win); a legacy metadata-free
        file is trusted as-is, because the caller constructed this module from
        the run configuration that produced it.
        """
        recorded = _checkpoint_constructor(Path(path))
        if recorded is not None:
            conflicts = _conflict_keys(recorded, self.model_config())
            if conflicts:
                # The self-describing checkpoint is authoritative; a mismatched
                # run configuration is reported but never blocks the load.
                warnings.warn(
                    f"Checkpoint model identity differs from the constructed model; "
                    f"keeping the checkpoint values for: {', '.join(conflicts)}",
                    stacklevel=2,
                )
        load_model(self, path, device=str(self.model_device), strict=True)

    @classmethod
    def checkpoint_metadata(cls, path: str | Path) -> dict[str, Any] | None:
        """Read the self-describing constructor metadata of a model export.

        Args:
            path: Safetensors model file to inspect.

        Returns:
            The constructor mapping recorded at save time, or ``None`` for a
            legacy metadata-free file.
        """
        return _checkpoint_constructor(Path(path))

    @classmethod
    def from_checkpoint(
        cls, path: str | Path, config: Config | dict[str, Any] | None = None, *, device: str | torch.device = "cpu"
    ) -> Self:
        """Construct a model from a checkpoint, with configuration as fallback.

        Priority: a self-describing checkpoint is authoritative — its recorded
        construction is used, and a provided run configuration is only
        cross-checked against it (any conflict warns while the checkpoint
        values win).  A legacy
        metadata-free checkpoint falls back entirely to the run
        configuration; with neither metadata nor configuration the load is
        rejected.

        Args:
            path: Exact safetensors model file selected by the calling pipeline.
            config: Run configuration; mandatory for legacy metadata-free
                checkpoints, optional identity cross-check otherwise.
            device: Destination parameter device.

        Returns:
            The strictly restored model in evaluation mode.

        Emits a ``UserWarning`` when the recorded identity differs from the
        provided run configuration; the checkpoint values win.

        Raises:
            ValueError: If the checkpoint has no metadata and no configuration
                was provided.
        """
        recorded = _checkpoint_constructor(Path(path))
        if recorded is not None:
            if config is not None:
                conflicts = _conflict_keys(recorded, _constructor_from_config(config))
                if conflicts:
                    # The self-describing checkpoint wins; the conflicting
                    # configuration fields are reported and then ignored.
                    warnings.warn(
                        f"Checkpoint model identity differs from run configuration; "
                        f"keeping the checkpoint values for: {', '.join(conflicts)}",
                        stacklevel=2,
                    )
            constructor = recorded
        else:
            if config is None:
                raise ValueError("Checkpoint carries no construction metadata and no run configuration was provided")
            constructor = _constructor_from_config(config)
        model = cls(**constructor, device=device)
        model.load_checkpoint(path)
        return model.eval()
