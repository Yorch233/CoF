"""Composable CoF update: Dynamic Rollout Correction plus transition consistency.

The class and configuration wiring are part of the released contract; the
objective implementation is withheld during peer review and will be released
upon acceptance.
"""

from __future__ import annotations

from torch import Tensor, nn

from cof.config.manager import Config
from cof.config.schemas import CtcConfig, DrcConfig, EndpointLossConfig
from cof.method.base import WITHHELD_MESSAGE, EndpointLoss, StepContext, StepResult, TrainingMethod
from cof.method.posttrain.cof.loss import CoFCleanLoss
from cof.method.registry import PostTrainingRegister
from cof.model import GenerativeModel4SE


@PostTrainingRegister.register("cof")
class CoFPosttraining(TrainingMethod):
    """Correct the current sampler's states with a shared clean-domain objective."""

    name = "cof"
    stage = "post_training"
    required_capabilities = frozenset({"clean_prediction", "clean_transition"})

    def __init__(
        self,
        model: GenerativeModel4SE,
        *,
        drc: DrcConfig | None = None,
        ctc: CtcConfig | None = None,
        endpoint_loss: EndpointLoss | None = None,
        solver: str | None = None,
        skip_type: str = "time_uniform",
        t_min: float | None = None,
    ) -> None:
        """Inject the DRC/CTC policies, loss and formulation-supported sampler."""
        super().__init__(model)
        self.drc = drc or DrcConfig()
        self.ctc = ctc or CtcConfig()
        self.endpoint_loss = endpoint_loss or CoFCleanLoss(model.transform, model.formulation.compute_weight)
        self.solver = solver or model.formulation.default_solver
        self.skip_type = skip_type
        self.t_min = max(model.formulation.training_time_start, t_min or 0.0)

    @classmethod
    def from_config(cls, model: GenerativeModel4SE, config: Config) -> CoFPosttraining:
        """Resolve only CoF-owned configuration before constructing the method."""
        drc = DrcConfig(
            int(config.get("posttrain.cof.drc.rollout.n_max", 16)), config.get("posttrain.cof.drc.rollout.fixed_n")
        )
        ctc = CtcConfig(
            float(config.get("posttrain.cof.ctc.lambda", 0.1)),
            int(config.get("posttrain.cof.ctc.steps", 1)),
            config.get("posttrain.cof.ctc.fa_model", "ema"),
            config.get("posttrain.cof.ctc.cf_model", "ema"),
            bool(config.get("posttrain.cof.ctc.shared_noise", True)),
        )
        loss_config = EndpointLossConfig(
            float(config.get("posttrain.cof.objective.magnitude_weight", 0.7)),
            float(config.get("posttrain.cof.objective.complex_weight", 0.3)),
            float(config.get("posttrain.cof.objective.si_sdr_weight", 0.01)),
            config.get("posttrain.cof.objective.compression", "power"),
        )
        loss = CoFCleanLoss(model.transform, model.formulation.compute_weight, loss_config)
        return cls(
            model,
            drc=drc,
            ctc=ctc,
            endpoint_loss=loss,
            solver=model.sampling_solver,
            skip_type=model.sampling_skip_type,
            t_min=config.get("formulation.sampling.t_min"),
        )

    def required_references(self) -> set[str]:
        """Declare the reference providers needed by either CTC branch."""
        return {self.ctc.factual_model, self.ctc.target_model} - {"online"}

    def auxiliary_modules(self) -> dict[str, nn.Module]:
        """Register an injected trainable endpoint objective when present."""
        return {"endpoint_loss": self.endpoint_loss} if isinstance(self.endpoint_loss, nn.Module) else {}

    def step(self, batch: tuple[Tensor, Tensor], context: StepContext) -> StepResult:
        """Run DRC and CTC with two differentiable student evaluations.

        The construction and configuration wiring above are fully functional;
        the objective implementation is withheld during peer review.
        """
        raise NotImplementedError(WITHHELD_MESSAGE)
