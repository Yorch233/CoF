"""OT-CFM sampling with safe-time and terminal-grid semantics."""

from __future__ import annotations

import math

import torch

from cof.formulation.base import Formulation, PredictionFunction, Sampler


class OTCFM_ODE_Solver(Sampler):
    """Deterministic ODE solver for FlowSE's optimal-transport CFM flow.

    OT-CFM keeps FlowSE's white-box grid: the network is evaluated on
    ``linspace(1, t_min, N)`` and a final transition reaches the clean
    endpoint at zero.  The path exposes that grid through
    :meth:`~cof.formulation.ot_cfm.definition.OTCFMFormulation.sampling_time_grid`,
    which the base solver consults only when the integration bounds equal the
    path's own bounds.  Resolving those bounds is therefore this solver's job:

    * the caller left ``t_min`` to the path, asked for the configured value, or
      passed zero -> keep the path bounds so the FlowSE grid applies;
    * the caller asked for a different lower time -> truncate the integration
      there instead, with a plain uniform grid.
    """

    expected_path_name = "OTCFM"

    def __init__(
        self,
        formulation: Formulation,
        predictor: PredictionFunction | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        """Bind the deterministic kernel under its explicit solver name."""
        super().__init__(formulation, predictor, device, solver="OTCFM_ODE_Solver")

    def integration_bounds(
        self,
        t_max: float | None,
        t_min: float | None,
        skip_type: str,
    ) -> tuple[float, float]:
        """Resolve integration bounds, preserving FlowSE's grid when appropriate.

        Args:
            t_max (float | None): Upper bound override; ``None`` uses the path start.
            t_min (float | None): Lower time requested by the caller; ``None``
                means the caller left it to the path's configured ``t_min``.
            skip_type (str): Time-grid spacing of the call.

        Returns:
            tuple[float, float]: The ``(start, end)`` integration bounds.
        """
        start = float(self.formulation.start_time) if t_max is None else float(t_max)
        configured = float(self.formulation.t_min)
        requested = configured if t_min is None else float(t_min)
        uses_flowse_grid = skip_type == "time_uniform" and (
            t_min is None
            or math.isclose(requested, 0.0, abs_tol=1e-12)
            or math.isclose(requested, configured, abs_tol=1e-12)
        )
        if uses_flowse_grid:
            return start, float(self.formulation.end_time)
        return start, max(float(self.formulation.end_time), requested)
