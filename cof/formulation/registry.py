"""Register the probability paths and the sampling solvers each accepts.

This module is the single source of truth for which probability paths exist
and which sampling solvers each of them accepts.  Configuration loading, model
construction, and the CLI all read the metadata from here instead of keeping
their own copies of the rules, so an unsupported path/solver pair is rejected
consistently everywhere.

Author: Qing Yao
Date: 2026/9/23
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from cof.formulation.base import Formulation
from cof.formulation.ot_cfm.definition import OTCFMFormulation
from cof.formulation.sb_ve.definition import SBVEFormulation

ProbabilityPathName = str
SolverName = Literal["SB_SDE_Solver", "SB_ODE_Solver", "OTCFM_ODE_Solver"]
ConfiguredSolverName = Literal["SB_SDE_Solver", "SB_ODE_Solver", "OTCFM_ODE_Solver", "AUTO"]

#: Every formulation that can be selected through ``formulation``.
PATH_TYPES: dict[str, type[Formulation]] = {
    "SBVE": SBVEFormulation,
    "OTCFM": OTCFMFormulation,
}


def register_formulation(name: str) -> Callable[[type[Formulation]], type[Formulation]]:
    """Register a formulation class under a normalized configuration name.

    Its metadata defines supported solvers, network target and preset directory.
    Applications import their component registrations before resolving configs.
    """
    normalized = name.upper().replace("-", "").replace("_", "")

    def register(component: type[Formulation]) -> type[Formulation]:
        """Reject duplicate names and expose the decorated formulation."""
        if normalized in PATH_TYPES:
            raise ValueError(f"Formulation already registered: {name}")
        if not issubclass(component, Formulation):
            raise TypeError("Registered formulation must derive from Formulation")
        PATH_TYPES[normalized] = component
        return component

    return register


def path_type_for(name: str) -> type[Formulation]:
    """Return the path class registered under one configuration name.

    The lookup normalises case and separators, so ``"sb-ve"`` and ``"SBVE"``
    select the same formulation.

    Args:
        name (str): Probability-path name from a configuration or a call.

    Returns:
        type[Formulation]: The registered formulation class.

    Raises:
        ValueError: If no formulation is registered under ``name``.
    """
    normalized = str(name).upper().replace("-", "").replace("_", "")
    try:
        return PATH_TYPES[normalized]
    except KeyError as error:
        raise ValueError(f"Unsupported probability path: {name!r}") from error


def default_solver_for(name: str) -> SolverName:
    """Return the solver a path uses when the configuration leaves it on ``AUTO``."""
    return path_type_for(name).default_solver  # type: ignore[return-value]


def validate_solver(name: str, solver: str) -> SolverName:
    """Validate one solver name against a probability path.

    Args:
        name (str): Canonical probability-path name, such as ``"OTCFM"``.
        solver (str): Solver name to validate.

    Returns:
        SolverName: The validated solver name.

    Raises:
        ValueError: If the solver does not exist or the path rejects it.
    """
    if solver not in {"SB_SDE_Solver", "SB_ODE_Solver", "OTCFM_ODE_Solver"}:
        raise ValueError(f"Unsupported sampling solver: {solver!r}")
    supported = path_type_for(name).allowed_solvers
    if solver not in supported:
        choices = ", ".join(sorted(supported))
        raise ValueError(f"{name} supports {choices}, got {solver!r}")
    return solver  # type: ignore[return-value]


__all__ = [
    "PATH_TYPES",
    "ConfiguredSolverName",
    "ProbabilityPathName",
    "SolverName",
    "default_solver_for",
    "path_type_for",
    "validate_solver",
]
