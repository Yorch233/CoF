"""Resolve canonical probability-path and sampling-protocol settings.

One resolver owns how a configuration's probability path, solver, step count,
skip schedule, and inference ``t_min`` combine: explicit CLI overrides win,
persisted run fields come next, and path-specific canonical defaults fill the
rest, so every consumer (training, inference, validation sampling) sees the
same protocol for the same configuration.

Author: Qing Yao
Date: 2026/9/23
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol

from cof.config.manager import Config
from cof.formulation.registry import (
    ProbabilityPathName,
    SolverName,
    default_solver_for,
    path_type_for,
    validate_solver,
)

SkipType = Literal["time_uniform", "time_quadratic"]


class ConfigLike(Protocol):
    """Minimal configuration interface required by protocol resolution."""

    def get(self, field: str, default: Any = None) -> Any:
        """Return a configured field or its default."""
        ...


@dataclass(frozen=True)
class SamplingProtocol:
    """Resolved deployment, validation, and post-training settings."""

    formulation: ProbabilityPathName
    solver: SolverName
    num_steps: int
    skip_type: SkipType


def _resolve_path(config: ConfigLike) -> ProbabilityPathName:
    """Return the configured probability path.

    ``formulation`` is required; a configuration without it is rejected
    rather than silently resolved to a default formulation.

    Args:
        config: Configuration-like mapping to read.

    Returns:
        The canonical probability-path name.

    Raises:
        ValueError: If no probability path is configured.
    """
    configured = config.get("formulation.name")
    if configured is None:
        raise ValueError("formulation is required")
    return path_type_for(configured).path_name  # type: ignore[return-value]


def resolve_sampling_protocol(
    config: ConfigLike,
    *,
    solver: str | None = None,
    num_steps: int | None = None,
    skip_type: str | None = None,
) -> SamplingProtocol:
    """Resolve explicit overrides and canonical path-specific settings.

    Args:
        config: Configuration-like mapping carrying persisted sampling fields.
        solver: Solver override; ``None`` or ``"AUTO"`` falls back to the
            configured or path-default solver.
        num_steps: Step count (NFE) override.
        skip_type: Time-skip schedule override.

    Returns:
        The fully resolved sampling protocol.

    Raises:
        ValueError: If the solver, step count, or skip type is missing or
            unsupported for the resolved path.
    """
    path = _resolve_path(config)
    configured_solver = config.get("formulation.sampling.solver")
    configured_steps = config.get("formulation.sampling.num_steps")
    configured_skip = config.get("formulation.sampling.skip_type")
    resolved_solver_name = solver if solver is not None else configured_solver
    if resolved_solver_name in {None, "AUTO"}:
        resolved_solver_name = default_solver_for(path)
    resolved_solver = validate_solver(path, str(resolved_solver_name))

    if num_steps is not None:
        resolved_steps = num_steps
    elif configured_steps is not None:
        resolved_steps = int(configured_steps)
    else:
        resolved_steps = 4
    if resolved_steps < 1:
        raise ValueError("sampling_num_steps must be at least 1")

    resolved_skip = skip_type if skip_type is not None else configured_skip or "time_uniform"
    if resolved_skip not in {"time_uniform", "time_quadratic"}:
        raise ValueError(f"Unsupported sampling skip type: {resolved_skip!r}")

    return SamplingProtocol(
        formulation=path,
        solver=resolved_solver,
        num_steps=int(resolved_steps),
        skip_type=resolved_skip,  # type: ignore[arg-type]
    )


def canonical_sampling_values(config: ConfigLike) -> dict[str, Any]:
    """Return canonical fields suitable for persistence in a run config.

    Args:
        config: Configuration-like mapping carrying the sampling fields.

    Returns:
        The canonical probability path, path kwargs (with OT-CFM defaults
        materialized), inference ``t_min``, and resolved sampling protocol
        fields.
    """
    protocol = resolve_sampling_protocol(config)
    component = path_type_for(protocol.formulation)
    path_kwargs = {**component.default_kwargs, **dict(Config.unwrap(config.get("formulation.kwargs") or {}))}
    definition = component(**path_kwargs)
    configured_t_min = config.get("formulation.sampling.t_min")
    configured_t_min = (
        definition.training_time_start
        if configured_t_min is None
        else max(float(configured_t_min), definition.training_time_start)
    )
    return {
        "formulation.name": protocol.formulation,
        "formulation.kwargs": path_kwargs,
        "formulation.sampling.t_min": float(configured_t_min),
        "formulation.sampling.solver": protocol.solver,
        "formulation.sampling.num_steps": protocol.num_steps,
        "formulation.sampling.skip_type": protocol.skip_type,
    }


def resolve_inference_t_min(config: ConfigLike, override: float | None = None) -> float:
    """Resolve the lower network-evaluation time used by inference.

    SB-VE and the other reverse bridge paths integrate all the way to the
    clean endpoint, so their default lower time is zero. FlowSE-style OT-CFM
    evaluates the vector field on ``[t_min, 1]`` and performs a final
    numerical transition from ``t_min`` to zero; its default is therefore the
    path value (``0.03``), rather than a truncated terminal endpoint.

    Args:
        config: Configuration-like mapping carrying the path kwargs.
        override: Explicit inference ``t_min`` taking precedence over every
            default.

    Returns:
        The resolved lower evaluation time, validated to ``0 <= t_min < 1``.

    Raises:
        ValueError: If the override or resolved value falls outside
            ``[0, 1)``, or the path kwargs are not a mapping.
    """
    path = _resolve_path(config)
    if override is not None:
        value = float(override)
    else:
        path_kwargs = Config.unwrap(config.get("formulation.kwargs") or {})
        if not isinstance(path_kwargs, dict):
            raise ValueError("formulation.kwargs must be a mapping")
        value = path_type_for(path)(**path_kwargs).inference_t_min
    value = float(value)
    if not 0.0 <= value < 1.0:
        raise ValueError("inference t_min must satisfy 0 <= t_min < 1")
    return value
