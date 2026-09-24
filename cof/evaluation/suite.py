"""One configurable metric interface shared by validation and offline evaluation."""

from __future__ import annotations

import math
from typing import Any

from cof.evaluation.registry import MetricRegister


def supported_metrics() -> tuple[str, ...]:
    """Return registered names, including components added by applications."""
    return tuple(MetricRegister.names())


class MetricSuite:
    """Instantiate named metrics and validate their output metadata and values.

    New metrics declare output_names, higher_is_better and requires_reference,
    and implement calculate(ref_wav, deg_wav, sample_rate). Non-intrusive
    metrics may ignore ref_wav. Both pipelines call this same interface.
    """

    def __init__(
        self,
        names: tuple[str, ...] | list[str],
        sample_rate: int = 16000,
        *,
        kwargs: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        """Resolve requested components and reject missing or colliding outputs."""
        self.names = tuple(dict.fromkeys(names))
        if not self.names:
            raise ValueError("At least one evaluation metric is required")
        self.sample_rate = sample_rate
        self.metrics = {
            name: MetricRegister.fetch(name)(sample_rate=sample_rate, **(kwargs or {}).get(name, {}))
            for name in self.names
        }
        self.output_names = tuple(output for metric in self.metrics.values() for output in metric.output_names)
        if not self.output_names or len(set(self.output_names)) != len(self.output_names):
            raise ValueError("Metric outputs must be nonempty and unique")
        self.modes = {
            output: "max" if metric.higher_is_better else "min"
            for metric in self.metrics.values()
            for output in metric.output_names
        }

    def calculate(self, reference: Any, estimate: Any) -> dict[str, float]:
        """Score one aligned utterance and fail on incomplete or nonfinite outputs."""
        result = {}
        for name, metric in self.metrics.items():
            if metric.requires_reference and reference is None:
                raise ValueError(f"Metric {name} requires reference audio")
            values = metric.calculate(ref_wav=reference, deg_wav=estimate, sample_rate=self.sample_rate)
            if set(values) != set(metric.output_names):
                raise ValueError(f"Metric {name} returned outputs inconsistent with its metadata")
            for output, value in values.items():
                value = float(value)
                if not math.isfinite(value):
                    raise FloatingPointError(f"Non-finite metric {output}")
                result[output] = value
        return result
