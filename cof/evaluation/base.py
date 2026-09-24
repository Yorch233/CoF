"""Common metadata and thread-safe scoring for audio metric components."""

from __future__ import annotations

import threading

import torch
from torch import Tensor


class AudioMetric:
    """Mixin exposing the one-file evaluation interface to TorchMetrics metrics."""

    output_names: tuple[str, ...] = ()
    higher_is_better = True
    requires_reference = True

    def __init__(self, *args: object, sample_rate: int = 16_000, **kwargs: object) -> None:
        """Initialize a TorchMetrics metric for one sample rate.

        Args:
            *args: Positional arguments forwarded to the TorchMetrics class.
            sample_rate: Sample rate this instance validates every call
                against.
            **kwargs: Keyword arguments forwarded to the TorchMetrics class.

        Raises:
            ValueError: If the sample rate is not positive.
        """
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        self.sample_rate = sample_rate
        self._calculation_lock = threading.RLock()
        super().__init__(*args, **kwargs)

    def _validate_sample_rate(self, sample_rate: int) -> None:
        """Reject audio at a rate other than the configured one.

        Args:
            sample_rate: Sample rate of the audio being scored.

        Raises:
            ValueError: If the rate differs from the configured one.
        """
        if sample_rate != self.sample_rate:
            raise ValueError(
                f"{type(self).__name__} was initialized for {self.sample_rate} Hz, but received {sample_rate} Hz audio"
            )

    def _score(self, *args: object) -> Tensor:
        """Run a stateful TorchMetrics instance safely for one utterance.

        Args:
            *args: Waveform tensors in the TorchMetrics metric's argument
                order.

        Returns:
            The detached scalar score on CPU.
        """
        with self._calculation_lock, torch.inference_mode():
            return self(*args).detach().cpu()  # type: ignore[operator]
