"""Adapt TorchMetrics implementations as registered speech-enhancement metrics.

Each metric wraps one TorchMetrics class with a uniform one-utterance
``calculate`` interface and registers itself under a canonical name, so the
evaluation workflow can instantiate metrics from names without importing the
implementations.

Author: Qing Yao
Date: 2026/9/23
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
from torchmetrics.audio.sdr import ScaleInvariantSignalDistortionRatio
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility

from cof.evaluation.base import AudioMetric
from cof.evaluation.registry import MetricRegister


def _waveform(value: np.ndarray | Tensor) -> Tensor:
    """Convert one waveform to the floating-point tensor format expected by TorchMetrics.

    Args:
        value: Waveform as a NumPy array or tensor.

    Returns:
        The waveform as a float32 tensor.
    """
    return torch.as_tensor(np.asarray(value), dtype=torch.float32)


@MetricRegister.register("pesq")
class PesqMetric(AudioMetric, PerceptualEvaluationSpeechQuality):
    """Perceptual Evaluation of Speech Quality via TorchMetrics."""

    output_names = ("PESQ",)

    def __init__(self, sample_rate: int = 16_000, **kwargs: Any) -> None:
        """Configure wide-band PESQ at 16 kHz and narrow-band PESQ at 8 kHz.

        Args:
            sample_rate: Sample rate; supported values are 8000 and 16000 Hz.
            **kwargs: Keyword arguments forwarded to TorchMetrics.

        Raises:
            ValueError: If the sample rate is unsupported.
        """
        if sample_rate not in (8_000, 16_000):
            raise ValueError("PESQ supports sample rates of 8000 or 16000 Hz")
        mode = "wb" if sample_rate == 16_000 else "nb"
        super().__init__(fs=sample_rate, mode=mode, sample_rate=sample_rate, **kwargs)

    def calculate(
        self,
        ref_wav: np.ndarray,
        deg_wav: np.ndarray,
        sample_rate: int = 16_000,
        **kwargs: object,
    ) -> dict[str, float]:
        """Calculate PESQ for one enhanced/reference pair.

        Args:
            ref_wav: Reference waveform samples.
            deg_wav: Enhanced (degraded) waveform samples.
            sample_rate: Sample rate of both waveforms.
            **kwargs: Unused; accepted for interface compatibility.

        Returns:
            Mapping with the ``PESQ`` score.
        """
        del kwargs
        self._validate_sample_rate(sample_rate)
        return {"PESQ": float(self._score(_waveform(deg_wav), _waveform(ref_wav)))}


@MetricRegister.register("estoi")
class EstoiMetric(AudioMetric, ShortTimeObjectiveIntelligibility):
    """Extended Short-Time Objective Intelligibility via TorchMetrics."""

    output_names = ("ESTOI",)

    def __init__(self, sample_rate: int = 16_000, **kwargs: Any) -> None:
        """Configure TorchMetrics STOI in extended mode.

        Args:
            sample_rate: Sample rate of the audio.
            **kwargs: Keyword arguments forwarded to TorchMetrics.
        """
        super().__init__(fs=sample_rate, extended=True, sample_rate=sample_rate, **kwargs)

    def calculate(
        self,
        ref_wav: np.ndarray,
        deg_wav: np.ndarray,
        sample_rate: int = 16_000,
        **kwargs: object,
    ) -> dict[str, float]:
        """Calculate ESTOI for one enhanced/reference pair.

        Args:
            ref_wav: Reference waveform samples.
            deg_wav: Enhanced (degraded) waveform samples.
            sample_rate: Sample rate of both waveforms.
            **kwargs: Unused; accepted for interface compatibility.

        Returns:
            Mapping with the ``ESTOI`` score.
        """
        del kwargs
        self._validate_sample_rate(sample_rate)
        return {"ESTOI": float(self._score(_waveform(deg_wav), _waveform(ref_wav)))}


@MetricRegister.register("si_sdr")
class SiSdrMetric(AudioMetric, ScaleInvariantSignalDistortionRatio):
    """Scale-invariant signal-to-distortion ratio via TorchMetrics."""

    output_names = ("SI_SDR",)

    def __init__(self, sample_rate: int = 16_000, *, zero_mean: bool = False, **kwargs: Any) -> None:
        """Configure the TorchMetrics SI-SDR implementation.

        The default keeps the raw (non-centered) convention: SI-SDR is a
        dB-scale ``10*log10`` log ratio that is validly negative, and no mean
        subtraction is applied to either waveform.

        Args:
            sample_rate: Sample rate of the audio.
            zero_mean: Whether TorchMetrics subtracts per-waveform means;
                the default preserves the deployment convention.
            **kwargs: Keyword arguments forwarded to TorchMetrics.
        """
        super().__init__(zero_mean=zero_mean, sample_rate=sample_rate, **kwargs)

    def calculate(
        self,
        ref_wav: np.ndarray,
        deg_wav: np.ndarray,
        sample_rate: int = 16_000,
        **kwargs: object,
    ) -> dict[str, float]:
        """Calculate SI-SDR for one enhanced/reference pair.

        Args:
            ref_wav: Reference waveform samples.
            deg_wav: Enhanced (degraded) waveform samples.
            sample_rate: Sample rate of both waveforms.
            **kwargs: Unused; accepted for interface compatibility.

        Returns:
            Mapping with the ``SI_SDR`` score in dB; the value is validly
            negative when distortion energy dominates.
        """
        del kwargs
        self._validate_sample_rate(sample_rate)
        return {"SI_SDR": float(self._score(_waveform(deg_wav), _waveform(ref_wav)))}
