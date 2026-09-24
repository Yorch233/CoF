"""Metric-registry and TorchMetrics-adapter contracts.

Author: Qing Yao
Date: 2026/9/23
"""

import numpy as np
import pytest
import torch
from torchmetrics import Metric
from torchmetrics.audio.sdr import ScaleInvariantSignalDistortionRatio

from cof.evaluation import EstoiMetric, MetricRegister, PesqMetric, SiSdrMetric


def test_exactly_three_intrusive_metrics_are_registered():
    registered = MetricRegister.fetch(["pesq", "estoi", "si_sdr"])

    assert registered == {"pesq": PesqMetric, "estoi": EstoiMetric, "si_sdr": SiSdrMetric}


def test_removed_metric_names_do_not_resolve():
    with pytest.warns(UserWarning, match="Unregistered"), pytest.raises(ValueError, match="No registered"):
        MetricRegister.fetch(["utmos", "dnsmos"])


@pytest.mark.parametrize("metric_type", [PesqMetric, EstoiMetric, SiSdrMetric])
def test_metrics_extend_torchmetrics(metric_type):
    assert issubclass(metric_type, Metric)


def test_pesq_rejects_unsupported_sample_rates():
    with pytest.raises(ValueError, match="8000 or 16000"):
        PesqMetric(sample_rate=22_050)


def test_estoi_uses_extended_mode():
    metric = EstoiMetric(sample_rate=16_000)

    assert metric.fs == 16_000
    assert metric.extended is True


def test_si_sdr_matches_torchmetrics_reference():
    reference = np.array([3.0, -0.5, 2.0, 7.0], dtype=np.float32)
    estimate = np.array([2.5, 0.0, 2.0, 8.0], dtype=np.float32)
    expected = ScaleInvariantSignalDistortionRatio()(torch.from_numpy(estimate), torch.from_numpy(reference))

    result = SiSdrMetric().calculate(ref_wav=reference, deg_wav=estimate)

    assert result == {"SI_SDR": pytest.approx(float(expected))}


def test_calculate_rejects_a_sample_rate_mismatch():
    metric = SiSdrMetric(sample_rate=16_000)
    waveform = np.zeros(8, dtype=np.float32)

    with pytest.raises(ValueError, match="8000 Hz"):
        metric.calculate(ref_wav=waveform, deg_wav=waveform, sample_rate=8_000)
