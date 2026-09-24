"""Probability-path and solver contracts exercised through the public model API.

These tests pin the mathematical invariants the training and sampling code
relies on: the bridge forward kernel ``q_sample`` shapes, the sampler NFE
contract, and the OT-CFM vector-field conversion.

Author: Qing Yao
Date: 2026/9/23
"""

import pytest
import torch

from cof.model import GenerativeModel4SE

TINY_BACKBONE = {"nf": 8, "ch_mult": [1, 1, 1, 1], "num_res_blocks": 1, "attn_resolutions": [], "image_size": 256}
SPECTRUM_SHAPE = (2, 1, 256, 64)


def make_model(formulation: str, solver: str) -> GenerativeModel4SE:
    """Build a tiny CPU model over the requested formulation."""
    return GenerativeModel4SE(
        formulation=formulation,
        backbone="ncsnpp_base",
        sampling_num_steps=4,
        sampling_solver=solver,
        sampling_skip_type="time_uniform",
        backbone_kwargs=TINY_BACKBONE,
    ).eval()


@pytest.fixture(scope="module")
def sbve_model() -> GenerativeModel4SE:
    return make_model("SBVE", "SB_SDE_Solver")


@pytest.fixture(scope="module")
def otcfm_model() -> GenerativeModel4SE:
    return make_model("OTCFM", "OTCFM_ODE_Solver")


@pytest.mark.parametrize("formulation", ["SBVE", "OTCFM"])
def test_q_sample_interpolates_clean_and_noisy(sbve_model, otcfm_model, formulation):
    """The bridge kernel maps (x0, x1, t) to a finite state of the input shape."""
    model = sbve_model if formulation == "SBVE" else otcfm_model
    clean = torch.randn(*SPECTRUM_SHAPE)
    noisy = torch.randn(*SPECTRUM_SHAPE)
    timestep = torch.full((SPECTRUM_SHAPE[0],), 0.5)

    state = model.formulation.sample_training_state(clean, noisy, timestep)

    assert state.shape == clean.shape
    assert torch.isfinite(state).all()


def test_sbve_training_target_is_data(sbve_model):
    assert sbve_model.formulation.sde.training_target == "data"


def test_otcfm_forces_vector_target(otcfm_model):
    assert otcfm_model.formulation.ode.training_target == "vector"


def test_otcfm_lower_time_bound(otcfm_model):
    """OT-CFM exposes the path-specific safe lower bound for network calls."""
    assert float(getattr(otcfm_model.formulation.ode, "training_time_start", 0.0)) == pytest.approx(0.03)


def test_sample_reports_consumed_nfe(sbve_model):
    """A four-step SDE sample must consume exactly four network evaluations."""
    observation = torch.randn(*SPECTRUM_SHAPE, dtype=torch.complex64)

    sample, trajectory, predictions = sbve_model.sample(observation, num_steps=4)

    assert sample.shape == observation.shape
    assert sbve_model.sampling_evaluations == 4
    assert torch.isfinite(sample).all()
    assert torch.isfinite(trajectory).all()
    assert torch.isfinite(predictions).all()


def test_vector_field_conversion_is_finite(otcfm_model):
    """The data-prediction-to-vector-field conversion yields finite fields."""
    state = torch.randn(*SPECTRUM_SHAPE)
    data_prediction = torch.randn(*SPECTRUM_SHAPE)
    timestep = torch.full((SPECTRUM_SHAPE[0],), 0.5)
    condition = torch.randn(*SPECTRUM_SHAPE)

    field = otcfm_model.formulation.ode.vector_field_from_data_prediction(state, timestep, condition, data_prediction)

    assert field.shape == state.shape
    assert torch.isfinite(field).all()
