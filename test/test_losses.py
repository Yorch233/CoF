"""Mathematical objectives and gradient boundaries through standalone methods.

The CoF objective implementation is withheld during peer review; its tests
assert the withheld contract instead of the mathematics.
"""

import pytest
import torch

from cof.config.schemas import DrcConfig
from cof.method.base import StepContext
from cof.method.posttrain.cof.method import CoFPosttraining
from cof.method.pretrain.base import BasePretraining
from cof.model import GenerativeModel4SE

TINY_BACKBONE = {"nf": 8, "ch_mult": [1, 1, 1, 1], "num_res_blocks": 1, "attn_resolutions": [], "image_size": 256}


@pytest.fixture(params=["SBVE", "OTCFM"])
def model(request):
    return GenerativeModel4SE(request.param, backbone_kwargs=TINY_BACKBONE).eval()


def make_batch():
    clean = torch.randn(1, 1, 256, 64, dtype=torch.complex64)
    return clean, clean + 0.05 * torch.randn_like(clean)


def test_base_loss_and_gradients(model):
    result = BasePretraining(model).step(make_batch(), StepContext(0, "train"))
    assert "prediction_loss" in result.terms
    assert all(torch.isfinite(value) for value in result.logging_values().values())
    result.loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_cof_step_is_withheld(model):
    method = CoFPosttraining(model)
    with pytest.raises(NotImplementedError, match="withheld during peer review"):
        method.step(make_batch(), StepContext(0, "train"))


@pytest.mark.parametrize("kwargs", [{"n_max": 0}, {"n_max": 2, "fixed_n": 3}])
def test_invalid_drc_budgets(kwargs):
    with pytest.raises(ValueError, match="DRC"):
        DrcConfig(**kwargs)
