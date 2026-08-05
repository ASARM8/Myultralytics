"""Unit tests for the conservative Refine V3.1 head."""

import math

import torch

from ultralytics.nn.modules.refine_v31 import OBBProposalRefinerV31


def build_module(*, use_quality_aux=False):
    return OBBProposalRefinerV31(
        p2_channels=16,
        p3_channels=24,
        roi_channels=8,
        roi_size=(3, 8),
        hidden_channels=16,
        use_quality_aux=use_quality_aux,
    )


def sample_inputs():
    p2 = torch.randn(1, 16, 32, 32)
    p3 = torch.randn(1, 24, 16, 16)
    proposals = torch.tensor([[[32.0, 40.0, 4.0, 40.0, 0.2], [60.0, 70.0, 30.0, 6.0, 0.8]]])
    scores = torch.full((1, 2), 0.8)
    return p2, p3, proposals, scores


def test_v31_geometry_only_is_zero_initialized_identity_without_quality_parameters():
    module = build_module().eval()
    p2, p3, proposals, scores = sample_inputs()
    output = module(p2, p3, proposals, scores, (128, 128))
    assert output["quality_logit"] is None
    assert not any(name.startswith("quality_head") for name in module.state_dict())
    assert torch.equal(output["residual"], torch.zeros_like(output["residual"]))
    assert torch.equal(module.apply_residual(proposals, output["residual"]), proposals)


def test_v31_quality_aux_is_training_only_but_has_a_real_head():
    module = build_module(use_quality_aux=True).eval()
    p2, p3, proposals, scores = sample_inputs()
    output = module(p2, p3, proposals, scores, (128, 128))
    assert output["quality_logit"].shape == (1, 2, 1)
    assert any(name.startswith("quality_head") for name in module.state_dict())
    assert torch.equal(output["residual"], torch.zeros_like(output["residual"]))


def test_v31_defaults_have_conservative_physical_scale_limits():
    module = build_module()
    assert module.short_negative_limit == 0.50
    assert module.short_positive_limit == 0.20
    assert module.long_negative_limit == 0.08
    assert module.long_positive_limit == 0.08
    assert math.exp(-module.short_negative_limit) > 0.60
    assert math.exp(module.short_positive_limit) < 1.23
    assert math.exp(-module.long_negative_limit) > 0.92
    assert math.exp(module.long_positive_limit) < 1.09


def test_v31_bounds_and_invalid_padding_are_respected():
    module = build_module().eval()
    with torch.no_grad():
        module.geometry_head.bias.copy_(torch.tensor([-10.0, 10.0]))
    p2, p3, proposals, scores = sample_inputs()
    valid = torch.tensor([[True, False]])
    residual = module(p2, p3, proposals, scores, (128, 128), valid)["residual"]
    assert residual[0, 0, 0] >= -module.short_negative_limit
    assert residual[0, 0, 1] <= module.long_positive_limit
    assert torch.equal(residual[0, 1], torch.zeros(4))
