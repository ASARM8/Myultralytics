"""Unit tests for Refine V3.1.1 target compression."""

import torch

from ultralytics.nn.modules.refine_v311 import OBBProposalRefinerV311


def build_module():
    return OBBProposalRefinerV311(
        p2_channels=16,
        p3_channels=24,
        roi_channels=8,
        roi_size=(3, 8),
        hidden_channels=16,
    )


def test_v311_is_geometry_only_and_zero_initialized():
    module = build_module().eval()
    assert module.quality_head is None
    assert not any(name.startswith("quality_head") for name in module.state_dict())
    p2 = torch.randn(1, 16, 32, 32)
    p3 = torch.randn(1, 24, 16, 16)
    proposals = torch.tensor([[[32.0, 40.0, 4.0, 40.0, 0.2]]])
    scores = torch.tensor([[0.8]])
    output = module(p2, p3, proposals, scores, (128, 128))
    assert output["quality_logit"] is None
    assert torch.equal(output["residual"], torch.zeros_like(output["residual"]))


def test_v311_smooth_targets_stay_inside_eighty_percent_supervision_range():
    module = build_module()
    raw = torch.tensor([[[-100.0, 100.0, 5.0, -5.0], [-0.1, -0.02, 0.0, 0.0]]])
    transformed = module.clip_target(raw)
    assert transformed[0, 0, 0] >= -0.50 * 0.80
    assert transformed[0, 0, 1] <= 0.08 * 0.80
    assert transformed[0, 1, 0] > -0.1
    assert transformed[0, 1, 1] > -0.02
    assert torch.equal(transformed[..., 2:], torch.zeros_like(transformed[..., 2:]))


def test_v311_smooth_target_has_unit_local_slope_and_no_hard_clip_point_mass():
    module = build_module()
    epsilon = torch.tensor(1e-5)
    positive_slope = module._smooth_target(epsilon, 0.4, 0.16) / epsilon
    negative_slope = module._smooth_target(-epsilon, 0.4, 0.16) / -epsilon
    assert torch.isclose(positive_slope, torch.tensor(1.0), atol=1e-5)
    assert torch.isclose(negative_slope, torch.tensor(1.0), atol=1e-5)
    first = module.clip_target(torch.tensor([[[-1.0, 0.0, 0.0, 0.0]]]))[..., 0]
    second = module.clip_target(torch.tensor([[[-2.0, 0.0, 0.0, 0.0]]]))[..., 0]
    assert not torch.equal(first, second)
