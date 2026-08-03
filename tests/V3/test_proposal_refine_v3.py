import torch

from ultralytics.nn.modules.refine_v3 import OBBProposalRefinerV3


def build_module():
    return OBBProposalRefinerV3(
        p2_channels=16,
        p3_channels=24,
        roi_channels=8,
        roi_size=(3, 8),
        hidden_channels=16,
    )


def test_v3_zero_initialization_is_identity():
    module = build_module().eval()
    p2 = torch.randn(2, 16, 32, 32)
    p3 = torch.randn(2, 24, 16, 16)
    proposals = torch.tensor(
        [
            [[32.0, 40.0, 4.0, 40.0, 0.2], [60.0, 70.0, 30.0, 6.0, 0.8]],
            [[45.0, 50.0, 8.0, 20.0, 0.1], [80.0, 64.0, 24.0, 12.0, 1.0]],
        ]
    )
    scores = torch.full((2, 2), 0.8)
    output = module(p2, p3, proposals, scores, image_size=(128, 128))
    assert torch.equal(output["residual"], torch.zeros_like(output["residual"]))
    assert torch.equal(module.apply_residual(proposals, output["residual"]), proposals)


def test_v3_encode_apply_recovers_scale_and_center():
    proposals = torch.tensor([[[50.0, 60.0, 8.0, 40.0, 0.3], [30.0, 20.0, 30.0, 6.0, 0.8]]])
    targets = proposals.clone()
    targets[..., 0] += 2.0
    targets[..., 1] -= 1.0
    targets[..., 2] *= 1.2
    targets[..., 3] *= 0.9
    residual = OBBProposalRefinerV3.encode_targets(proposals, targets)
    refined = OBBProposalRefinerV3.apply_residual(proposals, residual)
    assert torch.allclose(refined[..., :4], targets[..., :4], atol=1e-5, rtol=1e-5)
    assert torch.equal(refined[..., 4], proposals[..., 4])


def test_v3_invalid_proposals_are_identity_masked():
    module = build_module().eval()
    with torch.no_grad():
        module.geometry_head.bias.fill_(0.2)
    p2 = torch.randn(1, 16, 32, 32)
    p3 = torch.randn(1, 24, 16, 16)
    proposals = torch.tensor([[[32.0, 40.0, 4.0, 40.0, 0.2], [60.0, 70.0, 30.0, 6.0, 0.8]]])
    scores = torch.full((1, 2), 0.8)
    valid = torch.tensor([[True, False]])
    output = module(p2, p3, proposals, scores, image_size=(128, 128), valid_mask=valid)
    assert output["residual"][0, 0].abs().sum() > 0
    assert torch.equal(output["residual"][0, 1], torch.zeros(4))
    assert output["quality_logit"][0, 1, 0] == -20
