"""Focused tests for the bounded and fully decoupled OBB Refine V2 path."""

from types import SimpleNamespace

import pytest
import torch

from train_yolo11_obb_refine import (
    configure_refine_only_training,
    keep_frozen_batch_norm_eval,
    load_ca_weights_into_refine_v2,
)
from ultralytics import YOLO
from ultralytics.nn.modules.head import OBBRefine, OBBRefineV2
from ultralytics.nn.tasks import OBBModel
from ultralytics.utils.loss import v8OBBLoss


def build_head(experiment: str = "aligned_identity") -> OBBRefineV2:
    """Build a small standalone V2 head."""
    head = OBBRefineV2(nc=1, ne=1, ne_refine=2, reg_max=32, ch=(32, 64, 128))
    head.set_refine_experiment(experiment)
    return head


def test_refine_v2_zero_identity_and_local_gradient():
    """Zero output is identity and the smooth transform has unit slope at zero."""
    head = build_head()
    raw = torch.zeros(1, 2, 3, requires_grad=True)
    bounded = head.bound_refine(raw)
    assert torch.equal(bounded, torch.zeros_like(bounded))
    bounded.sum().backward()
    assert torch.allclose(raw.grad, torch.ones_like(raw))

    boxes = torch.tensor([[[10.0, 20.0, 30.0], [10.0, 20.0, 30.0], [2.0, 4.0, 8.0], [8.0, 4.0, 2.0]]])
    refined = head._apply_wh_refine(boxes, raw.detach())
    assert torch.equal(refined, boxes)


def test_refine_v2_short_long_mapping():
    """Short/long residuals map back to the correct width/height axes."""
    head = build_head("direct_short_long")
    boxes = torch.tensor([[[0.0, 0.0], [0.0, 0.0], [2.0, 8.0], [8.0, 2.0]]])
    delta = torch.tensor([[[0.05, 0.05], [-0.02, -0.02]]])
    refined = head._apply_refine_delta(boxes, delta)

    assert torch.allclose(refined[:, 2, 0], boxes[:, 2, 0] * torch.exp(torch.tensor(0.05)))
    assert torch.allclose(refined[:, 3, 0], boxes[:, 3, 0] * torch.exp(torch.tensor(-0.02)))
    assert torch.allclose(refined[:, 2, 1], boxes[:, 2, 1] * torch.exp(torch.tensor(-0.02)))
    assert torch.allclose(refined[:, 3, 1], boxes[:, 3, 1] * torch.exp(torch.tensor(0.05)))


def test_refine_v2_profile_validation():
    """Unknown experiment names fail instead of silently changing residual semantics."""
    head = build_head()
    with pytest.raises(ValueError, match="Unknown Refine V2 experiment"):
        head.set_refine_experiment("unknown")


def test_refine_v2_rejects_legacy_cv5_state():
    """Equal-shaped legacy cv5 tensors must not load under V2 channel semantics."""
    legacy = OBBRefine(nc=1, ne=1, ne_refine=2, reg_max=32, ch=(32, 64, 128))
    v2 = build_head()
    with pytest.raises(RuntimeError, match="without the V2 marker"):
        v2.load_state_dict(legacy.state_dict(), strict=False)


def test_refine_v2_direct_target_loss_and_gradient():
    """Direct short/long supervision is zero at its target and backpropagates otherwise."""
    head = build_head("direct_short_long")
    criterion = object.__new__(v8OBBLoss)
    criterion.refine_head = head

    coarse = torch.tensor([[[0.0, 0.0, 2.0, 8.0, 0.0], [0.0, 0.0, 4.0, 10.0, 0.0]]])
    target = coarse.clone()
    pred_refine = torch.zeros(1, 2, 2, requires_grad=True)
    fg_mask = torch.tensor([[True, True]])
    refine_mask = torch.tensor([True, True])
    weights = torch.ones(2)
    stride = torch.ones(2, 1)

    zero_loss = criterion.calculate_refine_v2_direct_loss(
        coarse,
        pred_refine,
        target,
        fg_mask,
        weights,
        stride,
        refine_mask,
    )
    assert zero_loss.item() == pytest.approx(0.0, abs=1e-8)

    smaller_target = target.clone()
    smaller_target[..., 2:4] *= 0.9
    nonzero_loss = criterion.calculate_refine_v2_direct_loss(
        coarse,
        pred_refine,
        smaller_target,
        fg_mask,
        weights,
        stride,
        refine_mask,
    )
    nonzero_loss.backward()
    assert nonzero_loss.item() > 0
    assert pred_refine.grad is not None
    assert pred_refine.grad.abs().sum().item() > 0


def test_refine_v2_predicted_gate_matches_inference_rule():
    """The training gate uses the same AR-or-short-side rule as inference."""
    criterion = object.__new__(v8OBBLoss)
    criterion.hyp = SimpleNamespace(aux_geo_ar=30.0, aux_geo_ws=16.0)
    coarse = torch.tensor(
        [
            [
                [0.0, 0.0, 1.0, 40.0, 0.0],
                [0.0, 0.0, 10.0, 20.0, 0.0],
                [0.0, 0.0, 20.0, 30.0, 0.0],
            ]
        ]
    )
    gate = criterion.build_predicted_refine_gate(coarse, torch.ones(3, 1))
    assert torch.equal(gate, torch.tensor([[True, True, False]]))


def test_refine_v2_identity_loss_targets_raw_unmatched_residual():
    """Identity regularization keeps a recovery gradient even near tanh saturation."""
    head = build_head("aligned_identity")
    criterion = object.__new__(v8OBBLoss)
    criterion.refine_head = head
    raw = torch.tensor([[[0.0, 2.0], [0.0, -2.0]]], requires_grad=True)
    predicted_gate = torch.tensor([[True, True]])
    fg_mask = torch.tensor([[True, False]])

    loss = criterion.calculate_refine_v2_identity_loss(raw, predicted_gate, fg_mask)
    loss.backward()
    assert loss.item() > 0
    assert raw.grad[:, :, 1].abs().sum().item() > 0


def test_refine_v2_yaml_is_explicit_and_legacy_yaml_stays_legacy():
    """V2 and legacy YAMLs resolve to different head classes."""
    legacy = OBBModel("ultralytics/cfg/models/11/yolo11-obb-ca-refine.yaml", nc=1, verbose=False)
    v2 = OBBModel("ultralytics/cfg/models/11/yolo11-obb-ca-refine-v2.yaml", nc=1, verbose=False)
    assert isinstance(legacy.model[-1], OBBRefine)
    assert not isinstance(legacy.model[-1], OBBRefineV2)
    assert isinstance(v2.model[-1], OBBRefineV2)
    assert v2.model[-1].reg_max == 32


def test_refine_v2_strict_ca_transfer_rebuilds_dataset_nc(tmp_path):
    """Pure CA transfer rebuilds V2 for the checkpoint class count and only leaves cv5 new."""
    ca_model = OBBModel("ultralytics/cfg/models/11/yolo11-obb-ca.yaml", nc=1, verbose=False)
    checkpoint = tmp_path / "ca.pt"
    torch.save({"model": ca_model, "ema": None}, checkpoint)

    v2 = YOLO("ultralytics/cfg/models/11/yolo11-obb-ca-refine-v2.yaml")
    summary = load_ca_weights_into_refine_v2(v2, checkpoint)
    assert summary["nc"] == 1
    assert summary["shared_tensors"] > 0
    assert summary["new_v2_tensors"] > 0
    assert isinstance(v2.model.model[-1], OBBRefineV2)
    assert v2.model.model[-1].nc == 1
    assert v2.ckpt["refine_v2_ca_init"] is True


def test_refine_only_freezes_base_and_preserves_only_cv5_batch_norm_training():
    """The refine-only callback removes every shared tensor from optimization and BN updates."""
    model = OBBModel("ultralytics/cfg/models/11/yolo11-obb-ca-refine-v2.yaml", nc=1, verbose=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    trainer = SimpleNamespace(model=model, optimizer=optimizer)

    summary = configure_refine_only_training(trainer)
    assert summary["trainable_parameters"] > 0
    assert all((".cv5." in f".{name}.") == parameter.requires_grad for name, parameter in model.named_parameters())
    optimized = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    assert optimized == {id(parameter) for parameter in model.parameters() if parameter.requires_grad}

    model.train()
    keep_frozen_batch_norm_eval(trainer)
    base_bn = [module.training for name, module in model.named_modules() if name.endswith("bn") and ".cv5." not in name]
    refine_bn = [module.training for name, module in model.named_modules() if name.endswith("bn") and ".cv5." in name]
    assert base_bn and not any(base_bn)
    assert refine_bn and all(refine_bn)
