"""Unit tests for Refine V3 matching and leakage controls."""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from myscripts.V3 import runtime
from myscripts.V3.runtime import (
    FrozenCAExtractor,
    align_equivalent_targets,
    greedy_class_match,
    group_key,
    prediction_to_raw,
    rerun_rotated_nms,
    split_dataset_indices,
)


def test_group_split_is_deterministic_and_has_no_overlap():
    paths = [f"/data/scene{i // 2}_frame{i}.jpg" for i in range(40)]
    pattern = r"(scene\d+)_"
    first = split_dataset_indices(paths, 0.25, 0, pattern)
    second = split_dataset_indices(paths, 0.25, 0, pattern)
    assert first[:2] == second[:2]
    assert not (first[2] & first[3])


def test_group_regex_requires_a_capture_group():
    with pytest.raises(ValueError, match="capture group"):
        group_key("scene1_frame2.jpg", r"scene\d+")


def test_greedy_match_is_class_aware_and_one_to_one():
    proposals = torch.tensor([[10.0, 10.0, 4.0, 20.0, 0.0], [10.0, 10.0, 4.0, 20.0, 0.0]])
    targets = proposals[:1].clone()
    proposal_index, gt_index = greedy_class_match(
        proposals,
        torch.tensor([0, 1]),
        targets,
        torch.tensor([0]),
        minimum_iou=0.3,
    )
    assert proposal_index.tolist() == [0]
    assert gt_index.tolist() == [0]


def test_equivalent_obb_target_prefers_width_height_swap_when_angle_is_closer():
    proposals = torch.tensor([[10.0, 10.0, 20.0, 4.0, 0.0]])
    targets = torch.tensor([[10.0, 10.0, 4.0, 20.0, -torch.pi / 2]])
    aligned = align_equivalent_targets(proposals, targets)
    assert torch.allclose(aligned[:, 2:4], proposals[:, 2:4])


def test_frozen_ca_features_are_normal_tensors_available_to_downstream_backward(monkeypatch):
    class FakeHead(nn.Module):
        def forward(self, features):
            batch = features[0].shape[0]
            return features[0].new_zeros((batch, 6, 1))

    class FakeCore(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.ModuleList((nn.Identity(), nn.Identity(), nn.Conv2d(3, 4, 1), FakeHead()))

        def forward(self, images):
            p2 = self.model[2](images)
            p3 = F.avg_pool2d(p2, 2)
            return self.model[-1]([p3])

    def fake_nms(prediction, *_args, **_kwargs):
        detection = prediction.new_tensor([[16.0, 16.0, 4.0, 20.0, 0.8, 0.0, 0.2]])
        return [detection.clone() for _ in range(prediction.shape[0])]

    monkeypatch.setattr(runtime.nms, "non_max_suppression", fake_nms)
    extractor = FrozenCAExtractor(
        FakeCore().eval(),
        device=torch.device("cpu"),
        nc=1,
        conf=0.01,
        nms_iou=0.7,
        max_det=300,
        amp=False,
    )
    try:
        batch = {"img": torch.zeros(2, 3, 32, 32, dtype=torch.uint8)}
        _images, p2, p3, _detections = extractor.infer(batch)
        assert not torch.is_inference(p2)
        assert not torch.is_inference(p3)
        downstream = nn.Conv2d(4, 2, 1)
        downstream(p2).sum().backward()
        assert downstream.weight.grad is not None
    finally:
        extractor.close()


def test_six_one_class_obb_proposals_do_not_trigger_end2end_shape_shortcut():
    boxes = torch.tensor(
        [[10.0 + 20.0 * index, 16.0, 4.0, 12.0, 0.1] for index in range(6)], dtype=torch.float32
    )
    scores = torch.linspace(0.9, 0.4, 6)
    classes = torch.zeros(6, dtype=torch.long)
    raw = prediction_to_raw(boxes, scores, classes, nc=1)
    assert raw.shape == (6, 7)
    assert torch.equal(raw[:, -1], torch.zeros(6))
    output = rerun_rotated_nms(boxes, scores, classes, nc=1, conf=0.01, nms_iou=0.7, max_det=300)
    assert output["bboxes"].shape[0] == 6
    assert torch.allclose(output["bboxes"], boxes)
    assert torch.allclose(output["conf"], scores)
