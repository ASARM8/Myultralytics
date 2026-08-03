"""CPU-only tests for proposal-oracle argument and aggregation helpers."""

import argparse

import pytest

from myscripts.V3.refine_proposal_oracle import aggregate_recall, parse_iou_thresholds


def test_parse_iou_thresholds_is_stable_and_unique():
    assert parse_iou_thresholds("0.5,0.75,0.5,0.9") == (0.5, 0.75, 0.9)


@pytest.mark.parametrize("value", ["", "0", "1.1", "abc"])
def test_parse_iou_thresholds_rejects_invalid_values(value):
    with pytest.raises(argparse.ArgumentTypeError):
        parse_iou_thresholds(value)


def test_aggregate_recall_uses_gt_weighted_counts():
    rows = [
        {"source": "pre", "gt_count": 1, "recalled_0.50": 1, "recalled_0.75": 1},
        {"source": "pre", "gt_count": 9, "recalled_0.50": 3, "recalled_0.75": 0},
        {"source": "post", "gt_count": 2, "recalled_0.50": 1, "recalled_0.75": 0},
    ]
    summary = aggregate_recall(rows, (0.5, 0.75))
    assert summary[0]["source"] == "pre"
    assert summary[0]["proposal_recall"] == 0.4
    assert summary[1]["proposal_recall"] == 0.1
    assert summary[2]["source"] == "post"
    assert summary[2]["proposal_recall"] == 0.5
