"""Torch-free unit tests for the Refine V3 training policy helpers."""

import argparse

import pytest

from myscripts.V3.train_refine_v3 import parse_thresholds, select_holdout_row


def test_quality_threshold_parser_is_sorted_unique_and_three_decimal():
    assert parse_thresholds("0.7,0.3004,0.7,0.5") == (0.3, 0.5, 0.7)


@pytest.mark.parametrize("value", ["", "0", "1", "-0.1", "abc"])
def test_quality_threshold_parser_rejects_invalid_values(value):
    with pytest.raises(argparse.ArgumentTypeError):
        parse_thresholds(value)


def test_holdout_selection_uses_map_then_strict_quality_tie_breaks():
    rows = [
        {"variant": "coarse", "map50_95": 0.4, "ap75": 0.4, "ap90": 0.2},
        {"variant": "quality_0.300", "map50_95": 0.41, "ap75": 0.42, "ap90": 0.20},
        {"variant": "quality_0.500", "map50_95": 0.41, "ap75": 0.42, "ap90": 0.20},
        {"variant": "quality_0.700", "map50_95": 0.409, "ap75": 0.43, "ap90": 0.21},
    ]
    assert select_holdout_row(rows)["variant"] == "quality_0.500"


def test_holdout_selection_ignores_degenerate_gate_when_a_viable_gate_exists():
    rows = [
        {"variant": "quality_0.300", "map50_95": 0.42, "ap75": 0.42, "ap90": 0.2, "gate_ratio": 1.0},
        {"variant": "quality_0.500", "map50_95": 0.41, "ap75": 0.41, "ap90": 0.2, "gate_ratio": 0.4},
    ]
    assert select_holdout_row(rows)["variant"] == "quality_0.500"
