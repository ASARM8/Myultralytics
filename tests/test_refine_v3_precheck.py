"""CPU-only tests for the Refine V3 precheck's frozen decision logic."""

import math

import pytest

from myscripts.refine_v3_precheck import binary_average_precision, binary_roc_auc, precheck_decision


def test_binary_metrics_are_perfect_for_separated_scores():
    labels = [0, 0, 1, 1]
    scores = [0.1, 0.2, 0.8, 0.9]
    assert binary_roc_auc(labels, scores) == pytest.approx(1.0)
    assert binary_average_precision(labels, scores) == pytest.approx(1.0)


def test_binary_auc_ties_are_neutral():
    labels = [0, 1, 0, 1]
    scores = [0.5, 0.5, 0.5, 0.5]
    assert binary_roc_auc(labels, scores) == pytest.approx(0.5)
    assert binary_average_precision(labels, scores) == pytest.approx(0.5)


def test_binary_metrics_reject_single_class_auc():
    assert math.isnan(binary_roc_auc([1, 1], [0.1, 0.2]))
    assert binary_average_precision([1, 1], [0.1, 0.2]) == pytest.approx(1.0)


def test_precheck_requires_both_predictability_checks():
    decision = precheck_decision(
        quality_pass=True,
        residual_pass=False,
        center_gain=0.0012,
        angle_gain=0.0009,
        extra_dof_min_gain=0.001,
    )
    assert decision == {
        "quality_pass": True,
        "residual_pass": False,
        "recommend_v3": False,
        "include_center": True,
        "include_angle": False,
    }
