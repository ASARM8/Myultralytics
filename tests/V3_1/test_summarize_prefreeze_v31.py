import pytest

from myscripts.V3_1.summarize_prefreeze_v31 import evaluate_scope, identity, noninferior


def row(map50_95, ap75, ap90):
    return {
        "precision": 0.8,
        "recall": 0.8,
        "map50": 0.8,
        "map50_95": map50_95,
        "ap75": ap75,
        "ap90": ap90,
        "ap95": 0.1,
    }


def test_noninferiority_is_not_best_score_selection():
    metrics = {"candidate": row(0.699, 0.790, 0.348), "reference": row(0.700, 0.791, 0.349)}
    passed, deltas = noninferior(metrics, "candidate", "reference", 0.002)
    assert passed
    assert deltas["map50_95"] == pytest.approx(-0.001)


def test_identity_checks_all_reported_metrics():
    metrics = {"candidate": row(0.7, 0.79, 0.35), "reference": row(0.7, 0.79, 0.35)}
    passed, _ = identity(metrics, "candidate", "reference", 5e-4)
    assert passed
    metrics["candidate"]["precision"] += 0.001
    passed, _ = identity(metrics, "candidate", "reference", 5e-4)
    assert not passed


def test_scope_requires_refine_gain_and_high_iou_safety():
    metrics = {
        "coarse": row(0.45, 0.44, 0.25),
        "all_refine": row(0.70, 0.79, 0.35),
        "short_only": row(0.699, 0.789, 0.349),
        "short_only_all": row(0.700, 0.790, 0.350),
        "short_only_all_no_renms": row(0.700, 0.790, 0.350),
    }
    result = evaluate_scope(metrics, noninferiority_tolerance=0.002, identity_tolerance=5e-4, minimum_gain=0.002)
    assert result["short_branch_noninferior_to_full_all"]
    assert result["all_proposals_noninferior_to_quality_gate"]
    assert result["renms_identity_for_short_all"]
    assert result["short_all_refine_gain_pass"]
