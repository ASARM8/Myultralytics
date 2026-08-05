"""Torch-free tests for Refine V3 truth-audit statistics."""

import math

import pytest

from myscripts.V3.audit_refine_v3 import (
    VARIANTS,
    assign_bin,
    binary_auc,
    binary_summary,
    pearson,
    read_excluded_images,
    retained_indices,
    stable_seed,
    summarize_rows,
)


def test_stable_seed_is_repeatable_and_key_sensitive():
    assert stable_seed("image-a", 7) == stable_seed("image-a", 7)
    assert stable_seed("image-a", 7) != stable_seed("image-b", 7)


def test_v31_prefreeze_variants_isolate_branch_gate_and_renms():
    assert "short_only_all" in VARIANTS
    assert "short_only_all_no_renms" in VARIANTS
    assert "all_refine_no_renms" in VARIANTS
    assert "long_only_all" in VARIANTS


def test_binary_auc_is_tie_aware_and_exact_for_perfect_ordering():
    assert binary_auc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == pytest.approx(1.0)
    assert binary_auc([0.5, 0.5], [0, 1]) == pytest.approx(0.5)
    assert math.isnan(binary_auc([0.1, 0.2], [0, 0]))


def test_binary_summary_uses_frozen_threshold():
    summary = binary_summary([0.1, 0.4, 0.8, 0.9], [0, 1, 0, 1], 0.3)
    assert summary["tp"] == 2
    assert summary["fp"] == 1
    assert summary["fn"] == 0
    assert summary["tn"] == 1
    assert summary["precision"] == pytest.approx(2 / 3)
    assert summary["recall"] == pytest.approx(1.0)


def test_pearson_and_bin_assignment():
    assert pearson([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0)
    assert assign_bin(3.9, (4.0, 8.0)) == "<4"
    assert assign_bin(4.0, (4.0, 8.0)) == "<8"
    assert assign_bin(8.0, (4.0, 8.0)) == ">=8"


def test_subgroup_summary_reports_improvement_and_gate_ratios():
    rows = [
        {"coarse_short": 3.0, "coarse_iou": 0.4, "selected_iou": 0.5, "selected_delta_iou": 0.1, "gate": 1},
        {
            "coarse_short": 3.5,
            "coarse_iou": 0.6,
            "selected_iou": 0.55,
            "selected_delta_iou": -0.05,
            "gate": 1,
        },
        {"coarse_short": 10.0, "coarse_iou": 0.7, "selected_iou": 0.7, "selected_delta_iou": 0.0, "gate": 0},
    ]
    summary = summarize_rows(rows, "coarse_short", (4.0, 8.0, 16.0))
    first = summary[0]
    assert first["bin"] == "<4"
    assert first["count"] == 2
    assert first["delta_iou_mean"] == pytest.approx(0.025)
    assert first["improved_ratio"] == pytest.approx(0.5)
    assert first["worsened_ratio"] == pytest.approx(0.5)
    assert first["gate_ratio"] == pytest.approx(1.0)


def test_image_exclusion_manifest_requires_every_path_to_match(tmp_path):
    excluded = tmp_path / "excluded.txt"
    excluded.write_text("/data/val/b.jpg\n", encoding="utf-8")
    rows = read_excluded_images(excluded)
    indices, found = retained_indices(["/data/val/a.jpg", "/data/val/b.jpg"], rows)
    assert indices == [0]
    assert found == ["/data/val/b.jpg"]
    with pytest.raises(RuntimeError, match="absent"):
        retained_indices(["/data/val/a.jpg"], rows)

    excluded.write_text("# no exact overlap\n", encoding="utf-8")
    assert read_excluded_images(excluded) == set()
