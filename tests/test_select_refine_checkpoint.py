"""Tests for deterministic Refine checkpoint selection."""

import pytest

from myscripts.select_refine_checkpoint import SelectionThresholds, analyze_curve, write_env


def make_pair(
    epoch: int,
    *,
    d_map: float,
    d_ap75: float,
    d_ap90: float,
    d_ap95: float,
    coarse_map: float = 0.45,
) -> list[dict[str, str]]:
    """Create one minimal paired curve record."""
    base = {
        "epoch": str(epoch),
        "weights": f"/runs/weights/epoch{epoch - 1}.pt",
        "data": "/data/dataset.yaml",
        "imgsz": "640",
        "split": "val",
        "profile": "curve",
        "fresh_load": "True",
        "refine_version": "2",
        "refine_experiment": "stable_raw_short_long",
        "refine_delta_max": "0.05",
        "refine_target_limit": "0.04",
        "precision": "0.8",
        "recall": "0.7",
        "map50": "0.75",
        "map50_95": str(coarse_map),
        "ap55": "0.7",
        "ap60": "0.65",
        "ap65": "0.6",
        "ap70": "0.55",
        "ap75": "0.44",
        "ap80": "0.4",
        "ap85": "0.3",
        "ap90": "0.24",
        "ap95": "0.1",
    }
    coarse = {**base, "variant": "coarse"}
    normal = {
        **base,
        "variant": "normal",
        "map50_95": str(coarse_map + d_map),
        "ap75": str(0.44 + d_ap75),
        "ap90": str(0.24 + d_ap90),
        "ap95": str(0.1 + d_ap95),
    }
    return [coarse, normal]


def analyze(rows):
    """Run the production V2.2 rule used by the paper experiment."""
    return analyze_curve(
        rows,
        thresholds=SelectionThresholds(),
        coarse_tolerance=5e-4,
        expect_profile="stable_raw_short_long",
        expect_delta_max=0.05,
        expect_target_limit=0.04,
    )


def test_selects_highest_valid_map_gain_and_rejects_high_iou_failure():
    """A larger mAP gain cannot win if AP95 violates the frozen safety boundary."""
    rows = []
    rows += make_pair(1, d_map=0.0025, d_ap75=0.002, d_ap90=0.0007, d_ap95=0.0001)
    rows += make_pair(2, d_map=0.0030, d_ap75=0.003, d_ap90=0.0010, d_ap95=-0.0011)
    rows += make_pair(3, d_map=0.0023, d_ap75=0.002, d_ap90=0.0010, d_ap95=0.0002)

    result = analyze(rows)

    assert result["candidate_count"] == 2
    assert result["selected"]["epoch"] == 1
    assert result["selected"]["weights"].endswith("epoch0.pt")


def test_exact_selection_tie_prefers_earlier_epoch():
    """Tie-breaking is deterministic and does not favor later training."""
    rows = []
    rows += make_pair(2, d_map=0.0025, d_ap75=0.001, d_ap90=0.0, d_ap95=0.0)
    rows += make_pair(1, d_map=0.0025, d_ap75=0.001, d_ap90=0.0, d_ap95=0.0)

    assert analyze(rows)["selected"]["epoch"] == 1


def test_rejects_coarse_drift_across_checkpoints():
    """Selection is invalid if supposedly frozen CA metrics move between checkpoints."""
    rows = []
    rows += make_pair(1, d_map=0.0025, d_ap75=0.001, d_ap90=0.0, d_ap95=0.0, coarse_map=0.45)
    rows += make_pair(2, d_map=0.0025, d_ap75=0.001, d_ap90=0.0, d_ap95=0.0, coarse_map=0.451)

    with pytest.raises(ValueError, match="coarse-only"):
        analyze(rows)


def test_rejects_wrong_checkpoint_profile():
    """A curve cannot silently mix V2.1 and V2.2 semantics."""
    rows = make_pair(1, d_map=0.0025, d_ap75=0.001, d_ap90=0.0, d_ap95=0.0)
    rows[0]["refine_experiment"] = "conservative_short_long"

    with pytest.raises(ValueError, match="profile"):
        analyze(rows)


def test_rejects_test_split_selection():
    """The test split must remain untouched until checkpoint selection is frozen."""
    rows = make_pair(1, d_map=0.0025, d_ap75=0.001, d_ap90=0.0, d_ap95=0.0)
    for row in rows:
        row["split"] = "test"

    with pytest.raises(ValueError, match="validation split"):
        analyze(rows)


def test_output_env_supports_version_specific_weight_variable(tmp_path):
    """V2.3 can reuse the selector without exporting a misleading V2.2 variable name."""
    result = analyze(make_pair(1, d_map=0.0025, d_ap75=0.001, d_ap90=0.0, d_ap95=0.0))
    output = tmp_path / "selected.env"
    write_env(output, result, "SELECTED_V23")
    assert output.read_text(encoding="utf-8") == (
        "SELECTED_EPOCH=1\nSELECTED_V23=/runs/weights/epoch0.pt\n"
    )


def test_output_env_rejects_shell_code_as_variable_name(tmp_path):
    """The configurable shell variable name must remain a plain identifier."""
    result = analyze(make_pair(1, d_map=0.0025, d_ap75=0.001, d_ap90=0.0, d_ap95=0.0))
    with pytest.raises(ValueError, match="shell"):
        write_env(tmp_path / "selected.env", result, "SELECTED;touch_x")
