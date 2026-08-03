import pytest

from myscripts.V3.refine_rotated_roi_probe import group_key, grouped_fit_holdout_indices


def test_grouped_split_has_no_group_overlap():
    groups = ["image_a"] * 3 + ["image_b"] * 2 + ["image_c"] * 4 + ["image_d"]
    fit, holdout = grouped_fit_holdout_indices(groups, 0.25, seed=7)
    assert {groups[index] for index in fit}.isdisjoint({groups[index] for index in holdout})
    assert sorted(fit + holdout) == list(range(len(groups)))


def test_group_key_defaults_to_complete_image_path():
    first = group_key("scene/a/image001.jpg")
    second = group_key("scene/b/image001.jpg")
    assert first != second


def test_group_key_can_extract_scene():
    assert group_key("/data/scene12_frame003.jpg", r"(scene\d+)_") == "scene12"
    with pytest.raises(ValueError):
        group_key("/data/other.jpg", r"(scene\d+)_")
