import pytest

from myscripts.V3_1.prepare_prefreeze_v31 import scene_id, source_frame_id


def test_source_frame_id_removes_only_the_tile_suffix():
    assert source_frame_id("/dataset/44_00701_x1920_y1280.jpg") == "44_00701"
    assert source_frame_id("/dataset/other_name.jpg") == "other_name"


def test_scene_id_requires_a_capture_group():
    path = "/dataset/44_00701_x1920_y1280.jpg"
    assert scene_id(path, r"^([^_]+)_") == "44"
    with pytest.raises(ValueError, match="capture group"):
        scene_id(path, r"^missing")
