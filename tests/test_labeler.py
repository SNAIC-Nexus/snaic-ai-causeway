import os
import json
import pytest
from causeway import db
from causeway.labeler import _parse_hour, _get_shift, _normalize_polygon, generate_lane_labels

@pytest.fixture(autouse=True)
def use_test_db(monkeypatch, tmp_path):
    test_db_path = str(tmp_path / "test.db")
    monkeypatch.setattr(db, "DB_PATH", test_db_path)
    db.init_db()
    yield

def test_parse_hour_from_filename():
    fname = "Woodlands_Causeway_Towards_Johor_2701_20260625_143022.jpg"
    assert _parse_hour(fname) == 14

def test_parse_hour_fallback(monkeypatch):
    from datetime import datetime
    monkeypatch.setattr("causeway.labeler.datetime", type("FakeDT", (), {"now": staticmethod(lambda: type("T", (), {"hour": 9})())}))
    result = _parse_hour("no_timestamp_here.jpg")
    assert isinstance(result, int)

def test_get_shift_morning():
    assert _get_shift(6) == "morning"
    assert _get_shift(11) == "morning"

def test_get_shift_afternoon():
    assert _get_shift(12) == "afternoon"
    assert _get_shift(18) == "afternoon"

def test_get_shift_night():
    assert _get_shift(19) == "night"
    assert _get_shift(5) == "night"
    assert _get_shift(0) == "night"

def test_normalize_polygon_basic():
    points = [[960, 540], [1920, 1080]]
    result = _normalize_polygon(points, w=1920, h=1080)
    assert result == pytest.approx([0.5, 0.5, 1.0, 1.0])

def test_normalize_polygon_clips_to_unit():
    points = [[0, 0], [1920, 1080]]
    result = _normalize_polygon(points)
    assert all(0.0 <= v <= 1.0 for v in result)

def test_generate_lane_labels_creates_txt_files(tmp_path, monkeypatch):
    import causeway.labeler as labeler_mod
    from causeway.config import CAMERA_CONFIG_PATH

    base_img_dir = tmp_path / "traffic_images"
    lane_lbl_dir = tmp_path / "traffic_lane_labels"
    cam_dir = base_img_dir / "20260625" / "2701"
    cam_dir.mkdir(parents=True)

    img_path = cam_dir / "Woodlands_Causeway_Towards_Johor_2701_20260625_143022.jpg"
    img_path.write_bytes(b"fake")

    monkeypatch.setattr(labeler_mod, "IMAGE_BASE_DIR", str(base_img_dir))
    monkeypatch.setattr(labeler_mod, "LANE_LABELS_DIR", str(lane_lbl_dir))

    count = generate_lane_labels(base_images_dir=str(base_img_dir))

    assert count == 1
    label_file = lane_lbl_dir / "20260625" / "2701" / "Woodlands_Causeway_Towards_Johor_2701_20260625_143022.txt"
    assert label_file.exists()
    content = label_file.read_text().strip().splitlines()
    assert len(content) == 2  # 2701 afternoon has 2 lanes
    assert content[0].startswith("0 ") or content[0].startswith("1 ")

def test_generate_vehicle_labels_accepts_model_path_param(tmp_path, monkeypatch):
    """generate_vehicle_labels must accept a model_path kwarg and pass it to YOLO()."""
    import causeway.labeler as labeler_mod
    from causeway import db as db_mod

    # Patch DB path
    test_db = str(tmp_path / "test.db")
    monkeypatch.setattr(db_mod, "DB_PATH", test_db)
    db_mod.init_db()

    # Empty image dir — no images to process, so YOLO is never actually loaded
    base = tmp_path / "traffic_images"
    base.mkdir()
    monkeypatch.setattr(labeler_mod, "IMAGE_BASE_DIR", str(base))

    # Should accept model_path without error (no images → YOLO never instantiated)
    count = labeler_mod.generate_vehicle_labels(
        base_images_dir=str(base),
        model_path="models/causeway_vehicle_v1.pt",
    )
    assert count == 0  # empty image dir, nothing processed


def test_generate_lane_labels_skips_already_labeled(tmp_path, monkeypatch):
    import causeway.labeler as labeler_mod

    base_img_dir = tmp_path / "traffic_images"
    lane_lbl_dir = tmp_path / "traffic_lane_labels"
    cam_dir = base_img_dir / "20260625" / "2701"
    cam_dir.mkdir(parents=True)
    lbl_dir = lane_lbl_dir / "20260625" / "2701"
    lbl_dir.mkdir(parents=True)

    fname = "Woodlands_Causeway_Towards_Johor_2701_20260625_143022"
    (cam_dir / f"{fname}.jpg").write_bytes(b"fake")
    (lbl_dir / f"{fname}.txt").write_text("0 0.5 0.5")

    monkeypatch.setattr(labeler_mod, "IMAGE_BASE_DIR", str(base_img_dir))
    monkeypatch.setattr(labeler_mod, "LANE_LABELS_DIR", str(lane_lbl_dir))

    count = generate_lane_labels(base_images_dir=str(base_img_dir))
    assert count == 0
