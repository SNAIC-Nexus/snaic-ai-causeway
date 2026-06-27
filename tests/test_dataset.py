import os
import yaml
import pytest
from causeway import db
from causeway.dataset import build_dataset_split

@pytest.fixture(autouse=True)
def use_test_db(monkeypatch, tmp_path):
    test_db_path = str(tmp_path / "test.db")
    monkeypatch.setattr(db, "DB_PATH", test_db_path)
    db.init_db()
    yield

def _make_image_tree(base, dates_cameras):
    for date_str, cam_id in dates_cameras:
        cam_dir = base / date_str / cam_id
        cam_dir.mkdir(parents=True, exist_ok=True)
        (cam_dir / f"img_{date_str}_{cam_id}.jpg").write_bytes(b"fake")

def test_split_assigns_last_day_to_val(tmp_path, monkeypatch):
    import causeway.dataset as dataset_mod
    base = tmp_path / "traffic_images"
    _make_image_tree(base, [("20260625", "2701"), ("20260626", "2701"), ("20260627", "2701")])
    monkeypatch.setattr(dataset_mod, "IMAGE_BASE_DIR", str(base))
    monkeypatch.setattr(dataset_mod, "LANE_LABELS_DIR", str(tmp_path / "ll"))
    monkeypatch.setattr(dataset_mod, "VEHICLE_LABELS_DIR", str(tmp_path / "vl"))
    monkeypatch.chdir(tmp_path)
    result = build_dataset_split(base_images_dir=str(base))
    assert result["lane_val"] == 1
    assert result["lane_train"] == 2

def test_split_warns_on_single_partition(tmp_path, monkeypatch, capsys):
    import causeway.dataset as dataset_mod
    base = tmp_path / "traffic_images"
    _make_image_tree(base, [("20260625", "2701")])
    monkeypatch.setattr(dataset_mod, "IMAGE_BASE_DIR", str(base))
    monkeypatch.chdir(tmp_path)
    build_dataset_split(base_images_dir=str(base))
    out = capsys.readouterr().out
    assert "WARNING" in out

def test_split_writes_yaml_files(tmp_path, monkeypatch):
    import causeway.dataset as dataset_mod
    base = tmp_path / "traffic_images"
    _make_image_tree(base, [("20260625", "2701"), ("20260627", "2701")])
    monkeypatch.setattr(dataset_mod, "IMAGE_BASE_DIR", str(base))
    monkeypatch.setattr(dataset_mod, "LANE_LABELS_DIR", str(tmp_path / "ll"))
    monkeypatch.setattr(dataset_mod, "VEHICLE_LABELS_DIR", str(tmp_path / "vl"))
    monkeypatch.chdir(tmp_path)
    build_dataset_split(base_images_dir=str(base))
    lane_yaml = tmp_path / "dataset_lane.yaml"
    assert lane_yaml.exists()
    data = yaml.safe_load(lane_yaml.read_text())
    assert "train" in data
    assert "val" in data
    assert data["nc"] == 2

def test_split_logs_to_sqlite(tmp_path, monkeypatch):
    import causeway.dataset as dataset_mod
    base = tmp_path / "traffic_images"
    _make_image_tree(base, [("20260625", "2701"), ("20260626", "2701")])
    monkeypatch.setattr(dataset_mod, "IMAGE_BASE_DIR", str(base))
    monkeypatch.setattr(dataset_mod, "LANE_LABELS_DIR", str(tmp_path / "ll"))
    monkeypatch.setattr(dataset_mod, "VEHICLE_LABELS_DIR", str(tmp_path / "vl"))
    monkeypatch.chdir(tmp_path)
    build_dataset_split(base_images_dir=str(base))
    summary = db.get_dataset_split_summary()
    assert len(summary) > 0
