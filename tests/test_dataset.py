import os
import shutil
import yaml
import pytest
from causeway import db
from causeway.dataset import build_dataset_split, export_curated_dataset

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

def _make_approved_vehicle_labels(tmp_path):
    """Create fake images + label files and log them as approved vehicle labels."""
    base = tmp_path / "traffic_images"
    labels_base = tmp_path / "traffic_vehicle_labels"

    entries = [
        ("20260625", "2701", "cam_2701_20260625_080000.jpg"),
        ("20260625", "2701", "cam_2701_20260625_083000.jpg"),
        ("20260626", "2701", "cam_2701_20260626_090000.jpg"),
        ("20260626", "2701", "cam_2701_20260626_093000.jpg"),
        ("20260627", "2701", "cam_2701_20260627_100000.jpg"),
    ]

    for date, cam, fname in entries:
        img_dir = base / date / cam
        img_dir.mkdir(parents=True, exist_ok=True)
        img_path = img_dir / fname
        img_path.write_bytes(b"\xff\xd8\xff")  # minimal JPEG magic bytes

        lbl_dir = labels_base / date / cam
        lbl_dir.mkdir(parents=True, exist_ok=True)
        lbl_path = lbl_dir / (fname.replace(".jpg", ".txt"))
        lbl_path.write_text("1 0.5 0.5 0.1 0.1\n")

        db.log_label(str(img_path), str(lbl_path), "vehicle", "morning")
        db.update_label_validation(str(img_path), "vehicle", "approved")

    return base, labels_base

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

def test_export_curated_creates_directory_layout(tmp_path, monkeypatch):
    import causeway.dataset as dataset_mod
    base, labels_base = _make_approved_vehicle_labels(tmp_path)
    monkeypatch.setattr(dataset_mod, "IMAGE_BASE_DIR", str(base))
    monkeypatch.setattr(dataset_mod, "VEHICLE_LABELS_DIR", str(labels_base))
    monkeypatch.chdir(tmp_path)

    out_dir = str(tmp_path / "dataset" / "curated")
    result = export_curated_dataset(out_dir)

    assert result["train"] > 0
    assert result["val"] > 0
    assert (tmp_path / "dataset" / "curated" / "images" / "train").exists()
    assert (tmp_path / "dataset" / "curated" / "images" / "val").exists()
    assert (tmp_path / "dataset" / "curated" / "labels" / "train").exists()
    assert (tmp_path / "dataset" / "curated" / "labels" / "val").exists()


def test_export_curated_writes_yaml(tmp_path, monkeypatch):
    import causeway.dataset as dataset_mod
    base, labels_base = _make_approved_vehicle_labels(tmp_path)
    monkeypatch.setattr(dataset_mod, "IMAGE_BASE_DIR", str(base))
    monkeypatch.setattr(dataset_mod, "VEHICLE_LABELS_DIR", str(labels_base))
    monkeypatch.chdir(tmp_path)

    out_dir = str(tmp_path / "dataset" / "curated")
    result = export_curated_dataset(out_dir)

    yaml_path = tmp_path / "dataset_vehicle_curated.yaml"
    assert yaml_path.exists()
    data = yaml.safe_load(yaml_path.read_text())
    assert data["nc"] == 4
    assert data["names"] == {0: "motorcycle", 1: "car", 2: "bus", 3: "truck"}
    assert result["yaml_path"] == str(yaml_path)


def test_export_curated_splits_by_day(tmp_path, monkeypatch):
    import causeway.dataset as dataset_mod
    base, labels_base = _make_approved_vehicle_labels(tmp_path)
    monkeypatch.setattr(dataset_mod, "IMAGE_BASE_DIR", str(base))
    monkeypatch.setattr(dataset_mod, "VEHICLE_LABELS_DIR", str(labels_base))
    monkeypatch.chdir(tmp_path)

    out_dir = str(tmp_path / "dataset" / "curated")
    result = export_curated_dataset(out_dir)

    # With 3 days: first 2 → train, last 1 → val (same day-split logic as build_dataset_split)
    val_images = list((tmp_path / "dataset" / "curated" / "images" / "val").glob("*.jpg"))
    train_images = list((tmp_path / "dataset" / "curated" / "images" / "train").glob("*.jpg"))
    assert len(val_images) > 0
    assert len(train_images) > 0
    assert result["train"] == len(train_images)
    assert result["val"] == len(val_images)


def test_export_curated_returns_zero_when_no_approved(tmp_path, monkeypatch):
    import causeway.dataset as dataset_mod
    monkeypatch.setattr(dataset_mod, "IMAGE_BASE_DIR", str(tmp_path / "traffic_images"))
    monkeypatch.setattr(dataset_mod, "VEHICLE_LABELS_DIR", str(tmp_path / "traffic_vehicle_labels"))
    monkeypatch.chdir(tmp_path)

    out_dir = str(tmp_path / "dataset" / "curated")
    result = export_curated_dataset(out_dir)
    assert result["train"] == 0
    assert result["val"] == 0
