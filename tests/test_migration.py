import os
import shutil
import pytest
from datetime import datetime
from causeway import db
from causeway.migration import _parse_image_metadata, migrate

@pytest.fixture(autouse=True)
def use_test_db(monkeypatch, tmp_path):
    test_db_path = str(tmp_path / "test.db")
    monkeypatch.setattr(db, "DB_PATH", test_db_path)
    db.init_db()
    yield

def test_parse_image_metadata_valid():
    fname = "Woodlands_Causeway_Towards_Johor_2701_20260625_221422.jpg"
    cam_id, date_str, dt = _parse_image_metadata(fname)
    assert cam_id == "2701"
    assert date_str == "20260625"
    assert dt == datetime(2026, 6, 25, 22, 14, 22)

def test_parse_image_metadata_unknown_camera():
    fname = "SomeOther_9999_20260625_221422.jpg"
    cam_id, date_str, dt = _parse_image_metadata(fname)
    assert cam_id is None

def test_parse_image_metadata_unrecognised_pattern():
    cam_id, date_str, dt = _parse_image_metadata("random_file.jpg")
    assert cam_id is None

def test_migrate_moves_file_to_correct_location(tmp_path, monkeypatch):
    import causeway.migration as migration_mod
    monkeypatch.setattr(migration_mod, "IMAGE_BASE_DIR", str(tmp_path / "traffic_images"))

    # Create a misplaced file at the root of traffic_images/
    base = tmp_path / "traffic_images"
    base.mkdir()
    src = base / "Woodlands_Causeway_Towards_Johor_2701_20260625_221422.jpg"
    src.write_bytes(b"fake_image_data")

    result = migrate(base_dir=str(base))

    assert result["moved"] == 1
    assert result["skipped"] == 0
    expected = base / "20260625" / "2701" / "Woodlands_Causeway_Towards_Johor_2701_20260625_221422.jpg"
    assert expected.exists()
    assert not src.exists()

def test_migrate_idempotent(tmp_path, monkeypatch):
    import causeway.migration as migration_mod
    base = tmp_path / "traffic_images"
    cam_dir = base / "20260625" / "2701"
    cam_dir.mkdir(parents=True)
    img = cam_dir / "Woodlands_Causeway_Towards_Johor_2701_20260625_221422.jpg"
    img.write_bytes(b"fake")

    result1 = migrate(base_dir=str(base))
    result2 = migrate(base_dir=str(base))

    assert result1["already_correct"] == 1
    assert result2["already_correct"] == 1
    assert img.exists()

def test_migrate_backfills_scrape_log(tmp_path, monkeypatch):
    import causeway.migration as migration_mod
    base = tmp_path / "traffic_images"
    cam_dir = base / "20260625" / "2701"
    cam_dir.mkdir(parents=True)
    img = cam_dir / "Woodlands_Causeway_Towards_Johor_2701_20260625_221422.jpg"
    img.write_bytes(b"fake")

    migrate(base_dir=str(base))

    last = db.get_last_scrape_timestamp("2701")
    assert last is not None
    assert "2026-06-25" in last
