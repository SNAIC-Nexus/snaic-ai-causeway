import os
import pytest
from causeway import db

@pytest.fixture(autouse=True)
def use_test_db(monkeypatch, tmp_path):
    test_db_path = str(tmp_path / "test.db")
    monkeypatch.setattr(db, "DB_PATH", test_db_path)
    db.init_db()
    yield


def _make_approved_images(tmp_path, entries):
    """Create fake images + vehicle labels and mark them approved in DB."""
    base = tmp_path / "traffic_images"
    labels_base = tmp_path / "traffic_vehicle_labels"
    for date, cam, fname in entries:
        img_dir = base / date / cam
        img_dir.mkdir(parents=True, exist_ok=True)
        img_path = img_dir / fname
        img_path.write_bytes(b"\xff\xd8\xff")
        lbl_dir = labels_base / date / cam
        lbl_dir.mkdir(parents=True, exist_ok=True)
        lbl_path = lbl_dir / fname.replace(".jpg", ".txt")
        lbl_path.write_text("1 0.5 0.5 0.1 0.1\n")
        db.log_label(str(img_path), str(lbl_path), "vehicle", "morning")
        db.update_label_validation(str(img_path), "vehicle", "approved")
    return base


def test_get_sample_images_returns_n_per_camera(tmp_path, monkeypatch):
    import causeway.db as db_mod
    monkeypatch.setattr(db_mod, "DB_PATH", str(tmp_path / "test.db"))
    db_mod.init_db()

    entries = [
        ("20260625", "2701", f"img_2701_{i:02d}.jpg") for i in range(8)
    ] + [
        ("20260625", "2702", f"img_2702_{i:02d}.jpg") for i in range(3)
    ]
    base = _make_approved_images(tmp_path, entries)

    # Import after monkeypatching DB
    import importlib, evaluate_labeler
    importlib.reload(evaluate_labeler)

    result = evaluate_labeler.get_sample_images(
        cameras=["2701", "2702"],
        n=5,
        images_base=str(base),
    )
    assert len(result["2701"]) == 5   # capped at n=5
    assert len(result["2702"]) == 3   # only 3 available


def test_get_sample_images_falls_back_to_filesystem(tmp_path, monkeypatch):
    """When no approved labels exist, fall back to any .jpg in traffic_images/."""
    import causeway.db as db_mod
    monkeypatch.setattr(db_mod, "DB_PATH", str(tmp_path / "test.db"))
    db_mod.init_db()

    base = tmp_path / "traffic_images"
    cam_dir = base / "20260625" / "2701"
    cam_dir.mkdir(parents=True, exist_ok=True)
    for i in range(4):
        (cam_dir / f"img_{i}.jpg").write_bytes(b"\xff\xd8\xff")

    import importlib, evaluate_labeler
    importlib.reload(evaluate_labeler)

    result = evaluate_labeler.get_sample_images(
        cameras=["2701"],
        n=5,
        images_base=str(base),
    )
    assert len(result["2701"]) == 4  # all 4 available (fewer than n)
