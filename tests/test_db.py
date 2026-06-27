import pytest
import sqlite3
from causeway import db

@pytest.fixture(autouse=True)
def use_test_db(monkeypatch, tmp_path):
    test_db_path = str(tmp_path / "test.db")
    monkeypatch.setattr(db, "DB_PATH", test_db_path)
    db.init_db()
    yield
    # cleanup handled by tmp_path

def test_init_db_creates_tables():
    conn = db.get_connection()
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    assert "scrape_log" in tables
    assert "label_log" in tables
    assert "dataset_splits" in tables

def test_log_scrape_and_retrieve():
    db.log_scrape("2026-06-27T10:00:00", "2701", "/some/path.jpg", "success")
    last = db.get_last_scrape_timestamp()
    assert last == "2026-06-27T10:00:00"

def test_get_last_scrape_timestamp_by_camera():
    db.log_scrape("2026-06-27T09:00:00", "2702", "/path/a.jpg", "success")
    db.log_scrape("2026-06-27T10:00:00", "2701", "/path/b.jpg", "success")
    assert db.get_last_scrape_timestamp("2702") == "2026-06-27T09:00:00"
    assert db.get_last_scrape_timestamp("2701") == "2026-06-27T10:00:00"

def test_get_last_scrape_timestamp_returns_none_when_empty():
    assert db.get_last_scrape_timestamp() is None

def test_log_label_and_retrieve():
    db.log_label("/img.jpg", "/lbl.txt", "lane", "morning")
    rows = db.get_label_logs(label_type="lane")
    assert len(rows) == 1
    assert rows[0][0] == "/img.jpg"
    assert rows[0][4] == "pending"

def test_update_label_validation():
    db.log_label("/img.jpg", "/lbl.txt", "lane", "morning")
    db.update_label_validation("/img.jpg", "lane", "approved")
    rows = db.get_label_logs(validated="approved")
    assert len(rows) == 1

def test_log_split_and_summary():
    db.log_split("/img.jpg", "lane", "train")
    db.log_split("/img2.jpg", "lane", "val")
    summary = db.get_dataset_split_summary()
    split_map = {(r[0], r[1]): r[2] for r in summary}
    assert split_map[("lane", "train")] == 1
    assert split_map[("lane", "val")] == 1

def test_get_recent_scrape_logs_respects_limit():
    for i in range(10):
        db.log_scrape(f"2026-06-27T{i:02d}:00:00", "2701", f"/p{i}.jpg", "success")
    logs = db.get_recent_scrape_logs(limit=5)
    assert len(logs) == 5
