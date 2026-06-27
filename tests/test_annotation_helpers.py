import os
import sqlite3
import pytest
from causeway.db import init_db, ensure_label_log_entry, get_label_logs


@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    db_file = str(tmp_path / "test.db")
    monkeypatch.setattr("causeway.db.DB_PATH", db_file)
    init_db()
    return db_file


def test_ensure_label_log_entry_inserts_when_absent(tmp_db):
    ensure_label_log_entry("/img/a.jpg", "/lbl/a.txt", "vehicle", "morning")
    rows = get_label_logs(label_type="vehicle")
    assert len(rows) == 1
    assert rows[0][0] == "/img/a.jpg"
    assert rows[0][4] == "pending"


def test_ensure_label_log_entry_does_not_duplicate(tmp_db):
    ensure_label_log_entry("/img/a.jpg", "/lbl/a.txt", "vehicle", "morning")
    ensure_label_log_entry("/img/a.jpg", "/lbl/a.txt", "vehicle", "morning")
    rows = get_label_logs(label_type="vehicle")
    assert len(rows) == 1


def test_ensure_label_log_entry_preserves_existing_validated(tmp_db):
    from causeway.db import update_label_validation
    ensure_label_log_entry("/img/a.jpg", "/lbl/a.txt", "vehicle", "morning")
    update_label_validation("/img/a.jpg", "vehicle", "approved")
    ensure_label_log_entry("/img/a.jpg", "/lbl/a.txt", "vehicle", "morning")
    rows = get_label_logs(label_type="vehicle", validated="approved")
    assert len(rows) == 1
