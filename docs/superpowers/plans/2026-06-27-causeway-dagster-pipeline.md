# Causeway Dagster Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Dagster-orchestrated backend pipeline that scrapes LTA traffic images every 5 minutes, generates YOLO lane-segmentation and vehicle-detection labels, splits the dataset by day, tracks everything in SQLite, and exposes a Streamlit validation app — all as new files without touching existing code.

**Architecture:** A `causeway/` Python package holds all domain logic (config, db, scraper, labeler, dataset, migration). `dagster_defs.py` at the root wires them into Dagster software-defined assets with a 5-minute scrape schedule on port 3001. `causeway_app.py` is a separate Streamlit app on port 8502 for human label review and pipeline health monitoring.

**Tech Stack:** Python 3.14, Dagster, Streamlit, Ultralytics YOLOv8, OpenCV, SQLite (stdlib), PyYAML, requests, uv (package manager)

## Global Constraints

- Do NOT modify any existing file (`app.py`, `pipeline.py`, `traffic_image_scrapper.py`, `auto_label.py`, `train.py`, `label_by_config.py`, `camera_config.json`, `dataset.yaml`, `pyproject.toml` may be updated for deps only)
- All new files live under `causeway/` package or project root
- Dagster runs on port **3001** (existing Dagster instance uses default port)
- Streamlit validation app runs on port **8502**
- SQLite database file: `causeway.db` at project root
- Image structure: `traffic_images/YYYYMMDD/CAMERA_ID/filename.jpg`
- Lane label output: `traffic_lane_labels/YYYYMMDD/CAMERA_ID/filename.txt`
- Vehicle label output: `traffic_vehicle_labels/YYYYMMDD/CAMERA_ID/filename.txt`
- YOLO lane segmentation format: `class_id x1 y1 x2 y2 ... xn yn` (normalized 0.0–1.0, image size 1920×1080)
- YOLO vehicle detection format: `class_id cx cy w h` (normalized xywh)
- Lane class 0 = Towards Woodlands/Singapore CIQ, class 1 = Towards Johor/Malaysia
- Vehicle classes: motorcycle=0, car=1, bus=2, truck=3
- Shift times: morning = 06:00–11:59, afternoon = 12:00–18:59, night = 19:00–05:59
- All modules import shared constants exclusively from `causeway/config.py`
- Tests live in `tests/` mirroring the `causeway/` structure

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `causeway/__init__.py` | Create | Package marker |
| `causeway/config.py` | Create | Shared constants: API URL, key, camera map, paths |
| `causeway/db.py` | Create | SQLite schema init + all query functions |
| `causeway/migration.py` | Create | One-shot: move images to canonical path + backfill SQLite |
| `causeway/scraper.py` | Create | Advanced scraper: exponential backoff + gap detection |
| `causeway/labeler.py` | Create | Lane seg label generation + vehicle det label generation |
| `causeway/dataset.py` | Create | Day-based train/val splitter + dataset YAML writer |
| `dagster_defs.py` | Create | Dagster assets, schedule, Definitions |
| `workspace.yaml` | Create | Dagster workspace config |
| `causeway_app.py` | Create | Streamlit 2-tab validation app |
| `README.md` | Modify | Educational explanations + setup + run instructions |
| `tests/__init__.py` | Create | Test package marker |
| `tests/test_db.py` | Create | SQLite layer tests |
| `tests/test_migration.py` | Create | Migration parsing + idempotency tests |
| `tests/test_scraper.py` | Create | Backoff, gap detection, run_once tests |
| `tests/test_labeler.py` | Create | Shift resolution, normalisation, label writing tests |
| `tests/test_dataset.py` | Create | Split logic + YAML output tests |

---

## Task 1: Project Scaffold, Config, and Dependencies

**Files:**
- Create: `causeway/__init__.py`
- Create: `causeway/config.py`
- Create: `tests/__init__.py`
- Modify: `pyproject.toml` (add dependencies)

**Interfaces:**
- Produces: `causeway.config.LTA_API_URL`, `causeway.config.API_KEY`, `causeway.config.TARGET_CAMERAS`, `causeway.config.IMAGE_BASE_DIR`, `causeway.config.LANE_LABELS_DIR`, `causeway.config.VEHICLE_LABELS_DIR`, `causeway.config.DB_PATH`, `causeway.config.CAMERA_CONFIG_PATH` — all imported by every other module

- [ ] **Step 1: Install dependencies**

```bash
cd /Users/chuan/Development/PythonProjects/snaic-ai-causeway
uv add dagster dagster-webserver streamlit ultralytics opencv-python pyyaml
```

Expected: uv resolves and writes to `uv.lock`, no errors.

- [ ] **Step 2: Create package markers**

Create `causeway/__init__.py`:
```python
```
(empty file)

Create `tests/__init__.py`:
```python
```
(empty file)

- [ ] **Step 3: Create `causeway/config.py`**

Open `traffic_image_scrapper.py` and copy the `API_KEY` value. Then create:

```python
# causeway/config.py
LTA_API_URL = "https://api.data.gov.sg/v1/transport/traffic-images"
API_KEY = "YOUR_DATA_GOV_SG_API_KEY"  # paste value from traffic_image_scrapper.py

TARGET_CAMERAS = {
    "2701": "Woodlands_Causeway_Towards_Johor",
    "2702": "Woodlands_Checkpoint_Towards_BKE",
    "2704": "Woodlands_Flyover_Towards_Checkpoint",
}

IMAGE_BASE_DIR = "traffic_images"
LANE_LABELS_DIR = "traffic_lane_labels"
VEHICLE_LABELS_DIR = "traffic_vehicle_labels"
DB_PATH = "causeway.db"
CAMERA_CONFIG_PATH = "camera_config.json"
```

- [ ] **Step 4: Verify imports resolve**

```bash
python -c "from causeway.config import LTA_API_URL, TARGET_CAMERAS, DB_PATH; print('OK', LTA_API_URL, list(TARGET_CAMERAS.keys()))"
```

Expected output:
```
OK https://api.data.gov.sg/v1/transport/traffic-images ['2701', '2702', '2704']
```

- [ ] **Step 5: Commit**

```bash
git add causeway/__init__.py causeway/config.py tests/__init__.py pyproject.toml uv.lock
git commit -m "feat: scaffold causeway package and shared config"
```

---

## Task 2: SQLite Database Layer

**Files:**
- Create: `causeway/db.py`
- Create: `tests/test_db.py`

**Interfaces:**
- Consumes: `causeway.config.DB_PATH`
- Produces:
  - `init_db()` → None
  - `get_connection()` → `sqlite3.Connection`
  - `log_scrape(scraped_at: str, camera_id: str, file_path: str | None, status: str, error_msg: str | None = None)` → None
  - `get_last_scrape_timestamp(camera_id: str | None = None)` → `str | None` (ISO datetime string)
  - `log_label(image_path: str, label_path: str, label_type: str, shift: str)` → None
  - `update_label_validation(image_path: str, label_type: str, validated: str)` → None
  - `log_split(image_path: str, label_type: str, split: str)` → None
  - `get_recent_scrape_logs(limit: int = 50)` → `list[tuple]`
  - `get_label_logs(label_type: str | None = None, validated: str | None = None)` → `list[tuple]`
  - `get_dataset_split_summary()` → `list[tuple]`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_db.py`:
```python
import os
import pytest
import sqlite3
from causeway import db

TEST_DB = "test_causeway.db"

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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_db.py -v
```

Expected: `ImportError` or `ModuleNotFoundError` for `causeway.db`.

- [ ] **Step 3: Implement `causeway/db.py`**

```python
# causeway/db.py
import sqlite3
from datetime import datetime
from causeway.config import DB_PATH


def get_connection() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH)


def init_db() -> None:
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS scrape_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scraped_at TEXT NOT NULL,
                camera_id TEXT NOT NULL,
                file_path TEXT,
                status TEXT NOT NULL,
                error_msg TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS label_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_path TEXT NOT NULL,
                label_path TEXT NOT NULL,
                label_type TEXT NOT NULL,
                shift TEXT,
                validated TEXT DEFAULT 'pending'
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS dataset_splits (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_path TEXT NOT NULL,
                label_type TEXT NOT NULL,
                split TEXT NOT NULL,
                assigned_at TEXT NOT NULL
            )
        """)
        conn.commit()


def log_scrape(scraped_at: str, camera_id: str, file_path, status: str, error_msg=None) -> None:
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO scrape_log (scraped_at, camera_id, file_path, status, error_msg) VALUES (?,?,?,?,?)",
            (scraped_at, camera_id, file_path, status, error_msg),
        )
        conn.commit()


def get_last_scrape_timestamp(camera_id=None):
    with get_connection() as conn:
        if camera_id:
            row = conn.execute(
                "SELECT scraped_at FROM scrape_log WHERE status IN ('success','migrated') AND camera_id=? ORDER BY scraped_at DESC LIMIT 1",
                (camera_id,),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT scraped_at FROM scrape_log WHERE status IN ('success','migrated') ORDER BY scraped_at DESC LIMIT 1"
            ).fetchone()
        return row[0] if row else None


def log_label(image_path: str, label_path: str, label_type: str, shift: str) -> None:
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO label_log (image_path, label_path, label_type, shift) VALUES (?,?,?,?)",
            (image_path, label_path, label_type, shift),
        )
        conn.commit()


def update_label_validation(image_path: str, label_type: str, validated: str) -> None:
    with get_connection() as conn:
        conn.execute(
            "UPDATE label_log SET validated=? WHERE image_path=? AND label_type=?",
            (validated, image_path, label_type),
        )
        conn.commit()


def log_split(image_path: str, label_type: str, split: str) -> None:
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO dataset_splits (image_path, label_type, split, assigned_at) VALUES (?,?,?,?)",
            (image_path, label_type, split, datetime.now().isoformat()),
        )
        conn.commit()


def get_recent_scrape_logs(limit: int = 50) -> list:
    with get_connection() as conn:
        return conn.execute(
            "SELECT scraped_at, camera_id, file_path, status, error_msg FROM scrape_log ORDER BY scraped_at DESC LIMIT ?",
            (limit,),
        ).fetchall()


def get_label_logs(label_type=None, validated=None) -> list:
    with get_connection() as conn:
        query = "SELECT image_path, label_path, label_type, shift, validated FROM label_log WHERE 1=1"
        params = []
        if label_type:
            query += " AND label_type=?"
            params.append(label_type)
        if validated:
            query += " AND validated=?"
            params.append(validated)
        return conn.execute(query, params).fetchall()


def get_dataset_split_summary() -> list:
    with get_connection() as conn:
        return conn.execute(
            "SELECT label_type, split, COUNT(*) FROM dataset_splits GROUP BY label_type, split"
        ).fetchall()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_db.py -v
```

Expected: all 8 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add causeway/db.py tests/test_db.py
git commit -m "feat: SQLite database layer with scrape/label/split tracking"
```

---

## Task 3: Migration Script

**Files:**
- Create: `causeway/migration.py`
- Create: `tests/test_migration.py`

**Interfaces:**
- Consumes: `causeway.config.IMAGE_BASE_DIR`, `causeway.config.TARGET_CAMERAS`; `causeway.db.init_db`, `causeway.db.get_connection`
- Produces:
  - `migrate(base_dir: str = IMAGE_BASE_DIR)` → `dict` with keys `moved`, `already_correct`, `skipped`
  - `_parse_image_metadata(filename: str)` → `tuple[str | None, str | None, datetime | None]` — `(camera_id, date_str, dt)`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_migration.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_migration.py -v
```

Expected: `ImportError` for `causeway.migration`.

- [ ] **Step 3: Implement `causeway/migration.py`**

```python
# causeway/migration.py
import os
import re
import shutil
from datetime import datetime
from causeway.config import IMAGE_BASE_DIR, TARGET_CAMERAS
from causeway.db import init_db, get_connection


def _parse_image_metadata(filename: str):
    """
    Parse camera_id, date_str, and datetime from an LTA filename.
    Expected pattern: ...PrefixName_CAMERAID_YYYYMMDD_HHMMSS.jpg
    Returns (camera_id, date_str, datetime) or (None, None, None) if unrecognised.
    """
    base = os.path.basename(filename)
    match = re.search(r"_(\d{4})_(\d{8})_(\d{6})\.jpg$", base, re.IGNORECASE)
    if not match:
        return None, None, None

    camera_id = match.group(1)
    date_str = match.group(2)
    time_str = match.group(3)

    if camera_id not in TARGET_CAMERAS:
        return None, None, None

    try:
        dt = datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")
    except ValueError:
        return None, None, None

    return camera_id, date_str, dt


def _backfill_scrape_log(file_path: str, camera_id: str, dt: datetime) -> None:
    with get_connection() as conn:
        exists = conn.execute(
            "SELECT 1 FROM scrape_log WHERE file_path=?", (file_path,)
        ).fetchone()
        if not exists:
            conn.execute(
                "INSERT INTO scrape_log (scraped_at, camera_id, file_path, status) VALUES (?,?,?,?)",
                (dt.isoformat(), camera_id, file_path, "migrated"),
            )
            conn.commit()


def migrate(base_dir: str = IMAGE_BASE_DIR) -> dict:
    """
    Walk all .jpg files under base_dir, move any that are not in the canonical
    YYYYMMDD/CAMERA_ID/ structure, and backfill the SQLite scrape_log.
    Safe to run multiple times (idempotent).
    """
    init_db()
    moved = 0
    already_correct = 0
    skipped = 0

    for root, dirs, files in os.walk(base_dir):
        # Avoid descending into directories we just created mid-walk
        dirs.sort()
        for fname in files:
            if not fname.lower().endswith(".jpg"):
                continue

            full_path = os.path.join(root, fname)
            camera_id, date_str, dt = _parse_image_metadata(fname)

            if camera_id is None:
                print(f"  SKIP (unrecognised filename): {fname}")
                skipped += 1
                continue

            target_dir = os.path.join(base_dir, date_str, camera_id)
            target_path = os.path.join(target_dir, fname)

            if os.path.abspath(full_path) == os.path.abspath(target_path):
                already_correct += 1
                _backfill_scrape_log(full_path, camera_id, dt)
                continue

            os.makedirs(target_dir, exist_ok=True)
            shutil.move(full_path, target_path)
            _backfill_scrape_log(target_path, camera_id, dt)
            print(f"  MOVED: {fname} -> {date_str}/{camera_id}/")
            moved += 1

    print(f"\nMigration complete: {moved} moved, {already_correct} already correct, {skipped} skipped.")
    return {"moved": moved, "already_correct": already_correct, "skipped": skipped}


if __name__ == "__main__":
    migrate()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_migration.py -v
```

Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add causeway/migration.py tests/test_migration.py
git commit -m "feat: migration script to canonicalise image paths and backfill SQLite"
```

---

## Task 4: Advanced Scraper

**Files:**
- Create: `causeway/scraper.py`
- Create: `tests/test_scraper.py`

**Interfaces:**
- Consumes: `causeway.config.LTA_API_URL`, `causeway.config.API_KEY`, `causeway.config.TARGET_CAMERAS`, `causeway.config.IMAGE_BASE_DIR`; `causeway.db.init_db`, `causeway.db.log_scrape`, `causeway.db.get_last_scrape_timestamp`
- Produces:
  - `run_once()` → None — one scrape cycle (fetch API, save images, log to SQLite)
  - `run_loop()` → None — calls `run_once()` in a 300s loop; blocks forever
  - `_log_startup_gap()` → None — queries SQLite, logs how long the scraper was offline
  - `_fetch_with_backoff()` → `dict | None` — returns parsed JSON payload or None on auth failure

- [ ] **Step 1: Write the failing tests**

Create `tests/test_scraper.py`:
```python
import pytest
from unittest.mock import patch, MagicMock, call
from datetime import datetime, timedelta
from causeway import db
import causeway.scraper as scraper_mod

@pytest.fixture(autouse=True)
def use_test_db(monkeypatch, tmp_path):
    test_db_path = str(tmp_path / "test.db")
    monkeypatch.setattr(db, "DB_PATH", test_db_path)
    db.init_db()
    yield

def test_log_startup_gap_no_history(capsys):
    scraper_mod._log_startup_gap()
    out = capsys.readouterr().out + capsys.readouterr().err
    # Just verify it doesn't raise

def test_log_startup_gap_with_history(capsys):
    recent = (datetime.now() - timedelta(minutes=10)).isoformat()
    db.log_scrape(recent, "2701", "/img.jpg", "success")
    scraper_mod._log_startup_gap()
    # No assertion on exact text — just verify it completes without error

def test_fetch_with_backoff_returns_payload():
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"items": [{"cameras": []}]}

    with patch("causeway.scraper.requests.get", return_value=mock_response):
        result = scraper_mod._fetch_with_backoff()

    assert result == {"items": [{"cameras": []}]}

def test_fetch_with_backoff_returns_none_on_auth_failure():
    mock_response = MagicMock()
    mock_response.status_code = 403

    with patch("causeway.scraper.requests.get", return_value=mock_response):
        result = scraper_mod._fetch_with_backoff()

    assert result is None

def test_fetch_with_backoff_retries_on_server_error():
    fail_response = MagicMock()
    fail_response.status_code = 500

    ok_response = MagicMock()
    ok_response.status_code = 200
    ok_response.json.return_value = {"items": []}

    with patch("causeway.scraper.requests.get", side_effect=[fail_response, ok_response]):
        with patch("causeway.scraper.time.sleep"):
            result = scraper_mod._fetch_with_backoff()

    assert result == {"items": []}

def test_run_once_saves_image_and_logs(tmp_path, monkeypatch):
    monkeypatch.setattr(scraper_mod, "IMAGE_BASE_DIR", str(tmp_path / "traffic_images"))

    payload = {
        "items": [{
            "cameras": [
                {"camera_id": "2701", "image": "http://fake-cdn.test/2701.jpg"}
            ]
        }]
    }

    fake_image_content = b"\xff\xd8\xff"  # minimal JPEG header bytes

    with patch.object(scraper_mod, "_fetch_with_backoff", return_value=payload):
        img_resp = MagicMock()
        img_resp.status_code = 200
        img_resp.content = fake_image_content
        with patch("causeway.scraper.requests.get", return_value=img_resp):
            scraper_mod.run_once()

    logs = db.get_recent_scrape_logs(limit=10)
    assert len(logs) == 1
    assert logs[0][1] == "2701"
    assert logs[0][3] == "success"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_scraper.py -v
```

Expected: `ImportError` for `causeway.scraper`.

- [ ] **Step 3: Implement `causeway/scraper.py`**

```python
# causeway/scraper.py
import os
import time
import logging
import requests
from datetime import datetime
from causeway.config import LTA_API_URL, API_KEY, TARGET_CAMERAS, IMAGE_BASE_DIR
from causeway.db import init_db, log_scrape, get_last_scrape_timestamp

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SCRAPE_INTERVAL = 300
MAX_BACKOFF = 300


def _log_startup_gap() -> None:
    last = get_last_scrape_timestamp()
    if last:
        last_dt = datetime.fromisoformat(last)
        gap_minutes = (datetime.now() - last_dt).total_seconds() / 60
        logger.info(f"Resuming. Gap since last scrape: {gap_minutes:.1f} minutes (last: {last})")
    else:
        logger.info("No prior scrape history. Starting fresh.")


def _fetch_with_backoff():
    delay = 1
    while True:
        try:
            headers = {"x-api-key": API_KEY}
            params = {"date_time": datetime.now().strftime("%Y-%m-%dT%H:%M:%S")}
            response = requests.get(LTA_API_URL, params=params, headers=headers, timeout=10)

            if response.status_code in (401, 403):
                logger.error(f"Auth failure HTTP {response.status_code}. Check API_KEY in causeway/config.py.")
                return None

            if response.status_code != 200:
                raise requests.RequestException(f"HTTP {response.status_code}")

            return response.json()

        except (requests.Timeout, requests.ConnectionError, requests.RequestException) as exc:
            logger.warning(f"Request failed: {exc}. Retrying in {delay}s...")
            time.sleep(delay)
            delay = min(delay * 2, MAX_BACKOFF)


def run_once() -> None:
    payload = _fetch_with_backoff()
    if payload is None:
        return

    items = payload.get("items", [])
    if not items or "cameras" not in items[0]:
        logger.warning("No camera data in payload.")
        return

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    date_str = datetime.now().strftime("%Y%m%d")

    for cam in items[0]["cameras"]:
        cam_id = str(cam.get("camera_id", ""))
        if cam_id not in TARGET_CAMERAS:
            continue

        image_url = cam.get("image")
        if not image_url:
            continue

        scraped_at = datetime.now().isoformat()
        try:
            img_resp = requests.get(image_url, timeout=15)
            if img_resp.status_code != 200:
                raise requests.RequestException(f"CDN HTTP {img_resp.status_code}")

            cam_dir = os.path.join(IMAGE_BASE_DIR, date_str, cam_id)
            os.makedirs(cam_dir, exist_ok=True)

            filename = f"{TARGET_CAMERAS[cam_id]}_{cam_id}_{timestamp_str}.jpg"
            filepath = os.path.join(cam_dir, filename)

            with open(filepath, "wb") as f:
                f.write(img_resp.content)

            log_scrape(scraped_at, cam_id, filepath, "success")
            logger.info(f"Saved cam {cam_id}: {filepath}")

        except Exception as exc:
            log_scrape(scraped_at, cam_id, None, "error", str(exc))
            logger.error(f"Failed cam {cam_id}: {exc}")


def run_loop() -> None:
    init_db()
    _log_startup_gap()
    logger.info("Scraper started. Cycle interval: 5 minutes.")
    while True:
        run_once()
        logger.info(f"Cycle complete. Sleeping {SCRAPE_INTERVAL}s...")
        time.sleep(SCRAPE_INTERVAL)


if __name__ == "__main__":
    run_loop()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_scraper.py -v
```

Expected: all 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add causeway/scraper.py tests/test_scraper.py
git commit -m "feat: advanced scraper with exponential backoff and startup gap detection"
```

---

## Task 5: Lane and Vehicle Labeler

**Files:**
- Create: `causeway/labeler.py`
- Create: `tests/test_labeler.py`

**Interfaces:**
- Consumes: `causeway.config.*`; `causeway.db.init_db`, `causeway.db.log_label`; `camera_config.json` on disk
- Produces:
  - `_parse_hour(filename: str)` → `int` (0–23)
  - `_get_shift(hour: int)` → `str` — `"morning"`, `"afternoon"`, or `"night"`
  - `_normalize_polygon(points: list[list[int]], w: int = 1920, h: int = 1080)` → `list[float]`
  - `generate_lane_labels(base_images_dir: str = IMAGE_BASE_DIR)` → `int` (count of new labels)
  - `generate_vehicle_labels(base_images_dir: str = IMAGE_BASE_DIR)` → `int` (count of new labels)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_labeler.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_labeler.py -v
```

Expected: `ImportError` for `causeway.labeler`.

- [ ] **Step 3: Implement `causeway/labeler.py`**

```python
# causeway/labeler.py
import os
import re
import json
import glob
from datetime import datetime
from causeway.config import (
    IMAGE_BASE_DIR, LANE_LABELS_DIR, VEHICLE_LABELS_DIR, CAMERA_CONFIG_PATH
)
from causeway.db import init_db, log_label

# Maps short camera_id -> camera_config.json top-level key
CONFIG_KEY_MAP = {
    "2701": "Woodlands_Checkpoint_Towards_Johor_2701",
    "2702": "Woodlands_Checkpoint_Towards_BKE_2702",
    "2704": "Woodlands_Flyover_Towards_Checkpoint_2704",
}

# Class IDs for lane segmentation labels
# Class 0 = Towards Woodlands/Singapore CIQ
# Class 1 = Towards Johor/Malaysia
# (2704 gets extra class 2 for PIE branch — static camera)
SHIFT_CLASS_MAP = {
    "2701": {
        "morning":   {"To Woodlands CIQ (3 Lanes)": 0, "To Johor (1 Lane)": 1},
        "afternoon": {"To Woodlands CIQ (2 Lanes)": 0, "To Johor (2 Lanes)": 1},
        "night":     {"To Woodlands CIQ (2 Lanes)": 0, "To Johor (2 Lanes)": 1},
    },
    "2702": {
        "static": {"Towards BKE": 0, "Towards Checkpoint Arrival": 1},
    },
    "2704": {
        "static": {"To Woodlands Checkpoint": 0, "To Woodlands Ave 3": 1, "To PIE": 2},
    },
}

IMG_W = 1920
IMG_H = 1080

# COCO class → project class for vehicle detection
COCO_TO_VEHICLE_CLASS = {2: 1, 3: 0, 5: 2, 7: 3}  # car→1, motorcycle→0, bus→2, truck→3


def _parse_hour(filename: str) -> int:
    base = os.path.basename(filename)
    match = re.search(r"_(\d{8})_(\d{2})\d{4}\.jpg$", base, re.IGNORECASE)
    if match:
        return int(match.group(2))
    return datetime.now().hour


def _get_shift(hour: int) -> str:
    if 6 <= hour < 12:
        return "morning"
    if 12 <= hour < 19:
        return "afternoon"
    return "night"


def _normalize_polygon(points: list, w: int = IMG_W, h: int = IMG_H) -> list:
    flat = [coord for pt in points for coord in pt]
    return [flat[i] / w if i % 2 == 0 else flat[i] / h for i in range(len(flat))]


def generate_lane_labels(base_images_dir: str = IMAGE_BASE_DIR) -> int:
    init_db()
    with open(CAMERA_CONFIG_PATH) as f:
        master_config = json.load(f)

    image_paths = glob.glob(os.path.join(base_images_dir, "**", "*.jpg"), recursive=True)
    processed = 0

    for img_path in image_paths:
        norm_path = img_path.replace("\\", "/")
        parts = norm_path.split("/")
        if len(parts) < 3:
            continue
        camera_id = parts[-2]
        if camera_id not in CONFIG_KEY_MAP:
            continue

        config_key = CONFIG_KEY_MAP[camera_id]
        camera_profiles = master_config.get(config_key)
        if not camera_profiles:
            continue

        rel_path = os.path.relpath(img_path, base_images_dir)
        rel_dir = os.path.dirname(rel_path)
        base_name = os.path.splitext(os.path.basename(img_path))[0]

        label_dir = os.path.join(LANE_LABELS_DIR, rel_dir)
        label_path = os.path.join(label_dir, f"{base_name}.txt")

        if os.path.exists(label_path):
            continue

        if "static" in camera_profiles:
            profile = camera_profiles["static"]
            shift = "static"
        else:
            shift = _get_shift(_parse_hour(img_path))
            profile = camera_profiles[shift]

        class_map = SHIFT_CLASS_MAP.get(camera_id, {}).get(shift, {})
        os.makedirs(label_dir, exist_ok=True)

        with open(label_path, "w") as f:
            for label, polygon in zip(profile["labels"], profile["polygons"]):
                class_id = class_map.get(label, 0)
                norm = _normalize_polygon(polygon)
                coords_str = " ".join(f"{v:.6f}" for v in norm)
                f.write(f"{class_id} {coords_str}\n")

        log_label(img_path, label_path, "lane", shift)
        processed += 1

    print(f"Lane labeling complete: {processed} new labels generated.")
    return processed


def generate_vehicle_labels(base_images_dir: str = IMAGE_BASE_DIR) -> int:
    from ultralytics import YOLO
    init_db()

    model = YOLO("yolov8x.pt")
    image_paths = glob.glob(os.path.join(base_images_dir, "**", "*.jpg"), recursive=True)
    processed = 0

    for img_path in image_paths:
        rel_path = os.path.relpath(img_path, base_images_dir)
        rel_dir = os.path.dirname(rel_path)
        base_name = os.path.splitext(os.path.basename(img_path))[0]

        label_dir = os.path.join(VEHICLE_LABELS_DIR, rel_dir)
        label_path = os.path.join(label_dir, f"{base_name}.txt")

        if os.path.exists(label_path):
            continue

        results = model.predict(source=img_path, imgsz=1280, conf=0.25, device="mps", verbose=False)[0]
        os.makedirs(label_dir, exist_ok=True)

        with open(label_path, "w") as f:
            for box in results.boxes:
                coco_cls = int(box.cls[0].item())
                if coco_cls in COCO_TO_VEHICLE_CLASS:
                    target_cls = COCO_TO_VEHICLE_CLASS[coco_cls]
                    xywhn = box.xywhn[0].cpu().numpy()
                    cx, cy, w, h = xywhn
                    f.write(f"{target_cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

        log_label(img_path, label_path, "vehicle", "N/A")
        processed += 1

    print(f"Vehicle labeling complete: {processed} new labels generated.")
    return processed
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_labeler.py -v
```

Expected: all 8 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add causeway/labeler.py tests/test_labeler.py
git commit -m "feat: YOLO lane segmentation and vehicle detection label generators"
```

---

## Task 6: Dataset Splitter

**Files:**
- Create: `causeway/dataset.py`
- Create: `tests/test_dataset.py`

**Interfaces:**
- Consumes: `causeway.config.IMAGE_BASE_DIR`, `causeway.config.LANE_LABELS_DIR`, `causeway.config.VEHICLE_LABELS_DIR`; `causeway.db.init_db`, `causeway.db.log_split`
- Produces:
  - `build_dataset_split(base_images_dir: str = IMAGE_BASE_DIR)` → `dict` with keys `lane_train`, `lane_val`, `vehicle_train`, `vehicle_val` (counts)
  - Writes `dataset_lane.yaml` and `dataset_vehicle.yaml` to project root

- [ ] **Step 1: Write the failing tests**

Create `tests/test_dataset.py`:
```python
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
    """Helper: create fake image files. dates_cameras = [('20260625', '2701'), ...]"""
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_dataset.py -v
```

Expected: `ImportError` for `causeway.dataset`.

- [ ] **Step 3: Implement `causeway/dataset.py`**

```python
# causeway/dataset.py
import os
import glob
import yaml
from causeway.config import IMAGE_BASE_DIR, LANE_LABELS_DIR, VEHICLE_LABELS_DIR
from causeway.db import init_db, log_split


def _write_yaml(path: str, train_imgs: list, val_imgs: list, names: dict) -> None:
    data = {
        "path": os.path.abspath("."),
        "train": [os.path.abspath(p) for p in train_imgs],
        "val": [os.path.abspath(p) for p in val_imgs],
        "nc": len(names),
        "names": names,
    }
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)
    print(f"Written: {path}")


def build_dataset_split(base_images_dir: str = IMAGE_BASE_DIR) -> dict:
    init_db()
    date_dirs = sorted(glob.glob(os.path.join(base_images_dir, "????????")))

    if len(date_dirs) < 2:
        print(f"WARNING: Only {len(date_dirs)} date partition(s) found. Need at least 2 to split meaningfully.")
        return {"lane_train": 0, "lane_val": 0, "vehicle_train": 0, "vehicle_val": 0}

    train_dates = date_dirs[:-1]
    val_dates = [date_dirs[-1]]

    print(f"Train: {[os.path.basename(d) for d in train_dates]}")
    print(f"Val:   {[os.path.basename(d) for d in val_dates]}")

    lane_train, lane_val, vehicle_train, vehicle_val = [], [], [], []

    for date_dir in train_dates:
        for img in glob.glob(os.path.join(date_dir, "**", "*.jpg"), recursive=True):
            lane_train.append(img)
            vehicle_train.append(img)
            log_split(img, "lane", "train")
            log_split(img, "vehicle", "train")

    for date_dir in val_dates:
        for img in glob.glob(os.path.join(date_dir, "**", "*.jpg"), recursive=True):
            lane_val.append(img)
            vehicle_val.append(img)
            log_split(img, "lane", "val")
            log_split(img, "vehicle", "val")

    if not lane_val:
        print("WARNING: Val set is empty.")

    _write_yaml(
        "dataset_lane.yaml", lane_train, lane_val,
        {0: "Towards Woodlands CIQ", 1: "Towards Johor"},
    )
    _write_yaml(
        "dataset_vehicle.yaml", vehicle_train, vehicle_val,
        {0: "motorcycle", 1: "car", 2: "bus", 3: "truck"},
    )

    print(f"Lane:    train={len(lane_train)}, val={len(lane_val)}")
    print(f"Vehicle: train={len(vehicle_train)}, val={len(vehicle_val)}")

    return {
        "lane_train": len(lane_train),
        "lane_val": len(lane_val),
        "vehicle_train": len(vehicle_train),
        "vehicle_val": len(vehicle_val),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_dataset.py -v
```

Expected: all 4 tests PASS.

- [ ] **Step 5: Run full test suite**

```bash
pytest tests/ -v
```

Expected: all tests PASS (Tasks 2–6).

- [ ] **Step 6: Commit**

```bash
git add causeway/dataset.py tests/test_dataset.py
git commit -m "feat: day-based dataset splitter with YAML output and SQLite logging"
```

---

## Task 7: Dagster Definitions and Workspace

**Files:**
- Create: `dagster_defs.py`
- Create: `workspace.yaml`

**Interfaces:**
- Consumes: all `causeway.*` modules
- Produces: Dagster assets `migrate_existing_images`, `scrape_images`, `generate_lane_labels`, `generate_vehicle_labels`, `build_dataset_split`; schedule `scrape_schedule` (every 5 min); `Definitions` object named `defs`

- [ ] **Step 1: Create `workspace.yaml`**

```yaml
# workspace.yaml
load_from:
  - python_file:
      relative_path: dagster_defs.py
      location_name: causeway_pipeline
```

- [ ] **Step 2: Create `dagster_defs.py`**

```python
# dagster_defs.py
from dagster import asset, Definitions, ScheduleDefinition, define_asset_job, AssetSelection


@asset(description="One-shot: migrate existing traffic images into canonical path and backfill SQLite.")
def migrate_existing_images():
    from causeway.migration import migrate
    result = migrate()
    return result


@asset(description="Scrape one cycle of LTA traffic camera images for cameras 2701, 2702, 2704.")
def scrape_images():
    from causeway.db import init_db
    from causeway.scraper import run_once
    init_db()
    run_once()


@asset(deps=[scrape_images], description="Generate YOLO segmentation labels for lane regions from camera_config.json.")
def generate_lane_labels():
    from causeway.labeler import generate_lane_labels as _gen
    count = _gen()
    return {"new_labels": count}


@asset(deps=[scrape_images], description="Generate YOLO detection labels for vehicles using YOLOv8x inference on MPS.")
def generate_vehicle_labels():
    from causeway.labeler import generate_vehicle_labels as _gen
    count = _gen()
    return {"new_labels": count}


@asset(
    deps=[generate_lane_labels, generate_vehicle_labels],
    description="Split dataset by day: all days except last → train, last day → val. Writes dataset_lane.yaml and dataset_vehicle.yaml.",
)
def build_dataset_split():
    from causeway.dataset import build_dataset_split as _split
    return _split()


scrape_job = define_asset_job(
    "scrape_job",
    selection=AssetSelection.assets(scrape_images),
    description="Runs one scrape cycle for all target cameras.",
)

scrape_schedule = ScheduleDefinition(
    name="scrape_every_5_minutes",
    job=scrape_job,
    cron_schedule="*/5 * * * *",
)

defs = Definitions(
    assets=[
        migrate_existing_images,
        scrape_images,
        generate_lane_labels,
        generate_vehicle_labels,
        build_dataset_split,
    ],
    schedules=[scrape_schedule],
)
```

- [ ] **Step 3: Verify Dagster loads the definitions**

```bash
dagster definitions validate -f dagster_defs.py
```

Expected output: `Definitions loaded successfully` (or equivalent success message). No import errors.

- [ ] **Step 4: Start Dagster on port 3001 and verify UI**

```bash
dagster dev -f dagster_defs.py --port 3001
```

Open `http://localhost:3001`. Verify:
- All 5 assets appear in the Asset Catalog
- `scrape_every_5_minutes` schedule appears under Automation
- No red error banners

Stop with `Ctrl+C` after verifying.

- [ ] **Step 5: Commit**

```bash
git add dagster_defs.py workspace.yaml
git commit -m "feat: Dagster asset definitions with 5-minute scrape schedule on port 3001"
```

---

## Task 8: Streamlit Validation App

**Files:**
- Create: `causeway_app.py`

**Interfaces:**
- Consumes: `causeway.db.*`; `causeway.config.IMAGE_BASE_DIR`, `LANE_LABELS_DIR`, `VEHICLE_LABELS_DIR`, `CAMERA_CONFIG_PATH`

- [ ] **Step 1: Create `causeway_app.py`**

```python
# causeway_app.py
import os
import json
import glob
from datetime import datetime, timedelta

import cv2
import numpy as np
import streamlit as st

from causeway.config import (
    IMAGE_BASE_DIR, LANE_LABELS_DIR, VEHICLE_LABELS_DIR, CAMERA_CONFIG_PATH
)
from causeway.db import (
    init_db, get_connection, update_label_validation,
    get_recent_scrape_logs, get_dataset_split_summary, get_label_logs
)

st.set_page_config(page_title="Causeway Pipeline Validator", page_icon="🚦", layout="wide")
init_db()

st.title("🚦 Causeway Pipeline Validator")
st.caption("Human validation tool for lane segmentation and vehicle detection labels.")

tab1, tab2 = st.tabs(["📋 Label Review", "📊 Pipeline Health"])


# ── Helpers ──────────────────────────────────────────────────────────────────

def _render_lane_annotation(img_path: str, label_path: str):
    """Draw normalised lane polygons onto the image and return as RGB numpy array."""
    image = cv2.imread(img_path)
    if image is None:
        return None
    h, w = image.shape[:2]

    if not os.path.exists(label_path):
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    colours = [(0, 0, 200), (0, 180, 0), (200, 140, 0)]
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:
                continue
            cls_id = int(parts[0])
            coords = list(map(float, parts[1:]))
            pts_x = [int(coords[i] * w) for i in range(0, len(coords), 2)]
            pts_y = [int(coords[i] * h) for i in range(1, len(coords), 2)]
            pts = np.array(list(zip(pts_x, pts_y)), np.int32).reshape((-1, 1, 2))
            colour = colours[cls_id % len(colours)]
            overlay = image.copy()
            cv2.fillPoly(overlay, [pts], colour)
            cv2.addWeighted(overlay, 0.3, image, 0.7, 0, image)
            cv2.polylines(image, [pts], isClosed=True, color=colour, thickness=2)

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _render_vehicle_annotation(img_path: str, label_path: str):
    """Draw normalised bounding boxes onto the image and return as RGB numpy array."""
    image = cv2.imread(img_path)
    if image is None:
        return None
    h, w = image.shape[:2]

    cls_colours = {0: (255, 100, 0), 1: (0, 200, 255), 2: (0, 255, 100), 3: (200, 0, 255)}
    cls_names = {0: "motorcycle", 1: "car", 2: "bus", 3: "truck"}

    if os.path.exists(label_path):
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls_id, cx, cy, bw, bh = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                x1 = int((cx - bw / 2) * w)
                y1 = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w)
                y2 = int((cy + bh / 2) * h)
                colour = cls_colours.get(cls_id, (255, 255, 255))
                cv2.rectangle(image, (x1, y1), (x2, y2), colour, 2)
                cv2.putText(image, cls_names.get(cls_id, str(cls_id)), (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _get_label_path(img_path: str, label_type: str) -> str:
    labels_dir = LANE_LABELS_DIR if label_type == "lane" else VEHICLE_LABELS_DIR
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    rel_dir = os.path.dirname(os.path.relpath(img_path, IMAGE_BASE_DIR))
    return os.path.join(labels_dir, rel_dir, f"{base_name}.txt")


# ── Tab 1: Label Review ───────────────────────────────────────────────────────

with tab1:
    st.header("Label Review")
    st.info(
        "**Why do we review labels?** AI labels are generated automatically, but computers can make mistakes. "
        "Reviewing labels ensures that only correct, high-quality data is used to train the model. "
        "If the model trains on bad data, it will learn bad habits — this is called 'garbage in, garbage out'."
    )

    with st.sidebar:
        st.header("🔍 Filters")
        camera_filter = st.selectbox("Camera", ["All", "2701", "2702", "2704"])
        label_type_filter = st.selectbox("Label Type", ["lane", "vehicle"])
        validation_filter = st.selectbox("Validation Status", ["pending", "approved", "rejected", "All"])

    rows = get_label_logs(
        label_type=label_type_filter,
        validated=None if validation_filter == "All" else validation_filter,
    )

    if camera_filter != "All":
        rows = [r for r in rows if f"/{camera_filter}/" in r[0].replace("\\", "/")]

    st.write(f"Found **{len(rows)}** label(s) matching filters.")

    for img_path, label_path, label_type, shift, validated in rows:
        expander_title = f"{os.path.basename(img_path)}  —  {validated.upper()}"
        with st.expander(expander_title, expanded=(validated == "pending")):
            col1, col2 = st.columns(2)

            with col1:
                st.caption("📷 Raw Image")
                if os.path.exists(img_path):
                    st.image(img_path, use_column_width=True)
                else:
                    st.error(f"Image not found: `{img_path}`")

            with col2:
                st.caption("🎨 Annotated Image")
                computed_label_path = _get_label_path(img_path, label_type)
                if label_type == "lane":
                    annotated = _render_lane_annotation(img_path, computed_label_path)
                else:
                    annotated = _render_vehicle_annotation(img_path, computed_label_path)

                if annotated is not None:
                    st.image(annotated, use_column_width=True)
                else:
                    st.warning("Could not render annotation.")

            st.caption(f"Shift: **{shift}** | Type: **{label_type}**")

            btn1, btn2, btn3 = st.columns(3)
            key_suffix = f"{img_path}_{label_type}"
            with btn1:
                if st.button("✅ Approve", key=f"approve_{key_suffix}"):
                    update_label_validation(img_path, label_type, "approved")
                    st.rerun()
            with btn2:
                if st.button("❌ Reject", key=f"reject_{key_suffix}"):
                    update_label_validation(img_path, label_type, "rejected")
                    st.rerun()
            with btn3:
                st.button("⏭ Skip", key=f"skip_{key_suffix}")


# ── Tab 2: Pipeline Health ────────────────────────────────────────────────────

with tab2:
    st.header("Pipeline Health")
    st.info(
        "**Why monitor pipeline health?** The scraper runs every 5 minutes. If your laptop sleeps or "
        "loses internet, cycles are missed and gaps appear in your dataset. Knowing where the gaps are "
        "helps you decide whether the dataset is clean enough to train on."
    )

    # Per-camera status cards
    st.subheader("Camera Status")
    cam_cols = st.columns(3)
    camera_ids = ["2701", "2702", "2704"]
    camera_labels = {
        "2701": "Woodlands → Johor",
        "2702": "Woodlands → BKE",
        "2704": "Flyover → Checkpoint",
    }

    for idx, cam_id in enumerate(camera_ids):
        with cam_cols[idx]:
            with get_connection() as conn:
                row = conn.execute(
                    "SELECT scraped_at FROM scrape_log WHERE camera_id=? AND status IN ('success','migrated') ORDER BY scraped_at DESC LIMIT 1",
                    (cam_id,),
                ).fetchone()
            if row:
                last_dt = datetime.fromisoformat(row[0])
                age_minutes = (datetime.now() - last_dt).total_seconds() / 60
                badge = "🟢 Healthy" if age_minutes < 10 else "🟡 Stale" if age_minutes < 30 else "🔴 Missing"
                st.metric(label=camera_labels[cam_id], value=badge, delta=f"{age_minutes:.0f} min ago")
            else:
                st.metric(label=camera_labels[cam_id], value="⚪ No data")

    # Hourly scrape timeline
    st.subheader("Scrape Timeline (last 24 h)")
    with get_connection() as conn:
        cutoff = (datetime.now() - timedelta(hours=24)).isoformat()
        timeline_rows = conn.execute(
            "SELECT scraped_at, status FROM scrape_log WHERE scraped_at >= ? ORDER BY scraped_at",
            (cutoff,),
        ).fetchall()

    if timeline_rows:
        import pandas as pd
        df = pd.DataFrame(timeline_rows, columns=["scraped_at", "status"])
        df["scraped_at"] = pd.to_datetime(df["scraped_at"])
        df["hour"] = df["scraped_at"].dt.floor("h")
        hourly = df.groupby(["hour", "status"]).size().reset_index(name="count")
        success_df = hourly[hourly["status"] == "success"]
        st.bar_chart(success_df.set_index("hour")["count"])
        error_count = len(df[df["status"] == "error"])
        if error_count:
            st.warning(f"⚠️ {error_count} scrape error(s) in the last 24 h.")
    else:
        st.info("No scrape activity recorded in the last 24 hours.")

    # Recent events table
    st.subheader("Recent Scrape Events")
    logs = get_recent_scrape_logs(limit=50)
    if logs:
        import pandas as pd
        df = pd.DataFrame(logs, columns=["Timestamp", "Camera", "File", "Status", "Error"])
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No scrape events logged yet.")

    # Dataset split summary
    st.subheader("Dataset Split Summary")
    summary = get_dataset_split_summary()
    if summary:
        import pandas as pd
        df = pd.DataFrame(summary, columns=["Label Type", "Split", "Count"])
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No dataset splits recorded yet. Run the build_dataset_split Dagster asset first.")
```

- [ ] **Step 2: Start the app and verify both tabs**

```bash
streamlit run causeway_app.py --server.port 8502
```

Open `http://localhost:8502`. Verify:
- Tab 1 "Label Review" loads without errors (may show "No labels found" if labeler hasn't run yet — that is correct)
- Tab 2 "Pipeline Health" loads without errors; camera cards show "No data" if scraper hasn't run yet
- No Python tracebacks in the terminal

Stop with `Ctrl+C`.

- [ ] **Step 3: Commit**

```bash
git add causeway_app.py
git commit -m "feat: Streamlit validation app with label review and pipeline health tabs"
```

---

## Task 9: README Update

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Rewrite `README.md`**

```markdown
# Causeway Time-Aware Tidal Flow

A real-time traffic lane monitoring pipeline that uses AI to track vehicle counts
and tidal flow directions at the Singapore-Malaysia Woodlands Checkpoint.

Built as an educational showcase for the **SNAIC AI Programme Week 4**.

---

## What This Project Does (and Why)

### The Problem
The Woodlands Causeway is one of the busiest border crossings in the world.
Traffic lanes change direction throughout the day — more lanes open towards
Malaysia in the morning (people going to work) and flip back towards Singapore
in the evening (people coming home). This is called **tidal flow**.

### What We're Building
We collect live traffic camera images from Singapore's [Data.gov.sg API](https://data.gov.sg)
every 5 minutes and use computer vision (YOLO — You Only Look Once) to:
1. **Detect vehicles** in each camera frame
2. **Identify which lane zone** each vehicle is in
3. **Count congestion** per direction in real time

### Why We Label Data
Before a YOLO model can detect something, it needs thousands of example images
where a human (or a smart algorithm) has already drawn boxes or outlines around
the objects of interest. This is called **labeling** or **annotation**.

In this project we generate two types of labels automatically:
- **Lane segmentation labels** — polygon outlines of each active traffic lane zone
- **Vehicle detection labels** — bounding boxes around each car, bus, truck, or motorcycle

**Why review them?** Automated labels can be wrong. A camera might be blurry.
The time-based logic that determines which lanes are active might misclassify a
late-night image as a morning one. Reviewing labels before training ensures the
model learns from correct examples only. This principle is called
*"garbage in, garbage out"* — a model is only as good as the data it trains on.

---

## Project Structure

```
causeway/
  config.py       — Shared constants (API URL, camera IDs, folder paths)
  db.py           — SQLite database layer (scrape log, label log, dataset splits)
  scraper.py      — Advanced scraper with auto-retry and gap detection
  labeler.py      — Automatic YOLO label generation (lane + vehicle)
  dataset.py      — Dataset splitter (train/val by day)
  migration.py    — One-shot script to migrate existing images and backfill the database
dagster_defs.py   — Dagster pipeline orchestration (run and schedule assets)
causeway_app.py   — Streamlit app for human label review and pipeline health
```

---

## Setup

### 1. Install dependencies

```bash
uv sync
```

### 2. Set your API key

Open `causeway/config.py` and paste your [Data.gov.sg API key](https://data.gov.sg/developer):

```python
API_KEY = "your-api-key-here"
```

### 3. Migrate existing images (first time only)

If you already have images in `traffic_images/`, run the migration to ensure
they are in the correct folder structure and registered in the database:

```bash
python -m causeway.migration
```

---

## Running the Pipeline

### Option A: Dagster (recommended)

Dagster is an orchestration tool that lets you run, schedule, and monitor
each step of the pipeline through a web UI.

```bash
dagster dev -f dagster_defs.py --port 3001
```

Open `http://localhost:3001`. You will see five assets:

| Asset | What it does |
|---|---|
| `migrate_existing_images` | Run once to backfill existing images |
| `scrape_images` | Download the latest camera frames |
| `generate_lane_labels` | Create lane polygon labels |
| `generate_vehicle_labels` | Create vehicle bounding box labels |
| `build_dataset_split` | Split data into train/val sets |

**To run the full pipeline:** click each asset in order and press **Materialize**.

The `scrape_images` asset runs automatically every 5 minutes via the built-in schedule.

### Option B: Scraper standalone

To run just the scraper (without Dagster):

```bash
python -m causeway.scraper
```

### Streamlit Validation App

```bash
streamlit run causeway_app.py --server.port 8502
```

Open `http://localhost:8502` to:
- **Tab 1 — Label Review:** See each image side-by-side with its generated labels.
  Approve good labels, reject bad ones.
- **Tab 2 — Pipeline Health:** See how often images were scraped, spot gaps, and
  check the current dataset split counts.

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Training the Model

Once you have labeled and reviewed your dataset, train YOLO:

```bash
python train.py
```

This uses `dataset_lane.yaml` or `dataset_vehicle.yaml` generated by
the `build_dataset_split` Dagster asset.

---

## Key Concepts Glossary

| Term | Meaning |
|---|---|
| **Tidal flow** | Traffic lanes that change direction at different times of day |
| **YOLO** | A real-time object detection AI model (You Only Look Once) |
| **Segmentation** | Drawing polygon outlines around objects (more precise than boxes) |
| **Labeling** | Annotating images so the AI knows what to look for |
| **Train / Val split** | Dividing data into training examples and unseen test examples |
| **Dagster** | A workflow tool that runs and schedules pipeline steps |
| **SQLite** | A lightweight local database that stores logs and metadata |
```

- [ ] **Step 2: Verify the README renders correctly**

Open `README.md` in the IDE preview. Confirm all tables, code blocks, and headings render cleanly.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: educational README with setup, run instructions, and key concepts"
```

---

## Self-Review Checklist

**Spec coverage:**
- ✅ `causeway/config.py` — Task 1
- ✅ `causeway/db.py` — Task 2
- ✅ `causeway/migration.py` + SQLite backfill + Dagster asset — Tasks 3, 7
- ✅ `causeway/scraper.py` + exponential backoff + gap detection — Task 4
- ✅ `causeway/labeler.py` + lane seg + vehicle det — Task 5
- ✅ `causeway/dataset.py` + YAML output — Task 6
- ✅ `dagster_defs.py` + `workspace.yaml` + port 3001 — Task 7
- ✅ `causeway_app.py` + Label Review tab + Pipeline Health tab — Task 8
- ✅ `README.md` educational content + setup + run instructions — Task 9
- ✅ No existing files modified

**Type consistency:**
- `log_scrape(scraped_at, camera_id, file_path, status, error_msg)` — consistent across db.py, scraper.py, migration.py
- `generate_lane_labels(base_images_dir)` — consistent between labeler.py and dagster_defs.py
- `build_dataset_split(base_images_dir)` — consistent between dataset.py and dagster_defs.py
- `get_label_logs(label_type, validated)` — consistent between db.py and causeway_app.py
