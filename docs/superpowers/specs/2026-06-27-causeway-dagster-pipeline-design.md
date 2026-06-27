# Causeway Time-Aware Tidal Flow — Dagster Pipeline Design

**Date:** 2026-06-27
**Scope:** New Dagster-orchestrated backend pipeline, advanced scraper, YOLO label generation, SQLite tracking, and Streamlit validation app.

---

## 1. Goals

Build a production-grade, locally-run data engineering pipeline that:
- Continuously scrapes LTA traffic camera images every 5 minutes
- Generates two types of YOLO labels: lane segmentation + vehicle detection
- Splits the dataset strictly by day to prevent data leakage
- Tracks all activity in SQLite
- Exposes a Streamlit app for human label validation and pipeline health monitoring

---

## 2. Package Structure

```
snaic-ai-causeway/
  causeway/
    __init__.py
    config.py           # Shared constants: LTA_API_URL, API_KEY, TARGET_CAMERAS
    scraper.py          # Advanced scraper: exponential backoff + SQLite gap detection
    labeler.py          # Lane seg labels + vehicle detection labels
    dataset.py          # Day-based train/val splitter, writes dataset YAML files
    db.py               # SQLite schema, connection, all query functions
    migration.py        # One-shot: migrate existing images into YYYYMMDD/CAMERA_ID/ + backfill SQLite
  dagster_defs.py       # All Dagster assets, schedule, Definitions entry point
  workspace.yaml        # Dagster workspace config (port 3001)
  causeway_app.py       # Streamlit validation app (port 8502)
  README.md             # Updated with setup, educational explanations, run instructions
```

No existing files are modified. All new files.

---

## 3. SQLite Schema (`causeway/db.py`)

**`scrape_log`**
| column | type | description |
|---|---|---|
| id | INTEGER PK | auto |
| scraped_at | TEXT | ISO timestamp of scrape cycle |
| camera_id | TEXT | 2701 / 2702 / 2704 |
| file_path | TEXT | saved image path |
| status | TEXT | success / skipped / error |
| error_msg | TEXT | null if success |

**`label_log`**
| column | type | description |
|---|---|---|
| id | INTEGER PK | auto |
| image_path | TEXT | source image |
| label_path | TEXT | output .txt path |
| label_type | TEXT | lane / vehicle |
| shift | TEXT | morning / afternoon / night / static |
| validated | TEXT | pending / approved / rejected |

**`dataset_splits`**
| column | type | description |
|---|---|---|
| id | INTEGER PK | auto |
| image_path | TEXT | source image |
| label_type | TEXT | lane / vehicle |
| split | TEXT | train / val |
| assigned_at | TEXT | ISO timestamp |

---

## 4. Module Responsibilities

### `causeway/config.py`
Shared constants extracted from `traffic_image_scrapper.py` (read-only reference, not modified):
```python
LTA_API_URL = "https://api.data.gov.sg/v1/transport/traffic-images"
API_KEY = "<extracted from traffic_image_scrapper.py>"
TARGET_CAMERAS = {
    "2701": "Woodlands_Causeway_Towards_Johor",
    "2702": "Woodlands_Checkpoint_Towards_BKE",
    "2704": "Woodlands_Flyover_Towards_Checkpoint",
}
IMAGE_BASE_DIR = "traffic_images"
LANE_LABELS_DIR = "traffic_lane_labels"
VEHICLE_LABELS_DIR = "traffic_vehicle_labels"
DB_PATH = "causeway.db"
```
All other modules import from here — no config duplication across files.

### `causeway/migration.py`
One-shot script. Run once to migrate any existing images into the canonical `traffic_images/YYYYMMDD/CAMERA_ID/` structure and backfill SQLite `scrape_log`.

**Logic:**
1. Walk all files currently under `traffic_images/` recursively
2. For each `.jpg`, parse `CAMERA_ID` and `YYYYMMDD` from the filename (e.g. `Woodlands_Causeway_Towards_Johor_2701_20260625_221422.jpg` → camera `2701`, date `20260625`)
3. If the file is not already inside `traffic_images/20260625/2701/`, move it there (create dirs as needed)
4. Insert a row into `scrape_log` with `status = 'migrated'` and `scraped_at` derived from filename timestamp — so gap detection works correctly from day one
5. Print a summary: N files moved, N already in correct location, N skipped (unrecognised filename)
6. Idempotent: re-running is safe (files already in correct location are skipped)

Run with: `python -m causeway.migration`

### `causeway/scraper.py`
- On startup: query SQLite for last successful scrape timestamp, compute and log gap duration
- Each 5-min cycle: hit Data.gov.sg API with exponential backoff (1s → 2s → 4s → … → 300s cap) for network/timeout errors
- Hard skip (no retry) on HTTP 401/403 with clear log message
- Save images to `traffic_images/YYYYMMDD/CAMERA_ID/` (same structure as existing scraper)
- Log every attempt (success/skip/error) to SQLite `scrape_log`
- Can be run standalone: `python -m causeway.scraper` or managed by Dagster schedule

### `causeway/labeler.py`
Two independent functions:

**`generate_lane_labels(image_path)`**
- Parse timestamp from filename to determine shift (morning 06–12 / afternoon 12–19 / night 19–06)
- Load polygon for that camera + shift from `camera_config.json`
- Normalize pixel coords to YOLO segmentation format: `class x1 y1 x2 y2 ... xn yn` (0.0–1.0)
- Class 0 = Towards Woodlands/Singapore CIQ, Class 1 = Towards Johor/Malaysia
- Write to `traffic_lane_labels/YYYYMMDD/CAMERA_ID/filename.txt`
- Skip if label already exists (resumable); log to SQLite `label_log`

**`generate_vehicle_labels(image_path)`**
- Run YOLOv8x inference at 1280p on MPS device
- Map COCO classes: car(2)→1, motorcycle(3)→0, bus(5)→2, truck(7)→3
- Write normalized bounding boxes to `traffic_vehicle_labels/YYYYMMDD/CAMERA_ID/filename.txt`
- Skip if label already exists; log to SQLite `label_log`

### `causeway/dataset.py`
- Scan `traffic_images/` for all date partitions
- Require minimum 2 date folders before splitting (warns and exits if not met)
- Assign: all days except the last → train, last day → val
- Write `dataset_lane.yaml` and `dataset_vehicle.yaml` (Ultralytics-compatible)
- Log all assignments to SQLite `dataset_splits`

### `dagster_defs.py`
Five software-defined assets:
1. **`migrate_existing_images`** — one-shot asset; calls `migration.py`; run once manually via Dagster UI to backfill existing images and SQLite log before the pipeline goes live
2. **`scrape_images`** — calls `causeway.scraper` for one cycle; scheduled every 5 minutes
3. **`generate_lane_labels`** — depends on `scrape_images`; calls `labeler.generate_lane_labels()` for all unlabeled images
4. **`generate_vehicle_labels`** — depends on `scrape_images`; calls `labeler.generate_vehicle_labels()` for all unlabeled images
5. **`build_dataset_split`** — depends on both label assets; calls `dataset.py`

Schedule: `scrape_images` runs every 5 minutes. Downstream assets materialize on-demand via Dagster UI.

Dagster runs on **port 3001** (`workspace.yaml` + `dagster dev --port 3001`).

---

## 5. Streamlit Validation App (`causeway_app.py`)

**Port:** 8502 (`streamlit run causeway_app.py --server.port 8502`)

**Tab 1 — Label Review**
- Sidebar filters: camera, date, shift, label type (lane/vehicle), validation status
- Left panel: raw image
- Right panel: annotated image with lane polygons (lane labels) or bounding boxes (vehicle labels) drawn on top
- Action buttons: **Approve** / **Reject** / **Skip** — writes back to SQLite `label_log.validated`
- Rejected queue: separate view listing all rejected images for re-labeling
- Shift override dropdown: correct misclassified time-window and regenerate label

**Tab 2 — Pipeline Health**
- Scrape gap timeline: hourly bar chart, gaps (>6 min between saves) highlighted red
- Per-camera status cards: last-seen timestamp + healthy/stale/missing badge
- Recent events table: last 50 `scrape_log` rows
- Dataset split summary: train/val count per camera per day

---

## 6. Error Handling Summary

| Component | Failure | Behaviour |
|---|---|---|
| Scraper | Network timeout | Exponential backoff, retry up to 300s |
| Scraper | HTTP 401/403 | Hard skip, log error, no retry |
| Scraper | Restart after gap | Log gap duration on startup |
| Labeler | Unparseable filename | Skip image, log warning |
| Labeler | Corrupted image | Skip, log as error in SQLite |
| Labeler | Already labeled | Skip (resumable by default) |
| Dataset splitter | < 2 date partitions | Warn and exit cleanly |
| Dataset splitter | Empty val set | Warn with count |

---

## 7. Run Instructions (Summary)

```bash
# 1. First-time setup: migrate existing images + backfill SQLite
python -m causeway.migration

# 2. Dagster (port 3001 — avoids conflict with existing Dagster instance)
dagster dev -f dagster_defs.py --port 3001
# Then open http://localhost:3001, materialize assets in order:
#   migrate_existing_images → scrape_images → generate_lane_labels
#   → generate_vehicle_labels → build_dataset_split

# 3. Scraper standalone (if not using Dagster schedule)
python -m causeway.scraper

# 4. Streamlit validation app (port 8502)
streamlit run causeway_app.py --server.port 8502
```

---

## 8. Out of Scope

- No modifications to existing files (`app.py`, `pipeline.py`, `traffic_image_scrapper.py`, etc.)
- No cloud infrastructure
- No web frontend beyond Streamlit
- No model training automation (training triggered manually via `train.py`)
