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

### Why We Fine-Tune the Model

The base YOLO model was trained on COCO — a dataset of street-level photos taken
close to vehicles. Causeway cameras shoot from overhead, at night, with dense
bumper-to-bumper queues and headlight glare. This **domain gap** means the base
model misses most vehicles in our images.

To fix this, we:
1. **Curate** a clean subset of daytime frames where the base model detects well
2. **Fine-tune** the model on those curated frames (adapting it to our cameras)
3. **Re-label** all images with the improved model — including night frames

The result is `models/causeway_vehicle_v1.pt` — a detector trained specifically
for Woodlands Causeway surveillance angles.

---

## Project Structure

```
.
├── causeway/                    # Core pipeline package (new)
│   ├── __init__.py
│   ├── config.py                — Shared constants: API URL, camera IDs, folder paths
│   ├── db.py                    — SQLite layer: scrape log, label log, dataset splits
│   ├── scraper.py               — LTA API scraper with exponential backoff + gap detection
│   ├── labeler.py               — YOLO label generators (lane segmentation + vehicle detection)
│   ├── dataset.py               — Day-based train/val splitter + dataset YAML writer
│   └── migration.py             — One-shot: canonicalise existing images + backfill SQLite
│
├── tests/                       # Test suite mirroring causeway/ structure
│   ├── test_db.py
│   ├── test_scraper.py
│   ├── test_labeler.py
│   ├── test_migration.py
│   ├── test_dataset.py
│   └── test_app_helpers.py      — Unit tests for Streamlit helper functions
│
├── dagster_defs.py              — Dagster assets + 5-minute scrape schedule (port 3001)
├── workspace.yaml               — Dagster workspace config
├── causeway_app.py              — Streamlit validation + curation app (port 8502)
│
├── traffic_image_scrapper.py    — Original standalone scraper script (legacy)
├── pipeline.py                  — MLX YOLO live counting pipeline (uses causeway_vehicle_v1.npz)
├── picker.py                    — Interactive polygon point picker for camera_config.json
├── camera_config.json           — Lane polygon definitions per camera and shift
└── train.py                     — Fine-tune script: curate → train on MPS → export MLX weights
```

### Data folders (created at runtime)

```
traffic_images/
  YYYYMMDD/
    CAMERA_ID/              ← e.g. 20260627/2701/
      filename.jpg

traffic_lane_labels/        ← YOLO segmentation labels (polygon per lane zone)
  YYYYMMDD/CAMERA_ID/filename.txt

traffic_vehicle_labels/     ← YOLO detection labels (bounding box per vehicle)
  YYYYMMDD/CAMERA_ID/filename.txt

causeway.db                 ← SQLite database (scrape log, label log, dataset splits)
dataset_lane.yaml           ← Written by build_dataset_split
dataset_vehicle.yaml        ← Written by build_dataset_split
dataset_vehicle_curated.yaml ← Written by train.py (curated frames only, for fine-tuning)

models/
  yolo26n.pt                ← Base COCO-pretrained YOLOv2.6n (starting point for fine-tune)
  causeway_vehicle_v1.pt    ← Fine-tuned detector (created by train.py)
  causeway_vehicle_v1.npz   ← MLX-converted weights for live counting in pipeline.py

dataset/curated/            ← Exported by train.py before training
  images/train/
  images/val/
  labels/train/
  labels/val/
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

If you already have images in `traffic_images/` from the legacy scraper, run this
once to move them into the canonical `YYYYMMDD/CAMERA_ID/` folder structure and
register them in the database:

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

Open `http://localhost:3001`. You will see six assets:

| Asset | What it does |
|---|---|
| `migrate_existing_images` | Run once to backfill existing images into canonical paths |
| `scrape_images` | Download the latest camera frames from the LTA API |
| `generate_lane_labels` | Write lane polygon labels from `camera_config.json` |
| `generate_vehicle_labels` | Run YOLO inference to write vehicle bounding box labels |
| `fine_tune_vehicle_model` | Export curated labels, fine-tune on MPS, save `causeway_vehicle_v1.pt/.npz` |
| `build_dataset_split` | Split all days into train/val and write YAML config files |

**To run the full pipeline:** click each asset in order and press **Materialize**.

> **Fine-tuning note:** `fine_tune_vehicle_model` requires approved vehicle labels in the
> Streamlit app before it will produce a useful model. Approve at least 200 daytime frames
> first (see [Curating Vehicle Labels](#curating-vehicle-labels) below).

The `scrape_images` asset runs automatically every 5 minutes via the built-in schedule.

### Option B: Scraper standalone

```bash
python -m causeway.scraper
```

Runs the scraper in an infinite loop (5-minute cycle). Logs a startup gap warning
if the scraper was offline for longer than one cycle.

### Streamlit Validation App

```bash
streamlit run causeway_app.py --server.port 8502
```

Open `http://localhost:8502` to:
- **Tab 1 — Label Review:** See each image side-by-side with its generated labels.
  Approve good labels, reject bad ones. Filter by camera, label type, or status.
  Enable **Daytime only (06:00–19:00)** to focus on frames where vehicle detection
  is most reliable. A curation progress counter shows how many frames you've reviewed.
- **Tab 2 — Pipeline Health:** Per-camera status cards, hourly scrape timeline,
  recent event log, and dataset split summary.

### Curating Vehicle Labels

Vehicle detection quality is lower at night (headlight glare, overhead camera angle).
Before fine-tuning, curate a clean training set:

1. Open the Streamlit app and go to **Tab 1 — Label Review**
2. Set **Label Type = vehicle** and tick **Daytime only (06:00–19:00)**
3. Work through the frames — click **✅ Approve** for frames where boxes look correct,
   **❌ Reject** for frames that are wrong or blurry
4. Aim for **200+ approved frames** across all three cameras
5. Run `python train.py` — it exports approved frames, fine-tunes the model on your
   M-series Mac (MPS), and saves the result to `models/causeway_vehicle_v1.pt`

---

## Legacy Scripts

These scripts predate the `causeway/` package and are preserved for reference.

| File | Purpose |
|---|---|
| `traffic_image_scrapper.py` | Original scraper — downloads images from the LTA API without SQLite tracking |
| `pipeline.py` | MLX-native YOLO inference pipeline for Apple Silicon using `yolo26mlx` |
| `picker.py` | Interactive OpenCV tool for clicking polygon points to populate `camera_config.json` |

The new `causeway/` package supersedes `traffic_image_scrapper.py` for day-to-day use.
`picker.py` is still useful when you need to re-draw lane polygons for a camera.

---

## Running Tests

```bash
uv run pytest tests/ -v
```

Tests cover the SQLite layer, migration idempotency, scraper backoff, lane label
generation, dataset splitting, curated export, and Streamlit helper functions.

---

## Training the Model

After curating at least 200 approved vehicle label frames in the Streamlit app:

```bash
python train.py
```

This script does four things automatically:
1. Exports approved frames to `dataset/curated/` (80/20 day-based split)
2. Fine-tunes `models/yolo26n.pt` on your M-series Mac via MPS (≈20–30 min)
3. Copies the best checkpoint to `models/causeway_vehicle_v1.pt`
4. Converts weights to MLX format (`models/causeway_vehicle_v1.npz`) for fast inference

After training, re-run vehicle labeling with the domain-adapted model:

```python
from causeway.labeler import generate_vehicle_labels
generate_vehicle_labels(model_path="models/causeway_vehicle_v1.pt")
```

Or materialise `generate_vehicle_labels` in the Dagster UI after deleting existing
`traffic_vehicle_labels/` files to force a clean re-run.

---

## Camera Reference

| Camera ID | Location | Shift-aware? |
|---|---|---|
| `2701` | Woodlands Causeway → Johor | Yes (morning / afternoon / night) |
| `2702` | Woodlands Checkpoint → BKE | No (static layout) |
| `2704` | Woodlands Flyover → Checkpoint | No (static layout) |

Lane classes: `0` = Towards Woodlands/Singapore CIQ, `1` = Towards Johor/Malaysia.
Vehicle classes: `0` = motorcycle, `1` = car, `2` = bus, `3` = truck.

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
| **Exponential backoff** | Retry strategy that waits longer after each failure to avoid hammering an API |
| **Gap detection** | Logging how long the scraper was offline so you know which time windows are missing |
| **Domain gap** | The performance drop when a model trained on one type of image (street-level COCO) is used on a different type (overhead surveillance) |
| **Fine-tuning** | Continuing training of a pre-trained model on a small domain-specific dataset to close the domain gap |
| **Curation** | Manually approving a clean subset of auto-generated labels to use as fine-tuning training data |
| **MLX** | Apple's machine learning framework optimised for Apple Silicon — used here for fast inference in `pipeline.py` |
| **MPS** | Metal Performance Shaders — Apple Silicon GPU backend used by `ultralytics` for training |
