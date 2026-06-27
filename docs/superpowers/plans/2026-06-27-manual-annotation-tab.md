# Manual Vehicle Annotation Tab — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an ✏️ Annotate tab to `causeway_app.py` that lets users draw, edit, and delete YOLO bounding boxes directly on camera images using `streamlit-drawable-canvas`.

**Architecture:** Session state is the authoritative source for boxes; a live OpenCV-rendered preview shows the current annotation state; the canvas is used solely for drawing new boxes and auto-resets after each draw by keying on box count. Box editing (class change, deletion) happens via a compact list below the canvas. On Save, the label `.txt` is overwritten and the DB record is marked `approved`.

**Tech Stack:** `streamlit-drawable-canvas`, `streamlit`, `opencv-python`, `Pillow` (for canvas background), SQLite via existing `causeway.db`.

## Global Constraints

- Display size: 640 × 480 px (canvas and preview image)
- Class IDs: 0=Motorcycle (Blue `#0000FF`), 1=Car (Yellow `#FFFF00`), 2=Bus (Green `#00FF00`), 3=Truck (Cyan `#00FFFF`)
- YOLO format: `class_id cx cy w h` normalised to original image dimensions (0–1)
- Label files live in `VEHICLE_LABELS_DIR` (from `causeway.config`); resolved via existing `_get_label_path()`
- DB operations via `causeway.db` functions only — no raw SQL in `causeway_app.py`
- Python ≥ 3.11 (project uses `str | None` union syntax)

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `pyproject.toml` | Modify | Add `streamlit-drawable-canvas` dependency |
| `causeway/annotation_helpers.py` | Create | Coordinate conversion, image listing, DB upsert helper |
| `causeway/db.py` | Modify | Add `ensure_label_log_entry()` for images never auto-labelled |
| `causeway_app.py` | Modify | Add tab3 with full Annotate UI |
| `tests/test_annotation_helpers.py` | Create | Unit tests for all helpers |

---

## Task 1: Add Dependency

**Files:**
- Modify: `pyproject.toml`

**Interfaces:**
- Produces: `streamlit_drawable_canvas` importable in subsequent tasks

- [ ] **Step 1: Add to pyproject.toml**

Edit `pyproject.toml`, add `"streamlit-drawable-canvas>=0.9.3"` to `dependencies`:

```toml
dependencies = [
    "dagster>=1.13.11",
    "dagster-webserver>=1.13.11",
    "opencv-python>=4.13.0.92",
    "pyyaml>=6.0.3",
    "requests>=2.34.2",
    "streamlit>=1.58.0",
    "streamlit-drawable-canvas>=0.9.3",
    "ultralytics>=8.4.80",
]
```

- [ ] **Step 2: Install**

```bash
pip install streamlit-drawable-canvas
```

Expected: `Successfully installed streamlit-drawable-canvas-...`

- [ ] **Step 3: Verify import**

```bash
python -c "from streamlit_drawable_canvas import st_canvas; print('ok')"
```

Expected: `ok`

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "feat: add streamlit-drawable-canvas dependency"
```

---

## Task 2: DB Helper for Unannotated Images

**Files:**
- Modify: `causeway/db.py` (add after `log_label`, around line 74)
- Test: `tests/test_annotation_helpers.py` (initial file, expanded in Task 3)

**Interfaces:**
- Consumes: existing `get_connection()`, `DB_PATH`, `label_log` schema
- Produces: `ensure_label_log_entry(image_path, label_path, label_type, shift) -> None`
  — guarantees exactly one `label_log` row for `(image_path, label_type)`, inserting if absent

- [ ] **Step 1: Write the failing test**

Create `tests/test_annotation_helpers.py`:

```python
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
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_annotation_helpers.py -v
```

Expected: `AttributeError: module 'causeway.db' has no attribute 'ensure_label_log_entry'`

- [ ] **Step 3: Implement in `causeway/db.py`**

Add after `log_label()` (after line 74):

```python
def ensure_label_log_entry(image_path: str, label_path: str, label_type: str, shift: str) -> None:
    """Insert a label_log row if none exists for (image_path, label_type)."""
    with get_connection() as conn:
        exists = conn.execute(
            "SELECT 1 FROM label_log WHERE image_path=? AND label_type=? LIMIT 1",
            (image_path, label_type),
        ).fetchone()
        if not exists:
            conn.execute(
                "INSERT INTO label_log (image_path, label_path, label_type, shift) VALUES (?,?,?,?)",
                (image_path, label_path, label_type, shift),
            )
            conn.commit()
```

Also add `ensure_label_log_entry` to the import in `causeway_app.py` later (Task 4).

- [ ] **Step 4: Run to verify pass**

```bash
pytest tests/test_annotation_helpers.py -v
```

Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add causeway/db.py tests/test_annotation_helpers.py
git commit -m "feat: add ensure_label_log_entry to db"
```

---

## Task 3: Annotation Helper Functions

**Files:**
- Create: `causeway/annotation_helpers.py`
- Test: `tests/test_annotation_helpers.py` (extend)

**Interfaces:**
- Produces (all used by Task 4):
  - `DISPLAY_W: int = 640`
  - `DISPLAY_H: int = 480`
  - `CLASS_NAMES: dict[int, str]` — `{0: "Motorcycle", 1: "Car", 2: "Bus", 3: "Truck"}`
  - `CLASS_COLOURS: dict[int, dict]` — `{cls_id: {"stroke": "#RRGGBB", "fill": "rgba(r,g,b,0.3)"}}`
  - `yolo_to_boxes(label_path, orig_w, orig_h) -> list[dict]`
    — Each box: `{"class_id": int, "x1_n": float, "y1_n": float, "x2_n": float, "y2_n": float}` (normalised)
  - `boxes_to_yolo_lines(boxes) -> list[str]`
    — Each line: `"class_id cx cy w h"` (6 decimal places)
  - `canvas_rect_to_box(rect, class_id, orig_w, orig_h) -> dict`
    — Converts one Fabric.js rect dict (with `left`, `top`, `width`, `height`, `scaleX`, `scaleY`) to a normalised box dict
  - `render_annotated_image(img_path, boxes, orig_w, orig_h) -> np.ndarray`
    — Returns 640×480 RGB array with coloured boxes drawn
  - `list_annotation_dates(camera_id, base_dir) -> list[str]`
    — Returns sorted YYYYMMDD strings (descending) for dates with images for that camera
  - `list_images_for_annotation(camera_id, date_str, base_dir) -> list[str]`
    — Returns sorted list of `.jpg` paths

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_annotation_helpers.py`:

```python
import numpy as np
from causeway.annotation_helpers import (
    yolo_to_boxes, boxes_to_yolo_lines, canvas_rect_to_box,
    render_annotated_image, list_annotation_dates, list_images_for_annotation,
    CLASS_NAMES, CLASS_COLOURS, DISPLAY_W, DISPLAY_H,
)


def test_class_names_and_colours():
    assert CLASS_NAMES[0] == "Motorcycle"
    assert CLASS_NAMES[1] == "Car"
    assert CLASS_NAMES[2] == "Bus"
    assert CLASS_NAMES[3] == "Truck"
    assert CLASS_COLOURS[0]["stroke"] == "#0000FF"
    assert CLASS_COLOURS[1]["stroke"] == "#FFFF00"
    assert CLASS_COLOURS[2]["stroke"] == "#00FF00"
    assert CLASS_COLOURS[3]["stroke"] == "#00FFFF"


def test_yolo_to_boxes_empty_when_no_file(tmp_path):
    boxes = yolo_to_boxes(str(tmp_path / "missing.txt"), 320, 240)
    assert boxes == []


def test_yolo_to_boxes_parses_correctly(tmp_path):
    label = tmp_path / "test.txt"
    label.write_text("1 0.5 0.5 0.4 0.6\n")
    boxes = yolo_to_boxes(str(label), 320, 240)
    assert len(boxes) == 1
    b = boxes[0]
    assert b["class_id"] == 1
    assert abs(b["x1_n"] - 0.3) < 1e-5   # cx - w/2 = 0.5 - 0.2
    assert abs(b["y1_n"] - 0.2) < 1e-5   # cy - h/2 = 0.5 - 0.3
    assert abs(b["x2_n"] - 0.7) < 1e-5
    assert abs(b["y2_n"] - 0.8) < 1e-5


def test_yolo_to_boxes_skips_malformed_lines(tmp_path):
    label = tmp_path / "test.txt"
    label.write_text("1 0.5 0.5 0.4 0.6\nbad line\n2 0.1 0.1 0.2 0.2\n")
    boxes = yolo_to_boxes(str(label), 320, 240)
    assert len(boxes) == 2


def test_boxes_to_yolo_lines_roundtrip(tmp_path):
    label = tmp_path / "test.txt"
    label.write_text("1 0.500000 0.500000 0.400000 0.600000\n")
    boxes = yolo_to_boxes(str(label), 320, 240)
    lines = boxes_to_yolo_lines(boxes)
    assert len(lines) == 1
    parts = lines[0].split()
    assert parts[0] == "1"
    assert abs(float(parts[1]) - 0.5) < 1e-5  # cx
    assert abs(float(parts[2]) - 0.5) < 1e-5  # cy
    assert abs(float(parts[3]) - 0.4) < 1e-5  # w
    assert abs(float(parts[4]) - 0.6) < 1e-5  # h


def test_boxes_to_yolo_lines_empty():
    assert boxes_to_yolo_lines([]) == []


def test_canvas_rect_to_box_normalises_correctly():
    # orig 640x480, display 640x480 (1:1 scale)
    rect = {"left": 64.0, "top": 48.0, "width": 128.0, "height": 96.0, "scaleX": 1.0, "scaleY": 1.0}
    box = canvas_rect_to_box(rect, class_id=2, orig_w=640, orig_h=480)
    assert box["class_id"] == 2
    assert abs(box["x1_n"] - 0.1) < 1e-5
    assert abs(box["y1_n"] - 0.1) < 1e-5
    assert abs(box["x2_n"] - 0.3) < 1e-5
    assert abs(box["y2_n"] - 0.3) < 1e-5


def test_canvas_rect_to_box_clamps_to_0_1():
    rect = {"left": -10.0, "top": -10.0, "width": 700.0, "height": 500.0, "scaleX": 1.0, "scaleY": 1.0}
    box = canvas_rect_to_box(rect, class_id=0, orig_w=640, orig_h=480)
    assert box["x1_n"] >= 0.0
    assert box["y1_n"] >= 0.0
    assert box["x2_n"] <= 1.0
    assert box["y2_n"] <= 1.0


def test_canvas_rect_to_box_respects_scale():
    # scaleX=2 means the rect was resized to double width in Fabric.js
    rect = {"left": 0.0, "top": 0.0, "width": 100.0, "height": 100.0, "scaleX": 2.0, "scaleY": 1.0}
    box = canvas_rect_to_box(rect, class_id=1, orig_w=640, orig_h=480)
    assert abs(box["x2_n"] - 200.0 / 640) < 1e-5


def test_render_annotated_image_returns_rgb_array(tmp_path):
    img = np.zeros((240, 320, 3), dtype=np.uint8)
    img_path = str(tmp_path / "frame.jpg")
    import cv2
    cv2.imwrite(img_path, img)
    boxes = [{"class_id": 1, "x1_n": 0.1, "y1_n": 0.1, "x2_n": 0.5, "y2_n": 0.5}]
    result = render_annotated_image(img_path, boxes, 320, 240)
    assert result.shape == (DISPLAY_H, DISPLAY_W, 3)
    assert result.dtype == np.uint8


def test_list_annotation_dates_returns_sorted_descending(tmp_path):
    for date in ["20260625", "20260627", "20260626"]:
        p = tmp_path / date / "2701"
        p.mkdir(parents=True)
        (p / "img.jpg").touch()
    dates = list_annotation_dates("2701", base_dir=str(tmp_path))
    assert dates == ["20260627", "20260626", "20260625"]


def test_list_annotation_dates_filters_by_camera(tmp_path):
    (tmp_path / "20260627" / "2701").mkdir(parents=True)
    (tmp_path / "20260627" / "2701" / "img.jpg").touch()
    (tmp_path / "20260627" / "2702").mkdir(parents=True)
    (tmp_path / "20260627" / "2702" / "img.jpg").touch()
    dates_2701 = list_annotation_dates("2701", base_dir=str(tmp_path))
    dates_2702 = list_annotation_dates("2702", base_dir=str(tmp_path))
    assert "20260627" in dates_2701
    assert "20260627" in dates_2702


def test_list_images_for_annotation_returns_sorted_jpgs(tmp_path):
    cam_dir = tmp_path / "20260627" / "2701"
    cam_dir.mkdir(parents=True)
    (cam_dir / "b.jpg").touch()
    (cam_dir / "a.jpg").touch()
    (cam_dir / "note.txt").touch()
    imgs = list_images_for_annotation("2701", "20260627", base_dir=str(tmp_path))
    assert len(imgs) == 2
    assert imgs[0].endswith("a.jpg")
    assert imgs[1].endswith("b.jpg")


def test_list_images_for_annotation_empty_when_no_match(tmp_path):
    imgs = list_images_for_annotation("2701", "20260627", base_dir=str(tmp_path))
    assert imgs == []
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_annotation_helpers.py -v
```

Expected: `ModuleNotFoundError: No module named 'causeway.annotation_helpers'`

- [ ] **Step 3: Create `causeway/annotation_helpers.py`**

```python
import os
import glob
import cv2
import numpy as np

DISPLAY_W = 640
DISPLAY_H = 480

CLASS_NAMES = {0: "Motorcycle", 1: "Car", 2: "Bus", 3: "Truck"}

CLASS_COLOURS = {
    0: {"stroke": "#0000FF", "fill": "rgba(0,0,255,0.3)"},
    1: {"stroke": "#FFFF00", "fill": "rgba(255,255,0,0.3)"},
    2: {"stroke": "#00FF00", "fill": "rgba(0,255,0,0.3)"},
    3: {"stroke": "#00FFFF", "fill": "rgba(0,255,255,0.3)"},
}

# BGR equivalents for OpenCV drawing (matches CLASS_COLOURS above)
_CV_COLOURS = {
    0: (255, 0, 0),    # Blue (BGR)
    1: (0, 255, 255),  # Yellow (BGR)
    2: (0, 255, 0),    # Green (BGR)
    3: (255, 255, 0),  # Cyan (BGR)
}


def yolo_to_boxes(label_path: str, orig_w: int, orig_h: int) -> list[dict]:
    """Read a YOLO vehicle label file and return normalised box dicts.

    Each box: {"class_id": int, "x1_n": float, "y1_n": float, "x2_n": float, "y2_n": float}
    Coordinates are normalised to [0, 1] relative to original image dimensions.
    Returns [] if the file does not exist or has no valid rows.
    """
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            try:
                cls_id = int(parts[0])
                cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            except ValueError:
                continue
            boxes.append({
                "class_id": cls_id,
                "x1_n": cx - bw / 2,
                "y1_n": cy - bh / 2,
                "x2_n": cx + bw / 2,
                "y2_n": cy + bh / 2,
            })
    return boxes


def boxes_to_yolo_lines(boxes: list[dict]) -> list[str]:
    """Convert normalised box dicts to YOLO format strings.

    Each output line: "class_id cx cy w h" with 6 decimal places.
    Returns [] for an empty box list (valid YOLO background image).
    """
    lines = []
    for box in boxes:
        cx = (box["x1_n"] + box["x2_n"]) / 2
        cy = (box["y1_n"] + box["y2_n"]) / 2
        bw = box["x2_n"] - box["x1_n"]
        bh = box["y2_n"] - box["y1_n"]
        lines.append(f"{box['class_id']} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return lines


def canvas_rect_to_box(rect: dict, class_id: int, orig_w: int, orig_h: int,
                        display_w: int = DISPLAY_W, display_h: int = DISPLAY_H) -> dict:
    """Convert a Fabric.js rect object from st_canvas output to a normalised box dict.

    rect must contain: left, top, width, height (in display px).
    scaleX and scaleY are applied if present (Fabric.js resize handles).
    Coordinates are clamped to [0, 1].
    """
    scale_x = rect.get("scaleX", 1.0)
    scale_y = rect.get("scaleY", 1.0)
    left = rect.get("left", 0.0)
    top = rect.get("top", 0.0)
    width = rect.get("width", 0.0) * scale_x
    height = rect.get("height", 0.0) * scale_y

    sx = display_w / orig_w
    sy = display_h / orig_h

    x1_n = left / (orig_w * sx)
    y1_n = top / (orig_h * sy)
    x2_n = (left + width) / (orig_w * sx)
    y2_n = (top + height) / (orig_h * sy)

    return {
        "class_id": class_id,
        "x1_n": max(0.0, min(1.0, x1_n)),
        "y1_n": max(0.0, min(1.0, y1_n)),
        "x2_n": max(0.0, min(1.0, x2_n)),
        "y2_n": max(0.0, min(1.0, y2_n)),
    }


def render_annotated_image(img_path: str, boxes: list[dict], orig_w: int, orig_h: int,
                            display_w: int = DISPLAY_W, display_h: int = DISPLAY_H) -> np.ndarray:
    """Render boxes onto the image, resized to display dimensions.

    Returns an RGB uint8 array of shape (display_h, display_w, 3).
    """
    image = cv2.imread(img_path)
    if image is None:
        image = np.zeros((display_h, display_w, 3), dtype=np.uint8)
    else:
        image = cv2.resize(image, (display_w, display_h))

    sx = display_w / orig_w
    sy = display_h / orig_h

    for box in boxes:
        x1 = int(box["x1_n"] * orig_w * sx)
        y1 = int(box["y1_n"] * orig_h * sy)
        x2 = int(box["x2_n"] * orig_w * sx)
        y2 = int(box["y2_n"] * orig_h * sy)
        colour = _CV_COLOURS.get(box["class_id"], (255, 255, 255))
        cv2.rectangle(image, (x1, y1), (x2, y2), colour, 2)
        label = CLASS_NAMES.get(box["class_id"], str(box["class_id"]))
        cv2.putText(image, label, (x1, max(y1 - 5, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def list_annotation_dates(camera_id: str, base_dir: str = "traffic_images") -> list[str]:
    """Return YYYYMMDD date strings (descending) that have images for the given camera."""
    pattern = os.path.join(base_dir, "*", camera_id, "*.jpg")
    paths = glob.glob(pattern)
    dates = sorted(
        {os.path.basename(os.path.dirname(os.path.dirname(p))) for p in paths},
        reverse=True,
    )
    return dates


def list_images_for_annotation(camera_id: str, date_str: str,
                                base_dir: str = "traffic_images") -> list[str]:
    """Return sorted list of .jpg paths for the given camera and date (YYYYMMDD)."""
    pattern = os.path.join(base_dir, date_str, camera_id, "*.jpg")
    return sorted(glob.glob(pattern))
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_annotation_helpers.py -v
```

Expected: all tests pass (the 3 DB tests from Task 2 + new annotation helper tests)

- [ ] **Step 5: Commit**

```bash
git add causeway/annotation_helpers.py tests/test_annotation_helpers.py
git commit -m "feat: add annotation helper functions with tests"
```

---

## Task 4: Annotate Tab UI

**Files:**
- Modify: `causeway_app.py`

**Interfaces:**
- Consumes:
  - `from streamlit_drawable_canvas import st_canvas`
  - `from causeway.annotation_helpers import (yolo_to_boxes, boxes_to_yolo_lines, canvas_rect_to_box, render_annotated_image, list_annotation_dates, list_images_for_annotation, CLASS_NAMES, CLASS_COLOURS, DISPLAY_W, DISPLAY_H)`
  - `from causeway.db import ensure_label_log_entry, update_label_validation`
  - `_get_label_path(img_path, "vehicle")` — existing helper in `causeway_app.py`
  - `from causeway.config import IMAGE_BASE_DIR, VEHICLE_LABELS_DIR`
- Produces: ✏️ Annotate tab in the Streamlit app

**Session state keys used:**
- `ann_img_idx` — `int`, current image index in the list for selected camera+date
- `ann_boxes_{img_path}` — `list[dict]`, annotated boxes for that image (source of truth)
- `ann_orig_dims_{img_path}` — `tuple[int, int]`, `(orig_w, orig_h)` for that image
- `ann_canvas_v_{img_path}` — `int`, canvas version counter; incrementing forces canvas reset

- [ ] **Step 1: Add imports to `causeway_app.py`**

After the existing imports block (after line 17), add:

```python
from PIL import Image as PILImage
from streamlit_drawable_canvas import st_canvas
from causeway.annotation_helpers import (
    yolo_to_boxes, boxes_to_yolo_lines, canvas_rect_to_box,
    render_annotated_image, list_annotation_dates, list_images_for_annotation,
    CLASS_NAMES, CLASS_COLOURS, DISPLAY_W, DISPLAY_H,
)
```

Also add `ensure_label_log_entry` to the existing `causeway.db` import line:

```python
from causeway.db import (
    init_db, get_connection, update_label_validation, ensure_label_log_entry,
    get_recent_scrape_logs, get_dataset_split_summary, get_label_logs
)
```

- [ ] **Step 2: Add the third tab to the tab definition (line 29)**

Change:
```python
tab1, tab2 = st.tabs(["📋 Label Review", "📊 Pipeline Health"])
```
To:
```python
tab1, tab2, tab3 = st.tabs(["📋 Label Review", "📊 Pipeline Health", "✏️ Annotate"])
```

- [ ] **Step 3: Add the Annotate tab block at the end of the file**

Append after the existing `with tab2:` block:

```python
with tab3:
    st.header("Manual Annotation")
    st.info(
        "Draw bounding boxes directly on camera images. "
        "Select a camera and date, pick a vehicle class, then drag to draw a box on the canvas. "
        "Use the box list below to change classes or delete boxes. Save writes the labels and marks the image approved."
    )

    # --- Sidebar controls (scoped to this tab via key prefix) ---
    with st.sidebar:
        st.markdown("---")
        st.subheader("✏️ Annotate Controls")
        ann_camera = st.selectbox("Camera", ["2701", "2702", "2704"], key="ann_camera")
        ann_dates = list_annotation_dates(ann_camera, base_dir=IMAGE_BASE_DIR)
        if not ann_dates:
            st.warning("No images found for this camera.")
            st.stop()
        ann_date = st.selectbox("Date", ann_dates, key="ann_date")

    ann_images = list_images_for_annotation(ann_camera, ann_date, base_dir=IMAGE_BASE_DIR)
    if not ann_images:
        st.info("No images found for the selected camera and date.")
        st.stop()

    # --- Image navigator ---
    if "ann_img_idx" not in st.session_state:
        st.session_state["ann_img_idx"] = 0
    st.session_state["ann_img_idx"] = min(st.session_state["ann_img_idx"], len(ann_images) - 1)

    nav_prev, nav_sel, nav_next = st.columns([1, 10, 1])
    with nav_prev:
        if st.button("◀", key="ann_prev") and st.session_state["ann_img_idx"] > 0:
            st.session_state["ann_img_idx"] -= 1
            st.rerun()
    with nav_sel:
        chosen_idx = st.selectbox(
            "Image",
            range(len(ann_images)),
            format_func=lambda i: os.path.basename(ann_images[i]),
            index=st.session_state["ann_img_idx"],
            key="ann_img_sel",
            label_visibility="collapsed",
        )
        if chosen_idx != st.session_state["ann_img_idx"]:
            st.session_state["ann_img_idx"] = chosen_idx
            st.rerun()
    with nav_next:
        if st.button("▶", key="ann_next") and st.session_state["ann_img_idx"] < len(ann_images) - 1:
            st.session_state["ann_img_idx"] += 1
            st.rerun()

    ann_img_path = ann_images[st.session_state["ann_img_idx"]]

    if not os.path.exists(ann_img_path):
        st.warning(f"Image file missing: `{ann_img_path}` — skipping.")
        if st.session_state["ann_img_idx"] < len(ann_images) - 1:
            st.session_state["ann_img_idx"] += 1
            st.rerun()
        st.stop()

    # --- Load boxes and image dims into session state ---
    boxes_key = f"ann_boxes_{ann_img_path}"
    dims_key = f"ann_orig_dims_{ann_img_path}"
    canvas_v_key = f"ann_canvas_v_{ann_img_path}"

    if boxes_key not in st.session_state:
        raw = cv2.imread(ann_img_path)
        orig_h, orig_w = raw.shape[:2] if raw is not None else (480, 640)
        label_path = _get_label_path(ann_img_path, "vehicle")
        st.session_state[boxes_key] = yolo_to_boxes(label_path, orig_w, orig_h)
        st.session_state[dims_key] = (orig_w, orig_h)
        st.session_state[canvas_v_key] = 0

    orig_w, orig_h = st.session_state[dims_key]
    boxes = st.session_state[boxes_key]
    canvas_version = st.session_state[canvas_v_key]

    # --- Main layout: preview left, canvas + controls right ---
    col_preview, col_canvas_ctrl = st.columns(2)

    with col_preview:
        st.caption("📷 Current annotations")
        preview = render_annotated_image(ann_img_path, boxes, orig_w, orig_h)
        st.image(preview, use_container_width=True)

    with col_canvas_ctrl:
        st.caption("🖊 Draw new box (drag a rectangle)")

        cls_names_ordered = [CLASS_NAMES[i] for i in range(4)]
        selected_cls_name = st.radio(
            "Vehicle class",
            cls_names_ordered,
            horizontal=True,
            key="ann_class",
        )
        selected_cls_id = cls_names_ordered.index(selected_cls_name)
        stroke_colour = CLASS_COLOURS[selected_cls_id]["stroke"]

        pil_bg = PILImage.open(ann_img_path).resize((DISPLAY_W, DISPLAY_H))

        canvas_result = st_canvas(
            fill_color="rgba(0,0,0,0)",
            stroke_width=2,
            stroke_color=stroke_colour,
            background_image=pil_bg,
            height=DISPLAY_H,
            width=DISPLAY_W,
            drawing_mode="rect",
            key=f"canvas_{ann_img_path}_{canvas_version}",
        )

        # Detect newly drawn box
        if (
            canvas_result.json_data is not None
            and canvas_result.json_data.get("objects")
        ):
            canvas_objects = canvas_result.json_data["objects"]
            if len(canvas_objects) > 0:
                # The canvas was just used — take the last rect as the new box
                new_rect = canvas_objects[-1]
                new_box = canvas_rect_to_box(new_rect, selected_cls_id, orig_w, orig_h)
                boxes.append(new_box)
                st.session_state[boxes_key] = boxes
                # Reset canvas for next draw
                st.session_state[canvas_v_key] = canvas_version + 1
                st.rerun()

    # --- Box list for class editing and deletion ---
    st.subheader(f"Boxes ({len(boxes)})")
    if not boxes:
        st.caption("No boxes yet — draw on the canvas above.")
    else:
        for i, box in enumerate(list(boxes)):
            bc1, bc2, bc3 = st.columns([3, 6, 1])
            with bc1:
                new_cls_name = st.selectbox(
                    f"box_{i}",
                    cls_names_ordered,
                    index=box["class_id"],
                    key=f"ann_cls_{ann_img_path}_{i}",
                    label_visibility="collapsed",
                )
                new_cls_id = cls_names_ordered.index(new_cls_name)
                if new_cls_id != box["class_id"]:
                    boxes[i]["class_id"] = new_cls_id
                    st.session_state[boxes_key] = boxes
                    st.session_state[canvas_v_key] += 1
                    st.rerun()
            with bc2:
                st.caption(
                    f"x:[{box['x1_n']:.2f}–{box['x2_n']:.2f}]  "
                    f"y:[{box['y1_n']:.2f}–{box['y2_n']:.2f}]"
                )
            with bc3:
                if st.button("×", key=f"ann_del_{ann_img_path}_{i}"):
                    boxes.pop(i)
                    st.session_state[boxes_key] = boxes
                    st.session_state[canvas_v_key] += 1
                    st.rerun()

    # --- Action buttons ---
    st.divider()
    act_clear, act_save = st.columns([1, 3])
    with act_clear:
        if st.button("🗑 Clear all", key="ann_clear"):
            st.session_state[boxes_key] = []
            st.session_state[canvas_v_key] += 1
            st.rerun()
    with act_save:
        if st.button("💾 Save & next", type="primary", key="ann_save"):
            label_path = _get_label_path(ann_img_path, "vehicle")
            os.makedirs(os.path.dirname(label_path), exist_ok=True)
            with open(label_path, "w") as f:
                f.write("\n".join(boxes_to_yolo_lines(boxes)))
            ensure_label_log_entry(ann_img_path, label_path, "vehicle", "")
            update_label_validation(ann_img_path, "vehicle", "approved")
            # Clear session state for this image
            for k in [boxes_key, dims_key, canvas_v_key]:
                st.session_state.pop(k, None)
            # Advance to next image
            if st.session_state["ann_img_idx"] < len(ann_images) - 1:
                st.session_state["ann_img_idx"] += 1
            st.success("Saved and marked approved.")
            st.rerun()
```

- [ ] **Step 4: Run all tests**

```bash
pytest tests/ -v
```

Expected: all existing + new tests pass

- [ ] **Step 5: Smoke-test the UI**

```bash
streamlit run causeway_app.py --server.port 8502
```

Manual checks:
1. Three tabs visible: 📋 Label Review, 📊 Pipeline Health, ✏️ Annotate
2. In Annotate tab: select camera 2701 and a date — image navigator appears
3. Existing auto-generated boxes appear in the left preview with correct colours
4. Draw a rectangle on the canvas → it appears in the left preview on rerun
5. Change a box class via the dropdown → preview updates
6. Delete a box via × → preview updates
7. Clear all → preview is blank
8. Save & next → advances to next image; check label `.txt` written and DB row approved
9. Reload Label Review tab for that image — status shows as approved

- [ ] **Step 6: Commit**

```bash
git add causeway_app.py
git commit -m "feat: add manual vehicle annotation tab with drawable canvas"
```

---

## Self-Review

**Spec coverage:**
- ✅ New ✏️ Annotate tab (Task 4)
- ✅ Sidebar camera selector + date picker (Task 4, Step 3 sidebar block)
- ✅ Image navigator prev/next + dropdown (Task 4, Step 3 navigator block)
- ✅ Canvas with class-coloured stroke (Task 4, Step 3 canvas block)
- ✅ Class radio: Motorcycle/Car/Bus/Truck (Task 4, Step 3)
- ✅ Draw new box (Task 4, Step 3 canvas detection block)
- ✅ Delete box via × button (Task 4, Step 3 box list)
- ✅ Class reassignment via dropdown (Task 4, Step 3 box list)
- ✅ Clear all (Task 4, Step 3 action buttons)
- ✅ Save overwrites `.txt`, marks approved, advances image (Task 4, Step 3 save block)
- ✅ No label file → canvas starts empty (handled by `yolo_to_boxes` returning `[]`)
- ✅ Empty save → writes empty `.txt` (handled by `boxes_to_yolo_lines([])` returning `[]`)
- ✅ Missing image → warning + skip (Task 4, Step 3 missing image guard)
- ✅ Images never auto-labelled → `ensure_label_log_entry` (Task 2)
- ✅ `streamlit-drawable-canvas` dependency (Task 1)

**No placeholders found.**

**Type consistency:** `canvas_rect_to_box` returns `dict` with `class_id, x1_n, y1_n, x2_n, y2_n` — same shape consumed by `boxes_to_yolo_lines` and `render_annotated_image`. All consistent.
