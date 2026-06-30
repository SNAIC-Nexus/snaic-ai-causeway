# Annotator App Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `annotator_app.py` — a dedicated full-screen Streamlit annotation app for drawing, resizing, and reclassifying vehicle bounding boxes on Causeway traffic images, saving labels in YOLO format.

**Architecture:** `streamlit-image-annotation`'s `detection()` component handles all canvas interactions natively (draw, resize, move, reclassify, delete). Sidebar handles navigation (camera → date → image) and save/skip actions. Two new helpers in `annotation_helpers.py` bridge normalised YOLO box dicts ↔ the pixel `[x, y, w, h]` format that `detection()` expects. The app reuses all existing DB and file helpers unchanged.

**Note on sidebar box list:** The `detection()` component has a built-in class selector per box — users click a box on the canvas to change its class. A duplicate sidebar dropdown would require complex state sync and would fight Streamlit's rerun model. The sidebar instead shows a compact per-class count summary and Save/Skip buttons.

**Tech Stack:** Python 3.14, `streamlit>=1.58`, `streamlit-image-annotation==0.8.0`, existing `causeway.annotation_helpers`, `causeway.db`

## Global Constraints

- Run with: `uv run streamlit run annotator_app.py --server.port 8503`
- All tests run with: `.venv/bin/pytest tests/ -v`
- No type annotations using `Optional` — use `X | None` style
- YOLO label files live at: `traffic_vehicle_labels/<date>/<camera_id>/<stem>.txt`
- Image files live at: `traffic_images/<date>/<camera_id>/<stem>.jpg`
- Classes: `{0: motorcycle, 1: car, 2: bus, 3: truck, 4: train}`
- `streamlit-image-annotation` `detection()` bbox format: `[x, y, width, height]` in **pixel** coordinates (origin top-left)
- Normalised box dict format (internal): `{"class_id": int, "x1_n": float, "y1_n": float, "x2_n": float, "y2_n": float}`

---

## File Map

| Action | File | Responsibility |
|---|---|---|
| Modify | `causeway/annotation_helpers.py` | Add Train class + two conversion helpers |
| Modify | `tests/test_annotation_helpers.py` | Tests for the two new helpers |
| Create | `annotator_app.py` | Full Streamlit annotation app |
| Modify | `dataset_vehicle_curated.yaml` | Add train as class 4 |
| Modify | `dataset_vehicle.yaml` | Add train as class 4 |
| Modify | `pyproject.toml` | Add `streamlit-image-annotation` dependency |

---

### Task 1: Add dependency + Train class to helpers

**Files:**
- Modify: `pyproject.toml`
- Modify: `causeway/annotation_helpers.py`

**Interfaces:**
- Produces: `CLASS_NAMES[4] == "Train"`, `CLASS_COLOURS[4]`, `_CV_COLOURS[4]` — used by Task 2 tests and Task 3 app

- [ ] **Step 1: Add `streamlit-image-annotation` to pyproject.toml**

In `pyproject.toml`, add to the `dependencies` list:

```toml
"streamlit-image-annotation>=0.8.0",
```

Run: `uv sync`
Expected: package resolves and installs without error.

- [ ] **Step 2: Add Train to CLASS_NAMES, CLASS_COLOURS, _CV_COLOURS in `causeway/annotation_helpers.py`**

In `annotation_helpers.py`, update the three dicts:

```python
CLASS_NAMES = {0: "Motorcycle", 1: "Car", 2: "Bus", 3: "Truck", 4: "Train"}

CLASS_COLOURS = {
    0: {"stroke": "#0000FF", "fill": "rgba(0,0,255,0.3)"},
    1: {"stroke": "#FFFF00", "fill": "rgba(255,255,0,0.3)"},
    2: {"stroke": "#00FF00", "fill": "rgba(0,255,0,0.3)"},
    3: {"stroke": "#00FFFF", "fill": "rgba(0,255,255,0.3)"},
    4: {"stroke": "#8B4513", "fill": "rgba(139,69,19,0.3)"},
}

_CV_COLOURS = {
    0: (255, 0, 0),      # Blue (BGR)
    1: (0, 255, 255),    # Yellow (BGR)
    2: (0, 255, 0),      # Green (BGR)
    3: (255, 255, 0),    # Cyan (BGR)
    4: (19, 69, 139),    # Brown (BGR)
}
```

- [ ] **Step 3: Verify existing tests still pass**

Run: `.venv/bin/pytest tests/test_annotation_helpers.py -v`
Expected: all 17 existing tests PASS (no regressions from dict additions).

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock causeway/annotation_helpers.py
git commit -m "feat: add Train class (id=4, brown) to annotation helpers and streamlit-image-annotation dep"
```

---

### Task 2: Add pixel ↔ normalised conversion helpers + tests

**Files:**
- Modify: `causeway/annotation_helpers.py`
- Modify: `tests/test_annotation_helpers.py`

**Interfaces:**
- Consumes: normalised box dict `{"class_id": int, "x1_n": float, "y1_n": float, "x2_n": float, "y2_n": float}`
- Produces:
  - `boxes_to_detection_args(boxes: list[dict], img_w: int, img_h: int) -> tuple[list[list[int]], list[int]]`
    - Returns `(bboxes, labels)` where `bboxes[i] = [x, y, w, h]` in pixels, `labels[i] = class_id`
  - `detection_result_to_boxes(result: list[dict], img_w: int, img_h: int) -> list[dict]`
    - `result[i]` has keys `bbox` (list `[x, y, w, h]` pixels) and `label_id` (int)
    - Returns normalised box dicts

- [ ] **Step 1: Write failing tests**

Add to `tests/test_annotation_helpers.py`:

```python
from causeway.annotation_helpers import (
    boxes_to_detection_args, detection_result_to_boxes,
)


def test_boxes_to_detection_args_converts_to_pixels():
    boxes = [{"class_id": 1, "x1_n": 0.1, "y1_n": 0.2, "x2_n": 0.5, "y2_n": 0.6}]
    bboxes, labels = boxes_to_detection_args(boxes, img_w=640, img_h=480)
    assert labels == [1]
    assert bboxes == [[64, 96, 256, 192]]  # x=0.1*640, y=0.2*480, w=0.4*640, h=0.4*480


def test_boxes_to_detection_args_empty():
    bboxes, labels = boxes_to_detection_args([], img_w=640, img_h=480)
    assert bboxes == []
    assert labels == []


def test_detection_result_to_boxes_converts_to_normalised():
    result = [{"bbox": [64, 96, 256, 192], "label_id": 1, "label": "Car"}]
    boxes = detection_result_to_boxes(result, img_w=640, img_h=480)
    assert len(boxes) == 1
    b = boxes[0]
    assert b["class_id"] == 1
    assert abs(b["x1_n"] - 0.1) < 1e-6
    assert abs(b["y1_n"] - 0.2) < 1e-6
    assert abs(b["x2_n"] - 0.5) < 1e-6
    assert abs(b["y2_n"] - 0.6) < 1e-6


def test_detection_result_to_boxes_empty():
    boxes = detection_result_to_boxes([], img_w=640, img_h=480)
    assert boxes == []


def test_roundtrip_normalised_to_pixel_and_back():
    original = [{"class_id": 3, "x1_n": 0.25, "y1_n": 0.1, "x2_n": 0.75, "y2_n": 0.9}]
    bboxes, labels = boxes_to_detection_args(original, img_w=640, img_h=480)
    result = [{"bbox": bboxes[0], "label_id": labels[0], "label": "Truck"}]
    recovered = detection_result_to_boxes(result, img_w=640, img_h=480)
    assert len(recovered) == 1
    b = recovered[0]
    assert b["class_id"] == 3
    assert abs(b["x1_n"] - 0.25) < 1e-3
    assert abs(b["y1_n"] - 0.1) < 1e-3
    assert abs(b["x2_n"] - 0.75) < 1e-3
    assert abs(b["y2_n"] - 0.9) < 1e-3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_annotation_helpers.py -k "detection_args or detection_result" -v`
Expected: FAIL with `ImportError: cannot import name 'boxes_to_detection_args'`

- [ ] **Step 3: Implement the two helpers in `causeway/annotation_helpers.py`**

Add after `canvas_rect_to_box`:

```python
def boxes_to_detection_args(
    boxes: list[dict], img_w: int, img_h: int
) -> tuple[list[list[int]], list[int]]:
    """Convert normalised box dicts to pixel [x, y, w, h] lists for detection().

    Returns (bboxes, labels) where bboxes[i] = [x, y, w, h] in pixels,
    labels[i] = class_id (int).
    """
    bboxes = []
    labels = []
    for box in boxes:
        x = int(round(box["x1_n"] * img_w))
        y = int(round(box["y1_n"] * img_h))
        w = int(round((box["x2_n"] - box["x1_n"]) * img_w))
        h = int(round((box["y2_n"] - box["y1_n"]) * img_h))
        bboxes.append([x, y, w, h])
        labels.append(box["class_id"])
    return bboxes, labels


def detection_result_to_boxes(
    result: list[dict], img_w: int, img_h: int
) -> list[dict]:
    """Convert detection() output to normalised box dicts.

    result[i] has keys: bbox ([x, y, w, h] pixels), label_id (int).
    Returns list of {"class_id": int, "x1_n": float, "y1_n": float,
                      "x2_n": float, "y2_n": float}.
    """
    boxes = []
    for item in result:
        x, y, w, h = item["bbox"]
        boxes.append({
            "class_id": item["label_id"],
            "x1_n": max(0.0, min(1.0, x / img_w)),
            "y1_n": max(0.0, min(1.0, y / img_h)),
            "x2_n": max(0.0, min(1.0, (x + w) / img_w)),
            "y2_n": max(0.0, min(1.0, (y + h) / img_h)),
        })
    return boxes
```

- [ ] **Step 4: Run all annotation helper tests**

Run: `.venv/bin/pytest tests/test_annotation_helpers.py -v`
Expected: all tests PASS (17 existing + 5 new = 22 total).

- [ ] **Step 5: Commit**

```bash
git add causeway/annotation_helpers.py tests/test_annotation_helpers.py
git commit -m "feat: add boxes_to_detection_args and detection_result_to_boxes helpers"
```

---

### Task 3: Build `annotator_app.py`

**Files:**
- Create: `annotator_app.py`

**Interfaces:**
- Consumes:
  - `annotation_helpers.yolo_to_boxes(label_path, orig_w, orig_h) -> list[dict]`
  - `annotation_helpers.boxes_to_yolo_lines(boxes) -> list[str]`
  - `annotation_helpers.boxes_to_detection_args(boxes, img_w, img_h) -> (bboxes, labels)`
  - `annotation_helpers.detection_result_to_boxes(result, img_w, img_h) -> list[dict]`
  - `annotation_helpers.list_annotation_dates(camera_id, base_dir) -> list[str]`
  - `annotation_helpers.list_images_for_annotation(camera_id, date_str, base_dir) -> list[str]`
  - `db.ensure_label_log_entry(img_path, label_path, label_type, time_of_day)`
  - `db.update_label_validation(img_path, label_type, status)`
  - `detection(image_path, label_list, bboxes, labels, height, width, key) -> list[dict]`

- [ ] **Step 1: Write `annotator_app.py`**

Create `/Users/chuan/Development/PythonProjects/snaic-ai-causeway/annotator_app.py`:

```python
import os
import streamlit as st
from PIL import Image
from streamlit_image_annotation import detection

from causeway.annotation_helpers import (
    CLASS_NAMES,
    yolo_to_boxes,
    boxes_to_yolo_lines,
    boxes_to_detection_args,
    detection_result_to_boxes,
    list_annotation_dates,
    list_images_for_annotation,
)
from causeway.db import ensure_label_log_entry, update_label_validation

IMAGE_BASE_DIR = "traffic_images"
LABEL_BASE_DIR = "traffic_vehicle_labels"
CAMERAS = ["2701", "2702", "2704"]
LABEL_LIST = [CLASS_NAMES[i] for i in sorted(CLASS_NAMES)]  # ["Motorcycle","Car","Bus","Truck","Train"]
CANVAS_W = 800
CANVAS_H = 600


def _label_path(img_path: str) -> str:
    """Derive YOLO .txt path from image path."""
    parts = img_path.split(os.sep)
    # parts: [..., date, camera_id, stem.jpg]
    date, camera, fname = parts[-3], parts[-2], parts[-1]
    stem = os.path.splitext(fname)[0]
    return os.path.join(LABEL_BASE_DIR, date, camera, stem + ".txt")


def _save(img_path: str, result: list[dict], img_w: int, img_h: int) -> None:
    """Write YOLO .txt and mark image approved in DB."""
    boxes = detection_result_to_boxes(result, img_w, img_h)
    lines = boxes_to_yolo_lines(boxes)
    lbl_path = _label_path(img_path)
    os.makedirs(os.path.dirname(lbl_path), exist_ok=True)
    with open(lbl_path, "w") as f:
        f.write("\n".join(lines))
    time_of_day = "morning"  # scraper images don't carry time-of-day metadata
    ensure_label_log_entry(img_path, lbl_path, "vehicle", time_of_day)
    update_label_validation(img_path, "vehicle", "approved")


def _clear_canvas_key() -> None:
    """Force canvas remount on image change by bumping a version counter."""
    st.session_state["_canvas_ver"] = st.session_state.get("_canvas_ver", 0) + 1


# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Causeway Annotator", layout="wide")
st.title("✏️ Causeway Vehicle Annotator")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Navigation")

    camera = st.selectbox("Camera", CAMERAS, key="ann_camera")
    dates = list_annotation_dates(camera, base_dir=IMAGE_BASE_DIR)
    if not dates:
        st.warning("No images found for this camera.")
        st.stop()

    date = st.selectbox("Date", dates, key="ann_date")
    images = list_images_for_annotation(camera, date, base_dir=IMAGE_BASE_DIR)
    if not images:
        st.warning("No images found for this date.")
        st.stop()

    # Reset index when camera or date changes
    nav_key = f"{camera}_{date}"
    if st.session_state.get("_nav_key") != nav_key:
        st.session_state["_nav_key"] = nav_key
        st.session_state["ann_idx"] = 0
        _clear_canvas_key()

    n = len(images)
    idx = st.session_state.get("ann_idx", 0)

    col_prev, col_counter, col_next = st.columns([1, 2, 1])
    with col_prev:
        if st.button("◀", disabled=idx == 0):
            st.session_state["ann_idx"] = idx - 1
            _clear_canvas_key()
            st.rerun()
    with col_counter:
        st.markdown(f"<div style='text-align:center;padding-top:6px'>{idx+1} / {n}</div>",
                    unsafe_allow_html=True)
    with col_next:
        if st.button("▶", disabled=idx == n - 1):
            st.session_state["ann_idx"] = idx + 1
            _clear_canvas_key()
            st.rerun()

    st.divider()
    st.caption("**Controls**")
    st.caption("• Drag on image to draw a box")
    st.caption("• Click a box label to change class")
    st.caption("• Right-click a box to delete it")

    st.divider()
    save_next = st.button("💾 Save & Next", use_container_width=True, type="primary")
    save_only = st.button("💾 Save", use_container_width=True)
    skip = st.button("⏭ Skip", use_container_width=True)

# ── Main canvas ───────────────────────────────────────────────────────────────
idx = st.session_state.get("ann_idx", 0)
img_path = images[idx]
img = Image.open(img_path)
img_w, img_h = img.size

lbl_path = _label_path(img_path)
existing_boxes = yolo_to_boxes(lbl_path, img_w, img_h)
init_bboxes, init_labels = boxes_to_detection_args(existing_boxes, img_w, img_h)

canvas_ver = st.session_state.get("_canvas_ver", 0)
result = detection(
    image_path=img_path,
    label_list=LABEL_LIST,
    bboxes=init_bboxes if init_bboxes else None,
    labels=init_labels if init_labels else None,
    height=CANVAS_H,
    width=CANVAS_W,
    key=f"det_{img_path}_{canvas_ver}",
)

# Show per-class box count below canvas
if result:
    counts: dict[str, int] = {}
    for item in result:
        counts[item["label"]] = counts.get(item["label"], 0) + 1
    summary = " · ".join(f"{v}× {k}" for k, v in counts.items())
    st.caption(f"Boxes: {summary}")
else:
    st.caption("No boxes — drag on the image above to draw one.")

# ── Actions ────────────────────────────────────────────────────────────────────
if save_next or save_only:
    if result is None:
        result = []
    _save(img_path, result, img_w, img_h)
    st.success("Saved ✓")
    if save_next and idx < n - 1:
        st.session_state["ann_idx"] = idx + 1
        _clear_canvas_key()
        st.rerun()

if skip:
    if idx < n - 1:
        st.session_state["ann_idx"] = idx + 1
        _clear_canvas_key()
        st.rerun()
```

- [ ] **Step 2: Smoke-test — launch the app and verify it loads without error**

Run: `uv run streamlit run annotator_app.py --server.port 8503`

Open `http://localhost:8503` in a browser. Check:
- Camera selector shows 2701 / 2702 / 2704
- Date selector populates from available images
- Canvas renders with pre-loaded RT-DETR boxes (if labels exist for selected image)
- Box count summary updates when a box is drawn
- Save & Next advances to the next image
- YOLO `.txt` file written to `traffic_vehicle_labels/<date>/<camera>/<stem>.txt`

- [ ] **Step 3: Commit**

```bash
git add annotator_app.py
git commit -m "feat: add annotator_app.py with streamlit-image-annotation canvas"
```

---

### Task 4: Update YAML configs for Train class

**Files:**
- Modify: `dataset_vehicle_curated.yaml`
- Modify: `dataset_vehicle.yaml`

**Interfaces:**
- No code interfaces — YAML consumed by `train.py` and Ultralytics

- [ ] **Step 1: Update `dataset_vehicle_curated.yaml`**

Replace the `names` and `nc` fields:

```yaml
names:
  0: motorcycle
  1: car
  2: bus
  3: truck
  4: train
nc: 5
```

- [ ] **Step 2: Update `dataset_vehicle.yaml`**

Open `dataset_vehicle.yaml` and apply the same change to `names` and `nc`:

```yaml
names:
  0: motorcycle
  1: car
  2: bus
  3: truck
  4: train
nc: 5
```

- [ ] **Step 3: Verify YAML loads correctly**

Run:
```bash
.venv/bin/python -c "
import yaml
for f in ['dataset_vehicle_curated.yaml', 'dataset_vehicle.yaml']:
    d = yaml.safe_load(open(f))
    assert d['nc'] == 5, f'{f}: nc should be 5'
    assert d['names'][4] == 'train', f'{f}: class 4 should be train'
    print(f'{f}: OK')
"
```
Expected:
```
dataset_vehicle_curated.yaml: OK
dataset_vehicle.yaml: OK
```

- [ ] **Step 4: Commit**

```bash
git add dataset_vehicle_curated.yaml dataset_vehicle.yaml
git commit -m "feat: add train as class 4 to vehicle dataset YAML configs"
```

---

## Post-Implementation Checklist

- [ ] All tests pass: `.venv/bin/pytest tests/ -v`
- [ ] App loads at `http://localhost:8503` with no console errors
- [ ] Drawing a box and pressing Save & Next writes a `.txt` file with correct YOLO format
- [ ] Existing RT-DETR pre-labels load as initial boxes on images that already have `.txt` files
- [ ] Train class (id 4, brown) appears in the label selector within the canvas
