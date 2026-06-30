# Annotator App Design — 2026-06-30

## Goal

Dedicated Streamlit annotation app (`annotator_app.py`) for drawing and correcting vehicle bounding boxes on Woodlands Causeway traffic images. Replaces the cramped ✏️ Annotate tab in `causeway_app.py` with a full-screen, user-friendly interface. Runs on port 8503 alongside the existing curation app on 8502.

---

## Library

**`streamlit-image-annotation` (v0.8.0)** — purpose-built for multi-class bbox annotation. Provides:
- Drag-to-draw new boxes
- Resize and move handles on existing boxes
- Built-in class label rendering per box

---

## Classes

| Class     | ID | Colour         |
|-----------|----|----------------|
| Motorcycle | 0 | Blue (`#0000FF`) |
| Car        | 1 | Yellow (`#FFFF00`) |
| Bus        | 2 | Green (`#00FF00`) |
| Truck      | 3 | Cyan (`#00FFFF`) |
| Train      | 4 | Brown (`#8B4513`) |

`dataset_vehicle_curated.yaml` must have `train` added as class 4 in its `names` list.

---

## Architecture

```
annotator_app.py
├── Sidebar
│   ├── Camera selector (2701 / 2702 / 2704)
│   ├── Date selector (dropdown of available dates)
│   ├── Image navigator (◀ image N of M ▶)
│   └── Box list — one row per box:
│       ├── Class dropdown (Motorcycle / Car / Bus / Truck / Train)
│       └── Delete button [×]
├── Main canvas
│   └── st_bbox_select (streamlit-image-annotation)
│       ├── Pre-loaded boxes from existing YOLO .txt (RT-DETR output or prior hand labels)
│       ├── Drag to add new box → appended to box list
│       └── Resize / move / reposition handles on each box
└── Footer action bar
    ├── [Save & Next] — write YOLO .txt, mark approved in DB, advance image
    ├── [Save]        — write YOLO .txt, mark approved, stay on image
    └── [Skip]        — advance without saving
```

---

## Data Flow

### On image load
1. Resolve YOLO `.txt` path under `traffic_vehicle_labels/`
2. If file exists: `yolo_to_boxes()` → list of `{x, y, w, h, class_id}` in pixel coords
3. If no file: start with empty box list (blank canvas)
4. Pass boxes + PIL image to `st_bbox_select` as `label_list` initial state
5. Render sidebar box list with class dropdowns pre-set to loaded class IDs

### On canvas interaction (draw / resize / move)
6. `st_bbox_select` returns updated bbox list on each Streamlit rerun
7. Merge canvas bbox coords with sidebar class selections → unified box state in `st.session_state`

### On class dropdown change (sidebar)
8. Update `st.session_state` class for that box index
9. Streamlit reruns; canvas re-renders boxes with updated class colours

### On [Delete ×]
10. Remove box at index from session state → canvas rerenders without it

### On [Save] / [Save & Next]
11. `boxes_to_yolo_lines()` → write YOLO `.txt` to `traffic_vehicle_labels/<camera>/<date>/<image>.txt`
12. `ensure_label_log_entry()` + `update_label_validation(..., "approved")`
13. If [Save & Next]: advance image index → repeat from step 1

### On [Skip]
14. Advance image index without writing

---

## Reused Helpers (no changes needed)

| Helper | Used for |
|---|---|
| `annotation_helpers.yolo_to_boxes` | Load existing YOLO labels → pixel boxes |
| `annotation_helpers.boxes_to_yolo_lines` | Convert pixel boxes → YOLO txt lines |
| `annotation_helpers.list_annotation_dates` | Populate date selector |
| `annotation_helpers.list_images_for_annotation` | Populate image navigator |
| `db.ensure_label_log_entry` | Create label_log row if missing |
| `db.update_label_validation` | Mark image as approved |

---

## Session State Keys

| Key | Type | Purpose |
|---|---|---|
| `ann_camera` | str | Selected camera ID |
| `ann_date` | str | Selected date |
| `ann_idx` | int | Current image index |
| `ann_boxes` | list[dict] | `{x,y,w,h,class_id}` for current image |

State is cleared on camera/date/image change to prevent stale boxes from a previous image bleeding in.

---

## YAML Update Required

`dataset_vehicle_curated.yaml` — add `train` as class 4:

```yaml
names:
  0: motorcycle
  1: car
  2: bus
  3: truck
  4: train
```

Same update needed in `dataset_vehicle.yaml` if used for training.

---

## Running

```bash
uv run streamlit run annotator_app.py --server.port 8503
```

Curation app continues on 8502 — both can run simultaneously.

---

## Out of Scope

- Lane annotation (remains in `causeway_app.py`)
- Removing the existing Annotate tab from `causeway_app.py` (leave as fallback)
- Polygon / segmentation masks (bounding boxes only)
