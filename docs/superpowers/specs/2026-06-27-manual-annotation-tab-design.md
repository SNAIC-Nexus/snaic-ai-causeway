# Manual Vehicle Annotation Tab — Design Spec

**Date:** 2026-06-27  
**Status:** Approved

## Context

The fine-tuning pipeline relies on human-reviewed vehicle labels. The existing Label Review tab only supports approve/reject of auto-generated YOLO detections. Certain classes — particularly motorcycles — are rarely detected by the base model, leaving too few curated examples to train on. This feature adds a manual bounding box annotation tab so users can draw, edit, and correct vehicle labels directly in the Streamlit app.

---

## Tab: ✏️ Annotate

A new third tab added to `causeway_app.py` alongside the existing "📋 Label Review" and "📊 Pipeline Health" tabs.

---

## Layout & Navigation

**Sidebar (scoped to Annotate tab):**
- Camera selector: 2701 / 2702 / 2704
- Date picker: defaults to today; lists only dates that have scraped images

**Main area:**
- Image navigator: dropdown (or prev/next buttons) listing all images for the selected camera + date
- Canvas + controls rendered below the navigator

---

## Canvas & Controls

The canvas uses `streamlit-drawable-canvas` (`st_canvas`). The selected image is rendered as the canvas background at a fixed display size of **640 × 480 px**. Existing auto-generated boxes are pre-loaded as coloured rectangles via `initial_drawing`.

### Class colours
| Class | Colour |
|---|---|
| Motorcycle | Blue |
| Car | Yellow |
| Bus | Green |
| Truck | Cyan |

### Controls
| Control | Behaviour |
|---|---|
| Class selector (radio) | Motorcycle / Car / Bus / Truck — stamped onto each new box drawn; in Select mode, changing the radio re-assigns the class of the currently selected box |
| Mode toggle | **Draw** (add new box) / **Select** (click existing box to re-assign its class or delete it) |
| Delete selected | Removes the currently selected box |
| Clear all | Wipes all boxes from the canvas |
| Save | Writes YOLO `.txt`, marks `validated='approved'` in DB, advances to next image |

---

## Data Flow

### Loading
1. Resolve label `.txt` path from image path (same logic as Label Review tab)
2. If the file exists, parse YOLO rows (`class_id cx cy w h`, normalised)
3. Convert to pixel coordinates at 640 × 480
4. Pass to `st_canvas` as Fabric.js rect objects with class-colour fill and a `class_id` custom property
5. If no label file exists, canvas starts empty

### Saving
1. Read rect objects from canvas output JSON
2. For each rect, read `class_id` from its custom property
3. Normalise pixel coords back to YOLO format (`class_id cx cy w h` in 0–1 range, relative to original image dimensions)
4. Overwrite the label `.txt` file (creates it if absent)
5. Call `update_label_validation(img_path, "vehicle", "approved")`
6. Auto-advance to the next image in the navigator

### Edge cases
- **No label file** → canvas starts empty; Save creates the file
- **User clears all and saves** → writes an empty `.txt` (valid YOLO background image)
- **Image file missing** → skip with an inline warning, show next image

---

## New Dependency

```
streamlit-drawable-canvas
```

Add to `pyproject.toml` under `[project.dependencies]`.

---

## Files Changed

| File | Change |
|---|---|
| `causeway_app.py` | Add Annotate tab with canvas UI and save logic |
| `causeway/app_helpers.py` (new or existing) | `yolo_to_canvas_rects()`, `canvas_rects_to_yolo()` coordinate conversion helpers |
| `pyproject.toml` | Add `streamlit-drawable-canvas` dependency |

---

## Verification

1. `pip install streamlit-drawable-canvas` and run `streamlit run causeway_app.py`
2. Open the ✏️ Annotate tab — select camera 2701, pick a date with images
3. Confirm existing auto-generated boxes appear pre-loaded on the canvas in correct colours
4. Draw a new box, assign class "Motorcycle", save — verify the `.txt` file contains the new YOLO row and `label_log.validated = 'approved'`
5. Reload the image in Label Review tab — confirm it shows as approved with the manually drawn box
6. Clear all boxes and save — verify `.txt` is empty and image is still marked approved
7. Select an existing box, change its class via the radio selector, save — verify the `.txt` reflects the updated class
