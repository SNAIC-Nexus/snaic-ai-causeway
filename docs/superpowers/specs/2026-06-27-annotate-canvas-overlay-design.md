---
name: annotate-canvas-overlay
description: Redesign the Annotate page so the drawable canvas is full-width with the traffic camera image as its background, replacing the current 2-column split layout.
---

# Annotate Page — Full-Width Canvas Overlay

## Problem

The current layout splits the Annotate page into two equal columns: a read-only annotated preview on the left and the drawable canvas on the right. This halves the canvas width (320 px effective display), reducing annotation precision and making it harder to place bounding boxes at the correct pixel location for YOLO training.

## Goal

Render the drawable canvas full-width with the traffic camera image as its background, so users draw bounding boxes at the exact pixel location that maps to YOLO normalised coordinates.

## Design

### Layout

Remove `st.columns(2)` and the left-column preview block. The page becomes a single-column flow:

1. `st_canvas` — rendered at `DISPLAY_W × DISPLAY_H` (640 × 480 px), image as background, existing boxes pre-loaded via `initial_drawing`
2. Vehicle class radio (horizontal) — below the canvas
3. Save / Delete / Clear buttons — below the radio

The separate annotated preview (`render_annotated_image` + `st.image`) is deleted. The canvas already renders existing boxes via `initial_drawing`, making the preview redundant.

### Canvas sizing

`DISPLAY_W = 640` and `DISPLAY_H = 480` remain unchanged in `causeway/annotation_helpers.py`. All coordinate normalisation (`x1_n = left / DISPLAY_W`, etc.) continues to reference these constants, so YOLO export coordinates are unaffected.

The canvas is rendered at fixed pixel dimensions rather than stretching to the container. Stretching would require a scaling factor on every coordinate read-back, introducing rounding errors and coupling UI width to annotation accuracy.

### Coordinate correctness

No changes to coordinate math, session state keys, or YOLO serialisation. The only change is removing the layout wrapper that restricted the canvas to half the container.

## Files Changed

| File | Change |
|------|--------|
| `causeway_app.py` | Remove `st.columns(2)`, `col_preview` block, and `render_annotated_image` / `st.image` preview call inside `_render_vehicle_annotation_tab`. Canvas, controls, and button logic move out of the column context manager. |

No changes to `causeway/annotation_helpers.py` or any test files.

## Out of Scope

- Responsive/dynamic canvas sizing
- Replacing the collapsible annotated preview (can be added later as an expander)
- Any changes to YOLO export, label saving, or session state logic
