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
    bboxes=init_bboxes if init_bboxes else [],
    labels=init_labels if init_labels else [],
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
