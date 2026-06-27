import os
import json
import glob
import re
from datetime import datetime, timedelta

import cv2
import numpy as np
import streamlit as st

from causeway.config import (
    IMAGE_BASE_DIR, LANE_LABELS_DIR, VEHICLE_LABELS_DIR, CAMERA_CONFIG_PATH
)
from causeway.db import (
    init_db, get_connection, update_label_validation, ensure_label_log_entry,
    get_recent_scrape_logs, get_dataset_split_summary, get_label_logs
)
from PIL import Image as PILImage
from streamlit_drawable_canvas import st_canvas
from causeway.annotation_helpers import (
    yolo_to_boxes, boxes_to_yolo_lines, canvas_rect_to_box,
    render_annotated_image, list_annotation_dates, list_images_for_annotation,
    CLASS_NAMES, CLASS_COLOURS, DISPLAY_W, DISPLAY_H,
)

def _parse_hour_from_path(img_path: str):
    match = re.search(r"_(\d{8})_(\d{2})\d{4}\.jpg$", img_path, re.IGNORECASE)
    return int(match.group(2)) if match else None

st.set_page_config(page_title="Causeway Pipeline Validator", page_icon="🚦", layout="wide")
init_db()

st.title("🚦 Causeway Pipeline Validator")
st.caption("Human validation tool for lane segmentation and vehicle detection labels.")

tab1, tab2, tab3 = st.tabs(["📋 Label Review", "📊 Pipeline Health", "✏️ Annotate"])


def _render_lane_annotation(img_path: str, label_path: str):
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
        daytime_only = st.checkbox(
            "Daytime only (06:00–19:00)",
            value=(label_type_filter == "vehicle"),
            help="Filter to images captured between 06:00 and 19:00, where vehicle detection quality is highest.",
        )

    rows = get_label_logs(
        label_type=label_type_filter,
        validated=None if validation_filter == "All" else validation_filter,
    )

    if camera_filter != "All":
        rows = [r for r in rows if f"/{camera_filter}/" in r[0].replace("\\", "/")]

    if daytime_only:
        rows = [r for r in rows if (h := _parse_hour_from_path(r[0])) is not None and 6 <= h < 19]

    # Curation progress for vehicle labels
    if label_type_filter == "vehicle":
        total_shown = len(rows)
        reviewed = sum(1 for r in rows if r[4] in ("approved", "rejected"))
        st.sidebar.metric("Curation progress", f"{reviewed} / {total_shown}", help="Reviewed / shown (after filters)")

    st.write(f"Found **{len(rows)}** label(s) matching filters.")

    for img_path, label_path, label_type, shift, validated in rows:
        expander_title = f"{os.path.basename(img_path)}  —  {validated.upper()}"
        with st.expander(expander_title, expanded=(validated == "pending")):
            col1, col2 = st.columns(2)

            with col1:
                st.caption("📷 Raw Image")
                if os.path.exists(img_path):
                    st.image(img_path, width="stretch")
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
                    st.image(annotated, width="stretch")
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


with tab2:
    st.header("Pipeline Health")
    st.info(
        "**Why monitor pipeline health?** The scraper runs every 5 minutes. If your laptop sleeps or "
        "loses internet, cycles are missed and gaps appear in your dataset. Knowing where the gaps are "
        "helps you decide whether the dataset is clean enough to train on."
    )

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
        df["scraped_at"] = pd.to_datetime(df["scraped_at"], format="ISO8601")
        df["hour"] = df["scraped_at"].dt.floor("h")
        hourly = df.groupby(["hour", "status"]).size().reset_index(name="count")
        success_df = hourly[hourly["status"] == "success"]
        st.bar_chart(success_df.set_index("hour")["count"])
        error_count = len(df[df["status"] == "error"])
        if error_count:
            st.warning(f"⚠️ {error_count} scrape error(s) in the last 24 h.")
    else:
        st.info("No scrape activity recorded in the last 24 hours.")

    st.subheader("Recent Scrape Events")
    logs = get_recent_scrape_logs(limit=50)
    if logs:
        import pandas as pd
        df = pd.DataFrame(logs, columns=["Timestamp", "Camera", "File", "Status", "Error"])
        st.dataframe(df, width="stretch")
    else:
        st.info("No scrape events logged yet.")

    st.subheader("Dataset Split Summary")
    summary = get_dataset_split_summary()
    if summary:
        import pandas as pd
        df = pd.DataFrame(summary, columns=["Label Type", "Split", "Count"])
        st.dataframe(df, width="stretch")
    else:
        st.info("No dataset splits recorded yet. Run the build_dataset_split Dagster asset first.")

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

    # Reset navigator when camera or date changes
    _cam_date_key = f"ann_cam_date__{ann_camera}__{ann_date}"
    if st.session_state.get("_ann_cam_date_key") != _cam_date_key:
        st.session_state["_ann_cam_date_key"] = _cam_date_key
        st.session_state["ann_img_sel"] = 0

    # Clamp stale value (e.g. list shrank)
    if int(st.session_state.get("ann_img_sel", 0)) >= len(ann_images):
        st.session_state["ann_img_sel"] = 0

    def _sync_sel_from_widget():
        path = st.session_state.get("_ann_sel_widget")
        st.session_state["ann_img_sel"] = ann_images.index(path) if path in ann_images else 0

    nav_prev, nav_sel, nav_next = st.columns([1, 10, 1])
    with nav_prev:
        if st.button("◀", key="ann_prev"):
            st.session_state["ann_img_sel"] = max(0, int(st.session_state.get("ann_img_sel", 0)) - 1)
            st.rerun()
    with nav_sel:
        st.selectbox(
            "Image",
            ann_images,
            format_func=os.path.basename,
            index=int(st.session_state.get("ann_img_sel", 0)),
            key="_ann_sel_widget",
            on_change=_sync_sel_from_widget,
            label_visibility="collapsed",
        )
    with nav_next:
        if st.button("▶", key="ann_next"):
            st.session_state["ann_img_sel"] = min(len(ann_images) - 1, int(st.session_state.get("ann_img_sel", 0)) + 1)
            st.rerun()

    ann_img_path = ann_images[int(st.session_state.get("ann_img_sel", 0))]

    if not os.path.exists(ann_img_path):
        st.warning(f"Image file missing: `{ann_img_path}` — skipping.")
        if int(st.session_state.get("ann_img_sel", 0)) < len(ann_images) - 1:
            st.session_state["ann_img_sel"] = int(st.session_state.get("ann_img_sel", 0)) + 1
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
        st.image(preview, width="stretch")

    with col_canvas_ctrl:
        st.caption("📷 Reference image")
        st.image(ann_img_path, width="stretch")
        st.caption("🖊 Drag to draw a rectangle around a vehicle")

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
            new_rect = canvas_result.json_data["objects"][-1]
            if new_rect.get("type") == "rect":
                new_box = canvas_rect_to_box(new_rect, selected_cls_id, orig_w, orig_h)
                boxes.append(new_box)
                st.session_state[boxes_key] = boxes
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
                yolo_lines = boxes_to_yolo_lines(boxes)
                f.write("\n".join(yolo_lines) + ("\n" if yolo_lines else ""))
            ensure_label_log_entry(ann_img_path, label_path, "vehicle", "manual")
            update_label_validation(ann_img_path, "vehicle", "approved")
            # Clear session state for this image
            for k in [boxes_key, dims_key, canvas_v_key]:
                st.session_state.pop(k, None)
            # Advance to next image
            current_idx = st.session_state.get("ann_img_sel", 0)
            st.session_state["ann_img_sel"] = min(len(ann_images) - 1, current_idx + 1)
            st.success("Saved and marked approved.")
            st.rerun()
