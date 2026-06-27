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
    init_db, get_connection, update_label_validation,
    get_recent_scrape_logs, get_dataset_split_summary, get_label_logs
)

def _parse_hour_from_path(img_path: str):
    match = re.search(r"_(\d{8})_(\d{2})\d{4}\.jpg$", img_path, re.IGNORECASE)
    return int(match.group(2)) if match else None

st.set_page_config(page_title="Causeway Pipeline Validator", page_icon="🚦", layout="wide")
init_db()

st.title("🚦 Causeway Pipeline Validator")
st.caption("Human validation tool for lane segmentation and vehicle detection labels.")

tab1, tab2 = st.tabs(["📋 Label Review", "📊 Pipeline Health"])


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
                    st.image(img_path, use_container_width=True)
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
                    st.image(annotated, use_container_width=True)
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
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No scrape events logged yet.")

    st.subheader("Dataset Split Summary")
    summary = get_dataset_split_summary()
    if summary:
        import pandas as pd
        df = pd.DataFrame(summary, columns=["Label Type", "Split", "Count"])
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No dataset splits recorded yet. Run the build_dataset_split Dagster asset first.")
