# causeway/labeler.py
import os
import re
import json
import glob
from datetime import datetime
from causeway.config import (
    IMAGE_BASE_DIR, LANE_LABELS_DIR, VEHICLE_LABELS_DIR, CAMERA_CONFIG_PATH
)
from causeway.db import init_db, log_label

# Maps short camera_id -> camera_config.json top-level key
CONFIG_KEY_MAP = {
    "2701": "Woodlands_Checkpoint_Towards_Johor_2701",
    "2702": "Woodlands_Checkpoint_Towards_BKE_2702",
    "2704": "Woodlands_Flyover_Towards_Checkpoint_2704",
}

# Class IDs for lane segmentation labels
# Class 0 = Towards Woodlands/Singapore CIQ
# Class 1 = Towards Johor/Malaysia
# (2704 gets extra class 2 for PIE branch — static camera)
SHIFT_CLASS_MAP = {
    "2701": {
        "morning":   {"To Woodlands CIQ (3 Lanes)": 0, "To Johor (1 Lane)": 1},
        "afternoon": {"To Woodlands CIQ (2 Lanes)": 0, "To Johor (2 Lanes)": 1},
        "night":     {"To Woodlands CIQ (2 Lanes)": 0, "To Johor (2 Lanes)": 1},
    },
    "2702": {
        "static": {"Towards BKE": 0, "Towards Checkpoint Arrival": 1},
    },
    "2704": {
        "static": {"To Woodlands Checkpoint": 0, "To Woodlands Ave 3": 1, "To PIE": 2},
    },
}

IMG_W = 1920
IMG_H = 1080

# COCO class → project class for vehicle detection
COCO_TO_VEHICLE_CLASS = {2: 1, 3: 0, 5: 2, 7: 3}  # car→1, motorcycle→0, bus→2, truck→3

# Lazy-import shim: assigned at module level so tests can monkeypatch
YOLO = None
RTDETR = None


def _load_models():
    global YOLO, RTDETR
    if YOLO is None:
        from ultralytics import YOLO as _YOLO, RTDETR as _RTDETR
        YOLO = _YOLO
        RTDETR = _RTDETR


def _parse_hour(filename: str) -> int:
    base = os.path.basename(filename)
    match = re.search(r"_(\d{8})_(\d{2})\d{4}\.jpg$", base, re.IGNORECASE)
    if match:
        return int(match.group(2))
    return datetime.now().hour


def _get_shift(hour: int) -> str:
    if 6 <= hour < 12:
        return "morning"
    if 12 <= hour < 19:
        return "afternoon"
    return "night"


def _normalize_polygon(points: list, w: int = IMG_W, h: int = IMG_H) -> list:
    flat = [coord for pt in points for coord in pt]
    return [flat[i] / w if i % 2 == 0 else flat[i] / h for i in range(len(flat))]


def generate_lane_labels(base_images_dir: str = IMAGE_BASE_DIR) -> int:
    init_db()
    with open(CAMERA_CONFIG_PATH) as f:
        master_config = json.load(f)

    image_paths = glob.glob(os.path.join(base_images_dir, "**", "*.jpg"), recursive=True)
    processed = 0

    for img_path in image_paths:
        norm_path = img_path.replace("\\", "/")
        parts = norm_path.split("/")
        if len(parts) < 3:
            continue
        camera_id = parts[-2]
        if camera_id not in CONFIG_KEY_MAP:
            continue

        config_key = CONFIG_KEY_MAP[camera_id]
        camera_profiles = master_config.get(config_key)
        if not camera_profiles:
            continue

        rel_path = os.path.relpath(img_path, base_images_dir)
        rel_dir = os.path.dirname(rel_path)
        base_name = os.path.splitext(os.path.basename(img_path))[0]

        label_dir = os.path.join(LANE_LABELS_DIR, rel_dir)
        label_path = os.path.join(label_dir, f"{base_name}.txt")

        if os.path.exists(label_path):
            continue

        if "static" in camera_profiles:
            profile = camera_profiles["static"]
            shift = "static"
        else:
            shift = _get_shift(_parse_hour(img_path))
            profile = camera_profiles[shift]

        class_map = SHIFT_CLASS_MAP.get(camera_id, {}).get(shift, {})
        os.makedirs(label_dir, exist_ok=True)

        with open(label_path, "w") as f:
            for label, polygon in zip(profile["labels"], profile["polygons"]):
                class_id = class_map.get(label, 0)
                norm = _normalize_polygon(polygon)
                coords_str = " ".join(f"{v:.6f}" for v in norm)
                f.write(f"{class_id} {coords_str}\n")

        log_label(img_path, label_path, "lane", shift)
        processed += 1

    print(f"Lane labeling complete: {processed} new labels generated.")
    return processed


def generate_vehicle_labels(
    base_images_dir: str = IMAGE_BASE_DIR,
    model_path: str = "rtdetr-x.pt",
) -> int:
    init_db()

    image_paths = glob.glob(os.path.join(base_images_dir, "**", "*.jpg"), recursive=True)
    processed = 0

    if not image_paths:
        print("Vehicle labeling complete: 0 new labels generated.")
        return 0

    _load_models()
    model = RTDETR(model_path) if model_path.startswith("rtdetr") else YOLO(model_path)

    for img_path in image_paths:
        rel_path = os.path.relpath(img_path, base_images_dir)
        rel_dir = os.path.dirname(rel_path)
        base_name = os.path.splitext(os.path.basename(img_path))[0]

        label_dir = os.path.join(VEHICLE_LABELS_DIR, rel_dir)
        label_path = os.path.join(label_dir, f"{base_name}.txt")

        if os.path.exists(label_path):
            continue

        results = model.predict(source=img_path, imgsz=1280, conf=0.25, device="mps", verbose=False)[0]
        os.makedirs(label_dir, exist_ok=True)

        with open(label_path, "w") as f:
            for box in results.boxes:
                coco_cls = int(box.cls[0].item())
                if coco_cls in COCO_TO_VEHICLE_CLASS:
                    target_cls = COCO_TO_VEHICLE_CLASS[coco_cls]
                    xywhn = box.xywhn[0].cpu().numpy()
                    cx, cy, w, h = xywhn
                    f.write(f"{target_cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

        log_label(img_path, label_path, "vehicle", "N/A")
        processed += 1

    print(f"Vehicle labeling complete: {processed} new labels generated.")
    return processed
