# causeway/dataset.py
import os
import glob
import yaml
from causeway.config import IMAGE_BASE_DIR, LANE_LABELS_DIR, VEHICLE_LABELS_DIR
from causeway.db import init_db, log_split


def _write_yaml(path: str, train_imgs: list, val_imgs: list, names: dict) -> None:
    data = {
        "path": os.path.abspath("."),
        "train": [os.path.abspath(p) for p in train_imgs],
        "val": [os.path.abspath(p) for p in val_imgs],
        "nc": len(names),
        "names": names,
    }
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)
    print(f"Written: {path}")


def build_dataset_split(base_images_dir: str = IMAGE_BASE_DIR) -> dict:
    init_db()
    date_dirs = sorted(glob.glob(os.path.join(base_images_dir, "????????")))

    if len(date_dirs) < 2:
        print(f"WARNING: Only {len(date_dirs)} date partition(s) found. Need at least 2 to split meaningfully.")
        return {"lane_train": 0, "lane_val": 0, "vehicle_train": 0, "vehicle_val": 0}

    train_dates = date_dirs[:-1]
    val_dates = [date_dirs[-1]]

    print(f"Train: {[os.path.basename(d) for d in train_dates]}")
    print(f"Val:   {[os.path.basename(d) for d in val_dates]}")

    lane_train, lane_val, vehicle_train, vehicle_val = [], [], [], []

    for date_dir in train_dates:
        for img in glob.glob(os.path.join(date_dir, "**", "*.jpg"), recursive=True):
            lane_train.append(img)
            vehicle_train.append(img)
            log_split(img, "lane", "train")
            log_split(img, "vehicle", "train")

    for date_dir in val_dates:
        for img in glob.glob(os.path.join(date_dir, "**", "*.jpg"), recursive=True):
            lane_val.append(img)
            vehicle_val.append(img)
            log_split(img, "lane", "val")
            log_split(img, "vehicle", "val")

    if not lane_val:
        print("WARNING: Val set is empty.")

    _write_yaml(
        "dataset_lane.yaml", lane_train, lane_val,
        {0: "Towards Woodlands CIQ", 1: "Towards Johor"},
    )
    _write_yaml(
        "dataset_vehicle.yaml", vehicle_train, vehicle_val,
        {0: "motorcycle", 1: "car", 2: "bus", 3: "truck"},
    )

    print(f"Lane:    train={len(lane_train)}, val={len(lane_val)}")
    print(f"Vehicle: train={len(vehicle_train)}, val={len(vehicle_val)}")

    return {
        "lane_train": len(lane_train),
        "lane_val": len(lane_val),
        "vehicle_train": len(vehicle_train),
        "vehicle_val": len(vehicle_val),
    }


def export_curated_dataset(output_dir: str) -> dict:
    """Copy approved vehicle labels + images into a standard ultralytics layout.

    Directory layout written to output_dir:
        images/train/<filename>.jpg
        images/val/<filename>.jpg
        labels/train/<filename>.txt
        labels/val/<filename>.txt

    Returns {"train": int, "val": int, "yaml_path": str}.
    """
    import shutil
    import re
    from causeway.db import get_label_logs

    rows = get_label_logs(label_type="vehicle", validated="approved")
    if not rows:
        print("No approved vehicle labels found. Review labels in the Streamlit app first.")
        return {"train": 0, "val": 0, "yaml_path": ""}

    # Group image paths by date (YYYYMMDD extracted from path)
    def _day(img_path):
        m = re.search(r"[/\\](\d{8})[/\\]", img_path)
        return m.group(1) if m else "00000000"

    by_day = {}
    for img_path, label_path, *_ in rows:
        day = _day(img_path)
        by_day.setdefault(day, []).append((img_path, label_path))

    sorted_days = sorted(by_day.keys())
    if len(sorted_days) < 2:
        print(f"WARNING: Only {len(sorted_days)} day(s) of approved labels found. Need at least 2 to split meaningfully.")
        train_days, val_days = sorted_days, []
    else:
        train_days = sorted_days[:-1]
        val_days = [sorted_days[-1]]

    train_pairs = [pair for d in train_days for pair in by_day[d]]
    val_pairs   = [pair for d in val_days   for pair in by_day[d]]

    def _copy_pairs(pairs, split):
        img_out = os.path.join(output_dir, "images", split)
        lbl_out = os.path.join(output_dir, "labels", split)
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(lbl_out, exist_ok=True)
        for img_path, label_path in pairs:
            if os.path.exists(img_path) and os.path.exists(label_path):
                shutil.copy2(img_path, os.path.join(img_out, os.path.basename(img_path)))
                shutil.copy2(label_path, os.path.join(lbl_out, os.path.basename(label_path)))

    _copy_pairs(train_pairs, "train")
    _copy_pairs(val_pairs, "val")

    yaml_path = "dataset_vehicle_curated.yaml"
    yaml_abs_path = os.path.abspath(yaml_path)
    data = {
        "path": os.path.abspath(output_dir),
        "train": "images/train",
        "val": "images/val",
        "nc": 4,
        "names": {0: "motorcycle", 1: "car", 2: "bus", 3: "truck"},
    }
    with open(yaml_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)

    print(f"Curated export: train={len(train_pairs)}, val={len(val_pairs)} → {output_dir}")
    print(f"Written: {yaml_path}")
    return {"train": len(train_pairs), "val": len(val_pairs), "yaml_path": yaml_abs_path}
