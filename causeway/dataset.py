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
