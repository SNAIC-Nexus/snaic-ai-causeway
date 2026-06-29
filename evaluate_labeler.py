"""Side-by-side comparison of YOLOv8x vs RT-DETR-X on sampled traffic images.

Usage:
    python evaluate_labeler.py [--n 5] [--conf 0.25] [--cameras 2701,2702,2704] [--out eval_output]
"""
import argparse
import glob
import os
import random

import cv2

from causeway.config import IMAGE_BASE_DIR
from causeway.db import get_label_logs

# COCO class index → project class name
COCO_CLASS_NAMES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
# Colour per class (BGR)
CLASS_COLORS = {"car": (0, 200, 0), "motorcycle": (0, 0, 255), "bus": (255, 140, 0), "truck": (0, 180, 255)}

TARGET_CAMERAS = ["2701", "2702", "2704"]


def get_sample_images(
    cameras: list[str],
    n: int,
    images_base: str = IMAGE_BASE_DIR,
) -> dict[str, list[str]]:
    """Return up to n image paths per camera.

    Prefers images with approved vehicle labels from the DB.
    Falls back to any .jpg under traffic_images/<date>/<camera_id>/.
    """
    # Collect approved image paths per camera from DB
    approved_rows = get_label_logs(label_type="vehicle", validated="approved")
    approved_by_camera: dict[str, list[str]] = {}
    for row in approved_rows:
        img_path = row[0]
        # Extract camera_id from path: .../YYYYMMDD/<camera_id>/filename.jpg
        parts = img_path.replace("\\", "/").split("/")
        if len(parts) >= 2:
            cam = parts[-2]
            if cam in cameras:
                approved_by_camera.setdefault(cam, []).append(img_path)

    result: dict[str, list[str]] = {}
    for cam in cameras:
        candidates = approved_by_camera.get(cam, [])
        if len(candidates) < n:
            # Fall back: glob all .jpg for this camera
            pattern = os.path.join(images_base, "**", cam, "*.jpg")
            all_imgs = glob.glob(pattern, recursive=True)
            # Merge, deduplicate, prefer approved first
            seen = set(candidates)
            for p in all_imgs:
                if p not in seen:
                    candidates.append(p)
                    seen.add(p)
        random.seed(42)
        result[cam] = random.sample(candidates, min(n, len(candidates)))
    return result


def _draw_boxes(img, results, label: str):
    """Draw COCO vehicle boxes on img in-place. Returns per-class counts."""
    counts = {name: 0 for name in COCO_CLASS_NAMES.values()}
    for box in results.boxes:
        coco_cls = int(box.cls[0].item())
        if coco_cls not in COCO_CLASS_NAMES:
            continue
        name = COCO_CLASS_NAMES[coco_cls]
        counts[name] += 1
        xyxy = box.xyxy[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = xyxy
        color = CLASS_COLORS[name]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, name, (x1, max(y1 - 6, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    # Model label in top-left
    cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    return counts


def run_comparison(
    cameras: list[str],
    n: int,
    conf: float,
    out_dir: str,
) -> None:
    from ultralytics import RTDETR, YOLO

    os.makedirs(out_dir, exist_ok=True)
    samples = get_sample_images(cameras, n)

    model_yolo = YOLO("yolov8x.pt")
    model_rtdetr = RTDETR("rtdetr-x.pt")

    # Summary table: camera -> model -> class -> count
    summary: dict[str, dict[str, dict[str, int]]] = {}

    for cam, paths in samples.items():
        summary[cam] = {"YOLOv8x": {}, "RTDETR-X": {}}
        for img_path in paths:
            img = cv2.imread(img_path)
            if img is None:
                print(f"  WARNING: could not read {img_path}, skipping")
                continue

            params = dict(source=img_path, imgsz=1280, conf=conf, device="mps", verbose=False)
            res_yolo = model_yolo.predict(**params)[0]
            res_rtdetr = model_rtdetr.predict(**params)[0]

            left = img.copy()
            right = img.copy()
            counts_yolo = _draw_boxes(left, res_yolo, "YOLOv8x")
            counts_rtdetr = _draw_boxes(right, res_rtdetr, "RTDETR-X")

            grid = cv2.hconcat([left, right])
            fname = os.path.splitext(os.path.basename(img_path))[0]
            out_path = os.path.join(out_dir, f"{cam}_{fname}_compare.jpg")
            cv2.imwrite(out_path, grid)

            for cls, cnt in counts_yolo.items():
                summary[cam]["YOLOv8x"][cls] = summary[cam]["YOLOv8x"].get(cls, 0) + cnt
            for cls, cnt in counts_rtdetr.items():
                summary[cam]["RTDETR-X"][cls] = summary[cam]["RTDETR-X"].get(cls, 0) + cnt

    # Print summary table
    header = f"{'Camera':<8}{'Model':<12}{'motorcycle':>12}{'car':>6}{'bus':>6}{'truck':>8}{'total':>8}"
    print("\n" + header)
    print("-" * len(header))
    for cam in cameras:
        for model_name in ("YOLOv8x", "RTDETR-X"):
            c = summary.get(cam, {}).get(model_name, {})
            mc = c.get("motorcycle", 0)
            car = c.get("car", 0)
            bus = c.get("bus", 0)
            truck = c.get("truck", 0)
            total = mc + car + bus + truck
            print(f"{cam:<8}{model_name:<12}{mc:>12}{car:>6}{bus:>6}{truck:>8}{total:>8}")
        print()

    print(f"Comparison images saved to: {out_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Compare YOLOv8x vs RT-DETR-X on traffic images.")
    parser.add_argument("--n", type=int, default=5, help="Images per camera (default: 5)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25)")
    parser.add_argument("--cameras", default="2701,2702,2704", help="Comma-separated camera IDs")
    parser.add_argument("--out", default="eval_output", help="Output directory (default: eval_output)")
    args = parser.parse_args()
    cameras = [c.strip() for c in args.cameras.split(",")]
    run_comparison(cameras=cameras, n=args.n, conf=args.conf, out_dir=args.out)


if __name__ == "__main__":
    main()
