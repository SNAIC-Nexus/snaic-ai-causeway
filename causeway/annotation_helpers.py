import os
import glob
import cv2
import numpy as np

DISPLAY_W = 640
DISPLAY_H = 480

CLASS_NAMES = {0: "Motorcycle", 1: "Car", 2: "Bus", 3: "Truck", 4: "Train"}

CLASS_COLOURS = {
    0: {"stroke": "#0000FF", "fill": "rgba(0,0,255,0.3)"},
    1: {"stroke": "#FFFF00", "fill": "rgba(255,255,0,0.3)"},
    2: {"stroke": "#00FF00", "fill": "rgba(0,255,0,0.3)"},
    3: {"stroke": "#00FFFF", "fill": "rgba(0,255,255,0.3)"},
    4: {"stroke": "#8B4513", "fill": "rgba(139,69,19,0.3)"},
}

# BGR equivalents for OpenCV drawing (matches CLASS_COLOURS above)
_CV_COLOURS = {
    0: (255, 0, 0),      # Blue (BGR)
    1: (0, 255, 255),    # Yellow (BGR)
    2: (0, 255, 0),      # Green (BGR)
    3: (255, 255, 0),    # Cyan (BGR)
    4: (19, 69, 139),    # Brown (BGR)
}


def yolo_to_boxes(label_path: str, orig_w: int, orig_h: int) -> list[dict]:
    """Read a YOLO vehicle label file and return normalised box dicts.

    Each box: {"class_id": int, "x1_n": float, "y1_n": float, "x2_n": float, "y2_n": float}
    Coordinates are normalised to [0, 1] relative to original image dimensions.
    Returns [] if the file does not exist or has no valid rows.
    """
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            try:
                cls_id = int(parts[0])
                cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            except ValueError:
                continue
            boxes.append({
                "class_id": cls_id,
                "x1_n": cx - bw / 2,
                "y1_n": cy - bh / 2,
                "x2_n": cx + bw / 2,
                "y2_n": cy + bh / 2,
            })
    return boxes


def boxes_to_yolo_lines(boxes: list[dict]) -> list[str]:
    """Convert normalised box dicts to YOLO format strings.

    Each output line: "class_id cx cy w h" with 6 decimal places.
    Returns [] for an empty box list (valid YOLO background image).
    """
    lines = []
    for box in boxes:
        cx = (box["x1_n"] + box["x2_n"]) / 2
        cy = (box["y1_n"] + box["y2_n"]) / 2
        bw = box["x2_n"] - box["x1_n"]
        bh = box["y2_n"] - box["y1_n"]
        lines.append(f"{box['class_id']} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return lines


def canvas_rect_to_box(rect: dict, class_id: int, orig_w: int, orig_h: int,
                        display_w: int = DISPLAY_W, display_h: int = DISPLAY_H) -> dict:
    """Convert a Fabric.js rect object from st_canvas output to a normalised box dict.

    rect must contain: left, top, width, height (in display px).
    scaleX and scaleY are applied if present (Fabric.js resize handles).
    Coordinates are clamped to [0, 1].
    """
    scale_x = rect.get("scaleX", 1.0)
    scale_y = rect.get("scaleY", 1.0)
    left = rect.get("left", 0.0)
    top = rect.get("top", 0.0)
    width = rect.get("width", 0.0) * scale_x
    height = rect.get("height", 0.0) * scale_y

    sx = display_w / orig_w
    sy = display_h / orig_h

    x1_n = left / (orig_w * sx)
    y1_n = top / (orig_h * sy)
    x2_n = (left + width) / (orig_w * sx)
    y2_n = (top + height) / (orig_h * sy)

    # Swap if inverted (right-to-left or bottom-to-top drawing)
    if x1_n > x2_n:
        x1_n, x2_n = x2_n, x1_n
    if y1_n > y2_n:
        y1_n, y2_n = y2_n, y1_n

    return {
        "class_id": class_id,
        "x1_n": max(0.0, min(1.0, x1_n)),
        "y1_n": max(0.0, min(1.0, y1_n)),
        "x2_n": max(0.0, min(1.0, x2_n)),
        "y2_n": max(0.0, min(1.0, y2_n)),
    }


def boxes_to_detection_args(
    boxes: list[dict], img_w: int, img_h: int
) -> tuple[list[list[int]], list[int]]:
    """Convert normalised box dicts to pixel [x, y, w, h] lists for detection().

    Returns (bboxes, labels) where bboxes[i] = [x, y, w, h] in pixels,
    labels[i] = class_id (int).
    """
    bboxes = []
    labels = []
    for box in boxes:
        x = int(round(box["x1_n"] * img_w))
        y = int(round(box["y1_n"] * img_h))
        w = int(round((box["x2_n"] - box["x1_n"]) * img_w))
        h = int(round((box["y2_n"] - box["y1_n"]) * img_h))
        bboxes.append([x, y, w, h])
        labels.append(box["class_id"])
    return bboxes, labels


def detection_result_to_boxes(
    result: list[dict], img_w: int, img_h: int
) -> list[dict]:
    """Convert detection() output to normalised box dicts.

    result[i] has keys: bbox ([x, y, w, h] pixels), label_id (int).
    Returns list of {"class_id": int, "x1_n": float, "y1_n": float,
                      "x2_n": float, "y2_n": float}.
    """
    boxes = []
    for item in result:
        x, y, w, h = item["bbox"]
        boxes.append({
            "class_id": item["label_id"],
            "x1_n": max(0.0, min(1.0, x / img_w)),
            "y1_n": max(0.0, min(1.0, y / img_h)),
            "x2_n": max(0.0, min(1.0, (x + w) / img_w)),
            "y2_n": max(0.0, min(1.0, (y + h) / img_h)),
        })
    return boxes


def render_annotated_image(img_path: str, boxes: list[dict], orig_w: int, orig_h: int,
                            display_w: int = DISPLAY_W, display_h: int = DISPLAY_H) -> np.ndarray:
    """Render boxes onto the image, resized to display dimensions.

    Returns an RGB uint8 array of shape (display_h, display_w, 3).
    """
    image = cv2.imread(img_path)
    if image is None:
        image = np.zeros((display_h, display_w, 3), dtype=np.uint8)
    else:
        image = cv2.resize(image, (display_w, display_h))

    sx = display_w / orig_w
    sy = display_h / orig_h

    for box in boxes:
        x1 = int(box["x1_n"] * orig_w * sx)
        y1 = int(box["y1_n"] * orig_h * sy)
        x2 = int(box["x2_n"] * orig_w * sx)
        y2 = int(box["y2_n"] * orig_h * sy)
        colour = _CV_COLOURS.get(box["class_id"], (255, 255, 255))
        cv2.rectangle(image, (x1, y1), (x2, y2), colour, 2)
        label = CLASS_NAMES.get(box["class_id"], str(box["class_id"]))
        cv2.putText(image, label, (x1, max(y1 - 5, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def list_annotation_dates(camera_id: str, base_dir: str = "traffic_images") -> list[str]:
    """Return YYYYMMDD date strings (descending) that have images for the given camera."""
    pattern = os.path.join(base_dir, "*", camera_id, "*.jpg")
    paths = glob.glob(pattern)
    dates = sorted(
        {os.path.basename(os.path.dirname(os.path.dirname(p))) for p in paths},
        reverse=True,
    )
    return dates


def list_images_for_annotation(camera_id: str, date_str: str,
                                base_dir: str = "traffic_images") -> list[str]:
    """Return sorted list of .jpg paths for the given camera and date (YYYYMMDD)."""
    pattern = os.path.join(base_dir, date_str, camera_id, "*.jpg")
    return sorted(glob.glob(pattern))
