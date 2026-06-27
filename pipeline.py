import os
import re
import datetime
import json
import cv2
import numpy as np

# Import native MLX YOLO packages
from yolo26mlx import YOLO
from yolo26mlx.converters import convert_model  # Built-in weights converter

# --- Model Preparation Configuration ---
weights_dir = "models"
os.makedirs(weights_dir, exist_ok=True)

pt_path = os.path.join(weights_dir, "yolov8n.pt")
mlx_path = os.path.join(weights_dir, "yolov8n.npz")

# 1. Initialize and convert the model to native MLX format safely
# This step runs once, creates 'yolov8n.npz', and bypasses PyTorch loops permanently!
if not os.path.exists(mlx_path):
    print("M-Series Mac Optimized: Converting PyTorch weights to native MLX format...")
    # Downloads the .pt model automatically if it doesn't exist locally and converts it
    convert_model(pt_path, output_path=mlx_path, verify=True)

# 2. Load the native MLX model variant for local high-speed inference
model = YOLO(mlx_path)

# Target class IDs for vehicles in the COCO dataset
VEHICLE_CLASSES = [2, 3, 5, 7] # 2: car, 3: motorcycle, 5: bus, 7: truck
font_scale = 0.6 # Added missing variable reference from original context

def extract_hour_from_filename(filename):
    base_name = os.path.basename(filename)
    match = re.search(r"_(\d{8})_(\d{2})\d{4}", base_name)
    if match:
        return int(match.group(2))
    return None

def get_vehicles_in_polygon(image, points):
    """Runs YOLO detection natively via MLX on Apple Silicon and tracks vehicle intersections."""
    if not points or len(points) < 3:
        return 0
        
    # Run prediction through the ultra-fast MLX framework backend
    results = model(image)
    
    count = 0
    polygon = np.array(points, np.int32)
    
    # yolo-mlx returns boxes inside a results object wrapper or a dedicated properties list
    # Safely access the underlying detected bounding boxes
    boxes = getattr(results, 'boxes', [])
    
    for box in boxes:
        # Resolve class ID natively
        cls_id = int(box.cls)
        
        if cls_id in VEHICLE_CLASSES:
            # Native MLX array bounding boxes coordinate handling [x1, y1, x2, y2]
            xyxy = box.xyxy
            
            # Calculate the bottom-center point of the vehicle (where tires touch pavement)
            bottom_center_x = int((xyxy[0] + xyxy[2]) / 2)
            bottom_center_y = int(xyxy[3])
            
            # Check if this coordinate sitting point falls inside your polygon bounds
            inside = cv2.pointPolygonTest(polygon, (bottom_center_x, bottom_center_y), False)
            if inside >= 0:
                count += 1
                
    return count

def draw_boundary_label(image, points, label, color, fixed_position, vehicle_count, alpha=0.35):
    if not points or len(points) < 3:
        return

    overlay = image.copy()
    pts = np.array(points, np.int32).reshape((-1, 1, 2))

    # Tint polygon lane regions
    cv2.fillPoly(overlay, [pts], color)
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    cv2.polylines(image, [pts], isClosed=True, color=color, thickness=2)

    # Append vehicle count to the display label text string
    display_text = f"{label} | Count: {vehicle_count}"

    # Unpack fixed label coordinates
    text_x, text_y = fixed_position
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    (text_w, text_h), baseline = cv2.getTextSize(display_text, font, font_scale, thickness)

    adjusted_x = text_x - (text_w // 2)
    adjusted_y = text_y + (text_h // 2)

    # Background capsule mask
    bg_overlay = image.copy()
    cv2.rectangle(
        bg_overlay,
        (adjusted_x - 10, adjusted_y - text_h - 6),
        (adjusted_x + text_w + 10, adjusted_y + baseline + 4),
        (0, 0, 0),
        -1,
    )
    cv2.addWeighted(bg_overlay, 0.65, image, 0.35, 0, image)

    # Render clean font text
    cv2.putText(image, display_text, (adjusted_x, adjusted_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

def process_single_image(image_path, config_id, short_cam_id, config_path="camera_config.json"):
    if not os.path.exists(image_path):
        return None, "Unknown", {}

    with open(config_path, "r") as f:
        master_config = json.load(f)

    if config_id not in master_config:
        return None, "Unknown", {}

    camera_profiles = master_config[config_id]
    image = cv2.imread(image_path)
    if image is None:
        return None, "Unknown", {}

    if "static" in camera_profiles:
        current_config = camera_profiles["static"]
        shift_mode = "Static"
    else:
        target_hour = extract_hour_from_filename(image_path)
        if target_hour is None:
            target_hour = datetime.datetime.now().hour

        shift_mode = "morning" if 6 <= target_hour < 12 else "afternoon" if 12 <= target_hour < 19 else "night"
        current_config = camera_profiles[shift_mode]

    # Analytics metrics container dictionary to return to Streamlit dashboard
    counts_summary = {}

    # Process graphics and compute detections loops
    for label, poly, color, pos in zip(
        current_config["labels"],
        current_config["polygons"],
        current_config["colors"],
        current_config["label_positions"],
    ):
        # 1. Compute how many vehicles are in this exact lane polygon
        v_count = get_vehicles_in_polygon(image, poly)
        counts_summary[label] = v_count
        
        # 2. Render graphics containing the live count
        draw_boundary_label(image, poly, label, tuple(color), pos, v_count)

    os.makedirs("processed_output", exist_ok=True)
    output_filename = f"labeled_{config_id}.jpg"
    output_path = os.path.join("processed_output", output_filename)
    cv2.imwrite(output_path, image)

    return output_path, shift_mode, counts_summary