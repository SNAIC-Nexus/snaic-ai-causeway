# causeway/config.py
LTA_API_URL = "https://api.data.gov.sg/v1/transport/traffic-images"
API_KEY = "YOUR_DATA_GOV_SG_API_KEY"

TARGET_CAMERAS = {
    "2701": "Woodlands_Causeway_Towards_Johor",
    "2702": "Woodlands_Checkpoint_Towards_BKE",
    "2704": "Woodlands_Flyover_Towards_Checkpoint",
}

IMAGE_BASE_DIR = "traffic_images"
LANE_LABELS_DIR = "traffic_lane_labels"
VEHICLE_LABELS_DIR = "traffic_vehicle_labels"
DB_PATH = "causeway.db"
CAMERA_CONFIG_PATH = "camera_config.json"
