import os
import sys
import time
import requests
from datetime import datetime

# =========================================================
# CONFIGURATION
# =========================================================
LTA_API_URL = "https://api.data.gov.sg/v1/transport/traffic-images"
API_KEY = "YOUR_DATA_GOV_SG_API_KEY"  # ⚠️ Paste your data.gov.sg API key here!

TARGET_CAMERAS = {
    "2701": "Woodlands_Causeway_Towards_Johor",
    "2702": "Woodlands_Checkpoint_Towards_BKE",
    "2704": "Woodlands_Flyover_Towards_Checkpoint"
}

# =========================================================
# SCRAPER CORE ENGINE
# =========================================================
def download_traffic_images(base_directory):
    current_time_str = datetime.now().strftime('%H:%M:%S')
    print(f"[{current_time_str}] 🛰️ Pinging Data.gov.sg Traffic API...")
    
    # 1. Handle directory creation safety checks
    try:
        os.makedirs(base_directory, exist_ok=True)
    except Exception as e:
        print(f"❌ DIRECTORY ERROR: Could not create base folder. {e}")
        return

    # 2. Format current time layout required by data.gov.sg (YYYY-MM-DDTHH:mm:ss)
    current_time_iso = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    
    params = {"date_time": current_time_iso}
    headers = {"x-api-key": API_KEY}

    try:
        response = requests.get(LTA_API_URL, params=params, headers=headers, timeout=10)
        
        if response.status_code == 403 or response.status_code == 401:
            print(f"❌ AUTHENTICATION ERROR [HTTP {response.status_code}]: Your data.gov.sg API key is invalid.")
            return
        elif response.status_code != 200:
            print(f"⚠️ SERVER ERROR [HTTP {response.status_code}]: Endpoint failed. Retrying next cycle...")
            return

        try:
            payload = response.json()
        except ValueError:
            print("⚠️ DATA ERROR: Server did not return a valid JSON payload.")
            return

        # 3. Parse Data.gov.sg unique JSON nesting rules
        # Layout: payload['items'][0]['cameras']
        items = payload.get("items", [])
        if not items or "cameras" not in items[0]:
            print("⚠️ PAYLOAD WARNING: Request succeeded, but no camera elements were returned.")
            return
            
        camera_list = items[0]["cameras"]
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        download_count = 0

        # 4. Extract and save our specific camera targets
        for cam in camera_list:
            cam_id = str(cam.get("camera_id"))
            
            if cam_id in TARGET_CAMERAS:
                prefix_name = TARGET_CAMERAS[cam_id]
                image_url = cam.get("image")  # Data.gov.sg uses 'image', not 'ImageLink'
                
                if not image_url:
                    continue

                try:
                    img_response = requests.get(image_url, timeout=15)
                    if img_response.status_code == 200:
                        # 🔹 FIX: Create a dedicated subfolder for this specific camera ID
                        cam_specific_dir = os.path.join(base_directory, cam_id)
                        os.makedirs(cam_specific_dir, exist_ok=True)
                        
                        filename = f"{prefix_name}_{cam_id}_{timestamp_str}.jpg"
                        filepath = os.path.join(cam_specific_dir, filename) # 🔹 Saved inside cam_id/
                        
                        with open(filepath, "wb") as f:
                            f.write(img_response.content)
                        print(f"   💾 Saved Cam {cam_id}: {cam_id}/{filename}")
                        download_count += 1
                except Exception as img_err:
                    print(f"   ⚠️ Could not fetch CDN image for Cam {cam_id}: {img_err}")
                    continue

        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Cycle complete. Successfully saved {download_count} images.")

    except requests.exceptions.Timeout:
        print(f"[{current_time_str}] 📡 Connection Timed Out. Data.gov.sg took too long to respond.")
    except requests.exceptions.ConnectionError:
        print(f"[{current_time_str}] 🌐 Network Offline. Your Mac cannot reach the API right now.")
    except Exception as e:
        print(f"[{current_time_str}] 💥 Unexpected Loop Error: {e}")

# =========================================================
# MAIN EXECUTION ENGINE
# =========================================================
if __name__ == "__main__":
    print("\n==================================================")
    print("🚀 Data.gov.sg Tri-Camera Scraper Active (Partitioned)")
    print("==================================================")

    # Fixed the sys.argv bug from earlier
    if len(sys.argv) > 1:
        output_base_dir = sys.argv[1]
    else:
        output_base_dir = "./traffic_images"
        
    print(f"📁 Root Output Directory: {os.path.abspath(output_base_dir)}")
    print(f"📸 Tracking Camera IDs: {list(TARGET_CAMERAS.keys())}")
    print("--------------------------------------------------")

    while True:
        # Dynamically generate partition path based on current date: root/YYYYMMDD
        current_date_str = datetime.now().strftime("%Y%m%d")
        daily_partition_dir = os.path.join(output_base_dir, current_date_str)
        
        download_traffic_images(daily_partition_dir)
        
        print("\n⏳ Sleeping 5 minutes before checking for updates...")
        time.sleep(300)