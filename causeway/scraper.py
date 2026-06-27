# causeway/scraper.py
import os
import time
import logging
import requests
from datetime import datetime
from causeway.config import LTA_API_URL, API_KEY, TARGET_CAMERAS, IMAGE_BASE_DIR
from causeway.db import init_db, log_scrape, get_last_scrape_timestamp

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SCRAPE_INTERVAL = 300
MAX_BACKOFF = 300


def _log_startup_gap() -> None:
    last = get_last_scrape_timestamp()
    if last:
        last_dt = datetime.fromisoformat(last)
        gap_minutes = (datetime.now() - last_dt).total_seconds() / 60
        logger.info(f"Resuming. Gap since last scrape: {gap_minutes:.1f} minutes (last: {last})")
    else:
        logger.info("No prior scrape history. Starting fresh.")


def _fetch_with_backoff(max_retries: int = 8) -> dict | None:
    delay = 1
    attempts = 0
    while True:
        try:
            headers = {"x-api-key": API_KEY}
            params = {"date_time": datetime.now().strftime("%Y-%m-%dT%H:%M:%S")}
            response = requests.get(LTA_API_URL, params=params, headers=headers, timeout=10)

            if response.status_code in (401, 403):
                logger.error(f"Auth failure HTTP {response.status_code}. Check API_KEY in causeway/config.py.")
                return None

            if response.status_code != 200:
                raise requests.RequestException(f"HTTP {response.status_code}")

            return response.json()

        except (requests.Timeout, requests.ConnectionError, requests.RequestException) as exc:
            attempts += 1
            if attempts >= max_retries:
                logger.warning(f"Max retries ({max_retries}) reached. Skipping cycle.")
                return None
            logger.warning(f"Request failed: {exc}. Retrying in {delay}s (attempt {attempts}/{max_retries})...")
            time.sleep(delay)
            delay = min(delay * 2, MAX_BACKOFF)


def run_once() -> None:
    payload = _fetch_with_backoff()
    if payload is None:
        return

    items = payload.get("items", [])
    if not items or "cameras" not in items[0]:
        logger.warning("No camera data in payload.")
        return

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    date_str = datetime.now().strftime("%Y%m%d")

    for cam in items[0]["cameras"]:
        cam_id = str(cam.get("camera_id", ""))
        if cam_id not in TARGET_CAMERAS:
            continue

        image_url = cam.get("image")
        if not image_url:
            continue

        scraped_at = datetime.now().isoformat()
        try:
            img_resp = requests.get(image_url, timeout=15)
            if img_resp.status_code != 200:
                raise requests.RequestException(f"CDN HTTP {img_resp.status_code}")

            cam_dir = os.path.join(IMAGE_BASE_DIR, date_str, cam_id)
            os.makedirs(cam_dir, exist_ok=True)

            filename = f"{TARGET_CAMERAS[cam_id]}_{cam_id}_{timestamp_str}.jpg"
            filepath = os.path.join(cam_dir, filename)

            with open(filepath, "wb") as f:
                f.write(img_resp.content)

            log_scrape(scraped_at, cam_id, filepath, "success")
            logger.info(f"Saved cam {cam_id}: {filepath}")

        except Exception as exc:
            log_scrape(scraped_at, cam_id, None, "error", str(exc))
            logger.error(f"Failed cam {cam_id}: {exc}")


def run_loop() -> None:
    init_db()
    _log_startup_gap()
    logger.info("Scraper started. Cycle interval: 5 minutes.")
    while True:
        run_once()
        logger.info(f"Cycle complete. Sleeping {SCRAPE_INTERVAL}s...")
        time.sleep(SCRAPE_INTERVAL)


if __name__ == "__main__":
    run_loop()
