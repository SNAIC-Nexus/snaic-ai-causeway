# causeway/migration.py
import os
import re
import shutil
from datetime import datetime
from causeway.config import IMAGE_BASE_DIR, TARGET_CAMERAS
from causeway.db import init_db, get_connection


def _parse_image_metadata(filename: str):
    """
    Parse camera_id, date_str, and datetime from an LTA filename.
    Expected pattern: ...PrefixName_CAMERAID_YYYYMMDD_HHMMSS.jpg
    Returns (camera_id, date_str, datetime) or (None, None, None) if unrecognised.
    """
    base = os.path.basename(filename)
    match = re.search(r"_(\d{4})_(\d{8})_(\d{6})\.jpg$", base, re.IGNORECASE)
    if not match:
        return None, None, None

    camera_id = match.group(1)
    date_str = match.group(2)
    time_str = match.group(3)

    if camera_id not in TARGET_CAMERAS:
        return None, None, None

    try:
        dt = datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")
    except ValueError:
        return None, None, None

    return camera_id, date_str, dt


def _backfill_scrape_log(file_path: str, camera_id: str, dt: datetime) -> None:
    with get_connection() as conn:
        exists = conn.execute(
            "SELECT 1 FROM scrape_log WHERE file_path=?", (file_path,)
        ).fetchone()
        if not exists:
            conn.execute(
                "INSERT INTO scrape_log (scraped_at, camera_id, file_path, status) VALUES (?,?,?,?)",
                (dt.isoformat(), camera_id, file_path, "migrated"),
            )
            conn.commit()


def migrate(base_dir: str = IMAGE_BASE_DIR) -> dict:
    """
    Walk all .jpg files under base_dir, move any that are not in the canonical
    YYYYMMDD/CAMERA_ID/ structure, and backfill the SQLite scrape_log.
    Safe to run multiple times (idempotent).
    """
    init_db()
    moved = 0
    already_correct = 0
    skipped = 0

    for root, dirs, files in os.walk(base_dir):
        # Avoid descending into directories we just created mid-walk
        dirs.sort()
        for fname in files:
            if not fname.lower().endswith(".jpg"):
                continue

            full_path = os.path.join(root, fname)
            camera_id, date_str, dt = _parse_image_metadata(fname)

            if camera_id is None:
                print(f"  SKIP (unrecognised filename): {fname}")
                skipped += 1
                continue

            target_dir = os.path.join(base_dir, date_str, camera_id)
            target_path = os.path.join(target_dir, fname)

            if os.path.abspath(full_path) == os.path.abspath(target_path):
                already_correct += 1
                _backfill_scrape_log(full_path, camera_id, dt)
                continue

            os.makedirs(target_dir, exist_ok=True)
            shutil.move(full_path, target_path)
            _backfill_scrape_log(target_path, camera_id, dt)
            print(f"  MOVED: {fname} -> {date_str}/{camera_id}/")
            moved += 1

    print(f"\nMigration complete: {moved} moved, {already_correct} already correct, {skipped} skipped.")
    return {"moved": moved, "already_correct": already_correct, "skipped": skipped}


if __name__ == "__main__":
    migrate()
