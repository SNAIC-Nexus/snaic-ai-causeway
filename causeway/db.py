# causeway/db.py
import sqlite3
from datetime import datetime
from typing import Optional
from causeway.config import DB_PATH


def get_connection() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH)


def init_db() -> None:
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS scrape_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scraped_at TEXT NOT NULL,
                camera_id TEXT NOT NULL,
                file_path TEXT,
                status TEXT NOT NULL,
                error_msg TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS label_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_path TEXT NOT NULL,
                label_path TEXT NOT NULL,
                label_type TEXT NOT NULL,
                shift TEXT,
                validated TEXT DEFAULT 'pending'
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS dataset_splits (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_path TEXT NOT NULL,
                label_type TEXT NOT NULL,
                split TEXT NOT NULL,
                assigned_at TEXT NOT NULL
            )
        """)
        conn.commit()


def log_scrape(scraped_at: str, camera_id: str, file_path: Optional[str], status: str, error_msg=None) -> None:
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO scrape_log (scraped_at, camera_id, file_path, status, error_msg) VALUES (?,?,?,?,?)",
            (scraped_at, camera_id, file_path, status, error_msg),
        )
        conn.commit()


def get_last_scrape_timestamp(camera_id=None) -> Optional[str]:
    with get_connection() as conn:
        if camera_id:
            row = conn.execute(
                "SELECT scraped_at FROM scrape_log WHERE status IN ('success','migrated') AND camera_id=? ORDER BY scraped_at DESC LIMIT 1",
                (camera_id,),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT scraped_at FROM scrape_log WHERE status IN ('success','migrated') ORDER BY scraped_at DESC LIMIT 1"
            ).fetchone()
        return row[0] if row else None


def log_label(image_path: str, label_path: str, label_type: str, shift: str) -> None:
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO label_log (image_path, label_path, label_type, shift) VALUES (?,?,?,?)",
            (image_path, label_path, label_type, shift),
        )
        conn.commit()


def ensure_label_log_entry(image_path: str, label_path: str, label_type: str, shift: str) -> None:
    """Insert a label_log row if none exists for (image_path, label_type)."""
    with get_connection() as conn:
        exists = conn.execute(
            "SELECT 1 FROM label_log WHERE image_path=? AND label_type=? LIMIT 1",
            (image_path, label_type),
        ).fetchone()
        if not exists:
            conn.execute(
                "INSERT INTO label_log (image_path, label_path, label_type, shift) VALUES (?,?,?,?)",
                (image_path, label_path, label_type, shift),
            )
            conn.commit()


def update_label_validation(image_path: str, label_type: str, validated: str) -> None:
    with get_connection() as conn:
        conn.execute(
            "UPDATE label_log SET validated=? WHERE image_path=? AND label_type=?",
            (validated, image_path, label_type),
        )
        conn.commit()


def log_split(image_path: str, label_type: str, split: str) -> None:
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO dataset_splits (image_path, label_type, split, assigned_at) VALUES (?,?,?,?)",
            (image_path, label_type, split, datetime.now().isoformat()),
        )
        conn.commit()


def get_recent_scrape_logs(limit: int = 50) -> list:
    with get_connection() as conn:
        return conn.execute(
            "SELECT scraped_at, camera_id, file_path, status, error_msg FROM scrape_log ORDER BY scraped_at DESC LIMIT ?",
            (limit,),
        ).fetchall()


def get_label_logs(label_type=None, validated=None) -> list:
    with get_connection() as conn:
        query = "SELECT image_path, label_path, label_type, shift, validated FROM label_log WHERE 1=1"
        params = []
        if label_type:
            query += " AND label_type=?"
            params.append(label_type)
        if validated:
            query += " AND validated=?"
            params.append(validated)
        return conn.execute(query, params).fetchall()


def get_dataset_split_summary() -> list:
    with get_connection() as conn:
        return conn.execute(
            "SELECT label_type, split, COUNT(*) FROM dataset_splits GROUP BY label_type, split"
        ).fetchall()
