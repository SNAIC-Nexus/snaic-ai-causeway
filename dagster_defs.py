import os
import subprocess
import sys

from dagster import asset, Definitions, ScheduleDefinition, define_asset_job, AssetSelection


@asset(description="One-shot: migrate existing traffic images into canonical path and backfill SQLite.")
def migrate_existing_images():
    from causeway.migration import migrate
    result = migrate()
    return result


@asset(description="Scrape one cycle of LTA traffic camera images for cameras 2701, 2702, 2704.")
def scrape_images():
    from causeway.db import init_db
    from causeway.scraper import run_once
    init_db()
    run_once()


@asset(deps=[scrape_images], description="Generate YOLO segmentation labels for lane regions from camera_config.json.")
def generate_lane_labels():
    from causeway.labeler import generate_lane_labels as _gen
    count = _gen()
    return {"new_labels": count}


@asset(deps=[scrape_images], description="Generate YOLO detection labels for vehicles using YOLOv8x inference on MPS.")
def generate_vehicle_labels():
    from causeway.labeler import generate_vehicle_labels as _gen
    count = _gen()
    return {"new_labels": count}


@asset(
    deps=[generate_vehicle_labels],
    description="Export curated vehicle labels, fine-tune a domain-adapted vehicle detector, and output models/causeway_vehicle_v1.pt and models/causeway_vehicle_v1.npz.",
)
def fine_tune_vehicle_model():
    result = subprocess.run(
        [sys.executable, "train.py"],
        capture_output=False,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError("train.py failed — check output above for details.")
    pt_exists = os.path.exists("models/causeway_vehicle_v1.pt")
    npz_exists = os.path.exists("models/causeway_vehicle_v1.npz")
    return {"pt": pt_exists, "npz": npz_exists}


@asset(
    deps=[generate_lane_labels, generate_vehicle_labels],
    description="Split dataset by day: all days except last → train, last day → val. Writes dataset_lane.yaml and dataset_vehicle.yaml.",
)
def build_dataset_split():
    from causeway.dataset import build_dataset_split as _split
    return _split()


scrape_job = define_asset_job(
    "scrape_job",
    selection=AssetSelection.assets(scrape_images),
    description="Runs one scrape cycle for all target cameras.",
)

scrape_schedule = ScheduleDefinition(
    name="scrape_every_5_minutes",
    job=scrape_job,
    cron_schedule="*/5 * * * *",
)

defs = Definitions(
    assets=[
        migrate_existing_images,
        scrape_images,
        generate_lane_labels,
        generate_vehicle_labels,
        fine_tune_vehicle_model,
        build_dataset_split,
    ],
    schedules=[scrape_schedule],
)
