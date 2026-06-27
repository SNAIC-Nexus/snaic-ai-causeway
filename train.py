"""Fine-tune yolo26n.pt on curated Causeway vehicle labels.

Usage:
    python train.py

Outputs:
    models/causeway_vehicle_v1.pt   — best fine-tuned weights (4-class)
    models/causeway_vehicle_v1.npz  — MLX-converted weights for live inference
"""
import os
import shutil
from causeway.dataset import export_curated_dataset

_HERE        = os.path.dirname(os.path.abspath(__file__))
CURATED_DIR  = "dataset/curated"
BASE_MODEL   = "models/yolo26n.pt"
YAML_PATH    = "dataset_vehicle_curated.yaml"
OUT_PT       = "models/causeway_vehicle_v1.pt"
OUT_NPZ      = "models/causeway_vehicle_v1.npz"
RUNS_DIR     = os.path.join(_HERE, "runs")


def main():
    import pathlib
    if not pathlib.Path(BASE_MODEL).exists():
        print(f"ERROR: Base model not found at {BASE_MODEL}")
        print("Download it from https://docs.ultralytics.com or copy yolo26n.pt from the yolo26mlx package.")
        return

    # --- Stage 1: Export curated dataset ---
    print("=== Stage 1: Exporting curated dataset ===")
    result = export_curated_dataset(CURATED_DIR)
    if result["train"] == 0:
        print("ERROR: No approved vehicle labels found.")
        print("Use the Streamlit app (streamlit run causeway_app.py --server.port 8502)")
        print("to review and approve vehicle labels before training.")
        return

    print(f"Train frames: {result['train']}, Val frames: {result['val']}")
    if result["train"] < 50:
        print(f"WARNING: Only {result['train']} training frames. Aim for 200+ for meaningful fine-tuning.")

    # --- Stage 2: Fine-tune ---
    print("\n=== Stage 2: Fine-tuning yolo26n.pt on MPS ===")
    from ultralytics import YOLO
    model = YOLO(BASE_MODEL)
    model.train(
        data=YAML_PATH,
        epochs=100,
        imgsz=640,
        batch=16,
        device="mps",
        patience=20,
        freeze=10,        # freeze first 10 backbone layers; train neck + head
        optimizer="auto", # ultralytics picks AdamW for fine-tuning
        project=os.path.join(RUNS_DIR, "causeway_vehicle"),
        name="v1",
        exist_ok=True,
    )

    # --- Stage 3: Copy best weights ---
    print("\n=== Stage 3: Saving best weights ===")
    best_pt = os.path.join(RUNS_DIR, "causeway_vehicle", "v1", "weights", "best.pt")
    if not os.path.exists(best_pt):
        print(f"ERROR: Expected best weights at {best_pt} but file not found.")
        return

    os.makedirs("models", exist_ok=True)
    shutil.copy2(best_pt, OUT_PT)
    print(f"Saved: {OUT_PT}")

    # --- Stage 4: Convert to MLX ---
    print("\n=== Stage 4: Converting to MLX format ===")
    try:
        from yolo26mlx.converters.convert import convert_yolo26_weights
        convert_yolo26_weights(OUT_PT, output_path=OUT_NPZ)
        print(f"Saved: {OUT_NPZ}")
    except ImportError:
        print("WARNING: yolo26mlx not installed — skipping MLX conversion.")
        print(f"To convert manually: from yolo26mlx.converters import convert_model; convert_model('{OUT_PT}', '{OUT_NPZ}')")
    except Exception as e:
        print(f"WARNING: MLX conversion failed: {e}")
        print(f"PyTorch model is at {OUT_PT} and can be used directly with ultralytics.")

    print("\nDone. Run 'dagster dev -f dagster_defs.py --port 3001' and materialise generate_vehicle_labels to re-label all images with the new model.")


if __name__ == "__main__":
    main()
