# YOLO Training Readiness & Active Learning Loop

**Date:** 2026-06-29  
**Status:** Approved

---

## Context

The snaic-ai-causeway pipeline scrapes images from 3 LTA cameras (2701, 2702, 2704) and fine-tunes a YOLO vehicle detection model. The base YOLO model misses many vehicles — especially motorcycles — so auto-labelled annotations are being replaced with hand-labelled ground truth via the ✏️ Annotate tab.

**Current state:**
- ~50 images manually labelled, all from camera 2701
- Cameras 2702 and 2704 have zero hand labels
- Curated dataset has an inverted train/val split (50 train / 461 val) — must be fixed
- Motorcycle recall: 0% in current fine-tuned model

**Conclusion: not yet ready to train.** ~50 images from a single camera will overfit to one viewpoint and still fail on motorcycles from other cameras.

---

## Design

### Phase 1 — Seed Labelling (minimum viable dataset)

**Target:** 50 hand-labelled images per camera = 150 total

| Camera | Current | Target | Remaining |
|--------|---------|--------|-----------|
| 2701   | ~50     | 50     | 0         |
| 2702   | 0       | 50     | ~50       |
| 2704   | 0       | 50     | ~50       |
| **Total** | **~50** | **150** | **~100** |

**Why 50/camera:**
- Motorcycles appear in nearly every frame, so every labelled image contributes motorcycle examples
- 50 images × 3 cameras = 150 total; at 80/20 split → ~120 training images
- Sufficient for fine-tuning a pretrained backbone without overfitting
- Estimated effort: ~100 images × 12 min = ~20 hours

**All 4 classes must be annotated in every image:** motorcycle (0), car (1), bus (2), truck (3).

---

### Dataset Export Fix (before Phase 2)

The current curated dataset export produces a broken split (50 train / 461 val). Before training:

- Fix the export script to perform a proper **80/20 stratified split** across all 3 cameras
- Ensure all 3 camera viewpoints appear in both train and val sets
- Output to `dataset/curated/` as currently configured in `dataset_vehicle_curated.yaml`

---

### Phase 2 — First Real Training Run

Once 150 seed images are labelled and the dataset export is fixed:

```bash
python train.py
```

- Input: `dataset_vehicle_curated.yaml` pointing to fixed 80/20 split
- Output: `models/causeway_vehicle_v2.pt` (and MLX variant)
- Expected improvement: non-zero motorcycle recall across all 3 cameras

---

### Phase 3 — Model-Assisted Labelling (active learning)

After Phase 2, use the improved model to accelerate further labelling:

1. Update Dagster's `generate_vehicle_labels` asset to use `causeway_vehicle_v2.pt`
2. Re-run auto-labelling on all un-annotated images
3. Human reviews/corrects in the ✏️ Annotate tab — expected time drops from ~12 min to ~2-3 min per image (correcting vs. drawing from scratch)
4. Retrain when ~150 additional corrections are accumulated → `causeway_vehicle_v3.pt`
5. Repeat until performance plateaus

This active learning loop compresses the annotation effort significantly for subsequent iterations.

---

## Success Criteria

| Milestone | Criterion |
|-----------|-----------|
| Phase 1 complete | 50 labelled images each from 2702 and 2704 |
| Dataset export fixed | 80/20 split, all 3 cameras in train and val |
| Phase 2 complete | `train.py` runs successfully; motorcycle recall > 0 on val set |
| Phase 3 viable | New model pre-labels images; correction time < 3 min/image |

---

## Out of Scope

- Lane detection model (separate `dataset_lane.yaml` pipeline)
- Annotate tab UI changes
- Deployment / inference optimisation
