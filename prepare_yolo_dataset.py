"""
prepare_yolo_dataset.py
=======================
Converts physics-based detections into a YOLOv8 instance-segmentation dataset.

Uses the spray_physics.py dual-branch pipeline:
  Branch A — hysteresis z-score + contour extraction + physics classification
  Branch B — LoG blob detector for fine droplets (no morphology)

Output format (YOLO-seg):
    <class_id> <x1> <y1> <x2> <y2> ... <xN> <yN>

where coordinates are normalised to [0, 1].

Usage:
    python prepare_yolo_dataset.py                       # physics pipeline (default)
    python prepare_yolo_dataset.py --pipeline legacy     # old spray_drop.py

Output:
    yolo_dataset/
    ├── data.yaml
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── val/
    │   ├── images/
    │   └── labels/
    └── test/
        ├── images/
        └── labels/
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Import both pipelines
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))
import spray_physics as sp

# Try importing the legacy pipeline too
try:
    import spray_drop as sd
    LEGACY_AVAILABLE = True
except ImportError:
    LEGACY_AVAILABLE = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATASET_DIR = Path("dataset")
OUTPUT_DIR = Path("yolo_dataset")

# Class mapping — main_jet and uncertain are NOT mapped → skipped
LABEL_MAP = {
    "droplet": 0,
    "ligament": 1,
}
CLASS_NAMES = {0: "droplet", 1: "ligament"}

# Train / val / test split ratios
TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
TEST_RATIO  = 0.10

MIN_POLYGON_POINTS = 3
MIN_AREA_PIXELS = 3  # fixed: was 5, must align with spray_physics min_area=2


def contour_to_yolo_polygon(
    contour: np.ndarray, img_w: int, img_h: int,
) -> list[float] | None:
    """Convert an OpenCV contour to a normalised YOLO polygon.

    Uses adaptive simplification: thin ligaments (high aspect ratio) get
    a tighter epsilon to preserve their narrow shape details.  Round
    droplets can tolerate more simplification.
    """
    perim = cv2.arcLength(contour, True)
    area = cv2.contourArea(contour)
    if area < 1.0 or perim < 1.0:
        return None

    # Compute aspect ratio for adaptive epsilon
    (_, _), (rw, rh), _ = cv2.minAreaRect(contour)
    short = max(min(rw, rh), 1e-6)
    aspect = max(rw, rh) / short

    # Thin ligaments (aspect > 3): very tight simplification (0.002)
    # Medium shapes: moderate (0.005)
    # Round droplets: more aggressive (0.01)
    if aspect > 3.0:
        eps_factor = 0.002
    elif aspect > 1.8:
        eps_factor = 0.005
    else:
        eps_factor = 0.01

    epsilon = eps_factor * perim
    approx = cv2.approxPolyDP(contour, epsilon, True)

    pts = approx.reshape(-1, 2)
    if len(pts) < MIN_POLYGON_POINTS:
        return None

    coords: list[float] = []
    for x, y in pts:
        coords.append(round(float(x) / img_w, 6))
        coords.append(round(float(y) / img_h, 6))
    return coords


def build_physics_args() -> argparse.Namespace:
    """Create default args for the physics pipeline."""
    parser = sp.build_parser()
    return parser.parse_args([])


def build_legacy_args() -> object:
    """Create default args for the legacy pipeline."""
    if not LEGACY_AVAILABLE:
        raise SystemExit("Legacy pipeline (spray_drop.py) not available.")
    parser = sd.build_parser()
    return parser.parse_args([])


def process_frame_physics(
    frame_path: Path,
    frame_id: int,
    background: sp.BackgroundModel | None,
    args: argparse.Namespace,
) -> list[str]:
    """Segment a single frame using the dual-branch PHYSICS pipeline.

    Branch A: hysteresis z-score → morphology → contour → physics classify
    Branch B: LoG blob detection on raw grayscale → fine droplets
    """
    frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
    if frame is None:
        return []

    img_h, img_w = frame.shape[:2]
    gray = sp.to_gray(frame)

    # ===== Branch A: contour pipeline =====
    binary, denoised = sp.extract_foreground(frame, background, args)
    binary = sp.clean_mask(binary, args)
    components = sp.find_component_contours(binary)

    lines: list[str] = []
    branch_a_mask = np.zeros((img_h, img_w), dtype=np.uint8)

    for contour, comp_mask, pixel_area in components:
        if pixel_area < MIN_AREA_PIXELS:
            continue

        feats = sp.compute_features(contour, comp_mask, pixel_area, gray, background)

        # Quality gate: edge strength
        if feats["edge_strength"] < args.min_edge_strength:
            continue

        # Quality gate: interior consistency
        if background is not None and feats["interior_ratio"] < args.min_interior_ratio:
            continue

        # Physics classification (with boundary-aware main jet detection)
        label = sp.classify_physics(feats, args, img_w=img_w, img_h=img_h, contour=contour)

        # Mark Branch A coverage for deduplication with Branch B
        cv2.drawContours(branch_a_mask, [contour], -1, 255, cv2.FILLED)

        # Map to YOLO class — main_jet and uncertain are skipped
        class_id = LABEL_MAP.get(label)
        if class_id is None:
            continue

        polygon = contour_to_yolo_polygon(contour, img_w, img_h)
        if polygon is None:
            continue

        coord_str = " ".join(f"{c:.6f}" for c in polygon)
        lines.append(f"{class_id} {coord_str}")

    # ===== Branch B: LoG fine-droplet detection =====
    # Use raw grayscale (not denoised) — preserves tiny blobs
    raw_gray = sp.to_gray(frame)
    fine_blobs = sp.detect_fine_droplets(raw_gray, background, args, branch_a_mask)

    for blob_contour, blob_area in fine_blobs:
        if blob_area < MIN_AREA_PIXELS:
            continue

        # All LoG blobs are classified as droplets
        class_id = 0  # droplet

        polygon = contour_to_yolo_polygon(blob_contour, img_w, img_h)
        if polygon is None:
            continue

        coord_str = " ".join(f"{c:.6f}" for c in polygon)
        lines.append(f"{class_id} {coord_str}")

    return lines


def process_frame_legacy(
    frame_path: Path,
    frame_id: int,
    background,
    args,
) -> list[str]:
    """Segment a single frame using the LEGACY pipeline (spray_drop.py)."""
    frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
    if frame is None:
        return []

    img_h, img_w = frame.shape[:2]
    binary = sd.segment_frame(frame, background, args)
    components = sd.find_component_contours(binary)

    lines: list[str] = []
    for contour, comp_mask, pixel_area in components:
        if pixel_area < MIN_AREA_PIXELS:
            continue
        feats = sd.compute_features(contour, comp_mask, pixel_area)
        label = sd.classify_contour(feats, args)
        class_id = LABEL_MAP.get(label)
        if class_id is None:
            continue
        polygon = contour_to_yolo_polygon(contour, img_w, img_h)
        if polygon is None:
            continue
        coord_str = " ".join(f"{c:.6f}" for c in polygon)
        lines.append(f"{class_id} {coord_str}")

    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare YOLO dataset from spray frames.")
    parser.add_argument(
        "--pipeline", choices=("physics", "legacy"), default="physics",
        help="Which segmentation pipeline to use for label generation (default: physics).",
    )
    parser.add_argument("--dataset", default="dataset", help="Input frames directory.")
    parser.add_argument("--output", default="yolo_dataset", help="Output dataset directory.")
    cli_args = parser.parse_args()

    dataset_dir = Path(cli_args.dataset)
    output_dir = Path(cli_args.output)
    use_physics = cli_args.pipeline == "physics"

    print("=" * 60)
    print("  YOLO Instance-Segmentation Dataset Preparation")
    print(f"  Pipeline: {'PHYSICS dual-branch (spray_physics.py)' if use_physics else 'LEGACY (spray_drop.py)'}")
    print("=" * 60)

    # Collect frames
    frame_paths = sorted(dataset_dir.glob("frame*.png"), key=sp.frame_number)
    if not frame_paths:
        sys.exit(f"No frames found in {dataset_dir}")
    print(f"\nFound {len(frame_paths)} frames in {dataset_dir}/")

    # Build background + args for the chosen pipeline
    if use_physics:
        args = build_physics_args()
        background = sp.load_background(args, frame_paths)
        process_fn = process_frame_physics
    else:
        if not LEGACY_AVAILABLE:
            sys.exit("Legacy pipeline not available. Use --pipeline physics.")
        args = build_legacy_args()
        background = sd.load_background(args, frame_paths)
        process_fn = process_frame_legacy

    if background is not None:
        print("Background model loaded successfully.")
    else:
        print("No background model (proceeding without subtraction).")

    if use_physics:
        print(f"\n  Branch A: hysteresis z-score [{args.background_zscore_lo}, {args.background_zscore_hi}]")
        print(f"            morph-close={args.morph_close_ksize}, edge-strength>={args.min_edge_strength}")
        print(f"  Branch B: LoG blobs sigma=[{args.log_sigma_min}, {args.log_sigma_max}], threshold={args.log_threshold}")
        print(f"  Main jet: boundary-margin={args.boundary_margin}px, boundary-min-area={args.boundary_min_area}px")
        print(f"  Export:   MIN_AREA_PIXELS={MIN_AREA_PIXELS}")

    # Shuffle and split
    random.seed(42)
    indices = list(range(len(frame_paths)))
    random.shuffle(indices)

    n_train = int(len(indices) * TRAIN_RATIO)
    n_val = int(len(indices) * VAL_RATIO)

    split_map: dict[int, str] = {}
    for i in indices[:n_train]:
        split_map[i] = "train"
    for i in indices[n_train:n_train + n_val]:
        split_map[i] = "val"
    for i in indices[n_train + n_val:]:
        split_map[i] = "test"

    split_counts = {"train": n_train, "val": n_val, "test": len(indices) - n_train - n_val}
    print(f"\nSplit: train={split_counts['train']}, val={split_counts['val']}, test={split_counts['test']}")

    # Create directories
    for split in ("train", "val", "test"):
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    # Process every frame
    stats = {"total_objects": 0, "droplets": 0, "ligaments": 0, "frames_processed": 0}

    for idx, frame_path in enumerate(frame_paths):
        frame_id = sp.frame_number(frame_path)
        split = split_map[idx]

        label_lines = process_fn(frame_path, frame_id, background, args)

        for line in label_lines:
            cls = int(line.split()[0])
            stats["total_objects"] += 1
            if cls == 0:
                stats["droplets"] += 1
            else:
                stats["ligaments"] += 1

        # Copy image
        dst_img = output_dir / split / "images" / frame_path.name
        if not dst_img.exists():
            shutil.copy2(frame_path, dst_img)

        # Write label
        dst_label = output_dir / split / "labels" / (frame_path.stem + ".txt")
        with open(dst_label, "w") as f:
            f.write("\n".join(label_lines))
            if label_lines:
                f.write("\n")

        stats["frames_processed"] += 1

        if (idx + 1) % 100 == 0 or idx == len(frame_paths) - 1:
            pct = 100.0 * (idx + 1) / len(frame_paths)
            print(
                f"  [{pct:5.1f}%] Processed {idx + 1}/{len(frame_paths)} frames "
                f"| Objects: {stats['total_objects']} "
                f"(droplets={stats['droplets']}, ligaments={stats['ligaments']})"
            )

    # Write data.yaml
    data_yaml = {
        "path": str(output_dir.resolve()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": len(CLASS_NAMES),
        "names": [CLASS_NAMES[i] for i in sorted(CLASS_NAMES)],
    }
    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)

    # Summary
    print("\n" + "=" * 60)
    print("  Dataset Preparation Complete!")
    print("=" * 60)
    print(f"  Pipeline       : {'PHYSICS dual-branch' if use_physics else 'LEGACY'}")
    print(f"  Output dir     : {output_dir.resolve()}")
    print(f"  data.yaml      : {yaml_path.resolve()}")
    print(f"  Frames         : {stats['frames_processed']}")
    print(f"  Total objects  : {stats['total_objects']}")
    print(f"    - droplets   : {stats['droplets']}")
    print(f"    - ligaments  : {stats['ligaments']}")
    print(f"  Train images   : {split_counts['train']}")
    print(f"  Val images     : {split_counts['val']}")
    print(f"  Test images    : {split_counts['test']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
