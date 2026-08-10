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
from dataclasses import dataclass
import math
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
    "main_jet": 2,
}
CLASS_NAMES = {0: "droplet", 1: "ligament", 2: "main_jet"}
PREVIEW_COLORS = {
    0: (0, 200, 255),    # droplet: orange (BGR)
    1: (255, 0, 255),    # ligament: magenta (BGR)
    2: (255, 255, 0),    # main jet: cyan (BGR)
}

# Train / val / test split ratios
TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
TEST_RATIO  = 0.10

MIN_POLYGON_POINTS = 3
MIN_AREA_PIXELS = 3  # fixed: was 5, must align with spray_physics min_area=2


@dataclass
class FineTrack:
    centroid: tuple[float, float]
    last_frame: int
    hits: int = 1
    moving_hits: int = 0
    missed: int = 0


class FineDropletTracker:
    """Confirm tiny LoG candidates through short, non-static tracks."""

    def __init__(self, max_distance: float, min_displacement: float, min_hits: int, max_age: int):
        self.max_distance = max_distance
        self.min_displacement = min_displacement
        self.min_hits = min_hits
        self.max_age = max_age
        self.tracks: dict[int, FineTrack] = {}
        self.next_track_id = 1

    @staticmethod
    def _centroid(contour: np.ndarray) -> tuple[float, float]:
        moments = cv2.moments(contour)
        if moments["m00"]:
            return moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]
        x, y, width, height = cv2.boundingRect(contour)
        return x + width / 2.0, y + height / 2.0

    def update(self, contours: list[np.ndarray], frame_id: int) -> list[bool]:
        centres = [self._centroid(contour) for contour in contours]
        candidates: list[tuple[float, int, int]] = []
        for contour_idx, centre in enumerate(centres):
            for track_id, track in self.tracks.items():
                distance = math.dist(centre, track.centroid)
                if distance <= self.max_distance:
                    candidates.append((distance, contour_idx, track_id))
        candidates.sort()

        assignments: dict[int, int] = {}
        used_contours: set[int] = set()
        used_tracks: set[int] = set()
        for _, contour_idx, track_id in candidates:
            if contour_idx not in used_contours and track_id not in used_tracks:
                assignments[contour_idx] = track_id
                used_contours.add(contour_idx)
                used_tracks.add(track_id)

        confirmed: list[bool] = []
        for contour_idx, centre in enumerate(centres):
            track_id = assignments.get(contour_idx)
            if track_id is None:
                self.tracks[self.next_track_id] = FineTrack(centre, frame_id)
                self.next_track_id += 1
                confirmed.append(False)
                continue

            track = self.tracks[track_id]
            displacement = math.dist(centre, track.centroid)
            track.centroid = centre
            track.last_frame = frame_id
            track.hits += 1
            track.missed = 0
            if displacement >= self.min_displacement:
                track.moving_hits += 1
            confirmed.append(track.hits >= self.min_hits and track.moving_hits > 0)

        for track_id, track in list(self.tracks.items()):
            if track.last_frame != frame_id:
                track.missed += 1
                if track.missed > self.max_age:
                    del self.tracks[track_id]

        return confirmed


def contour_to_yolo_polygon(
    contour: np.ndarray, img_w: int, img_h: int, class_id: int | None = None,
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

    # Main jet / source ligament needs a very faithful outline. Round droplets
    # can tolerate more simplification.
    if class_id == LABEL_MAP["main_jet"]:
        eps_factor = 0.001
    elif aspect > 3.0:
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


def contour_overlap_fraction(contour: np.ndarray, owner_mask: np.ndarray) -> float:
    """Fraction of a candidate contour area covered by an owning liquid mask."""
    if not np.any(owner_mask):
        return 0.0
    candidate_mask = np.zeros_like(owner_mask)
    cv2.drawContours(candidate_mask, [contour], -1, 255, cv2.FILLED)
    area = int(cv2.countNonZero(candidate_mask))
    if area == 0:
        return 0.0
    overlap = cv2.bitwise_and(candidate_mask, owner_mask)
    return float(cv2.countNonZero(overlap)) / float(area)


def write_label_preview(image_path: Path, label_path: Path, output_path: Path) -> dict[int, int]:
    """Render final YOLO-seg polygons without per-instance text clutter.

    All exported masks receive a translucent fill. Main jet is drawn first
    and with a stronger outline so overlap mistakes are easy to spot.
    """
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return {}
    height, width = image.shape[:2]
    overlay = image.copy()
    counts = {class_id: 0 for class_id in CLASS_NAMES}
    outlines: list[tuple[np.ndarray, tuple[int, int, int], int]] = []
    entries: list[tuple[int, np.ndarray, tuple[int, int, int]]] = []

    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        parts = raw_line.split()
        if len(parts) < 7:
            continue
        try:
            class_id = int(parts[0])
            coords = np.asarray(parts[1:], dtype=np.float32).reshape(-1, 2)
        except (ValueError, TypeError):
            continue
        color = PREVIEW_COLORS.get(class_id)
        if color is None or len(coords) < MIN_POLYGON_POINTS:
            continue
        coords[:, 0] *= width - 1
        coords[:, 1] *= height - 1
        polygon = np.rint(coords).astype(np.int32).reshape(-1, 1, 2)
        entries.append((class_id, polygon, color))
        counts[class_id] = counts.get(class_id, 0) + 1

    entries.sort(key=lambda item: 0 if item[0] == LABEL_MAP["main_jet"] else 1)
    for class_id, polygon, color in entries:
        cv2.fillPoly(overlay, [polygon], color)
        outlines.append((polygon, color, 2 if class_id == LABEL_MAP["main_jet"] else 1))

    annotated = cv2.addWeighted(overlay, 0.22, image, 0.78, 0)
    for polygon, color, thickness in outlines:
        cv2.polylines(annotated, [polygon], True, color, thickness, cv2.LINE_AA)
    cv2.rectangle(annotated, (8, 8), (330, 82), (24, 24, 24), cv2.FILLED)
    cv2.putText(
        annotated,
        f"{image_path.stem} final YOLO-seg labels",
        (14, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    legend = [(0, "droplet"), (1, "ligament"), (2, "main jet")]
    for index, (class_id, name) in enumerate(legend):
        x = 14 + index * 104
        cv2.circle(annotated, (x, 53), 4, PREVIEW_COLORS[class_id], cv2.FILLED)
        cv2.putText(
            annotated,
            f"{name}: {counts[class_id]}",
            (x + 8, 57),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    cv2.imwrite(str(output_path), annotated)
    return counts


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
    fine_tracker: FineDropletTracker | None = None,
    medium_tracker: FineDropletTracker | None = None,
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
    binary = sp.split_touching_components(binary, args)
    components = sp.find_component_contours(binary)

    lines: list[str] = []
    branch_a_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    large_liquid_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    main_jet_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    branch_a_candidates: list[tuple[int, np.ndarray, list[float]]] = []

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

        # Large connected liquid structures are exported as labels, but their
        # interiors veto small/medium recovery so internal texture is not
        # turned into separate droplets.
        if label != "main_jet":
            cv2.drawContours(branch_a_mask, [contour], -1, 255, cv2.FILLED)
        if label in {"main_jet", "main_ligament", "ligament"}:
            cv2.drawContours(large_liquid_mask, [contour], -1, 255, cv2.FILLED)
        if label == "main_jet":
            cv2.drawContours(main_jet_mask, [contour], -1, 255, cv2.FILLED)

        # Map the physics label to a YOLO class.
        class_id = LABEL_MAP.get(label)
        if class_id is None:
            continue

        polygon = contour_to_yolo_polygon(contour, img_w, img_h, class_id)
        if polygon is None:
            continue

        branch_a_candidates.append((class_id, contour, polygon))

    for class_id, contour, polygon in branch_a_candidates:
        if class_id != LABEL_MAP["main_jet"]:
            overlap = contour_overlap_fraction(contour, main_jet_mask)
            if overlap >= getattr(args, "main_jet_child_overlap_veto", 0.35):
                continue
        coord_str = " ".join(f"{c:.6f}" for c in polygon)
        lines.append(f"{class_id} {coord_str}")

    # ===== Branch B: LoG fine-droplet detection =====
    # Use raw grayscale (not denoised) — preserves tiny blobs
    raw_gray = sp.to_gray(frame)
    interior_veto_mask = sp.erode_interior_veto(
        large_liquid_mask,
        getattr(args, "interior_veto_erosion", 0),
    )
    fine_exclusion_mask = cv2.bitwise_or(branch_a_mask, interior_veto_mask)
    fine_blobs = [
        (contour, area)
        for contour, area in sp.detect_fine_droplets(raw_gray, background, args, fine_exclusion_mask)
        if area >= MIN_AREA_PIXELS
    ]
    confirmed = (
        fine_tracker.update([contour for contour, _ in fine_blobs], frame_id)
        if fine_tracker is not None
        else [True] * len(fine_blobs)
    )
    branch_b_mask = np.zeros((img_h, img_w), dtype=np.uint8)

    for (blob_contour, blob_area), is_confirmed in zip(fine_blobs, confirmed):
        cv2.drawContours(branch_b_mask, [blob_contour], -1, 255, cv2.FILLED)
        if not is_confirmed:
            continue

        # All LoG blobs are classified as droplets
        class_id = 0  # droplet

        polygon = contour_to_yolo_polygon(blob_contour, img_w, img_h, class_id)
        if polygon is None:
            continue

        coord_str = " ".join(f"{c:.6f}" for c in polygon)
        lines.append(f"{class_id} {coord_str}")

    # ===== Branch C: medium irregular fragments =====
    claimed_mask = cv2.bitwise_or(fine_exclusion_mask, branch_b_mask)
    medium_mask = sp.extract_medium_fragment_mask(raw_gray, background, args, claimed_mask)
    medium_mask = sp.split_touching_components(medium_mask, args)
    medium_components = [
        component
        for component in sp.find_component_contours(medium_mask)
        if args.medium_min_area <= component[2] <= args.medium_max_area
    ]
    medium_confirmed = (
        medium_tracker.update([contour for contour, _, _ in medium_components], frame_id)
        if medium_tracker is not None
        else [True] * len(medium_components)
    )

    for (contour, component_mask, pixel_area), is_confirmed in zip(medium_components, medium_confirmed):
        if not is_confirmed:
            continue
        feats = sp.compute_features(contour, component_mask, pixel_area, raw_gray, background)
        if feats["edge_strength"] < args.medium_min_edge_strength:
            continue
        if background is not None and feats["interior_ratio"] < args.medium_min_interior_ratio:
            continue
        label = sp.classify_physics(feats, args, img_w=img_w, img_h=img_h, contour=contour)
        class_id = LABEL_MAP.get(label)
        if class_id is None:
            continue
        if class_id != LABEL_MAP["main_jet"]:
            overlap = contour_overlap_fraction(contour, main_jet_mask)
            if overlap >= getattr(args, "main_jet_child_overlap_veto", 0.35):
                continue
        polygon = contour_to_yolo_polygon(contour, img_w, img_h, class_id)
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
        polygon = contour_to_yolo_polygon(contour, img_w, img_h, class_id)
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
    parser.add_argument("--fine-track-distance", type=float, default=20.0,
                        help="Maximum fine-droplet displacement between adjacent frames.")
    parser.add_argument("--fine-track-min-displacement", type=float, default=0.5,
                        help="Minimum movement needed to reject static fine candidates.")
    parser.add_argument("--fine-track-min-hits", type=int, default=2,
                        help="Observations required before exporting a fine-droplet mask.")
    parser.add_argument("--fine-track-max-age", type=int, default=2,
                        help="Missed frames retained for a fine-droplet track.")
    parser.add_argument("--medium-track-distance", type=float, default=20.0,
                        help="Maximum medium-fragment displacement between adjacent frames.")
    parser.add_argument("--medium-track-min-displacement", type=float, default=0.5,
                        help="Minimum movement needed to reject static medium candidates.")
    parser.add_argument("--medium-track-min-hits", type=int, default=2,
                        help="Observations required before exporting a medium-fragment mask.")
    parser.add_argument("--medium-track-max-age", type=int, default=1,
                        help="Missed frames retained for a medium-fragment track.")
    parser.add_argument("--interior-veto-erosion", type=int, default=None,
                        help="Override physics interior veto erosion in pixels. Lower is stricter.")
    parser.add_argument("--main-jet-child-overlap-veto", type=float, default=None,
                        help="Override overlap fraction used to suppress droplet/ligament labels inside main jet.")
    parser.add_argument("--preview-count", type=int, default=10,
                        help="Number of representative coloured label previews to write.")
    parser.add_argument("--preview-frames", nargs="*", type=int,
                        help="Specific frame numbers to render instead of evenly spaced previews.")
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
        if cli_args.interior_veto_erosion is not None:
            args.interior_veto_erosion = cli_args.interior_veto_erosion
        if cli_args.main_jet_child_overlap_veto is not None:
            args.main_jet_child_overlap_veto = cli_args.main_jet_child_overlap_veto
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
        print(f"  Branch C: medium fragments area=[{args.medium_min_area}, {args.medium_max_area}]")
        print(f"  Main jet: left source ROI={args.source_roi_width_fraction:.0%}, min-area={args.source_min_area}px")
        print("  Watershed: enabled for compact multi-peak components")
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
    stats = {
        "total_objects": 0,
        "droplets": 0,
        "ligaments": 0,
        "main_jets": 0,
        "frames_processed": 0,
    }
    fine_tracker = (
        FineDropletTracker(
            cli_args.fine_track_distance,
            cli_args.fine_track_min_displacement,
            cli_args.fine_track_min_hits,
            cli_args.fine_track_max_age,
        )
        if use_physics
        else None
    )
    medium_tracker = (
        FineDropletTracker(
            cli_args.medium_track_distance,
            cli_args.medium_track_min_displacement,
            cli_args.medium_track_min_hits,
            cli_args.medium_track_max_age,
        )
        if use_physics
        else None
    )

    for idx, frame_path in enumerate(frame_paths):
        frame_id = sp.frame_number(frame_path)
        split = split_map[idx]

        if use_physics:
            label_lines = process_frame_physics(
                frame_path,
                frame_id,
                background,
                args,
                fine_tracker,
                medium_tracker,
            )
        else:
            label_lines = process_fn(frame_path, frame_id, background, args)

        for line in label_lines:
            cls = int(line.split()[0])
            stats["total_objects"] += 1
            if cls == 0:
                stats["droplets"] += 1
            elif cls == 1:
                stats["ligaments"] += 1
            elif cls == 2:
                stats["main_jets"] += 1

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
                f"(droplets={stats['droplets']}, ligaments={stats['ligaments']}, "
                f"main_jets={stats['main_jets']})"
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

    # Render a small, readable sample from the exact label files written above.
    preview_dir = output_dir / "preview"
    preview_dir.mkdir(parents=True, exist_ok=True)
    if cli_args.preview_frames:
        requested = set(cli_args.preview_frames)
        preview_indices = [
            index for index, path in enumerate(frame_paths)
            if sp.frame_number(path) in requested
        ]
    elif cli_args.preview_count > 0:
        preview_total = min(cli_args.preview_count, len(frame_paths))
        preview_indices = np.linspace(0, len(frame_paths) - 1, preview_total, dtype=int).tolist()
    else:
        preview_indices = []

    for index in dict.fromkeys(preview_indices):
        frame_path = frame_paths[index]
        split = split_map[index]
        image_path = output_dir / split / "images" / frame_path.name
        label_path = output_dir / split / "labels" / f"{frame_path.stem}.txt"
        preview_path = preview_dir / f"preview_{frame_path.name}"
        write_label_preview(image_path, label_path, preview_path)
    if preview_indices:
        print(f"  Previews       : {len(dict.fromkeys(preview_indices))} in {preview_dir}")

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
    print(f"    - main jets  : {stats['main_jets']}")
    print(f"  Train images   : {split_counts['train']}")
    print(f"  Val images     : {split_counts['val']}")
    print(f"  Test images    : {split_counts['test']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
