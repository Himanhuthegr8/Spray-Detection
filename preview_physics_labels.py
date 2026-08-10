"""
preview_physics_labels.py
=========================

Generate coloured YOLO-seg label previews for only a few frames, without
building the full dataset split.

The script reuses prepare_yolo_dataset.py, including the current physics
pipeline, main-jet overlap veto, and preview renderer.

Example:
    python preview_physics_labels.py --frames 1935 1535 1030 599 1280
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np

import prepare_yolo_dataset as prep
import spray_physics as sp


def select_preview_frames(frame_paths: list[Path], frames: list[int] | None, count: int) -> list[Path]:
    if frames:
        requested = set(frames)
        selected = [path for path in frame_paths if sp.frame_number(path) in requested]
        missing = sorted(requested - {sp.frame_number(path) for path in selected})
        if missing:
            print(f"Warning: requested frames not found: {missing}")
        return selected

    preview_total = min(max(count, 1), len(frame_paths))
    indices = np.linspace(0, len(frame_paths) - 1, preview_total, dtype=int)
    return [frame_paths[int(index)] for index in indices]


def local_window(frame_paths: list[Path], target: Path, warmup: int) -> list[Path]:
    target_index = frame_paths.index(target)
    start = max(0, target_index - max(warmup, 0))
    return frame_paths[start:target_index + 1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview physics YOLO labels for a few frames.")
    parser.add_argument("--dataset", default="dataset", help="Input frame directory.")
    parser.add_argument("--output", default="physics_label_preview_5", help="Preview output directory.")
    parser.add_argument("--frames", nargs="*", type=int, help="Specific frame numbers to preview.")
    parser.add_argument("--count", type=int, default=5, help="Number of evenly spaced frames when --frames is omitted.")
    parser.add_argument("--warmup", type=int, default=5,
                        help="Previous frames used only to warm fine/medium tracking.")
    parser.add_argument("--interior-veto-erosion", type=int, default=0,
                        help="Large-liquid interior veto erosion in pixels. 0 is strict.")
    parser.add_argument("--main-jet-child-overlap-veto", type=float, default=0.20,
                        help="Suppress child labels when this fraction overlaps main jet.")
    parser.add_argument("--fine-track-min-hits", type=int, default=2,
                        help="Observations required before exporting a fine-droplet mask.")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    output_dir = Path(args.output)
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    preview_dir = output_dir / "preview"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    frame_paths = sorted(dataset_dir.glob("frame*.png"), key=sp.frame_number)
    if not frame_paths:
        raise SystemExit(f"No frame*.png images found in {dataset_dir}")

    selected = select_preview_frames(frame_paths, args.frames, args.count)
    if not selected:
        raise SystemExit("No preview frames selected.")

    physics_args = prep.build_physics_args()
    # --- Balanced values: loosen small-droplet detection, tighten background rejection ---
    physics_args.interior_veto_erosion = args.interior_veto_erosion
    physics_args.main_jet_child_overlap_veto = args.main_jet_child_overlap_veto
    # LoG (Branch B): loosened to recover small droplets
    physics_args.log_threshold = 0.45
    physics_args.log_min_center_z = 2.0
    physics_args.log_min_center_delta_z = 1.2
    # Quality gates: tightened to reject background blobs
    physics_args.min_edge_strength = 5.0
    physics_args.min_interior_ratio = 0.25
    physics_args.min_foreground_contrast = 8

    print(f"Estimating/loading background from {len(frame_paths)} available frames...")
    background = sp.load_background(physics_args, frame_paths)

    print(f"Writing {len(selected)} preview(s) to {preview_dir.resolve()}")
    for target in selected:
        fine_tracker = prep.FineDropletTracker(
            max_distance=20.0,
            min_displacement=0.5,
            min_hits=args.fine_track_min_hits,
            max_age=2,
        )
        medium_tracker = prep.FineDropletTracker(
            max_distance=20.0,
            min_displacement=0.5,
            min_hits=2,
            max_age=1,
        )

        target_lines: list[str] = []
        for frame_path in local_window(frame_paths, target, args.warmup):
            frame_id = sp.frame_number(frame_path)
            lines = prep.process_frame_physics(
                frame_path,
                frame_id,
                background,
                physics_args,
                fine_tracker,
                medium_tracker,
            )
            if frame_path == target:
                target_lines = lines

        image_path = images_dir / target.name
        label_path = labels_dir / f"{target.stem}.txt"
        preview_path = preview_dir / f"preview_{target.name}"

        shutil.copy2(target, image_path)
        label_path.write_text("\n".join(target_lines) + ("\n" if target_lines else ""), encoding="utf-8")
        counts = prep.write_label_preview(image_path, label_path, preview_path)
        print(
            f"  frame {sp.frame_number(target):04d}: "
            f"droplet={counts.get(0, 0)}, ligament={counts.get(1, 0)}, main_jet={counts.get(2, 0)}"
        )

    print("Done.")


if __name__ == "__main__":
    main()
