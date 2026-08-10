"""
debug_rejections.py
===================
Diagnostic tool: draws ALL detected contours on spray frames, color-coded
by their acceptance/rejection reason.  This makes threshold tuning a
one-minute visual diagnosis instead of trial-and-error guessing.

Colour legend:
    🟢 Green  = accepted as droplet
    🟣 Purple = accepted as ligament
    🔵 Blue   = excluded as main_jet
    ⚪ Gray   = rejected as uncertain classification
    🟡 Yellow = rejected by edge strength
    🟤 Brown  = rejected by interior ratio
    🟠 Orange = rejected by area (too small)
    🔴 Red    = rejected by foreground threshold (below z-score)

Usage:
    python debug_rejections.py                    # 10 random frames
    python debug_rejections.py --count 20         # 20 random frames
    python debug_rejections.py --frames 231 465   # specific frames
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import spray_physics as sp

# Colour legend (BGR)
C_DROPLET   = (0, 200, 0)      # green — accepted droplet
C_LIGAMENT  = (200, 0, 200)    # purple — accepted ligament
C_MAIN_JET  = (255, 200, 0)    # blue — excluded main_jet
C_UNCERTAIN = (180, 180, 180)  # gray — uncertain classification
C_EDGE      = (0, 220, 220)    # yellow — rejected by edge strength
C_INTERIOR  = (50, 100, 180)   # brown — rejected by interior ratio
C_AREA      = (0, 140, 255)    # orange — rejected by area
C_LOG_BLOB  = (0, 255, 128)    # light green — LoG fine droplet

C_MEDIUM = (255, 255, 0)

LEGEND = [
    ("accepted: droplet", C_DROPLET),
    ("accepted: ligament", C_LIGAMENT),
    ("accepted: LoG blob", C_LOG_BLOB),
    ("candidate: medium fragment", C_MEDIUM),
    ("excluded: main_jet", C_MAIN_JET),
    ("rejected: uncertain", C_UNCERTAIN),
    ("rejected: edge strength", C_EDGE),
    ("rejected: interior ratio", C_INTERIOR),
    ("rejected: area too small", C_AREA),
]


def draw_legend(img: np.ndarray) -> None:
    """Draw colour legend in the top-right corner."""
    x0 = img.shape[1] - 230
    y0 = 10
    for i, (label, color) in enumerate(LEGEND):
        y = y0 + i * 18
        cv2.rectangle(img, (x0, y), (x0 + 12, y + 12), color, cv2.FILLED)
        cv2.putText(img, label, (x0 + 18, y + 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.32, (255, 255, 255), 1, cv2.LINE_AA)


def analyze_frame_with_rejections(
    frame: np.ndarray,
    background: sp.BackgroundModel | None,
    args: argparse.Namespace,
) -> np.ndarray:
    """Process a frame and draw ALL contours with rejection reasons."""
    img_h, img_w = frame.shape[:2]
    gray = sp.to_gray(frame)
    output = frame.copy()

    # ===== Branch A =====
    binary, denoised = sp.extract_foreground(frame, background, args)
    binary_clean = sp.clean_mask(binary, args)
    binary_clean = sp.split_touching_components(binary_clean, args)
    components = sp.find_component_contours(binary_clean)

    branch_a_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    large_liquid_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    stats = {k: 0 for k in ["droplet", "ligament", "main_jet", "uncertain",
                              "rej_edge", "rej_interior", "rej_area", "log_blob", "medium"]}

    for contour, comp_mask, pixel_area in components:
        # --- Rejection: area ---
        if pixel_area < args.min_area:
            cv2.drawContours(output, [contour], -1, C_AREA, 1)
            stats["rej_area"] += 1
            continue

        feats = sp.compute_features(contour, comp_mask, pixel_area, gray, background)

        # --- Rejection: edge strength ---
        if feats["edge_strength"] < args.min_edge_strength:
            cv2.drawContours(output, [contour], -1, C_EDGE, 1)
            stats["rej_edge"] += 1
            continue

        # --- Rejection: interior ratio ---
        if background is not None and feats["interior_ratio"] < args.min_interior_ratio:
            cv2.drawContours(output, [contour], -1, C_INTERIOR, 1)
            stats["rej_interior"] += 1
            continue

        # --- Classification ---
        label = sp.classify_physics(feats, args, img_w=img_w, img_h=img_h, contour=contour)
        # Large connected liquid structures are shown as detections, but their
        # interiors veto small/medium recovery so internal texture is not
        # shown as separate droplets.
        if label != "main_jet":
            cv2.drawContours(branch_a_mask, [contour], -1, 255, cv2.FILLED)
        if label in {"main_jet", "main_ligament", "ligament"}:
            cv2.drawContours(large_liquid_mask, [contour], -1, 255, cv2.FILLED)

        if label == "main_jet" or label == "main_ligament":
            cv2.drawContours(output, [contour], -1, C_MAIN_JET, 2)
            stats["main_jet"] += 1
        elif label == "uncertain":
            cv2.drawContours(output, [contour], -1, C_UNCERTAIN, 1)
            stats["uncertain"] += 1
        elif label == "droplet":
            overlay = output.copy()
            cv2.drawContours(overlay, [contour], -1, C_DROPLET, cv2.FILLED)
            cv2.addWeighted(overlay, 0.3, output, 0.7, 0, output)
            cv2.drawContours(output, [contour], -1, C_DROPLET, 1)
            stats["droplet"] += 1
        elif label == "ligament":
            overlay = output.copy()
            cv2.drawContours(overlay, [contour], -1, C_LIGAMENT, cv2.FILLED)
            cv2.addWeighted(overlay, 0.3, output, 0.7, 0, output)
            cv2.drawContours(output, [contour], -1, C_LIGAMENT, 2)
            stats["ligament"] += 1

    # ===== Branch B: LoG blobs =====
    raw_gray = sp.to_gray(frame)
    interior_veto_mask = sp.erode_interior_veto(
        large_liquid_mask,
        getattr(args, "interior_veto_erosion", 0),
    )
    fine_exclusion_mask = cv2.bitwise_or(branch_a_mask, interior_veto_mask)
    fine_blobs = sp.detect_fine_droplets(raw_gray, background, args, fine_exclusion_mask)
    branch_b_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    for blob_contour, blob_area in fine_blobs:
        if blob_area < 3:
            continue
        cv2.drawContours(branch_b_mask, [blob_contour], -1, 255, cv2.FILLED)
        overlay = output.copy()
        cv2.drawContours(overlay, [blob_contour], -1, C_LOG_BLOB, cv2.FILLED)
        cv2.addWeighted(overlay, 0.3, output, 0.7, 0, output)
        cv2.drawContours(output, [blob_contour], -1, C_LOG_BLOB, 1)
        stats["log_blob"] += 1

    # ===== Branch C: medium irregular fragments =====
    claimed_mask = cv2.bitwise_or(fine_exclusion_mask, branch_b_mask)
    medium_mask = sp.extract_medium_fragment_mask(raw_gray, background, args, claimed_mask)
    medium_mask = sp.split_touching_components(medium_mask, args)
    for contour, component_mask, pixel_area in sp.find_component_contours(medium_mask):
        if not args.medium_min_area <= pixel_area <= args.medium_max_area:
            continue
        feats = sp.compute_features(contour, component_mask, pixel_area, raw_gray, background)
        if feats["edge_strength"] < args.medium_min_edge_strength:
            continue
        if background is not None and feats["interior_ratio"] < args.medium_min_interior_ratio:
            continue
        overlay = output.copy()
        cv2.drawContours(overlay, [contour], -1, C_MEDIUM, cv2.FILLED)
        cv2.addWeighted(overlay, 0.25, output, 0.75, 0, output)
        cv2.drawContours(output, [contour], -1, C_MEDIUM, 1)
        stats["medium"] += 1

    # Draw legend
    draw_legend(output)

    # Draw stats in bottom-left
    y = img_h - 10
    for key in reversed(list(stats.keys())):
        text = f"{key}: {stats[key]}"
        cv2.putText(output, text, (5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    (255, 255, 255), 1, cv2.LINE_AA)
        y -= 16

    return output


def main():
    parser = argparse.ArgumentParser(description="Debug rejection overlay for spray detection.")
    parser.add_argument("--dataset", default="dataset", help="Input frames directory.")
    parser.add_argument("--output", default="yolo_dataset/debug_rejections", help="Output directory.")
    parser.add_argument("--count", type=int, default=10, help="Number of random frames to process.")
    parser.add_argument("--frames", nargs="*", type=int, help="Specific frame numbers to process.")
    cli_args = parser.parse_args()

    dataset_dir = Path(cli_args.dataset)
    output_dir = Path(cli_args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build physics args and background
    physics_parser = sp.build_parser()
    args = physics_parser.parse_args([])

    frame_paths = sorted(dataset_dir.glob("frame*.png"), key=sp.frame_number)
    if not frame_paths:
        sys.exit(f"No frames in {dataset_dir}")

    background = sp.load_background(args, frame_paths)
    print(f"Background model loaded. Processing frames...")

    # Select frames
    if cli_args.frames:
        # Specific frame numbers
        frame_map = {sp.frame_number(p): p for p in frame_paths}
        selected = [frame_map[n] for n in cli_args.frames if n in frame_map]
    else:
        random.seed(99)
        selected = random.sample(frame_paths, min(cli_args.count, len(frame_paths)))

    for frame_path in selected:
        frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if frame is None:
            continue

        result = analyze_frame_with_rejections(frame, background, args)
        out_path = output_dir / f"debug_{frame_path.name}"
        cv2.imwrite(str(out_path), result)
        print(f"  Saved: {out_path.name}")

    print(f"\nDone! {len(selected)} debug images saved to {output_dir}/")


if __name__ == "__main__":
    main()
