"""
infer_video.py
==============
Run the trained YOLO segmentation model on Drop.avi, frame by frame.

Each annotated frame is saved to disk immediately after processing so
that RAM/VRAM usage stays flat regardless of video length.
All frames are then stitched into a final output video.

Usage:
    python infer_video.py
    python infer_video.py --model runs/segment/spray_seg/weights/best.pt
    python infer_video.py --source Drop.avi --output output_video.avi --conf 0.25
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# -----------------------------------------------------------------------
# Color palette — BGR format (OpenCV convention)
# Class 0 = droplet  (orange)
# Class 1 = ligament (magenta)
# Class 2 = main_jet (cyan)
# -----------------------------------------------------------------------
COLOR_MAP = {
    0: (0, 200, 255),    # droplet  -> orange
    1: (255, 0, 255),    # ligament -> magenta
    2: (255, 255, 0),    # main_jet -> cyan
}
MASK_ALPHA = 0.45        # transparency of filled masks


def draw_masks_only(frame: np.ndarray, result) -> np.ndarray:
    """Draw ONLY colored segmentation masks on the frame (no boxes, no text)."""
    annotated = frame.copy()

    if result.masks is None or len(result.masks) == 0:
        return annotated

    masks   = result.masks.data.cpu().numpy()          # (N, H, W) float32
    classes = result.boxes.cls.cpu().numpy().astype(int)

    h, w = frame.shape[:2]

    for mask, cls in zip(masks, classes):
        color = COLOR_MAP.get(int(cls), (128, 128, 128))

        # Resize mask from model resolution back to original frame size
        mask_resized = cv2.resize(
            mask.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR
        )
        mask_bool = mask_resized > 0.5

        # Semi-transparent fill
        overlay = annotated.copy()
        overlay[mask_bool] = color
        annotated = cv2.addWeighted(annotated, 1 - MASK_ALPHA, overlay, MASK_ALPHA, 0)

        # Solid contour outline
        contours, _ = cv2.findContours(
            mask_bool.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(annotated, contours, -1, color, 1)

    return annotated


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Frame-by-frame YOLO inference on a spray video.")
    p.add_argument("--model",  default="runs/segment/spray_seg/weights/best.pt",
                   help="Path to trained model weights.")
    p.add_argument("--source", default="Drop.avi",
                   help="Input video file.")
    p.add_argument("--output", default="annotated_spray.avi",
                   help="Output annotated video filename.")
    p.add_argument("--frames-dir", default="annotated_frames",
                   help="Temporary directory for individual annotated frames.")
    p.add_argument("--conf",   type=float, default=0.25,
                   help="Detection confidence threshold.")
    p.add_argument("--imgsz",  type=int,   default=480,
                   help="Inference image size (must match training size).")
    p.add_argument("--keep-frames", action="store_true",
                   help="Keep individual frame PNGs after video is assembled.")
    return p


def main() -> None:
    args = build_parser().parse_args()

    source_path = Path(args.source)
    output_path = Path(args.output)
    frames_dir  = Path(args.frames_dir)

    if not source_path.exists():
        raise SystemExit(f"[ERROR] Source video not found: {source_path}")
    if not Path(args.model).exists():
        raise SystemExit(f"[ERROR] Model weights not found: {args.model}")

    # Create fresh temp directory for frames
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True)

    # ------------------------------------------------------------------
    # Step 1: Load model
    # ------------------------------------------------------------------
    print(f"Loading model: {args.model}")
    model = YOLO(args.model)

    # ------------------------------------------------------------------
    # Step 2: Open video and get properties
    # ------------------------------------------------------------------
    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        raise SystemExit(f"[ERROR] Cannot open video: {source_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
    orig_w       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    print(f"Video:  {source_path.name}  ({total_frames} frames, {fps:.1f} fps, {orig_w}x{orig_h})")
    print(f"Output: {output_path}")
    print(f"Frames: {frames_dir}/")
    print()

    # ------------------------------------------------------------------
    # Step 3: Infer frame by frame and save each frame immediately
    # ------------------------------------------------------------------
    print("Running inference frame by frame...")
    frame_paths: list[Path] = []

    for frame_idx, result in enumerate(
        model.predict(
            source=str(source_path),
            stream=True,           # process one frame at a time -- flat memory usage
            half=True,             # FP16 to save VRAM
            conf=args.conf,
            imgsz=args.imgsz,
            verbose=False,
        )
    ):
        # Get the original frame from the result
        orig_frame = result.orig_img   # numpy BGR array at full resolution

        # Draw masks (no boxes, no text)
        annotated = draw_masks_only(orig_frame, result)

        # Save frame immediately to disk
        frame_filename = frames_dir / f"frame_{frame_idx:06d}.png"
        cv2.imwrite(str(frame_filename), annotated)
        frame_paths.append(frame_filename)

        # Progress every 100 frames
        if (frame_idx + 1) % 100 == 0 or (frame_idx + 1) == total_frames:
            pct = 100.0 * (frame_idx + 1) / max(total_frames, 1)
            print(f"  [{frame_idx + 1:4d}/{total_frames}]  {pct:.1f}%")

    print(f"\nAll {len(frame_paths)} frames saved to {frames_dir}/")

    # ------------------------------------------------------------------
    # Step 4: Stitch frames into output video
    # ------------------------------------------------------------------
    print(f"\nAssembling video: {output_path}")

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (orig_w, orig_h))

    if not writer.isOpened():
        raise SystemExit("[ERROR] Could not open VideoWriter. Check codec availability.")

    for i, fp in enumerate(sorted(frame_paths)):
        img = cv2.imread(str(fp))
        if img is None:
            print(f"  [WARN] Could not read {fp}, skipping.")
            continue
        writer.write(img)
        if (i + 1) % 200 == 0:
            print(f"  Written {i + 1}/{len(frame_paths)} frames...")

    writer.release()
    print(f"\nDone! Output video saved to: {output_path.resolve()}")

    # ------------------------------------------------------------------
    # Step 5: Clean up frame PNGs (unless --keep-frames is set)
    # ------------------------------------------------------------------
    if not args.keep_frames:
        shutil.rmtree(frames_dir)
        print(f"  Temporary frames directory removed: {frames_dir}")
    else:
        print(f"  Individual frames kept in: {frames_dir.resolve()}")

    print("\nLegend:")
    print("  Orange  = droplet")
    print("  Magenta = ligament")
    print("  Cyan    = main jet")


if __name__ == "__main__":
    main()
