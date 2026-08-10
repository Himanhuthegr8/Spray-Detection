"""
evaluate_model.py
=================
Evaluate a trained YOLOv8-seg model on the test split and produce:

  1. Per-class precision, recall, mAP@50, mAP@50:95
  2. A grid of sample predictions overlaid on test images
  3. A confusion matrix

Usage:
    python evaluate_model.py
    python evaluate_model.py --model runs/segment/spray_seg/weights/best.pt
    python evaluate_model.py --predict-samples 20   # save 20 sample predictions
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate a trained YOLOv8-seg model.")
    p.add_argument(
        "--model",
        default="runs/segment/spray_seg/weights/best.pt",
        help="Path to the trained model weights.",
    )
    p.add_argument(
        "--data",
        default="yolo_dataset/data.yaml",
        help="Path to data.yaml.",
    )
    p.add_argument(
        "--imgsz", type=int, default=512, help="Inference image size."
    )
    p.add_argument(
        "--conf", type=float, default=0.25, help="Confidence threshold for predictions."
    )
    p.add_argument(
        "--predict-samples",
        type=int,
        default=16,
        help="Number of test images to run prediction on and save.",
    )
    p.add_argument(
        "--output",
        default="evaluation_results",
        help="Directory to save evaluation outputs.",
    )
    return p


def run_validation(model: YOLO, data_yaml: str, imgsz: int) -> None:
    """Run YOLO validation on the test split and print metrics."""
    print("\n" + "─" * 60)
    print("  Running validation on TEST split...")
    print("─" * 60)

    results = model.val(
        data=data_yaml,
        split="test",
        imgsz=imgsz,
        plots=True,
        save_json=False,
        verbose=True,
    )

    print("\n" + "─" * 60)
    print("  Validation Metrics Summary")
    print("─" * 60)

    # Box metrics
    box = results.box
    print(f"\n  {'='*40}")
    print(f"  BOUNDING BOX METRICS")
    print(f"  {'='*40}")
    print(f"  mAP@50     : {box.map50:.4f}")
    print(f"  mAP@50:95  : {box.map:.4f}")
    if hasattr(box, 'mp') and hasattr(box, 'mr'):
        print(f"  Precision  : {box.mp:.4f}")
        print(f"  Recall     : {box.mr:.4f}")

    # Segmentation mask metrics
    seg = results.seg
    print(f"\n  {'='*40}")
    print(f"  SEGMENTATION MASK METRICS")
    print(f"  {'='*40}")
    print(f"  mAP@50     : {seg.map50:.4f}")
    print(f"  mAP@50:95  : {seg.map:.4f}")
    if hasattr(seg, 'mp') and hasattr(seg, 'mr'):
        print(f"  Precision  : {seg.mp:.4f}")
        print(f"  Recall     : {seg.mr:.4f}")

    # Per-class breakdown
    class_names = results.names if hasattr(results, 'names') else {
        0: "droplet", 1: "ligament", 2: "main_jet",
    }
    if hasattr(seg, 'maps') and seg.maps is not None:
        print(f"\n  {'='*40}")
        print(f"  PER-CLASS MASK mAP@50:95")
        print(f"  {'='*40}")
        for i, m in enumerate(seg.maps):
            name = class_names.get(i, f"class_{i}")
            print(f"    {name:20s} : {m:.4f}")


def run_predictions(
    model: YOLO,
    data_yaml: str,
    imgsz: int,
    conf: float,
    n_samples: int,
    output_dir: Path,
) -> None:
    """Run inference on a subset of test images and save visualisations."""
    print("\n" + "─" * 60)
    print(f"  Generating {n_samples} sample predictions...")
    print("─" * 60)

    # Find test images
    import yaml
    with open(data_yaml, "r") as f:
        cfg = yaml.safe_load(f)

    test_images_dir = Path(cfg["path"]) / cfg.get("test", "test/images")
    if not test_images_dir.exists():
        print(f"  Test images directory not found: {test_images_dir}")
        return

    image_paths = sorted(test_images_dir.glob("*.png"))
    if not image_paths:
        print("  No test images found.")
        return

    # Sample a subset
    import random
    random.seed(123)
    sample_paths = random.sample(image_paths, min(n_samples, len(image_paths)))

    # Run prediction
    predictions_dir = output_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    color_map = {
        0: (0, 200, 255),   # droplet = orange (BGR)
        1: (255, 0, 255),   # ligament = magenta (BGR)
        2: (255, 255, 0),   # main jet = cyan (BGR)
    }
    class_names = {0: "droplet", 1: "ligament", 2: "main_jet"}

    for img_path in sample_paths:
        results = model.predict(
            source=str(img_path),
            imgsz=imgsz,
            conf=conf,
            save=False,
            verbose=False,
        )

        result = results[0]
        img = cv2.imread(str(img_path))
        annotated = img.copy()

        # Draw segmentation masks
        if result.masks is not None and len(result.masks) > 0:
            masks = result.masks.data.cpu().numpy()      # (N, H, W)
            classes = result.boxes.cls.cpu().numpy().astype(int)
            confs = result.boxes.conf.cpu().numpy()

            for mask, cls, c in zip(masks, classes, confs):
                color = color_map.get(cls, (128, 128, 128))
                name = class_names.get(cls, f"cls{cls}")

                # Resize mask to image dimensions
                mask_resized = cv2.resize(
                    mask.astype(np.float32),
                    (img.shape[1], img.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
                mask_bool = mask_resized > 0.5

                # Semi-transparent overlay
                overlay = annotated.copy()
                overlay[mask_bool] = color
                annotated = cv2.addWeighted(annotated, 0.6, overlay, 0.4, 0)

                # Find contour for outline
                contours, _ = cv2.findContours(
                    mask_bool.astype(np.uint8),
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE,
                )
                cv2.drawContours(annotated, contours, -1, color, 1)

        # Save
        out_path = predictions_dir / f"pred_{img_path.name}"
        cv2.imwrite(str(out_path), annotated)

    print(f"  Saved {len(sample_paths)} prediction images to {predictions_dir}/")

    # Create a grid preview
    create_preview_grid(predictions_dir, output_dir)


def create_preview_grid(predictions_dir: Path, output_dir: Path) -> None:
    """Stitch predicted images into a grid for quick review."""
    images = sorted(predictions_dir.glob("pred_*.png"))[:16]
    if not images:
        return

    loaded = [cv2.imread(str(p)) for p in images]
    loaded = [img for img in loaded if img is not None]
    if not loaded:
        return

    # Resize all to same size
    target_h, target_w = 256, 256
    resized = [cv2.resize(img, (target_w, target_h)) for img in loaded]

    # Build grid (4 columns)
    cols = 4
    rows_needed = (len(resized) + cols - 1) // cols
    # Pad with black if needed
    while len(resized) < rows_needed * cols:
        resized.append(np.zeros((target_h, target_w, 3), dtype=np.uint8))

    grid_rows = []
    for r in range(rows_needed):
        row_imgs = resized[r * cols : (r + 1) * cols]
        grid_rows.append(np.hstack(row_imgs))
    grid = np.vstack(grid_rows)

    grid_path = output_dir / "prediction_grid.png"
    cv2.imwrite(str(grid_path), grid)
    print(f"  Saved prediction grid: {grid_path}")


def main() -> None:
    args = build_parser().parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(
            f"Model not found: {model_path}\n"
            "Run  python train_yolo.py  first."
        )

    data_yaml = Path(args.data)
    if not data_yaml.exists():
        raise SystemExit(f"data.yaml not found at {data_yaml}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  YOLOv8-Seg Model Evaluation")
    print("=" * 60)
    print(f"  Model   : {model_path}")
    print(f"  Data    : {data_yaml}")
    print(f"  Output  : {output_dir}")
    print("=" * 60)

    # Load model
    model = YOLO(str(model_path))

    # 1. Validation metrics
    run_validation(model, str(data_yaml.resolve()), args.imgsz)

    # 2. Sample predictions with overlays
    run_predictions(
        model,
        str(data_yaml),
        args.imgsz,
        args.conf,
        args.predict_samples,
        output_dir,
    )

    print("\n" + "=" * 60)
    print("  Evaluation Complete!")
    print("=" * 60)
    print(f"  Results saved to: {output_dir.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
