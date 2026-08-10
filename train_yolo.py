"""
train_yolo.py
=============
Train a YOLOv8 instance-segmentation model on the prepared spray dataset.

Usage:
    python train_yolo.py                          # defaults (yolov8n-seg, 100 epochs)
    python train_yolo.py --model yolov8s-seg      # larger model
    python train_yolo.py --epochs 50 --batch 8    # custom training

The trained model is saved under  runs/segment/spray_*  inside this directory.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train YOLOv8-seg on the spray droplet / ligament dataset."
    )
    p.add_argument(
        "--model",
        default="yolov8s-seg.pt",
        help="Pre-trained YOLO model checkpoint (default: yolov8s-seg.pt = small).",
    )
    p.add_argument(
        "--data",
        default="yolo_dataset/data.yaml",
        help="Path to data.yaml created by prepare_yolo_dataset.py.",
    )
    p.add_argument("--epochs", type=int, default=100, help="Maximum training epochs.")
    p.add_argument("--batch", type=int, default=8, help="Batch size.")
    p.add_argument("--imgsz", type=int, default=640, help="Training image size.")
    p.add_argument("--patience", type=int, default=20, help="Early-stopping patience.")
    p.add_argument("--workers", type=int, default=4, help="Dataloader workers.")
    p.add_argument("--name", default="spray_seg", help="Run name (under runs/segment/).")
    p.add_argument("--resume", action="store_true", help="Resume from last checkpoint.")
    return p


def main() -> None:
    args = build_parser().parse_args()

    data_yaml = Path(args.data)
    if not data_yaml.exists():
        raise SystemExit(
            f"data.yaml not found at {data_yaml}.\n"
            "Run  python prepare_yolo_dataset.py  first."
        )

    print("=" * 60)
    print("  YOLOv8 Instance-Segmentation Training")
    print("=" * 60)
    print(f"  Model      : {args.model}")
    print(f"  Data       : {data_yaml.resolve()}")
    print(f"  Epochs     : {args.epochs}")
    print(f"  Batch size : {args.batch}")
    print(f"  Image size : {args.imgsz}")
    print(f"  Patience   : {args.patience}")
    print(f"  Run name   : {args.name}")
    print("=" * 60 + "\n")

    # Load model (downloads pre-trained weights on first run)
    model = YOLO(args.model)

    # Train
    results = model.train(
        data=str(data_yaml.resolve()),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        patience=args.patience,
        workers=args.workers,
        name=args.name,
        project=str(Path("runs/segment").resolve()),
        exist_ok=True,
        # ---- Mask settings (better for thin ligaments) ----
        mask_ratio=4,          # default — mask_ratio=2 requires too much VRAM for 6GB GPU
        overlap_mask=False,    # separate mask per instance — avoids merging ligaments
        # ---- Augmentation (tuned for spray + class imbalance) ----
        flipud=0.5,        # vertical flip (spray orientation varies)
        fliplr=0.5,        # horizontal flip
        mosaic=0.5,        # reduced from 1.0 — mosaic can fragment long ligaments
        scale=0.5,         # random scale ±50%
        hsv_h=0.015,       # hue shift (subtle — grayscale-ish images)
        hsv_s=0.3,         # saturation (minimal effect on backlit spray)
        hsv_v=0.4,         # value/brightness shift (helps with exposure variation)
        translate=0.1,     # random translation
        degrees=10.0,      # slight rotation
        copy_paste=0.3,    # paste instances to combat 4:1 class imbalance
        # ---- Logging / saving ----
        save=True,
        save_period=10,     # save checkpoint every 10 epochs
        plots=True,         # generate training plots
        verbose=True,
    )

    # Print final results
    print("\n" + "=" * 60)
    print("  Training Complete!")
    print("=" * 60)

    best_model = Path("runs/segment") / args.name / "weights" / "best.pt"
    last_model = Path("runs/segment") / args.name / "weights" / "last.pt"
    print(f"  Best model : {best_model.resolve()}")
    print(f"  Last model : {last_model.resolve()}")
    print(f"\n  To evaluate, run:")
    print(f"    python evaluate_model.py --model {best_model}")
    print("=" * 60)


if __name__ == "__main__":
    main()
