"""
train_yolo.py
=============
Train a YOLO11 instance-segmentation model on the prepared spray dataset.

Usage:
    python train_yolo.py                                    # defaults (yolo11m-seg, batch=8, imgsz=640)
    python train_yolo.py --batch 4                          # even lower VRAM (~4GB GPU)
    python train_yolo.py --model yolo11s-seg.pt --batch 4   # small model, extremely fast & low VRAM
    python train_yolo.py --model yolo11n-seg.pt --batch 2   # nano model (ultra low VRAM)

The trained model is saved under  runs/segment/spray_*  inside this directory.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train YOLO11-seg on the spray droplet / ligament / main_jet dataset."
    )
    p.add_argument(
        "--model",
        default="yolo11m-seg.pt",
        help="Pre-trained YOLO model checkpoint (default: yolo11m-seg.pt = medium, memory efficient).",
    )
    p.add_argument(
        "--data",
        default="yolo_dataset_v3/data.yaml",
        help="Path to data.yaml created by prepare_yolo_dataset.py.",
    )
    p.add_argument("--epochs", type=int, default=80, help="Maximum training epochs.")
    p.add_argument("--batch", type=int, default=8, help="Batch size (default: 8, lower to 4 or 2 if VRAM is low).")
    p.add_argument("--imgsz", type=int, default=640, help="Training image size.")
    p.add_argument("--patience", type=int, default=20, help="Early-stopping patience.")
    p.add_argument("--workers", type=int, default=4, help="Dataloader workers (set to 8 for max CPU throughput).")
    p.add_argument("--mask-ratio", type=int, default=4, help="Mask downsample ratio (default: 4 saves VRAM).")
    p.add_argument("--cache", action="store_true", help="Cache dataset in RAM to eliminate disk I/O bottlenecks and maximize GPU speed.")
    p.add_argument("--name", default="spray_seg_titan", help="Run name (under runs/segment/).")
    p.add_argument("--resume", action="store_true", help="Resume from last checkpoint.")
    return p


def main() -> None:
    args = build_parser().parse_args()

    data_yaml = Path(args.data)
    if not data_yaml.exists():
        raise SystemExit(
            f"data.yaml not found at {data_yaml}.\n"
            "Run  python prepare_yolo_dataset.py --output yolo_dataset_v3  first."
        )

    # Free memory in PyTorch CUDA cache before initializing training
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("=" * 60)
    print("  YOLO11 Instance-Segmentation Training")
    print("=" * 60)
    print(f"  Model      : {args.model}")
    print(f"  Data       : {data_yaml.resolve()}")
    print(f"  Epochs     : {args.epochs}")
    print(f"  Batch size : {args.batch}")
    print(f"  Image size : {args.imgsz}")
    print(f"  Mask ratio : {args.mask_ratio}")
    print(f"  RAM Cache  : {args.cache}")
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
        amp=True,              # Automatic Mixed Precision (FP16) - significantly reduces VRAM footprint
        cache=True if args.cache else False,  # RAM cache eliminates disk read stalls
        # ---- Mask settings ----
        mask_ratio=args.mask_ratio,  # default 4 (standard resolution, lower VRAM)
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
