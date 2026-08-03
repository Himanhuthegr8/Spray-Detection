import csv
from pathlib import Path

rows = list(csv.DictReader(open("All_Images/detections.csv")))
print(f"Total detections: {len(rows)}")

labels = {}
for r in rows:
    labels[r["label"]] = labels.get(r["label"], 0) + 1
print("Label distribution:")
for k, v in sorted(labels.items(), key=lambda x: -x[1]):
    print(f"  {k}: {v}")

frames = set(int(r["frame"]) for r in rows)
print(f"Frame range: {min(frames)} - {max(frames)} ({len(frames)} frames)")

dataset_frames = len(list(Path("dataset").glob("frame*.png")))
seg_images = len(list(Path("All_Images").glob("seg_*.png")))
print(f"Total frames in dataset folder: {dataset_frames}")
print(f"Total seg images: {seg_images}")

# Check image dimensions
import cv2
img = cv2.imread(str(list(Path("dataset").glob("frame*.png"))[0]))
print(f"Image dimensions: {img.shape}")
