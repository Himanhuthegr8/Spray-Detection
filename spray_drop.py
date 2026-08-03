"""
spray_drop.py
=============

Physically-motivated droplet / ligament detection pipeline for backlit spray
imaging. Reads a folder of extracted frames (frame0000.png, frame0001.png, ...),
segments the liquid regions, extracts shape features for every region, tracks
regions across frames, classifies them as droplet / ligament / main_ligament /
uncertain, and writes:

    - an annotated .mp4 video (color-coded only, no text overlays)
    - detections.csv            (one row per tracked object per frame)
    - frame_summary.csv         (one row per frame: counts, areas, averages)
    - optional debug PNG mosaics (original | mask | annotated) per frame

Run it exactly like the original script, e.g.:

    python spray_drop.py --dataset dataset --output All_Images \
        --video spray_detection.mp4 --video-view annotated --no-save-frames

New capabilities (background correction, CLAHE, denoising, Otsu/adaptive
thresholding, morphology, watershed splitting, richer features, tracking,
temporal label smoothing, per-frame summaries, and a run-comparison mode) are
all optional / configurable via CLI flags with sensible defaults, so the
command above keeps working unchanged.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import math
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable

import cv2
import numpy as np

try:
    from skimage.morphology import skeletonize

    SKELETON_AVAILABLE = True
except ImportError:  # scikit-image is optional
    SKELETON_AVAILABLE = False


# --------------------------------------------------------------------------
# CLASSIFICATION DEFAULTS -- tune these first (also exposed as CLI flags).
#
#   droplet         : compact, round, solid  -> high circularity + solidity,
#                      low aspect ratio
#   ligament        : elongated, thin, may be bent (several convexity
#                      defects) -> high aspect ratio, low circularity
#   main_ligament   : the large core liquid body / jet -> area alone decides
#   uncertain       : doesn't clearly match either rule -> neutral color
# --------------------------------------------------------------------------
DEFAULT_DROPLET_MIN_CIRCULARITY = 0.70
DEFAULT_DROPLET_MIN_SOLIDITY = 0.85
DEFAULT_DROPLET_MAX_ASPECT = 1.8
DEFAULT_LIGAMENT_MIN_ASPECT = 1.8
DEFAULT_LIGAMENT_MAX_CIRCULARITY = 0.55
DEFAULT_LIGAMENT_MIN_DEFECTS = 1
DEFAULT_MAIN_AREA = 5000
DEFAULT_SMALL_DROPLET_MAX_PIXELS = 30

# Color-only visual encoding (BGR). No text is ever drawn on the video.
COLORS = {
    "droplet": (0, 200, 255),        # orange
    "ligament": (255, 0, 255),       # magenta
    "main_ligament": (255, 255, 0),  # cyan
    "uncertain": (160, 160, 160),    # neutral gray
}


# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------
@dataclass
class Detection:
    frame: int
    track_id: int
    label: str
    contour_index: int
    pixel_area: int
    area: float
    perimeter: float
    equiv_diameter: float
    bbox_x: int
    bbox_y: int
    bbox_w: int
    bbox_h: int
    rot_w: float
    rot_h: float
    rot_angle: float
    aspect_ratio: float
    circularity: float
    solidity: float
    eccentricity: float
    centroid_x: float
    centroid_y: float
    defect_count: int
    defect_depth_avg: float
    skeleton_length: float
    thickness_est: float
    velocity_x: float
    velocity_y: float
    speed: float


@dataclass
class FrameSummary:
    frame: int
    droplet_count: int
    ligament_count: int
    main_ligament_count: int
    uncertain_count: int
    main_region_area: float
    total_detected_area: float
    avg_droplet_diameter: float
    avg_ligament_aspect_ratio: float
    avg_ligament_length: float


@dataclass
class BackgroundModel:
    """Per-pixel bright background and its normal temporal variation."""

    image: np.ndarray
    noise: np.ndarray


# --------------------------------------------------------------------------
# Frame I/O helpers
# --------------------------------------------------------------------------
def frame_number(path: Path) -> int:
    digits = "".join(ch for ch in path.stem if ch.isdigit())
    return int(digits) if digits else -1


def iter_frames(dataset_dir: Path, start: int, end: int | None) -> Iterable[Path]:
    paths = sorted(dataset_dir.glob("frame*.png"), key=frame_number)
    for path in paths:
        number = frame_number(path)
        if number < start:
            continue
        if end is not None and number > end:
            continue
        yield path


def iter_image_paths(directory: Path) -> list[Path]:
    image_suffixes = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
    return sorted(
        (path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in image_suffixes),
        key=frame_number,
    )


def to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# --------------------------------------------------------------------------
# Background estimation
# --------------------------------------------------------------------------
def estimate_background(
    frame_paths: list[Path], n_samples: int, estimator: str, percentile: float,
    noise_floor: float,
) -> BackgroundModel | None:
    """Build a per-pixel static-scene and background-noise model.

    In a backlit recording the liquid is darker than the background. A high
    percentile therefore gives a better estimate of the clean bright scene
    than a median when a jet visits the same pixels in many frames. Median is
    retained for recordings where the foreground can be brighter than the
    background. The per-pixel noise estimate uses only the brightest quarter
    of samples, so dark liquid does not inflate the tolerated background
    variation.
    """
    if not frame_paths or n_samples <= 0:
        return None
    sample_count = min(len(frame_paths), n_samples)
    sample_indices = np.linspace(0, len(frame_paths) - 1, sample_count, dtype=int)
    sampled = [frame_paths[index] for index in sample_indices]
    stack = []
    for path in sampled:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            stack.append(img)
    if not stack:
        return None
    frames = np.stack(stack, axis=0)
    if estimator == "median":
        image = np.median(frames, axis=0).astype(np.uint8)
    else:
        image = np.percentile(frames, np.clip(percentile, 0.0, 100.0), axis=0).astype(np.uint8)

    # Liquid is dark in this experiment. The brightest samples are therefore
    # the best estimate of normal background fluctuation at each pixel.
    bright_count = max(1, int(math.ceil(len(frames) * 0.25)))
    bright_samples = np.partition(frames, len(frames) - bright_count, axis=0)[-bright_count:]
    noise = np.maximum(np.std(bright_samples.astype(np.float32), axis=0), max(noise_floor, 1e-6))
    return BackgroundModel(image=image, noise=noise.astype(np.float32))


def load_background(args: argparse.Namespace, frame_paths: list[Path]) -> BackgroundModel | None:
    if args.background_dir:
        background_paths = iter_image_paths(Path(args.background_dir))
        if not background_paths:
            raise SystemExit(f"No image frames found in background folder: {args.background_dir}")
        print(
            f"Building a noise-aware background model from up to {args.background_frames} "
            f"clean frames in {args.background_dir}..."
        )
        return estimate_background(
            background_paths,
            args.background_frames,
            args.background_estimator,
            args.background_percentile,
            args.background_noise_floor,
        )

    if args.background:
        bg_path = Path(args.background)
        if bg_path.is_dir():
            background_paths = iter_image_paths(bg_path)
            if not background_paths:
                raise SystemExit(f"No image frames found in background folder: {bg_path}")
            print(f"Building a noise-aware background model from {bg_path}...")
            return estimate_background(
                background_paths,
                args.background_frames,
                args.background_estimator,
                args.background_percentile,
                args.background_noise_floor,
            )
        background = cv2.imread(str(bg_path), cv2.IMREAD_GRAYSCALE)
        if background is not None:
            noise = np.full(background.shape, max(args.background_noise_floor, 1e-6), dtype=np.float32)
            return BackgroundModel(image=background, noise=noise)
        print(f"Warning: could not read background image {bg_path}; estimating from frames instead.")
    if args.background_frames > 0:
        print(
            f"Estimating a {args.background_estimator} background from up to "
            f"{args.background_frames} sampled frames..."
        )
        return estimate_background(
            frame_paths,
            args.background_frames,
            args.background_estimator,
            args.background_percentile,
            args.background_noise_floor,
        )
    return None


# --------------------------------------------------------------------------
# Preprocessing: denoise -> illumination-align -> background subtract -> contrast enhance
# --------------------------------------------------------------------------
def denoise_image(gray: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    if args.denoise == "median":
        ksize = args.median_ksize if args.median_ksize % 2 == 1 else args.median_ksize + 1
        return cv2.medianBlur(gray, max(3, ksize))
    if args.denoise == "bilateral":
        return cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)
    return gray


def apply_clahe(gray: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=args.clahe_clip, tileGridSize=(args.clahe_tile, args.clahe_tile))
    return clahe.apply(gray)


def align_background_brightness(
    gray: np.ndarray, background: np.ndarray, args: argparse.Namespace
) -> np.ndarray:
    """Correct a small frame-to-frame exposure shift before subtraction.

    The bright percentile is dominated by the backlight rather than dark
    droplets. Without this correction a few levels of lamp flicker can become
    foreground across a large part of the image.
    """
    if not args.background_normalize:
        return background
    q = args.background_align_percentile
    frame_level = float(np.percentile(gray, q))
    background_level = float(np.percentile(background, q))
    shift = np.clip(
        frame_level - background_level,
        -args.background_max_shift,
        args.background_max_shift,
    )
    aligned = background.astype(np.int16) + int(round(shift))
    return np.clip(aligned, 0, 255).astype(np.uint8)


def preprocess_frame(
    frame: np.ndarray, background: BackgroundModel | None, args: argparse.Namespace
) -> tuple[np.ndarray, bool, np.ndarray | None]:
    gray = to_gray(frame)
    denoised = denoise_image(gray, args)
    if background is not None:
        aligned_background = align_background_brightness(denoised, background.image, args)
        if args.foreground_polarity == "dark":
            # Backlit liquid is darker than the scene. Directional subtraction
            # rejects bright background fluctuations that absdiff would keep.
            diff = cv2.subtract(aligned_background, denoised)
        elif args.foreground_polarity == "light":
            diff = cv2.subtract(denoised, aligned_background)
        else:
            diff = cv2.absdiff(denoised, aligned_background)
        foreground_is_bright = True
        background_noise = background.noise
    else:
        diff = denoised
        foreground_is_bright = False
        background_noise = None

    # CLAHE can magnify one- or two-level subtraction residuals into false
    # foreground. Keep it for raw-image segmentation, but require an explicit
    # flag when a background model is already doing the contrast separation.
    if args.clahe and (background is None or args.clahe_after_background):
        diff = apply_clahe(diff, args)
    return diff, foreground_is_bright, background_noise


# --------------------------------------------------------------------------
# Segmentation: threshold -> conservative morphology -> optional watershed split
# --------------------------------------------------------------------------
def threshold_image(
    diff: np.ndarray,
    foreground_is_bright: bool,
    background_noise: np.ndarray | None,
    args: argparse.Namespace,
) -> np.ndarray:
    """Threshold the image according to the foreground intensity polarity.

    Otsu adapts to every frame, which is useful for changing sprays but can
    also choose a very low threshold in a quiet frame. For background-
    subtracted images, the contrast floor prevents weak residual texture from
    becoming foreground even when Otsu's choice fluctuates.
    """
    threshold_type = cv2.THRESH_BINARY if foreground_is_bright else cv2.THRESH_BINARY_INV
    if args.threshold_mode == "fixed":
        _, binary = cv2.threshold(diff, args.threshold, 255, threshold_type)
    elif args.threshold_mode == "adaptive":
        block = args.adaptive_block_size if args.adaptive_block_size % 2 == 1 else args.adaptive_block_size + 1
        block = max(3, block)
        binary = cv2.adaptiveThreshold(
            diff,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            threshold_type,
            block,
            args.adaptive_c,
        )
    else:  # otsu (default)
        _, binary = cv2.threshold(diff, 0, 255, threshold_type + cv2.THRESH_OTSU)

    if foreground_is_bright and args.min_foreground_contrast > 0:
        _, confident = cv2.threshold(
            diff,
            args.min_foreground_contrast - 1,
            255,
            cv2.THRESH_BINARY,
        )
        binary = cv2.bitwise_and(binary, confident)

    if foreground_is_bright and background_noise is not None and args.background_zscore > 0:
        noise = np.maximum(background_noise, max(args.background_noise_floor, 1e-6))
        z_score = diff.astype(np.float32) / noise
        binary[z_score < args.background_zscore] = 0
    return binary


def fill_holes(binary: np.ndarray) -> np.ndarray:
    """Classic flood-fill hole filling: flood the background from a corner,
    invert, and OR back into the original mask to close interior holes."""
    h, w = binary.shape[:2]
    flood = binary.copy()
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, mask, (0, 0), 255)
    flood_inv = cv2.bitwise_not(flood)
    return cv2.bitwise_or(binary, flood_inv)


def clean_mask(binary: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    if args.morph_open_ksize > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (args.morph_open_ksize,) * 2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    if args.morph_close_ksize > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (args.morph_close_ksize,) * 2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    if args.fill_holes:
        binary = fill_holes(binary)
    return binary


def watershed_split(binary: np.ndarray) -> np.ndarray:
    """Distance-transform + watershed splitting for touching/merged droplets."""
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    if dist.max() <= 0:
        return binary
    _, sure_fg = cv2.threshold(dist, 0.4 * dist.max(), 255, 0)
    sure_fg = sure_fg.astype(np.uint8)
    sure_bg = cv2.dilate(binary, np.ones((3, 3), np.uint8), iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    color_img = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(color_img, markers)
    result = np.zeros_like(binary)
    result[markers > 1] = 255
    return result


def segment_frame(frame: np.ndarray, background: BackgroundModel | None, args: argparse.Namespace) -> np.ndarray:
    diff, foreground_is_bright, background_noise = preprocess_frame(frame, background, args)
    binary = threshold_image(diff, foreground_is_bright, background_noise, args)
    binary = clean_mask(binary, args)
    if args.watershed:
        binary = watershed_split(binary)
    return binary


def find_contours(binary: np.ndarray, mode: int) -> tuple[list[np.ndarray], np.ndarray | None]:
    result = cv2.findContours(binary.copy(), mode, cv2.CHAIN_APPROX_SIMPLE)
    if len(result) == 2:
        contours, hierarchy = result
    else:
        _, contours, hierarchy = result
    return list(contours), hierarchy


def find_component_contours(binary: np.ndarray) -> list[tuple[np.ndarray, np.ndarray, int]]:
    """Return one outer contour per connected foreground component.

    RETR_EXTERNAL on the whole image hides a distinct droplet if it happens
    to sit inside the outline of a transparent sheet. Connected components
    keeps these physically separate pieces available for detection.
    """
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    components: list[tuple[np.ndarray, np.ndarray, int]] = []
    for label in range(1, count):
        pixel_area = int(stats[label, cv2.CC_STAT_AREA])
        component_mask = np.zeros_like(binary)
        component_mask[labels == label] = 255
        contours, _ = find_contours(component_mask, cv2.RETR_EXTERNAL)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        components.append((contour, component_mask, pixel_area))
    return components


# --------------------------------------------------------------------------
# Feature extraction
# --------------------------------------------------------------------------
def contour_centroid(contour: np.ndarray) -> tuple[float, float]:
    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        x, y, w, h = cv2.boundingRect(contour)
        return x + w / 2.0, y + h / 2.0
    return moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]


def convexity_defect_stats(contour: np.ndarray) -> tuple[int, float]:
    if len(contour) < 4:
        return 0, 0.0
    hull_idx = cv2.convexHull(contour, returnPoints=False)
    if hull_idx is None or len(hull_idx) < 3:
        return 0, 0.0
    hull_idx = np.sort(hull_idx, axis=0)  # convexityDefects needs monotonic indices
    try:
        defects = cv2.convexityDefects(contour, hull_idx)
    except cv2.error:
        return 0, 0.0
    if defects is None or len(defects) == 0:
        return 0, 0.0
    # Some OpenCV builds return shape (N, 1, 4), others (N, 4); reshape handles both.
    depths = defects.reshape(-1, 4)[:, 3] / 256.0
    return int(len(depths)), float(np.mean(depths))


def compute_skeleton_length(binary_mask: np.ndarray, x: int, y: int, w: int, h: int) -> float:
    if not SKELETON_AVAILABLE:
        return 0.0
    pad = 2
    y0, y1 = max(0, y - pad), min(binary_mask.shape[0], y + h + pad)
    x0, x1 = max(0, x - pad), min(binary_mask.shape[1], x + w + pad)
    crop = binary_mask[y0:y1, x0:x1] > 0
    if crop.sum() == 0:
        return 0.0
    skeleton = skeletonize(crop)
    return float(np.sum(skeleton))


def estimate_thickness(binary_mask: np.ndarray, x: int, y: int, w: int, h: int,
                        skeleton_length: float, equiv_diameter: float) -> float:
    crop = binary_mask[y:y + h, x:x + w]
    if crop.size == 0:
        return 0.0
    dist = cv2.distanceTransform((crop > 0).astype(np.uint8), cv2.DIST_L2, 5)
    max_dist = float(dist.max())
    if max_dist <= 0:
        return 0.0
    if skeleton_length > 0:
        # Roughly elongated: thickness ~ diameter at the widest skeleton point.
        return 2.0 * max_dist
    # Roughly compact: thickness ~ overall equivalent diameter.
    return equiv_diameter


def compute_features(contour: np.ndarray, binary_mask: np.ndarray, pixel_area: int) -> dict:
    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))
    equiv_diameter = float(np.sqrt(4.0 * area / np.pi)) if area > 0 else 0.0

    x, y, w, h = cv2.boundingRect(contour)
    (_, _), (rw, rh), rangle = cv2.minAreaRect(contour)
    short_side = max(min(rw, rh), 1e-6)
    long_side = max(rw, rh)
    aspect_ratio = float(long_side / short_side)

    circularity = float(4.0 * np.pi * area / (perimeter * perimeter)) if perimeter > 0 else 0.0

    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))
    solidity = float(area / hull_area) if hull_area > 0 else 0.0

    eccentricity = 0.0
    if len(contour) >= 5:
        try:
            (_, _), (ax1, ax2), _ = cv2.fitEllipse(contour)
            major, minor = max(ax1, ax2), min(ax1, ax2)
            if major > 0:
                eccentricity = float(np.sqrt(max(0.0, 1.0 - (minor / major) ** 2)))
        except cv2.error:
            eccentricity = 0.0

    cx, cy = contour_centroid(contour)
    defect_count, defect_depth_avg = convexity_defect_stats(contour)
    skeleton_length = compute_skeleton_length(binary_mask, x, y, w, h)
    thickness_est = estimate_thickness(binary_mask, x, y, w, h, skeleton_length, equiv_diameter)

    return {
        "pixel_area": pixel_area,
        "area": area,
        "perimeter": perimeter,
        "equiv_diameter": equiv_diameter,
        "bbox_x": x,
        "bbox_y": y,
        "bbox_w": w,
        "bbox_h": h,
        "rot_w": float(rw),
        "rot_h": float(rh),
        "rot_angle": float(rangle),
        "aspect_ratio": aspect_ratio,
        "circularity": circularity,
        "solidity": solidity,
        "eccentricity": eccentricity,
        "centroid_x": cx,
        "centroid_y": cy,
        "defect_count": defect_count,
        "defect_depth_avg": defect_depth_avg,
        "skeleton_length": skeleton_length,
        "thickness_est": thickness_est,
    }


# --------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------
def classify_contour(feats: dict, args: argparse.Namespace) -> str:
    if feats["area"] >= args.area:
        return "main_ligament"

    aspect = feats["aspect_ratio"]
    circ = feats["circularity"]
    solidity = feats["solidity"]
    defects = feats["defect_count"]

    is_elongated = aspect >= args.ligament_min_aspect and circ <= args.ligament_max_circularity
    is_bent_thread = defects >= args.ligament_min_defects and aspect >= args.ligament_min_aspect * 0.7
    if is_elongated or is_bent_thread:
        return "ligament"

    # At a diameter of only a few pixels, perimeter/circularity are dominated
    # by the square pixel grid. Use compactness instead of demanding the same
    # strict circularity required for resolved droplets.
    is_small_compact_object = (
        feats["pixel_area"] <= args.small_droplet_max_pixels
        and aspect <= args.small_droplet_max_aspect
    )
    if is_small_compact_object:
        return "droplet"

    is_round = (
        circ >= args.droplet_min_circularity
        and solidity >= args.droplet_min_solidity
        and aspect <= args.droplet_max_aspect
    )
    if is_round:
        return "droplet"

    # ---- Soft fallback: close the gap zone instead of discarding ----
    # Moderately elongated with low roundness → lean toward ligament.
    if aspect > 1.5 and circ < 0.60:
        return "ligament"
    # Compact but didn't pass the strict droplet test → lean toward droplet.
    if solidity > 0.80 and aspect < 2.0:
        return "droplet"

    return "uncertain"


# --------------------------------------------------------------------------
# Frame analysis: segment -> contours -> features -> classify
# --------------------------------------------------------------------------
def analyze_frame(frame: np.ndarray, frame_id: int, background: BackgroundModel | None,
                   args: argparse.Namespace) -> tuple[list[Detection], list[np.ndarray], np.ndarray]:
    binary = segment_frame(frame, background, args)
    components = find_component_contours(binary)

    detections: list[Detection] = []
    kept_contours: list[np.ndarray] = []
    for idx, (contour, component_mask, pixel_area) in enumerate(components):
        if pixel_area < args.min_area:
            continue
        # Features such as skeleton length and thickness must use only this
        # component, not every dark pixel that happens to lie in its bounding box.
        feats = compute_features(contour, component_mask, pixel_area)
        label = classify_contour(feats, args)
        detections.append(
            Detection(
                frame=frame_id,
                track_id=-1,
                label=label,
                contour_index=idx,
                velocity_x=0.0,
                velocity_y=0.0,
                speed=0.0,
                **feats,
            )
        )
        kept_contours.append(contour)

    return detections, kept_contours, binary


# --------------------------------------------------------------------------
# Temporal tracking: centroid distance + area-similarity gating,
# track IDs, velocity, and majority-vote label smoothing.
# --------------------------------------------------------------------------
@dataclass
class Track:
    track_id: int
    centroid: tuple[float, float]
    area: float
    label_history: deque
    velocity: tuple[float, float] = (0.0, 0.0)
    last_frame: int = -1
    missed: int = 0
    hits: int = 1


class Tracker:
    def __init__(self, max_distance: float, max_age: int, smooth_window: int, min_hits: int):
        self.max_distance = max_distance
        self.max_age = max_age
        self.smooth_window = max(1, smooth_window)
        self.min_hits = max(1, min_hits)
        self.tracks: dict[int, Track] = {}
        self.next_id = 1

    def update(self, detections: list[Detection], frame_id: int):
        """Simple greedy nearest-neighbor matching (adequate for moderate
        object counts; swap for the Hungarian algorithm if you need optimal
        assignment on dense frames)."""
        candidates = []
        for di, det in enumerate(detections):
            for tid, track in self.tracks.items():
                dist = math.hypot(det.centroid_x - track.centroid[0], det.centroid_y - track.centroid[1])
                if dist > self.max_distance:
                    continue
                area_ratio = max(det.area, track.area) / max(min(det.area, track.area), 1e-6)
                if area_ratio <= 3.0:
                    candidates.append((dist, di, tid))
        candidates.sort(key=lambda item: item[0])

        matched_det: set[int] = set()
        matched_track: set[int] = set()
        assignment: dict[int, int] = {}
        for dist, di, tid in candidates:
            if di in matched_det or tid in matched_track:
                continue
            matched_det.add(di)
            matched_track.add(tid)
            assignment[di] = tid

        results = []
        for di, det in enumerate(detections):
            if di in assignment:
                tid = assignment[di]
                track = self.tracks[tid]
                velocity = (det.centroid_x - track.centroid[0], det.centroid_y - track.centroid[1])
                track.centroid = (det.centroid_x, det.centroid_y)
                track.area = det.area
                track.velocity = velocity
                track.label_history.append(det.label)
                track.last_frame = frame_id
                track.missed = 0
                track.hits += 1
            else:
                tid = self.next_id
                self.next_id += 1
                track = Track(
                    track_id=tid,
                    centroid=(det.centroid_x, det.centroid_y),
                    area=det.area,
                    label_history=deque([det.label], maxlen=self.smooth_window),
                    velocity=(0.0, 0.0),
                    last_frame=frame_id,
                    missed=0,
                )
                self.tracks[tid] = track

            smoothed_label = Counter(track.label_history).most_common(1)[0][0]
            speed = math.hypot(*track.velocity)
            confirmed = track.hits >= self.min_hits
            results.append((det, tid, track.velocity, speed, smoothed_label, confirmed, track.hits))

        # Age out tracks that found no match this frame.
        for tid, track in list(self.tracks.items()):
            if track.last_frame != frame_id:
                track.missed += 1
                if track.missed > self.max_age:
                    del self.tracks[tid]

        return results


# --------------------------------------------------------------------------
# Drawing -- color only, no text anywhere on the video.
# --------------------------------------------------------------------------
def draw_detections(
    frame: np.ndarray,
    tracked_with_contours: list[tuple[Detection, np.ndarray]],
    fill_main_ligament: bool,
) -> np.ndarray:
    output = frame.copy()

    if fill_main_ligament:
        # This is an inferred liquid region, not direct dark-pixel evidence.
        # It is off by default because transparent sheets can enclose visible
        # background that should not be presented as a confident detection.
        fill_mask = np.zeros(frame.shape[:2], np.uint8)
        for det, contour in tracked_with_contours:
            if det.label == "main_ligament":
                cv2.drawContours(fill_mask, [contour], -1, 255, thickness=cv2.FILLED)

        if fill_mask.any():
            colored = np.zeros_like(frame)
            colored[:] = COLORS["main_ligament"]
            blended = cv2.addWeighted(frame, 0.65, colored, 0.35, 0)
            output[fill_mask == 255] = blended[fill_mask == 255]

    for det, contour in tracked_with_contours:
        color = COLORS.get(det.label, COLORS["uncertain"])
        thickness = 2 if det.label != "main_ligament" else 1
        cv2.drawContours(output, [contour], -1, color, thickness)
        if det.label in ("ligament", "droplet"):
            box = cv2.boxPoints(cv2.minAreaRect(contour))
            box = np.intp(box)
            cv2.polylines(output, [box], True, color, 1)

    return output


def side_by_side(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if left.shape != right.shape:
        right = cv2.resize(right, (left.shape[1], left.shape[0]))
    return np.hstack([left, right])


def debug_mosaic(frame: np.ndarray, binary: np.ndarray, annotated: np.ndarray) -> np.ndarray:
    mask_bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    panels = [frame, mask_bgr, annotated]
    h, w = frame.shape[:2]
    prepared = []
    for panel in panels:
        if panel.shape[:2] != (h, w):
            panel = cv2.resize(panel, (w, h))
        panel = panel.copy()
        cv2.rectangle(panel, (0, 0), (w - 1, h - 1), (255, 255, 255), 1)
        prepared.append(panel)
    return np.hstack(prepared)


# --------------------------------------------------------------------------
# Per-frame summary
# --------------------------------------------------------------------------
def build_frame_summary(frame_id: int, detections: list[Detection]) -> FrameSummary:
    droplets = [d for d in detections if d.label == "droplet"]
    ligaments = [d for d in detections if d.label == "ligament"]
    mains = [d for d in detections if d.label == "main_ligament"]
    uncertain = [d for d in detections if d.label == "uncertain"]

    return FrameSummary(
        frame=frame_id,
        droplet_count=len(droplets),
        ligament_count=len(ligaments),
        main_ligament_count=len(mains),
        uncertain_count=len(uncertain),
        main_region_area=sum(d.area for d in mains),
        total_detected_area=sum(d.area for d in detections),
        avg_droplet_diameter=mean([d.equiv_diameter for d in droplets]) if droplets else 0.0,
        avg_ligament_aspect_ratio=mean([d.aspect_ratio for d in ligaments]) if ligaments else 0.0,
        avg_ligament_length=mean([d.skeleton_length for d in ligaments]) if ligaments else 0.0,
    )


# --------------------------------------------------------------------------
# CSV output
# --------------------------------------------------------------------------
def save_detections(path: Path, detections: list[Detection]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(Detection.__dataclass_fields__.keys()))
        writer.writeheader()
        for det in detections:
            writer.writerow(dataclasses.asdict(det))


def save_frame_summaries(path: Path, summaries: list[FrameSummary]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(FrameSummary.__dataclass_fields__.keys()))
        writer.writeheader()
        for summary in summaries:
            writer.writerow(dataclasses.asdict(summary))


# --------------------------------------------------------------------------
# Comparing two prior runs (no ground truth needed -- see explanation).
# --------------------------------------------------------------------------
def compare_runs(summary_a: Path, summary_b: Path) -> None:
    def load(path: Path) -> dict[int, dict]:
        rows: dict[int, dict] = {}
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                rows[int(row["frame"])] = row
        return rows

    rows_a = load(summary_a)
    rows_b = load(summary_b)
    common_frames = sorted(set(rows_a) & set(rows_b))
    if not common_frames:
        print("No overlapping frames between the two summary files.")
        return

    print(f"Comparing {len(common_frames)} overlapping frames:\n  A: {summary_a}\n  B: {summary_b}")
    fields = ["droplet_count", "ligament_count", "main_region_area", "total_detected_area"]
    for field in fields:
        diffs = [float(rows_b[f][field]) - float(rows_a[f][field]) for f in common_frames]
        print(f"  {field}: mean diff (B-A) = {mean(diffs):.3f}, mean |diff| = {mean(abs(d) for d in diffs):.3f}")


# --------------------------------------------------------------------------
# Video writer helper
# --------------------------------------------------------------------------
def make_video_writer(path: Path, fps: float, frame: np.ndarray) -> cv2.VideoWriter:
    height, width = frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create video writer for {path}")
    return writer


# --------------------------------------------------------------------------
# Main processing loop
# --------------------------------------------------------------------------
def process_frames(args: argparse.Namespace) -> None:
    dataset_dir = Path(args.dataset)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_paths = list(iter_frames(dataset_dir, args.start, args.end))
    if not frame_paths:
        raise SystemExit(f"No frames found in {dataset_dir}")

    background = load_background(args, frame_paths)
    if background is not None:
        reference = cv2.imread(str(frame_paths[0]), cv2.IMREAD_GRAYSCALE)
        if reference is None:
            raise SystemExit(f"Could not read first dataset frame: {frame_paths[0]}")
        if background.image.shape != reference.shape:
            raise SystemExit(
                "Background and dataset frame dimensions do not match: "
                f"{background.image.shape} vs {reference.shape}"
            )
        background_path = output_dir / "background_model.png"
        noise_path = output_dir / "background_noise.png"
        noise_display = cv2.normalize(background.noise, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        cv2.imwrite(str(background_path), background.image)
        cv2.imwrite(str(noise_path), noise_display)
        print(f"Saved background model: {background_path}")
        print(f"Saved background noise map: {noise_path}")

    tracker = Tracker(
        args.max_track_distance,
        args.track_max_age,
        args.classify_smooth_window,
        args.min_track_hits,
    )

    all_detections: list[Detection] = []
    all_summaries: list[FrameSummary] = []
    video_writer: cv2.VideoWriter | None = None
    video_path = output_dir / args.video

    for frame_path in frame_paths:
        frame_id = frame_number(frame_path)
        frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if frame is None:
            print(f"Skipping unreadable image: {frame_path}")
            continue

        detections, contours, binary = analyze_frame(frame, frame_id, background, args)
        tracked = tracker.update(detections, frame_id)

        final_detections: list[Detection] = []
        visible_pairs: list[tuple[Detection, np.ndarray]] = []
        for tracked_detection, contour in zip(tracked, contours):
            det, tid, velocity, speed, smoothed_label, confirmed, track_hits = tracked_detection
            final_det = dataclasses.replace(
                det,
                track_id=tid,
                label=smoothed_label,
                velocity_x=velocity[0],
                velocity_y=velocity[1],
                speed=speed,
            )
            # Fine droplets are deliberately detected with little or no
            # erosion. Reject only the ones that subsequently prove static,
            # which is the usual signature of residual background texture.
            is_static_small_object = (
                final_det.pixel_area <= args.small_droplet_max_pixels
                and track_hits >= 2
                and speed < args.min_small_droplet_speed
            )
            if is_static_small_object:
                continue
            if confirmed or args.draw_tentative:
                final_detections.append(final_det)
                visible_pairs.append((final_det, contour))

        annotated = draw_detections(frame, visible_pairs, args.fill_main_ligament)

        all_detections.extend(final_detections)
        all_summaries.append(build_frame_summary(frame_id, final_detections))

        if args.video_view == "side_by_side":
            video_frame = side_by_side(frame, annotated)
        else:
            video_frame = annotated

        if not args.no_save_frames:
            out_path = output_dir / f"seg_{frame_id:04d}.png"
            cv2.imwrite(str(out_path), debug_mosaic(frame, binary, annotated))

        if video_writer is None:
            video_writer = make_video_writer(video_path, args.fps, video_frame)
        video_writer.write(video_frame)

        if args.show:
            cv2.imshow("Result", video_frame)
            if cv2.waitKey(1) == 27:
                break

        print(
            f"frame={frame_id:04d} droplets={all_summaries[-1].droplet_count} "
            f"ligaments={all_summaries[-1].ligament_count} main={all_summaries[-1].main_ligament_count}"
        )

    save_detections(output_dir / "detections.csv", all_detections)
    save_frame_summaries(output_dir / "frame_summary.csv", all_summaries)

    if video_writer is not None:
        video_writer.release()
        print(f"Saved video: {video_path}")
    if args.show:
        cv2.destroyAllWindows()

    print(f"Saved detections: {output_dir / 'detections.csv'}")
    print(f"Saved frame summary: {output_dir / 'frame_summary.csv'}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Detect spray droplets and ligaments from extracted frames.")

    # Original I/O flags (kept for backward compatibility).
    parser.add_argument("--dataset", default="dataset", help="Folder containing frame0000.png style images.")
    parser.add_argument("--output", default="All_Images", help="Folder for detections.csv, summaries, and video.")
    parser.add_argument("--start", type=int, default=0, help="First frame number to process.")
    parser.add_argument("--end", type=int, default=None, help="Last frame number to process.")
    parser.add_argument("--video", default="spray_detection.mp4", help="Output video filename inside the output folder.")
    parser.add_argument("--fps", type=float, default=30.0, help="Frames per second for the output video.")
    parser.add_argument(
        "--video-view",
        choices=("annotated", "side_by_side"),
        default="annotated",
        help="annotated: color-coded overlay only. side_by_side: original frame next to the overlay.",
    )
    parser.add_argument("--no-save-frames", action="store_true", help="Only save video and CSVs, not per-frame debug PNGs.")
    parser.add_argument("--show", action="store_true", help="Display the live annotated frame while processing.")

    # Background correction. For normal backlit spray, liquid is darker than
    # the scene, so the high-percentile model and dark polarity are defaults.
    parser.add_argument(
        "--background",
        default=None,
        help="Path to one clean background image, or a folder of clean background frames.",
    )
    parser.add_argument(
        "--background-dir",
        default=None,
        help="Folder of clean background frames. Takes priority over --background and is the recommended calibration input.",
    )
    parser.add_argument(
        "--background-frames", type=int, default=80,
        help="Frames sampled to estimate a background when --background is not given. 0 disables background subtraction.",
    )
    parser.add_argument(
        "--background-estimator", choices=("percentile", "median"), default="percentile",
        help="percentile keeps a bright clean background despite recurring dark spray; median is the legacy option.",
    )
    parser.add_argument(
        "--background-percentile", type=float, default=99.0,
        help="Bright percentile (0-100) used by the percentile background estimator. Higher values reject recurring dark droplets.",
    )
    parser.add_argument(
        "--foreground-polarity", choices=("dark", "light", "absolute"), default="dark",
        help="Expected liquid intensity relative to background. dark is correct for the supplied backlit data.",
    )
    parser.add_argument("--background-normalize", dest="background_normalize", action="store_true", default=True,
                        help="Match small global brightness changes before background subtraction (default on).")
    parser.add_argument("--no-background-normalize", dest="background_normalize", action="store_false",
                        help="Disable brightness matching before background subtraction.")
    parser.add_argument("--background-align-percentile", type=float, default=90.0,
                        help="Bright percentile used to estimate the per-frame brightness shift.")
    parser.add_argument("--background-max-shift", type=int, default=25,
                        help="Largest allowed brightness correction in grayscale levels.")
    parser.add_argument(
        "--background-noise-floor", type=float, default=4.0,
        help="Minimum normal background variation in grayscale levels. Prevents unrealistically confident decisions.",
    )
    parser.add_argument(
        "--background-zscore", type=float, default=3.5,
        help="Per-pixel signal-to-background-noise ratio required for liquid. Raise to reject more background; 0 disables this test.",
    )

    # Preprocessing.
    parser.add_argument("--denoise", choices=("median", "bilateral", "none"), default="median")
    parser.add_argument("--median-ksize", type=int, default=5, help="Kernel size for median blur (odd number).")
    parser.add_argument("--clahe", dest="clahe", action="store_true", default=True, help="Enable CLAHE contrast enhancement (default on).")
    parser.add_argument("--no-clahe", dest="clahe", action="store_false", help="Disable CLAHE.")
    parser.add_argument("--clahe-after-background", action="store_true",
                        help="Also apply CLAHE after background subtraction. Off by default to avoid amplifying residue.")
    parser.add_argument("--clahe-clip", type=float, default=2.0)
    parser.add_argument("--clahe-tile", type=int, default=8)

    # Segmentation.
    parser.add_argument("--threshold-mode", choices=("otsu", "adaptive", "fixed"), default="otsu")
    parser.add_argument("--threshold", type=int, default=120, help="Fixed binary threshold, used when --threshold-mode fixed.")
    parser.add_argument(
        "--min-foreground-contrast", type=int, default=20,
        help="Minimum background-subtracted intensity required for a liquid pixel. Raise this to reject more background; lower it to retain faint liquid.",
    )
    parser.add_argument("--adaptive-block-size", type=int, default=35)
    parser.add_argument("--adaptive-c", type=float, default=5.0)

    # Morphology.
    parser.add_argument("--morph-open-ksize", type=int, default=0, help="0 disables opening. Default preserves thin ligaments and tiny droplets.")
    parser.add_argument("--morph-close-ksize", type=int, default=3, help="0 disables closing; keep small to avoid bridging nearby ligaments.")
    parser.add_argument(
        "--fill-holes", dest="fill_holes", action="store_true", default=False,
        help="Fill enclosed transparent interiors in the evidence mask. Off by default to avoid treating transparent sheets as solid foreground.",
    )
    parser.add_argument("--no-fill-holes", dest="fill_holes", action="store_false")
    parser.add_argument("--watershed", action="store_true", help="Split touching droplets via distance-transform watershed.")

    # Contour filtering / classification.
    parser.add_argument("--min-area", type=float, default=3.0, help="Foreground pixels required for a component. Temporal motion filtering rejects static specks.")
    parser.add_argument("--area", type=int, default=5000, help="Area threshold for the main/core ligament body.")
    parser.add_argument(
        "--small-droplet-max-pixels", type=int, default=DEFAULT_SMALL_DROPLET_MAX_PIXELS,
        help="Largest foreground-pixel area treated as an unresolved satellite-droplet candidate.",
    )
    parser.add_argument(
        "--small-droplet-max-aspect", type=float, default=2.5,
        help="Maximum aspect ratio for an unresolved compact satellite droplet.",
    )
    parser.add_argument("--droplet-min-circularity", type=float, default=DEFAULT_DROPLET_MIN_CIRCULARITY)
    parser.add_argument("--droplet-min-solidity", type=float, default=DEFAULT_DROPLET_MIN_SOLIDITY)
    parser.add_argument("--droplet-max-aspect", type=float, default=DEFAULT_DROPLET_MAX_ASPECT)
    parser.add_argument("--ligament-min-aspect", type=float, default=DEFAULT_LIGAMENT_MIN_ASPECT)
    parser.add_argument("--ligament-max-circularity", type=float, default=DEFAULT_LIGAMENT_MAX_CIRCULARITY)
    parser.add_argument("--ligament-min-defects", type=int, default=DEFAULT_LIGAMENT_MIN_DEFECTS)

    # Tracking / temporal smoothing.
    parser.add_argument("--max-track-distance", type=float, default=40.0, help="Max centroid displacement (px) allowed between frames for the same track.")
    parser.add_argument("--track-max-age", type=int, default=5, help="Frames a track may go unmatched before it is dropped.")
    parser.add_argument("--classify-smooth-window", type=int, default=3, help="Frames of label history used for majority-vote smoothing.")
    parser.add_argument(
        "--min-small-droplet-speed", type=float, default=0.5,
        help="Minimum motion (pixels/frame) for a tracked tiny object. Removes static background specks; 0 disables this filter.",
    )
    parser.add_argument(
        "--min-track-hits", type=int, default=2,
        help="Track observations required before an object is drawn or saved. 2 removes one-frame background flicker; use 1 to disable.",
    )
    parser.add_argument(
        "--draw-tentative", action="store_true",
        help="Draw and save first-frame, unconfirmed detections. Off by default for a cleaner result.",
    )
    parser.add_argument(
        "--fill-main-ligament", action="store_true",
        help="Fill the interior of a main-ligament contour. Off by default; outlines better represent transparent sheets.",
    )

    # Compare two previous runs (skips detection entirely when both are set).
    parser.add_argument("--compare-a", default=None, help="frame_summary.csv from a previous run.")
    parser.add_argument("--compare-b", default=None, help="frame_summary.csv from another run, to compare against --compare-a.")

    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.compare_a and args.compare_b:
        compare_runs(Path(args.compare_a), Path(args.compare_b))
        return
    process_frames(args)


if __name__ == "__main__":
    main()
