"""
spray_physics.py
================

Physics-motivated droplet / ligament detection pipeline for backlit spray
imaging.  Uses proper contour extraction with edge-aware quality gates and
physics-based classification instead of a neural network.

Pipeline stages:
  1. Noise-aware background subtraction (z-score per pixel)
  2. Edge-aware contour extraction with quality gates
  3. Physics-based classification (circularity, solidity, aspect ratio,
     Weber-number proxy)
  4. Temporal tracking with label persistence
  5. Interior-consistency rejection of false contours

Produces the same outputs as spray_drop.py for drop-in compatibility:
  - annotated .mp4 video
  - detections.csv
  - frame_summary.csv
  - optional debug PNG mosaics

Usage:
    python spray_physics.py --dataset dataset --output All_Images_physics \\
        --video spray_physics.mp4
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
except ImportError:
    SKELETON_AVAILABLE = False


# --------------------------------------------------------------------------
# Data model (identical to spray_drop.py for CSV compatibility)
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
    edge_strength: float      # mean gradient along contour
    interior_ratio: float     # fraction of interior darker than background


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
# Color palette (BGR). No text overlays on the video.
# --------------------------------------------------------------------------
COLORS = {
    "droplet": (0, 200, 255),       # orange
    "ligament": (255, 0, 255),      # magenta
    "main_jet": (255, 255, 0),      # cyan — excluded from YOLO
    "main_ligament": (255, 255, 0), # cyan (alias)
    "uncertain": (160, 160, 160),   # gray
}

# --------------------------------------------------------------------------
# Classification defaults
# --------------------------------------------------------------------------
DEFAULT_DROPLET_MIN_CIRCULARITY = 0.65
DEFAULT_DROPLET_MIN_SOLIDITY = 0.80
DEFAULT_DROPLET_MAX_ASPECT = 2.0
DEFAULT_LIGAMENT_MIN_ASPECT = 2.5
DEFAULT_LIGAMENT_MAX_CIRCULARITY = 0.50
DEFAULT_LIGAMENT_MIN_DEFECTS = 1
DEFAULT_MAIN_AREA = 5000
DEFAULT_SMALL_DROPLET_MAX_PIXELS = 30
DEFAULT_SOURCE_ROI_WIDTH_FRACTION = 0.10
DEFAULT_SOURCE_MIN_AREA = 500        # don't absorb small droplets near the inlet


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
        (p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in image_suffixes),
        key=frame_number,
    )


def to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# --------------------------------------------------------------------------
# Stage 0: Background estimation (reused logic, improved)
# --------------------------------------------------------------------------
def estimate_background(
    frame_paths: list[Path], n_samples: int, percentile: float,
    noise_floor: float,
) -> BackgroundModel | None:
    """Build a per-pixel background model from the brightest percentile.

    For backlit sprays the background is brighter than the liquid.
    A high percentile rejects the recurring dark spray pixels.
    """
    if not frame_paths or n_samples <= 0:
        return None
    sample_count = min(len(frame_paths), n_samples)
    sample_indices = np.linspace(0, len(frame_paths) - 1, sample_count, dtype=int)
    sampled = [frame_paths[idx] for idx in sample_indices]
    stack = []
    for path in sampled:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            stack.append(img)
    if not stack:
        return None
    frames = np.stack(stack, axis=0)
    image = np.percentile(frames, np.clip(percentile, 0.0, 100.0), axis=0).astype(np.uint8)

    # Noise estimate from the brightest quarter of samples (where liquid is absent)
    bright_count = max(1, int(math.ceil(len(frames) * 0.25)))
    bright_samples = np.partition(frames, len(frames) - bright_count, axis=0)[-bright_count:]
    noise = np.maximum(
        np.std(bright_samples.astype(np.float32), axis=0),
        max(noise_floor, 1e-6)
    )
    return BackgroundModel(image=image, noise=noise.astype(np.float32))


def load_background(args: argparse.Namespace, frame_paths: list[Path]) -> BackgroundModel | None:
    if args.background_dir:
        bg_paths = iter_image_paths(Path(args.background_dir))
        if not bg_paths:
            raise SystemExit(f"No images in background dir: {args.background_dir}")
        print(f"Building background from {args.background_dir}...")
        return estimate_background(bg_paths, args.background_frames, args.background_percentile, args.background_noise_floor)

    if args.background:
        bg_path = Path(args.background)
        if bg_path.is_dir():
            bg_paths = iter_image_paths(bg_path)
            if bg_paths:
                print(f"Building background from {bg_path}...")
                return estimate_background(bg_paths, args.background_frames, args.background_percentile, args.background_noise_floor)
        else:
            bg_img = cv2.imread(str(bg_path), cv2.IMREAD_GRAYSCALE)
            if bg_img is not None:
                noise = np.full(bg_img.shape, max(args.background_noise_floor, 1e-6), dtype=np.float32)
                return BackgroundModel(image=bg_img, noise=noise)

    if args.background_frames > 0:
        print(f"Estimating background from {args.background_frames} sampled frames...")
        return estimate_background(frame_paths, args.background_frames, args.background_percentile, args.background_noise_floor)
    return None


# --------------------------------------------------------------------------
# Stage 1: Preprocessing and z-score foreground extraction
# --------------------------------------------------------------------------
def denoise_image(gray: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    if args.denoise == "median":
        ksize = args.median_ksize if args.median_ksize % 2 == 1 else args.median_ksize + 1
        return cv2.medianBlur(gray, max(3, ksize))
    if args.denoise == "bilateral":
        return cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)
    return gray


def align_brightness(gray: np.ndarray, background: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    """Match global brightness shift before subtraction."""
    q = args.background_align_percentile
    shift = float(np.percentile(gray, q)) - float(np.percentile(background, q))
    shift = np.clip(shift, -args.background_max_shift, args.background_max_shift)
    aligned = background.astype(np.int16) + int(round(shift))
    return np.clip(aligned, 0, 255).astype(np.uint8)


def extract_foreground(
    frame: np.ndarray, background: BackgroundModel | None, args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (binary_mask, grayscale_preprocessed).

    Uses **hysteresis** z-score thresholding when a background model is
    available.  Two thresholds are applied:
      - **Strong** (z_hi): creates high-confidence seed pixels.
      - **Weak**   (z_lo): grows from seeds to recover faint edges.
    A pixel enters the final mask only if it passes the weak threshold
    AND is connected (8-way) to at least one seed pixel.
    """
    gray = to_gray(frame)
    denoised = denoise_image(gray, args)

    if background is not None:
        aligned_bg = align_brightness(denoised, background.image, args)
        # Directional subtraction: liquid is DARKER than backlight
        diff = cv2.subtract(aligned_bg, denoised)
        diff_float = diff.astype(np.float32)
        noise = np.maximum(background.noise, max(args.background_noise_floor, 1e-6))

        # Z-score map: how many noise-sigmas above zero is each pixel?
        z_map = diff_float / noise

        z_hi = args.background_zscore_hi
        z_lo = args.background_zscore_lo

        # Strong seeds
        strong = (z_map >= z_hi).astype(np.uint8) * 255
        # Weak candidates
        weak = (z_map >= z_lo).astype(np.uint8) * 255

        # Also require minimum absolute contrast on both masks
        if args.min_foreground_contrast > 0:
            contrast_fail = diff < args.min_foreground_contrast
            strong[contrast_fail] = 0
            weak[contrast_fail] = 0

        # Hysteresis: grow from strong seeds into weak candidates
        # Find connected components in the weak mask, keep only those
        # that contain at least one strong seed pixel.
        n_labels, labels = cv2.connectedComponents(weak, connectivity=8)
        # Which labels contain strong seeds?
        seed_labels = set(np.unique(labels[strong > 0]).tolist()) - {0}
        binary = np.zeros_like(denoised)
        for lbl in seed_labels:
            binary[labels == lbl] = 255

        return binary, denoised

    # No background model: fall back to Otsu on the denoised image
    _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary, denoised


# --------------------------------------------------------------------------
# Stage 2: Morphological cleanup and edge-aware contour extraction
# --------------------------------------------------------------------------
def clean_mask(binary: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    """Morphological opening → closing → optional hole-fill."""
    if args.morph_open_ksize > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (args.morph_open_ksize,) * 2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k)
    if args.morph_close_ksize > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (args.morph_close_ksize,) * 2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)
    return binary


def is_source_contour(
    contour: np.ndarray, img_w: int,
    roi_width_fraction: float = DEFAULT_SOURCE_ROI_WIDTH_FRACTION,
    min_area: int = DEFAULT_SOURCE_MIN_AREA,
) -> bool:
    """Return True when a large contour intersects the left-side source ROI.

    The nozzle is on the LEFT (and sometimes TOP) of the frame.
    Only contours touching those edges are the main jet/sheet.
    Right/bottom edge contact is ignored — those are droplets
    leaving the frame, not the source jet.
    """
    area = cv2.contourArea(contour)
    if area < min_area:
        return False
    x, y, w, h = cv2.boundingRect(contour)
    source_limit = max(1, int(round(img_w * roi_width_fraction)))
    return x <= source_limit


def find_component_contours(binary: np.ndarray) -> list[tuple[np.ndarray, np.ndarray, int]]:
    """Return (contour, component_mask, pixel_area) for each connected component."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    components: list[tuple[np.ndarray, np.ndarray, int]] = []
    for label in range(1, count):
        pixel_area = int(stats[label, cv2.CC_STAT_AREA])
        comp_mask = np.zeros_like(binary)
        comp_mask[labels == label] = 255
        result = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = result[0] if len(result) == 2 else result[1]
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        components.append((contour, comp_mask, pixel_area))
    return components


def split_touching_components(binary: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    """Split compact multi-peak blobs with marker-controlled watershed.

    The operation is restricted to compact, mid-sized components so genuine
    ligaments are not chopped into short fragments.
    """
    if not getattr(args, "enable_watershed", True):
        return binary

    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    split = np.zeros_like(binary)

    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        aspect = max(width, height) / max(min(width, height), 1)
        component = labels == label

        should_split = (
            args.watershed_min_component_area <= area <= args.watershed_max_component_area
            and aspect <= args.watershed_max_aspect
        )
        if not should_split:
            split[component] = 255
            continue

        # Pad the crop so watershed has a background border around each blob.
        x0, y0 = max(0, x - 1), max(0, y - 1)
        x1 = min(binary.shape[1], x + width + 1)
        y1 = min(binary.shape[0], y + height + 1)
        crop = component[y0:y1, x0:x1].astype(np.uint8)
        distance = cv2.distanceTransform(crop, cv2.DIST_L2, 5)
        max_distance = float(distance.max())
        if max_distance < args.watershed_min_peak_height:
            split[component] = 255
            continue

        kernel_size = 2 * args.watershed_min_peak_distance + 1
        peak_kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        local_max = distance == cv2.dilate(distance, peak_kernel)
        peak_threshold = max(
            args.watershed_min_peak_height,
            max_distance * args.watershed_peak_height_fraction,
        )
        peaks = (local_max & (distance >= peak_threshold) & (crop > 0)).astype(np.uint8)
        peak_count, peak_labels = cv2.connectedComponents(peaks, connectivity=8)
        if peak_count <= 2:  # Background plus one peak cannot be split.
            split[component] = 255
            continue

        # Peaks expand over the inverse distance surface. Watershed boundaries
        # remain zero so later connected-component extraction keeps the blobs
        # separate.
        topography = cv2.normalize(
            max_distance - distance, None, 0, 255, cv2.NORM_MINMAX,
        ).astype(np.uint8)
        watershed_image = cv2.cvtColor(topography, cv2.COLOR_GRAY2BGR)
        markers = np.zeros(crop.shape, dtype=np.int32)
        markers[crop == 0] = 1
        for peak_label in range(1, peak_count):
            markers[peak_labels == peak_label] = peak_label + 1
        markers = cv2.watershed(watershed_image, markers)
        split_crop = markers > 1
        target = split[y0:y1, x0:x1]
        target[split_crop] = 255

    return split


# --------------------------------------------------------------------------
# Branch B: Laplacian-of-Gaussian fine-droplet detector
# --------------------------------------------------------------------------
def detect_fine_droplets(
    gray: np.ndarray,
    background: BackgroundModel | None,
    args: argparse.Namespace,
    branch_a_mask: np.ndarray | None = None,
) -> list[tuple[np.ndarray, int]]:
    """Detect small dark circular blobs using scale-space LoG.

    This branch operates on the **raw** grayscale (no median filter, no
    morphological closing) to preserve tiny droplets that would be erased
    by Branch A's preprocessing.

    Key design decisions:
      - Laplacian is **negated** so that dark-blob centres (positive
        curvature in diff-space) give positive response peaks.
      - Normalised against **local background noise**, not the strongest
        object in the frame, so faint droplets aren't masked by the jet.
      - Each blob must have a **dark centre** relative to its immediate
        surroundings (validation against noise peaks).

    Returns a list of (contour, pixel_area) tuples.
    """
    if not getattr(args, 'enable_log_blobs', False):
        return []
    if background is None:
        return []

    h, w = gray.shape[:2]

    # Background-subtracted image (dark objects -> positive)
    aligned_bg = align_brightness(gray, background.image, args)
    diff = cv2.subtract(aligned_bg, gray).astype(np.float32)

    # Normalise by LOCAL NOISE (per-pixel sigma), not by diff_max.
    # This prevents the strong jet from suppressing faint droplet response.
    noise = np.maximum(background.noise, max(args.background_noise_floor, 1e-6))
    diff_norm = diff / noise  # now in units of "noise sigmas"

    # Multi-scale LoG with CORRECTED polarity
    sigmas = np.linspace(args.log_sigma_min, args.log_sigma_max, args.log_num_scales)
    responses = []
    for sigma in sigmas:
        ksize = int(np.ceil(sigma * 6)) | 1  # ensure odd
        blurred = cv2.GaussianBlur(diff_norm, (ksize, ksize), sigma)
        # Laplacian of a bright blob (in diff space) is NEGATIVE at centre.
        # Negate so blob centres become POSITIVE peaks.
        lap = -cv2.Laplacian(blurred, cv2.CV_32F)
        # Scale-normalise (multiply by sigma^2 for scale invariance)
        responses.append(lap * (sigma ** 2))

    # Stack and find per-pixel maximum across scales
    stack = np.stack(responses, axis=0)  # (n_scales, H, W)
    best_scale_idx = np.argmax(stack, axis=0)  # (H, W)
    best_response = np.max(stack, axis=0)      # (H, W)

    # Threshold (now in noise-sigma units, not [0,1])
    threshold = args.log_threshold
    candidates_mask = best_response > threshold

    # Non-maximum suppression: find local maxima in a 5x5 neighbourhood
    kernel = np.ones((5, 5), np.float32)
    dilated = cv2.dilate(best_response, kernel)
    local_max = (best_response == dilated) & candidates_mask

    # Extract blob centres
    ys, xs = np.where(local_max)
    blobs: list[tuple[np.ndarray, int]] = []

    for y_c, x_c in zip(ys, xs):
        scale_idx = best_scale_idx[y_c, x_c]
        sigma = sigmas[scale_idx]
        radius = float(sigma * np.sqrt(2))  # LoG radius ~ sigma*sqrt(2)

        if radius > args.log_max_radius:
            continue
        if radius < 1.0:
            radius = 1.0

        # Skip if already covered by Branch A
        if branch_a_mask is not None and branch_a_mask[y_c, x_c] > 0:
            continue

        # ---- Dark-centre validation ----
        # Validate in local background-noise units, rather than raw greyscale,
        # so the same thresholds work across uneven illumination.
        r_int = max(int(round(radius)), 1)
        r_outer = r_int + 2
        y0 = max(0, y_c - r_outer)
        y1 = min(h, y_c + r_outer + 1)
        x0 = max(0, x_c - r_outer)
        x1 = min(w, x_c + r_outer + 1)
        patch = diff_norm[y0:y1, x0:x1]
        if patch.size < 4:
            continue
        centre_z = float(diff_norm[y_c, x_c])
        # Annulus = pixels in patch that are > radius from centre
        yy, xx = np.mgrid[y0:y1, x0:x1]
        dist_sq = (yy - y_c) ** 2 + (xx - x_c) ** 2
        annulus_mask = dist_sq > (r_int * r_int)
        annulus_pixels = patch[annulus_mask]
        if len(annulus_pixels) < 2:
            continue
        annulus_z = float(np.median(annulus_pixels))
        if (
            centre_z < args.log_min_center_z
            or centre_z - annulus_z < args.log_min_center_delta_z
        ):
            continue

        # Create circular contour polygon
        n_pts = max(8, int(2 * np.pi * radius))
        n_pts = min(n_pts, 32)
        angles = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
        pts = np.stack([
            np.clip(x_c + radius * np.cos(angles), 0, w - 1),
            np.clip(y_c + radius * np.sin(angles), 0, h - 1),
        ], axis=1).astype(np.int32).reshape(-1, 1, 2)

        pixel_area = int(np.pi * radius * radius)
        if pixel_area < 2:
            pixel_area = 2

        blobs.append((pts, pixel_area))

    return blobs

def compute_edge_strength(gray: np.ndarray, contour: np.ndarray) -> float:
    """Mean Sobel gradient magnitude along the contour pixels.

    Real liquid boundaries produce strong intensity edges.  Background
    texture and sensor noise produce weak gradients.  A high edge-strength
    score confirms that the contour sits on a genuine intensity boundary.
    """
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(grad_x, grad_y)

    # Sample gradient values at contour points
    pts = contour.reshape(-1, 2)
    h, w = gray.shape[:2]
    values = []
    for x, y in pts:
        if 0 <= y < h and 0 <= x < w:
            values.append(float(grad_mag[y, x]))
    return float(np.mean(values)) if values else 0.0


def compute_interior_ratio(
    gray: np.ndarray, background: BackgroundModel | None, contour: np.ndarray,
    component_mask: np.ndarray,
) -> float:
    """Fraction of interior pixels that are genuinely darker than the background.

    For a real liquid contour, the interior should be consistently darker
    than the backlit background.  If most interior pixels are at background
    brightness (e.g., transparent sheet or noise), the ratio is low and we
    should reject the detection.
    """
    if background is None:
        return 1.0  # cannot check without a background model

    interior_pixels = component_mask > 0
    n_interior = int(np.sum(interior_pixels))
    if n_interior == 0:
        return 0.0

    bg_values = background.image[interior_pixels].astype(np.float32)
    fg_values = gray[interior_pixels].astype(np.float32)
    noise_values = background.noise[interior_pixels]

    # A pixel is "genuinely dark" if BG - FG > 1.5 * local_noise
    dark = (bg_values - fg_values) > 1.5 * noise_values
    return float(np.sum(dark)) / n_interior


# --------------------------------------------------------------------------
# Stage 3: Feature extraction
# --------------------------------------------------------------------------
def contour_centroid(contour: np.ndarray) -> tuple[float, float]:
    m = cv2.moments(contour)
    if m["m00"] == 0:
        x, y, w, h = cv2.boundingRect(contour)
        return x + w / 2.0, y + h / 2.0
    return m["m10"] / m["m00"], m["m01"] / m["m00"]


def convexity_defect_stats(contour: np.ndarray) -> tuple[int, float]:
    if len(contour) < 4:
        return 0, 0.0
    hull_idx = cv2.convexHull(contour, returnPoints=False)
    if hull_idx is None or len(hull_idx) < 3:
        return 0, 0.0
    hull_idx = np.sort(hull_idx, axis=0)
    try:
        defects = cv2.convexityDefects(contour, hull_idx)
    except cv2.error:
        return 0, 0.0
    if defects is None or len(defects) == 0:
        return 0, 0.0
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
        return 2.0 * max_dist
    return equiv_diameter


def compute_features(
    contour: np.ndarray, component_mask: np.ndarray, pixel_area: int,
    gray: np.ndarray, background: BackgroundModel | None,
) -> dict:
    """Compute all geometric + physics features for one contour."""
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
    skeleton_length = compute_skeleton_length(component_mask, x, y, w, h)
    thickness_est = estimate_thickness(component_mask, x, y, w, h, skeleton_length, equiv_diameter)

    # Physics quality metrics
    edge_strength = compute_edge_strength(gray, contour)
    interior_ratio = compute_interior_ratio(gray, background, contour, component_mask)

    return {
        "pixel_area": pixel_area,
        "area": area,
        "perimeter": perimeter,
        "equiv_diameter": equiv_diameter,
        "bbox_x": x, "bbox_y": y, "bbox_w": w, "bbox_h": h,
        "rot_w": float(rw), "rot_h": float(rh), "rot_angle": float(rangle),
        "aspect_ratio": aspect_ratio,
        "circularity": circularity,
        "solidity": solidity,
        "eccentricity": eccentricity,
        "centroid_x": cx, "centroid_y": cy,
        "defect_count": defect_count,
        "defect_depth_avg": defect_depth_avg,
        "skeleton_length": skeleton_length,
        "thickness_est": thickness_est,
        "edge_strength": edge_strength,
        "interior_ratio": interior_ratio,
    }


# --------------------------------------------------------------------------
# Stage 3: Physics-based classification
# --------------------------------------------------------------------------
def classify_physics(
    feats: dict, args: argparse.Namespace,
    img_w: int = 0, img_h: int = 0,
    contour: np.ndarray | None = None,
) -> str:
    """Classify a contour using physics-based decision cascade.

    The cascade order reflects physical certainty:
      0. Main jet body (boundary contact — position-based)
      1. Main jet body (area fallback — very large)
      2. Round compact object → droplet (surface tension → sphere)
      3. Elongated thin object → ligament (Rayleigh-Plateau instability)
      4. Small compact object → droplet (sub-pixel circle)
      5. Soft fallback using aspect + circularity
      6. Uncertain — ambiguous shapes are excluded from labels
    """
    # Gate 0: main jet by boundary contact (position-based)
    if contour is not None and img_w > 0 and img_h > 0:
        if is_source_contour(contour, img_w,
                             roi_width_fraction=args.source_roi_width_fraction,
                             min_area=args.source_min_area):
            return "main_jet"

    # Gate 1: main liquid body by area (fallback for huge fragments)
    if feats["area"] >= args.area:
        return "main_jet"

    circ = feats["circularity"]
    solidity = feats["solidity"]
    aspect = feats["aspect_ratio"]
    defects = feats["defect_count"]

    # Gate 2: round compact object → droplet (surface tension → sphere)
    is_round = (
        circ >= args.droplet_min_circularity
        and solidity >= args.droplet_min_solidity
        and aspect <= args.droplet_max_aspect
    )
    if is_round:
        return "droplet"

    # Gate 3: elongated thread → ligament (Rayleigh-Plateau instability)
    is_elongated = aspect >= args.ligament_min_aspect and circ <= args.ligament_max_circularity
    is_bent_thread = defects >= args.ligament_min_defects and aspect >= args.ligament_min_aspect * 0.7
    if is_elongated or is_bent_thread:
        return "ligament"

    # Gate 4: sub-pixel compact objects → droplet
    if feats["pixel_area"] <= args.small_droplet_max_pixels and aspect <= args.small_droplet_max_aspect:
        return "droplet"

    # Gate 5: soft fallback — use the strongest remaining signal
    if aspect > 1.8 and circ < 0.55:
        return "ligament"
    if solidity > 0.70 and aspect < 2.5:
        return "droplet"

    # Gate 6: truly ambiguous — fewer clean labels > many noisy labels
    return "uncertain"


# --------------------------------------------------------------------------
# Stage 4: Tracking with label persistence
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


class PhysicsTracker:
    """Centroid tracker with physics-based label persistence.

    A droplet cannot spontaneously become a ligament between frames
    (conservation of topology). Label flips are suppressed by majority
    vote over the track's history.
    """

    def __init__(self, max_distance: float, max_age: int, smooth_window: int, min_hits: int):
        self.max_distance = max_distance
        self.max_age = max_age
        self.smooth_window = max(1, smooth_window)
        self.min_hits = max(1, min_hits)
        self.tracks: dict[int, Track] = {}
        self.next_id = 1

    def update(self, detections: list[Detection], frame_id: int):
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

        for tid, track in list(self.tracks.items()):
            if track.last_frame != frame_id:
                track.missed += 1
                if track.missed > self.max_age:
                    del self.tracks[tid]

        return results


# --------------------------------------------------------------------------
# Stage 5: Full frame analysis
# --------------------------------------------------------------------------
def analyze_frame(
    frame: np.ndarray, frame_id: int, background: BackgroundModel | None,
    args: argparse.Namespace,
) -> tuple[list[Detection], list[np.ndarray], np.ndarray]:
    """Segment → contours → quality gate → features → classify."""
    img_h, img_w = frame.shape[:2]
    binary, gray = extract_foreground(frame, background, args)
    binary = clean_mask(binary, args)
    binary = split_touching_components(binary, args)
    components = find_component_contours(binary)

    detections: list[Detection] = []
    kept_contours: list[np.ndarray] = []
    branch_a_mask = np.zeros((img_h, img_w), dtype=np.uint8)

    for idx, (contour, comp_mask, pixel_area) in enumerate(components):
        if pixel_area < args.min_area:
            continue

        feats = compute_features(contour, comp_mask, pixel_area, gray, background)

        # ----- Quality gate: edge strength -----
        if feats["edge_strength"] < args.min_edge_strength:
            continue

        # ----- Quality gate: interior consistency -----
        if background is not None and feats["interior_ratio"] < args.min_interior_ratio:
            continue

        label = classify_physics(feats, args, img_w=img_w, img_h=img_h, contour=contour)

        detections.append(
            Detection(
                frame=frame_id,
                track_id=-1,
                label=label,
                contour_index=idx,
                velocity_x=0.0, velocity_y=0.0, speed=0.0,
                **feats,
            )
        )
        kept_contours.append(contour)
        cv2.drawContours(branch_a_mask, [contour], -1, 255, thickness=cv2.FILLED)

    # Fine blobs use the raw image and bypass Branch A's median filtering.
    # They still enter the common tracker below, so static specks are removed
    # by the existing persistence and motion checks.
    raw_gray = to_gray(frame)
    fine_blobs = detect_fine_droplets(raw_gray, background, args, branch_a_mask)
    for fine_idx, (contour, pixel_area) in enumerate(fine_blobs):
        if pixel_area < args.min_area:
            continue
        component_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        cv2.drawContours(component_mask, [contour], -1, 255, thickness=cv2.FILLED)
        feats = compute_features(contour, component_mask, pixel_area, raw_gray, background)
        detections.append(
            Detection(
                frame=frame_id,
                track_id=-1,
                label="droplet",
                contour_index=len(components) + fine_idx,
                velocity_x=0.0,
                velocity_y=0.0,
                speed=0.0,
                **feats,
            )
        )
        kept_contours.append(contour)
        cv2.drawContours(binary, [contour], -1, 255, thickness=cv2.FILLED)

    return detections, kept_contours, binary


# --------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------
def draw_detections(
    frame: np.ndarray,
    tracked_with_contours: list[tuple[Detection, np.ndarray]],
    fill_main: bool,
) -> np.ndarray:
    output = frame.copy()
    if fill_main:
        fill_mask = np.zeros(frame.shape[:2], np.uint8)
        for det, cnt in tracked_with_contours:
            if det.label == "main_jet":
                cv2.drawContours(fill_mask, [cnt], -1, 255, thickness=cv2.FILLED)
        if fill_mask.any():
            colored = np.zeros_like(frame)
            colored[:] = COLORS["main_jet"]
            blended = cv2.addWeighted(frame, 0.65, colored, 0.35, 0)
            output[fill_mask == 255] = blended[fill_mask == 255]

    for det, cnt in tracked_with_contours:
        color = COLORS.get(det.label, COLORS["uncertain"])
        thickness = 2 if det.label != "main_jet" else 1
        cv2.drawContours(output, [cnt], -1, color, thickness)
    return output


def debug_mosaic(frame: np.ndarray, binary: np.ndarray, annotated: np.ndarray) -> np.ndarray:
    mask_bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    h, w = frame.shape[:2]
    panels = []
    for panel in [frame, mask_bgr, annotated]:
        if panel.shape[:2] != (h, w):
            panel = cv2.resize(panel, (w, h))
        panel = panel.copy()
        cv2.rectangle(panel, (0, 0), (w - 1, h - 1), (255, 255, 255), 1)
        panels.append(panel)
    return np.hstack(panels)


# --------------------------------------------------------------------------
# Per-frame summary & CSV output
# --------------------------------------------------------------------------
def build_frame_summary(frame_id: int, detections: list[Detection]) -> FrameSummary:
    droplets = [d for d in detections if d.label == "droplet"]
    ligaments = [d for d in detections if d.label == "ligament"]
    mains = [d for d in detections if d.label == "main_jet"]
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


def save_detections(path: Path, detections: list[Detection]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(Detection.__dataclass_fields__.keys()))
        writer.writeheader()
        for det in detections:
            writer.writerow(dataclasses.asdict(det))


def save_frame_summaries(path: Path, summaries: list[FrameSummary]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(FrameSummary.__dataclass_fields__.keys()))
        writer.writeheader()
        for s in summaries:
            writer.writerow(dataclasses.asdict(s))


# --------------------------------------------------------------------------
# Video writer
# --------------------------------------------------------------------------
def make_video_writer(path: Path, fps: float, frame: np.ndarray) -> cv2.VideoWriter:
    h, w = frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create video writer: {path}")
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
        ref = cv2.imread(str(frame_paths[0]), cv2.IMREAD_GRAYSCALE)
        if ref is None:
            raise SystemExit(f"Could not read: {frame_paths[0]}")
        if background.image.shape != ref.shape:
            raise SystemExit(
                f"Background / frame dimension mismatch: {background.image.shape} vs {ref.shape}"
            )
        bg_path = output_dir / "background_model.png"
        cv2.imwrite(str(bg_path), background.image)
        noise_disp = cv2.normalize(background.noise, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        cv2.imwrite(str(output_dir / "background_noise.png"), noise_disp)
        print(f"Saved background model: {bg_path}")

    tracker = PhysicsTracker(
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
        fid = frame_number(frame_path)
        frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if frame is None:
            print(f"Skipping unreadable: {frame_path}")
            continue

        detections, contours, binary = analyze_frame(frame, fid, background, args)
        tracked = tracker.update(detections, fid)

        final_dets: list[Detection] = []
        visible: list[tuple[Detection, np.ndarray]] = []
        for trk, cnt in zip(tracked, contours):
            det, tid, velocity, speed, smoothed_label, confirmed, hits = trk
            final = dataclasses.replace(
                det,
                track_id=tid,
                label=smoothed_label,
                velocity_x=velocity[0],
                velocity_y=velocity[1],
                speed=speed,
            )
            # Reject static tiny objects (background texture)
            is_static_small = (
                final.pixel_area <= args.small_droplet_max_pixels
                and hits >= 2
                and speed < args.min_small_droplet_speed
            )
            if is_static_small:
                continue
            if confirmed or args.draw_tentative:
                final_dets.append(final)
                visible.append((final, cnt))

        annotated = draw_detections(frame, visible, args.fill_main_ligament)
        all_detections.extend(final_dets)
        all_summaries.append(build_frame_summary(fid, final_dets))

        video_frame = annotated
        if not args.no_save_frames:
            cv2.imwrite(str(output_dir / f"seg_{fid:04d}.png"), debug_mosaic(frame, binary, annotated))

        if video_writer is None:
            video_writer = make_video_writer(video_path, args.fps, video_frame)
        video_writer.write(video_frame)

        print(
            f"frame={fid:04d}  droplets={all_summaries[-1].droplet_count}  "
            f"ligaments={all_summaries[-1].ligament_count}  main={all_summaries[-1].main_ligament_count}"
        )

    save_detections(output_dir / "detections.csv", all_detections)
    save_frame_summaries(output_dir / "frame_summary.csv", all_summaries)
    if video_writer is not None:
        video_writer.release()
        print(f"Saved video: {video_path}")
    print(f"Saved detections: {output_dir / 'detections.csv'}")
    print(f"Saved frame summary: {output_dir / 'frame_summary.csv'}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Physics-based spray droplet/ligament detection.")

    # I/O
    p.add_argument("--dataset", default="dataset")
    p.add_argument("--output", default="All_Images_physics")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=None)
    p.add_argument("--video", default="spray_physics.mp4")
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--no-save-frames", action="store_true")
    p.add_argument("--show", action="store_true")

    # Background
    p.add_argument("--background", default=None)
    p.add_argument("--background-dir", default=None)
    p.add_argument("--background-frames", type=int, default=80)
    p.add_argument("--background-percentile", type=float, default=99.0)
    p.add_argument("--background-noise-floor", type=float, default=4.0)
    p.add_argument("--background-align-percentile", type=float, default=90.0)
    p.add_argument("--background-max-shift", type=int, default=25)
    p.add_argument("--background-zscore", type=float, default=1.5,
                    help="(Deprecated, use --background-zscore-lo) Alias for z_lo.")
    p.add_argument("--background-zscore-hi", type=float, default=3.0,
                    help="Strong z-score threshold (seeds). Higher = stricter.")
    p.add_argument("--background-zscore-lo", type=float, default=1.5,
                    help="Weak z-score threshold (growth). Lower = captures fainter edges.")
    p.add_argument("--min-foreground-contrast", type=int, default=4,
                    help="Minimum absolute BG-FG intensity difference.")

    # Preprocessing
    p.add_argument("--denoise", choices=("median", "bilateral", "none"), default="median")
    p.add_argument("--median-ksize", type=int, default=3)
    p.add_argument("--morph-open-ksize", type=int, default=1,
                    help="Opening kernel size. Removes tiny noise specks.")
    p.add_argument("--morph-close-ksize", type=int, default=1,
                    help="Closing kernel size. Bridges small gaps.")

    # Quality gates
    p.add_argument("--min-edge-strength", type=float, default=3.0,
                    help="Minimum mean Sobel gradient along the contour. Rejects soft-gradient false contours.")
    p.add_argument("--min-interior-ratio", type=float, default=0.05,
                    help="Minimum fraction of interior pixels darker than background. Rejects transparent/noise contours.")

    # Classification
    p.add_argument("--min-area", type=float, default=2.0)
    p.add_argument("--area", type=int, default=DEFAULT_MAIN_AREA)
    p.add_argument("--source-roi-width-fraction", type=float, default=DEFAULT_SOURCE_ROI_WIDTH_FRACTION,
                    help="Width of the left-side source ROI as a fraction of the frame width.")
    p.add_argument("--source-min-area", type=int, default=DEFAULT_SOURCE_MIN_AREA,
                    help="Min area (px) for a source-ROI contour to be classified as main jet.")
    p.add_argument("--small-droplet-max-pixels", type=int, default=DEFAULT_SMALL_DROPLET_MAX_PIXELS)
    p.add_argument("--small-droplet-max-aspect", type=float, default=2.5)
    p.add_argument("--droplet-min-circularity", type=float, default=DEFAULT_DROPLET_MIN_CIRCULARITY)
    p.add_argument("--droplet-min-solidity", type=float, default=DEFAULT_DROPLET_MIN_SOLIDITY)
    p.add_argument("--droplet-max-aspect", type=float, default=DEFAULT_DROPLET_MAX_ASPECT)
    p.add_argument("--ligament-min-aspect", type=float, default=DEFAULT_LIGAMENT_MIN_ASPECT)
    p.add_argument("--ligament-max-circularity", type=float, default=DEFAULT_LIGAMENT_MAX_CIRCULARITY)
    p.add_argument("--ligament-min-defects", type=int, default=DEFAULT_LIGAMENT_MIN_DEFECTS)

    # Branch B: LoG fine-droplet detector
    p.add_argument("--enable-log-blobs", action="store_true", default=True,
                    help="Enable LoG blob detector for fine droplets (Branch B).")
    p.add_argument("--log-sigma-min", type=float, default=1.0,
                    help="Smallest LoG sigma (detects ~2px blobs).")
    p.add_argument("--log-sigma-max", type=float, default=4.0,
                    help="Largest LoG sigma (detects ~12px blobs).")
    p.add_argument("--log-num-scales", type=int, default=5,
                    help="Number of LoG scales between sigma-min and sigma-max.")
    p.add_argument("--log-threshold", type=float, default=0.20,
                    help="Minimum noise-normalised LoG response to accept a blob.")
    p.add_argument("--log-max-radius", type=float, default=15.0,
                    help="Maximum blob radius in pixels.")
    p.add_argument("--log-min-center-z", type=float, default=1.0,
                    help="Minimum background-noise-normalised darkness at a blob centre.")
    p.add_argument("--log-min-center-delta-z", type=float, default=0.5,
                    help="Minimum centre-to-annulus darkness difference in local noise units.")

    # Split compact, multi-peak droplet clusters without fragmenting ligaments.
    p.add_argument("--disable-watershed", action="store_false", dest="enable_watershed",
                    help="Disable watershed splitting for touching compact droplets.")
    p.set_defaults(enable_watershed=True)
    p.add_argument("--watershed-min-component-area", type=int, default=20)
    p.add_argument("--watershed-max-component-area", type=int, default=2000)
    p.add_argument("--watershed-max-aspect", type=float, default=2.5)
    p.add_argument("--watershed-min-peak-distance", type=int, default=3)
    p.add_argument("--watershed-min-peak-height", type=float, default=1.2)
    p.add_argument("--watershed-peak-height-fraction", type=float, default=0.45)

    # Tracking
    p.add_argument("--max-track-distance", type=float, default=40.0)
    p.add_argument("--track-max-age", type=int, default=5)
    p.add_argument("--classify-smooth-window", type=int, default=5,
                    help="Frames of label history for majority-vote smoothing.")
    p.add_argument("--min-small-droplet-speed", type=float, default=0.5)
    p.add_argument("--min-track-hits", type=int, default=2)
    p.add_argument("--draw-tentative", action="store_true")
    p.add_argument("--fill-main-ligament", action="store_true")

    return p


def main() -> None:
    args = build_parser().parse_args()
    process_frames(args)


if __name__ == "__main__":
    main()
