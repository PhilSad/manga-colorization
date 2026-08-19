"""Line-art fidelity scoring (the sanity step's magic function).

Compares a black & white panel with its colorized output through a
*structural line representation* that is mostly invariant to line tinting
and to flat shading changes:

1. Both images are downscaled to a common analysis grid (long edge capped,
   `max_edge`, default 1024).
2. A thin-stroke line map is extracted per image: pixels that are (a) clearly
   darker than their large-scale local neighborhood (so a tinted line still
   counts as ink, whatever its color) and (b) part of a structure thinner
   than the fill scale (so flat shading fills are not ink, and isolated
   screentone dots are mostly eroded away).
3. The two maps are scored with four complementary metrics:

   - ``line_iou``: ink-mask intersection over union (content added/removed)
   - ``chamfer_sim``: symmetric chamfer distance, exp-scaled (lines moved)
   - ``component_sim``: agreement of the *large* connected-component counts
     (big content added/removed; screentone-sized components are ignored)
   - ``drift_sim``: phase-correlation shift of the grays (composition moved
     as a whole)

The composite ``line_fidelity`` is the weighted mean. A panel is flagged
when ``line_fidelity`` drops below the threshold or trips any hard rule
(very low IoU, large chamfer, or large drift).

Everything is pure numpy + OpenCV (no torch). OpenCV is imported at module
import time and is a hard dependency of this module.
"""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Analysis parameters (all in the common analysis grid)

DEFAULT_MAX_EDGE = 1024    # long-edge cap for the analysis grid
DARK_DELTA = 25            # ink px must be >= this many levels below local mean
STROKE_RADIUS = 2          # thin-stroke structuring element radius (px)
LOCAL_SIGMA_FRAC = 0.02    # blur sigma = frac * max edge (min 5 px)
CHAMFER_TAU_FRAC = 0.004   # chamfer similarity scale = frac * diagonal
DRIFT_FRAC = 0.02          # drift soft-rule scale = frac * diagonal
COMPONENT_MIN_AREA = 30    # components below this area (px) are noise
COMPONENT_CONNECTIVITY = 8

# Composite weights (must sum to 1)
WEIGHTS: dict[str, float] = {
    "line_iou": 0.45,
    "chamfer_sim": 0.30,
    "component_sim": 0.15,
    "drift_sim": 0.10,
}

# Hard rules: trip a flag regardless of the composite
HARD_LINE_IOU = 0.25        # < this IoU -> flagged
HARD_CHAMFER_PX = 4.0       # > this chamfer (px in analysis grid) -> flagged
HARD_DRIFT_FRAC = 0.03      # > this fraction of the diagonal -> flagged


def analysis_size(width: int, height: int,
                  max_edge: int = DEFAULT_MAX_EDGE) -> tuple[int, int]:
    """(w, h) of the analysis grid: longest edge capped at `max_edge`."""
    longest = max(width, height)
    if longest <= max_edge:
        return (width, height)
    scale = max_edge / longest
    return (max(1, round(width * scale)), max(1, round(height * scale)))


def gray_grid(image: Image.Image, size: tuple[int, int]) -> np.ndarray:
    """float32 grayscale at the analysis grid (resampled when needed)."""
    if (image.width, image.height) != size:
        image = image.resize(size, Image.Resampling.LANCZOS)
    return np.asarray(image.convert("L"), dtype=np.float32)


def thin_line_map(gray: np.ndarray, *,
                  dark_delta: float = DARK_DELTA,
                  stroke_radius: int = STROKE_RADIUS) -> np.ndarray:
    """Boolean ink-stroke map: locally-dark thin structures.

    A pixel counts as line art when it is clearly darker than its
    large-scale neighborhood (tint-invariant) and its structure is thinner
    than the fill scale. Thick flat fills survive the opening and are
    excluded; their boundary contour is kept (it is meaningful line art).
    Isolated screentone dots are small and round, so the elliptical opening
    erodes most of them away.
    """
    height, width = gray.shape
    sigma = max(5.0, LOCAL_SIGMA_FRAC * max(height, width))
    local_mean = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
    dark = gray < (local_mean - dark_delta)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * stroke_radius + 1, 2 * stroke_radius + 1)
    )
    opened = cv2.morphologyEx(dark.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    return dark & (opened == 0)


def _line_metrics(bw_map: np.ndarray, col_map: np.ndarray,
                  diag: float) -> dict[str, float]:
    inter = int(np.count_nonzero(bw_map & col_map))
    union = int(np.count_nonzero(bw_map | col_map))
    line_iou = inter / union if union else 1.0

    # cv2.distanceTransform reports, at each pixel, the distance to the
    # nearest *zero* pixel (0 on foreground) — so to get the distance to the
    # nearest *ink* pixel we transform the inverted masks.
    db = cv2.distanceTransform((bw_map == 0).astype(np.uint8), cv2.DIST_L2, 3)
    dc = cv2.distanceTransform((col_map == 0).astype(np.uint8), cv2.DIST_L2, 3)
    n_b = int(bw_map.sum())
    n_c = int(col_map.sum())
    half_b = float(dc[bw_map].sum()) / n_b if n_b else 0.0
    half_c = float(db[col_map].sum()) / n_c if n_c else 0.0
    chamfer = 0.5 * (half_b + half_c)
    tau = CHAMFER_TAU_FRAC * diag
    chamfer_sim = math.exp(-chamfer / tau) if tau > 0 else 0.0

    def _large_components(mask: np.ndarray) -> int:
        n_labels, _, stats, _ = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=COMPONENT_CONNECTIVITY
        )
        if n_labels <= 1:
            return 0
        return int((stats[1:, cv2.CC_STAT_AREA] >= COMPONENT_MIN_AREA).sum())

    n_bc = _large_components(bw_map)
    n_cc = _large_components(col_map)
    component_sim = min(n_bc, n_cc) / max(n_bc, n_cc) if max(n_bc, n_cc) else 1.0

    return {
        "line_iou": round(float(line_iou), 4),
        "chamfer_px": round(float(chamfer), 3),
        "chamfer_sim": round(float(chamfer_sim), 4),
        "component_sim": round(float(component_sim), 4),
        "components_bw": int(n_bc),
        "components_color": int(n_cc),
        "ink_ratio_bw": round(float(bw_map.mean()), 5),
        "ink_ratio_color": round(float(col_map.mean()), 5),
    }


def _drift_metrics(gray_bw: np.ndarray, gray_col: np.ndarray,
                   diag: float) -> dict[str, float]:
    height, width = gray_bw.shape
    window = cv2.createHanningWindow((width, height), cv2.CV_32F)
    (dx, dy), response = cv2.phaseCorrelate(gray_bw * window, gray_col * window)
    drift_px = math.hypot(float(dx), float(dy))
    drift_sim = max(0.0, 1.0 - drift_px / (DRIFT_FRAC * diag))
    return {
        "drift_px": round(float(drift_px), 3),
        "drift_response": round(float(response), 4),
        "drift_sim": round(float(drift_sim), 4),
    }


def _midgray_detail_ratio(gray: np.ndarray) -> float:
    """Screentone proxy: fraction of mid-gray pixels with strong local
    detail. High values mean the B&W page is heavy on screentone texture,
    which the colorizer routinely removes — a known false-positive source
    for the line-fidelity metrics (recorded for review, not a flag)."""
    detail = np.abs(gray - cv2.GaussianBlur(gray, (0, 0), sigmaX=3))
    mid = (gray > 80) & (gray < 200)
    if not mid.any():
        return 0.0
    return round(float((detail[mid] > 18).mean()), 4)


def score_pair(bw: Image.Image, color: Image.Image, *,
               max_edge: int = DEFAULT_MAX_EDGE,
               threshold: float = 0.45) -> dict[str, Any]:
    """Full scoring of one panel pair; returns the metrics record including
    the verdict (`flagged` + `reasons`).

    `bw` and `color` may have different sizes (FLUX/gpt-image-2 round to
    multiples of 16); both are resampled onto the B&W analysis grid.
    """
    size = analysis_size(bw.width, bw.height, max_edge)
    bw_g = gray_grid(bw, size)
    col_g = gray_grid(color, size)
    diag = math.hypot(*bw_g.shape)

    metrics: dict[str, Any] = {}
    metrics.update(_line_metrics(thin_line_map(bw_g), thin_line_map(col_g), diag))
    metrics.update(_drift_metrics(bw_g, col_g, diag))
    metrics["bw_detail_ratio"] = _midgray_detail_ratio(bw_g)

    fidelity = sum(WEIGHTS[key] * metrics[key] for key in WEIGHTS)
    metrics["line_fidelity"] = round(fidelity, 4)

    hard_iou = metrics["line_iou"] < HARD_LINE_IOU
    hard_chamfer = metrics["chamfer_px"] > HARD_CHAMFER_PX
    hard_drift = metrics["drift_px"] > HARD_DRIFT_FRAC * diag
    below = fidelity < threshold

    reasons: list[str] = []
    if below:
        reasons.append(
            f"line_fidelity {fidelity:.3f} < threshold {threshold:.3f}"
        )
    if hard_iou:
        reasons.append(f"line_iou {metrics['line_iou']:.3f} < {HARD_LINE_IOU:.2f}")
    if hard_chamfer:
        reasons.append(
            f"chamfer {metrics['chamfer_px']:.2f}px > {HARD_CHAMFER_PX:.1f}px"
        )
    if hard_drift:
        reasons.append(
            f"drift {metrics['drift_px']:.1f}px > "
            f"{HARD_DRIFT_FRAC * diag:.1f}px ({HARD_DRIFT_FRAC * 100:.0f}% "
            "of diagonal)"
        )
    metrics["flagged"] = bool(reasons)
    metrics["reasons"] = reasons
    metrics["threshold"] = threshold
    return metrics
