from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class RegionStats:
    count: int
    total_pixels: int
    largest_pixels: int
    largest_diameter_px: float
    per_region_pixels: list[int] = field(default_factory=list)


def analyse_regions(binary_mask: np.ndarray) -> RegionStats:
    mask_u8 = (binary_mask > 0).astype(np.uint8)
    total_pixels = int(mask_u8.sum())

    if total_pixels == 0:
        return RegionStats(
            count=0,
            total_pixels=0,
            largest_pixels=0,
            largest_diameter_px=0.0,
            per_region_pixels=[],
        )

    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)

    areas = [
        int(stats[label, cv2.CC_STAT_AREA])
        for label in range(1, n_labels)
    ]
    areas.sort(reverse=True)

    largest = areas[0] if areas else 0
    diameter = 2.0 * math.sqrt(largest / math.pi) if largest > 0 else 0.0

    return RegionStats(
        count=len(areas),
        total_pixels=total_pixels,
        largest_pixels=largest,
        largest_diameter_px=diameter,
        per_region_pixels=areas,
    )


def compute_gt_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    smooth: float = 1e-6,
) -> dict[str, float]:
    p = (pred > 0).astype(np.float32).ravel()
    g = (gt > 0).astype(np.float32).ravel()

    tp = float((p * g).sum())
    fp = float((p * (1.0 - g)).sum())
    fn = float(((1.0 - p) * g).sum())

    dice = (2.0 * tp + smooth) / (2.0 * tp + fp + fn + smooth)
    iou = (tp + smooth) / (tp + fp + fn + smooth)
    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)

    return {"dice": dice, "iou": iou, "precision": precision, "recall": recall}


def create_outline_overlay(
    original_img: np.ndarray,
    binary_mask: np.ndarray,
    color: tuple[int, int, int] = (255, 80, 80),
    thickness: int = 2,
) -> np.ndarray:
    bgr = cv2.cvtColor(original_img.copy(), cv2.COLOR_RGB2BGR)
    mask_u8 = (binary_mask > 0).astype(np.uint8)

    h, w = bgr.shape[:2]
    if mask_u8.shape != (h, w):
        mask_u8 = cv2.resize(mask_u8, (w, h), interpolation=cv2.INTER_NEAREST)

    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        bgr_color = (color[2], color[1], color[0])
        cv2.drawContours(bgr, contours, -1, bgr_color, thickness)

    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
