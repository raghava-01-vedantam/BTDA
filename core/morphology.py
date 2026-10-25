from __future__ import annotations

import cv2
import numpy as np

KERNEL_SIZE = 3
MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (KERNEL_SIZE, KERNEL_SIZE))
MIN_COMPONENT_PIXELS = 30


def clean_mask(
    binary_mask: np.ndarray,
    min_pixels: int = MIN_COMPONENT_PIXELS,
) -> np.ndarray:
    mask = binary_mask.astype(np.uint8)

    # Opening and closing
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, MORPH_KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, MORPH_KERNEL)

    # Remove small specks
    if min_pixels > 0 and mask.any():
        mask = _remove_small_components(mask, min_pixels)

    return mask


def _remove_small_components(mask: np.ndarray, min_pixels: int) -> np.ndarray:
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask, dtype=np.uint8)
    for label in range(1, n_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_pixels:
            cleaned[labels == label] = 1
    return cleaned
