from __future__ import annotations

import cv2
import numpy as np

# Physical constants
DEFAULT_PIXEL_SPACING_CM = 0.115
PIXEL_EDGE_CM = DEFAULT_PIXEL_SPACING_CM
PIXEL_AREA_CM2 = PIXEL_EDGE_CM ** 2


def calculate_tumor_area(
    binary_mask: np.ndarray,
    pixel_spacing_cm: float = DEFAULT_PIXEL_SPACING_CM,
) -> tuple[int, float]:
    n_pixels = int(np.sum(binary_mask > 0))
    area_cm2 = n_pixels * (pixel_spacing_cm ** 2)
    return n_pixels, area_cm2


def create_overlay(
    original_img: np.ndarray,
    binary_mask: np.ndarray,
    alpha: float = 0.5,
    mask_color: tuple[int, int, int] = (255, 80, 80),
) -> np.ndarray:
    if original_img.dtype != np.uint8:
        original_img = (original_img * 255).clip(0, 255).astype(np.uint8)

    if original_img.ndim == 2:
        original_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2RGB)

    h, w = original_img.shape[:2]
    mask_u8 = binary_mask.astype(np.uint8)
    if mask_u8.shape != (h, w):
        mask_u8 = cv2.resize(mask_u8, (w, h), interpolation=cv2.INTER_NEAREST)

    overlay = original_img.copy()
    tumor_pixels = mask_u8 > 0

    if np.any(tumor_pixels):
        color_arr = np.array(mask_color, dtype=np.uint8)
        orig_tumor = original_img[tumor_pixels].astype(np.float32)
        blended = alpha * color_arr.astype(np.float32) + (1.0 - alpha) * orig_tumor
        overlay[tumor_pixels] = blended.clip(0, 255).astype(np.uint8)

    return overlay
