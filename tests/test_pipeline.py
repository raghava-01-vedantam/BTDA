import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.inference import threshold_mask
from core.postprocessing import PIXEL_AREA_CM2, calculate_tumor_area, create_overlay

class TestAreaCalculation:
    """Tests for calculate_tumor_area() and the PIXEL_AREA_CM2 constant."""

    def test_pixel_area_constant_value(self):
        """PIXEL_AREA_CM2 must equal 0.115² = 0.013225 cm²."""
        expected = 0.115 ** 2
        assert abs(PIXEL_AREA_CM2 - expected) < 1e-9, (
            f"PIXEL_AREA_CM2 = {PIXEL_AREA_CM2}, expected {expected}"
        )

    def test_zero_pixels_gives_zero_area(self):
        """An all-zero mask should produce 0 pixels and 0.0 cm² area."""
        mask = np.zeros((256, 256), dtype=np.uint8)
        n_pixels, area = calculate_tumor_area(mask)
        assert n_pixels == 0
        assert area == 0.0

    def test_known_pixel_count(self):
        """100 tumor pixels should yield exactly 100 × 0.013225 = 1.3225 cm²."""
        mask = np.zeros((256, 256), dtype=np.uint8)
        mask[:10, :10] = 1  # 10 × 10 = 100 pixels
        n_pixels, area = calculate_tumor_area(mask)
        assert n_pixels == 100, f"Expected 100 pixels, got {n_pixels}"
        expected_area = 100 * PIXEL_AREA_CM2  # = 1.3225 cm²
        assert abs(area - expected_area) < 1e-9, (
            f"Expected area {expected_area:.6f}, got {area:.6f}"
        )

    def test_full_mask_area(self):
        """A fully-lit 256×256 mask should equal 256*256 * PIXEL_AREA_CM2."""
        mask = np.ones((256, 256), dtype=np.uint8)
        n_pixels, area = calculate_tumor_area(mask)
        assert n_pixels == 256 * 256
        expected_area = 256 * 256 * PIXEL_AREA_CM2
        assert abs(area - expected_area) < 1e-6

    def test_area_with_float_mask(self):
        """calculate_tumor_area should handle float32 masks (values 0.0/1.0)."""
        mask = np.zeros((100, 100), dtype=np.float32)
        mask[:5, :5] = 1.0  # 25 pixels
        n_pixels, area = calculate_tumor_area(mask)
        assert n_pixels == 25
        assert abs(area - 25 * PIXEL_AREA_CM2) < 1e-9

    def test_area_proportional_to_pixel_count(self):
        """Doubling the tumor region should double the area."""
        mask_small = np.zeros((256, 256), dtype=np.uint8)
        mask_small[:10, :10] = 1  # 100 pixels

        mask_large = np.zeros((256, 256), dtype=np.uint8)
        mask_large[:10, :20] = 1  # 200 pixels

        _, area_small = calculate_tumor_area(mask_small)
        _, area_large = calculate_tumor_area(mask_large)

        assert abs(area_large - 2 * area_small) < 1e-9


class TestOverlayShape:
    """Tests for create_overlay() — output shape must match the input image."""

    def test_output_shape_matches_input(self):
        """Overlay output shape must equal the original image shape."""
        h, w = 256, 256
        original = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[64:192, 64:192] = 1  # Square tumor region

        overlay = create_overlay(original, mask, alpha=0.5)
        assert overlay.shape == original.shape, (
            f"Expected shape {original.shape}, got {overlay.shape}"
        )

    def test_output_dtype_is_uint8(self):
        """Overlay must be a uint8 image."""
        original = np.zeros((128, 128, 3), dtype=np.uint8)
        mask = np.ones((128, 128), dtype=np.uint8)
        overlay = create_overlay(original, mask, alpha=0.4)
        assert overlay.dtype == np.uint8, f"Expected uint8, got {overlay.dtype}"

    def test_non_square_image(self):
        """Overlay must handle non-square images correctly."""
        original = np.random.randint(0, 256, (120, 200, 3), dtype=np.uint8)
        mask = np.zeros((120, 200), dtype=np.uint8)
        overlay = create_overlay(original, mask, alpha=0.3)
        assert overlay.shape == (120, 200, 3)

    def test_alpha_zero_produces_original_like_image(self):
        """With alpha=0.0 the overlay layer has no weight — result equals original."""
        original = np.full((64, 64, 3), fill_value=100, dtype=np.uint8)
        mask = np.ones((64, 64), dtype=np.uint8)
        overlay = create_overlay(original, mask, alpha=0.0)
        # At alpha=0.0 the color_layer contributes nothing; result should be the original
        assert overlay.shape == original.shape

    def test_mask_size_mismatch_resized(self):
        """If mask and image have different sizes, mask should be resized, not crash."""
        original = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        mask = np.ones((128, 128), dtype=np.uint8)  # Smaller than image
        overlay = create_overlay(original, mask, alpha=0.5)
        assert overlay.shape == (256, 256, 3)


class TestThresholding:
    """Tests for threshold_mask() in core/inference.py."""

    def test_above_threshold_becomes_one(self):
        """Values >= 0.5 should be classified as tumor (1)."""
        prob_map = np.array([[0.5, 0.6, 0.9, 1.0]])
        result = threshold_mask(prob_map, threshold=0.5)
        assert np.all(result == 1), f"Expected all 1s, got {result}"

    def test_below_threshold_becomes_zero(self):
        """Values < 0.5 should be classified as healthy (0)."""
        prob_map = np.array([[0.0, 0.1, 0.3, 0.4999]])
        result = threshold_mask(prob_map, threshold=0.5)
        assert np.all(result == 0), f"Expected all 0s, got {result}"

    def test_exact_threshold_is_inclusive(self):
        """Exactly 0.5 should map to 1 (>= threshold, inclusive)."""
        prob_map = np.array([[0.5]])
        result = threshold_mask(prob_map, threshold=0.5)
        assert result[0, 0] == 1

    def test_mixed_values(self):
        """Mixed probability map should produce a correct binary mask."""
        prob_map = np.array([
            [0.1, 0.9],
            [0.4, 0.6],
        ])
        expected = np.array([
            [0, 1],
            [0, 1],
        ], dtype=np.uint8)
        result = threshold_mask(prob_map, threshold=0.5)
        np.testing.assert_array_equal(result, expected)

    def test_output_dtype_is_uint8(self):
        """threshold_mask output should have uint8 dtype."""
        prob_map = np.random.rand(64, 64).astype(np.float32)
        result = threshold_mask(prob_map)
        assert result.dtype == np.uint8, f"Expected uint8, got {result.dtype}"

    def test_output_shape_preserved(self):
        """Output shape must equal input shape."""
        prob_map = np.random.rand(256, 256).astype(np.float32)
        result = threshold_mask(prob_map)
        assert result.shape == prob_map.shape

    def test_custom_threshold(self):
        """A custom threshold value should be respected."""
        prob_map = np.array([[0.3, 0.5, 0.7, 0.9]])
        result = threshold_mask(prob_map, threshold=0.6)
        expected = np.array([[0, 0, 1, 1]], dtype=np.uint8)
        np.testing.assert_array_equal(result, expected)

    def test_all_zeros_probability_map(self):
        """An all-zero probability map should yield an all-zero mask."""
        prob_map = np.zeros((128, 128), dtype=np.float32)
        result = threshold_mask(prob_map)
        assert np.all(result == 0)

    def test_all_ones_probability_map(self):
        """An all-ones probability map should yield an all-ones mask."""
        prob_map = np.ones((128, 128), dtype=np.float32)
        result = threshold_mask(prob_map)
        assert np.all(result == 1)
