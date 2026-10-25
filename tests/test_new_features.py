import io
import os
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.morphology import clean_mask
from core.postprocessing import calculate_tumor_area, create_overlay
from core.preprocessing import decode_image_as_rgb, preprocess_image
from core.regions import (
    RegionStats,
    analyse_regions,
    compute_gt_metrics,
    create_outline_overlay,
)
from db.database import init_db
from db.repository import (
    get_or_create_model_version,
    get_or_create_user,
    has_report_for_result,
    insert_report,
    insert_scan,
    insert_segmentation_result,
    list_history,
)


def test_empty_mask_overlay_unchanged():
    orig = np.random.randint(50, 200, (256, 256, 3), dtype=np.uint8)
    empty_mask = np.zeros((256, 256), dtype=np.uint8)
    overlay = create_overlay(orig, empty_mask, alpha=0.6)
    assert np.array_equal(orig, overlay)


def test_positive_mask_selective_blend():
    orig = np.full((100, 100, 3), 100, dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:40, 20:40] = 1

    overlay = create_overlay(orig, mask, mask_color=(255, 0, 0), alpha=0.5)
    assert np.all(overlay[0:10, 0:10] == 100)
    assert overlay[25, 25, 0] > 100


def test_tiff_8bit_decode():
    gray_arr = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
    pil_gray = Image.fromarray(gray_arr, mode="L")
    buf = io.BytesIO()
    pil_gray.save(buf, format="TIFF")
    buf.seek(0)

    decoded = decode_image_as_rgb(buf)
    assert decoded.shape == (64, 64, 3)
    assert decoded.dtype == np.uint8
    assert np.array_equal(decoded[:, :, 0], decoded[:, :, 1])


def test_tiff_16bit_decode():
    arr_16 = np.linspace(0, 65535, 10000, dtype=np.uint16).reshape((100, 100))
    pil_16 = Image.fromarray(arr_16, mode="I;16")
    buf = io.BytesIO()
    pil_16.save(buf, format="TIFF")
    buf.seek(0)

    decoded = decode_image_as_rgb(buf)
    assert decoded.shape == (100, 100, 3)
    assert decoded.dtype == np.uint8
    assert decoded.min() == 0
    assert decoded.max() == 255


def test_morphology_noise_removal():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10, 10] = 1
    mask[20, 20] = 1
    mask[40:80, 40:80] = 1

    cleaned = clean_mask(mask, min_pixels=30)
    assert cleaned[10, 10] == 0
    assert cleaned[20, 20] == 0
    assert cleaned[50, 50] == 1
    assert cleaned.sum() >= 1550


def test_morphology_hole_filling():
    mask = np.ones((20, 20), dtype=np.uint8)
    mask[10, 10] = 0

    cleaned = clean_mask(mask, min_pixels=10)
    assert cleaned[10, 10] == 1


def test_analyse_regions_synthetic():
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:20, 10:20] = 1
    mask[50:70, 50:70] = 1

    stats = analyse_regions(mask)
    assert stats.count == 2
    assert stats.total_pixels == 500
    assert stats.largest_pixels == 400
    assert stats.per_region_pixels == [400, 100]
    expected_diameter = 2.0 * np.sqrt(400.0 / np.pi)
    assert abs(stats.largest_diameter_px - expected_diameter) < 1e-4


def test_analyse_regions_empty():
    empty = np.zeros((64, 64), dtype=np.uint8)
    stats = analyse_regions(empty)
    assert stats.count == 0
    assert stats.total_pixels == 0
    assert stats.largest_pixels == 0
    assert stats.largest_diameter_px == 0.0
    assert stats.per_region_pixels == []


def test_outline_overlay():
    orig = np.zeros((100, 100, 3), dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:80, 20:80] = 1

    outline = create_outline_overlay(orig, mask, color=(255, 80, 80), thickness=2)
    assert outline.shape == (100, 100, 3)
    assert outline.dtype == np.uint8
    assert np.all(outline[50, 50] == 0)
    assert outline[20, 50, 0] == 255


def test_pixel_spacing_area_calculation():
    mask = np.ones((10, 10), dtype=np.uint8)
    spacing = 0.115
    expected_area = 100 * (0.115 ** 2)

    n_px, area_cm2 = calculate_tumor_area(mask, pixel_spacing_cm=spacing)
    assert n_px == 100
    assert abs(area_cm2 - expected_area) < 1e-6


def test_gt_metrics_perfect_and_imperfect():
    pred = np.zeros((50, 50), dtype=np.uint8)
    gt = np.zeros((50, 50), dtype=np.uint8)
    pred[10:30, 10:30] = 1
    gt[10:30, 10:30] = 1

    m_perfect = compute_gt_metrics(pred, gt)
    assert abs(m_perfect["dice"] - 1.0) < 1e-4
    assert abs(m_perfect["iou"] - 1.0) < 1e-4

    pred_half = np.zeros((50, 50), dtype=np.uint8)
    pred_half[10:30, 10:20] = 1
    m_half = compute_gt_metrics(pred_half, gt)
    assert abs(m_half["precision"] - 1.0) < 1e-4
    assert abs(m_half["recall"] - 0.5) < 1e-4
    assert abs(m_half["dice"] - 0.66666) < 1e-3


def test_database_fk_and_snapshot(monkeypatch, tmp_path):
    test_db = tmp_path / "test_app.db"
    monkeypatch.setenv("BRAIN_DB_PATH", str(test_db))

    init_db()

    user_id = get_or_create_user("Dr. Smith", "Administrator")
    assert user_id > 0

    model_id = get_or_create_model_version("UNet-Test", "resnet34", 0.75, "2026-10-09")
    assert model_id > 0

    scan_id = insert_scan(user_id, "test_slice_01.png")
    assert scan_id > 0

    result_id = insert_segmentation_result(scan_id, model_id, 450, 5.95, 0.6)
    assert result_id > 0
    assert not has_report_for_result(result_id)

    report_id = insert_report(result_id, "report_01.pdf", 5.95, 450, "Segmented region detected")
    assert report_id > 0
    assert has_report_for_result(result_id)

    history = list_history(user_id=user_id)
    assert len(history) == 1
    assert history[0]["filename"] == "test_slice_01.png"
    assert history[0]["report"] == "Yes"

    con = sqlite3.connect(test_db)
    con.execute("PRAGMA foreign_keys = ON")
    with pytest.raises(sqlite3.IntegrityError):
        con.execute(
            "INSERT INTO mri_scans (user_id, scan_filename, scan_sequence, scan_upload_date)"
            " VALUES (99999, 'bad.png', 'T1', '2026-10-09')"
        )
    con.close()


def test_train_smoke_run(tmp_path):
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "model" / "train.py"),
        "--epochs", "1",
        "--max-samples", "16",
        "--batch-size", "8",
        "--run-name", "pytest_smoke",
        "--checkpoint-dir", str(tmp_path / "checkpoints"),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"train.py failed: {res.stderr}\n{res.stdout}"
    assert (tmp_path / "checkpoints" / "runs" / "pytest_smoke" / "best_model.pth").exists()
