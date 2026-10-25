import io
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.inference import load_model, run_inference, threshold_mask
from core.morphology import clean_mask
from core.postprocessing import calculate_tumor_area, create_overlay
from core.preprocessing import decode_image_as_rgb, preprocess_image
from core.regions import analyse_regions, compute_gt_metrics, create_outline_overlay
from core.report import generate_pdf_report
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

print("[1] Initializing and verifying database...")
init_db()
user_id = get_or_create_user("Dr. Tester", "Administrator")
model_id = get_or_create_model_version("UNet-ResNet34", "resnet34", 0.7316, "2026-10-09")
assert user_id > 0 and model_id > 0
print("    => DB initialized and seeded.")

print("[2] Loading model checkpoint...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt_path = PROJECT_ROOT / "model" / "checkpoints" / "best_model.pth"
assert ckpt_path.exists(), "Checkpoint does not exist"
model = load_model(str(ckpt_path), device=device)
print(f"    => Model loaded successfully on {device}.")

print("[3] Testing tumor-positive sample from sample_index.json...")
sample_index_file = PROJECT_ROOT / "data" / "processed" / "test" / "sample_index.json"
assert sample_index_file.exists()
with open(sample_index_file) as f:
    samples = json.load(f)

pos_sample = next(s for s in samples if s["label"] in ["Tumor, medium", "Tumor, large"])
img_path = PROJECT_ROOT / pos_sample["image"]
mask_path = PROJECT_ROOT / pos_sample["mask"]
assert img_path.exists() and mask_path.exists()

img_rgb = decode_image_as_rgb(img_path)
tensor = preprocess_image(img_rgb)
prob = run_inference(model, tensor, device)
bin_mask = threshold_mask(prob, threshold=0.5)
clean = clean_mask(bin_mask)
n_px, area = calculate_tumor_area(clean)
regions = analyse_regions(clean)

assert n_px > 0, f"Expected tumor pixels > 0, got {n_px}"
assert area > 0.0, f"Expected tumor area > 0, got {area}"
assert regions.count >= 1, f"Expected region count >= 1, got {regions.count}"
print(f"    => Positive sample: {pos_sample['image']} -> Pixels={n_px}, Area={area:.2f} cm2, Regions={regions.count}")

print("[4] Testing tumor-negative sample...")
neg_sample = next(s for s in samples if not s["positive"])
neg_img_path = PROJECT_ROOT / neg_sample["image"]
neg_rgb = decode_image_as_rgb(neg_img_path)
neg_tensor = preprocess_image(neg_rgb)
neg_prob = run_inference(model, neg_tensor, device)
neg_bin = threshold_mask(neg_prob, threshold=0.5)
neg_clean = clean_mask(neg_bin)
neg_px, neg_area = calculate_tumor_area(neg_clean)
neg_regions = analyse_regions(neg_clean)
neg_overlay = create_overlay(neg_rgb, neg_clean)

assert neg_px == 0, f"Expected 0 pixels, got {neg_px}"
assert neg_area == 0.0, f"Expected 0.00 area, got {neg_area}"
assert neg_regions.count == 0, f"Expected 0 regions, got {neg_regions.count}"
assert np.array_equal(neg_rgb, neg_overlay), "Negative overlay changed original image pixels!"
print(f"    => Negative sample: {neg_sample['image']} -> Pixels={neg_px}, Area={neg_area:.2f} cm2, Overlay identical: TRUE")

print("[5] Testing TIFF upload handling (8-bit and 16-bit)...")
# Test synthetic 16-bit TIFF
arr16 = np.random.randint(0, 65535, (256, 256), dtype=np.uint16)
pil16 = Image.fromarray(arr16, mode="I;16")
buf16 = io.BytesIO()
pil16.save(buf16, format="TIFF")
buf16.seek(0)
decoded16 = decode_image_as_rgb(buf16)
assert decoded16.shape == (256, 256, 3) and decoded16.dtype == np.uint8
print("    => 16-bit TIFF decoded to uint8 RGB (256, 256, 3).")

# Test real raw dataset .tif if available
raw_tif_candidates = list(PROJECT_ROOT.glob("raw_data/**/*.tif"))
if raw_tif_candidates:
    real_tif = raw_tif_candidates[0]
    decoded_real = decode_image_as_rgb(real_tif)
    assert decoded_real.shape == (256, 256, 3) and decoded_real.dtype == np.uint8
    print(f"    => Real dataset TIF decoded: {real_tif.name} -> shape {decoded_real.shape}")

print("[6] Testing report generation & DB snapshot persistence...")
overlay_pos = create_overlay(img_rgb, clean)
pdf_buf = generate_pdf_report(
    original_img=img_rgb,
    overlay_img=overlay_pos,
    n_pixels=n_px,
    area_cm2=area,
    user_name="Dr. Tester",
    model_name="UNet-ResNet34",
    n_regions=regions.count,
    largest_px=regions.largest_pixels,
    pixel_spacing_cm=0.115,
)
pdf_bytes = pdf_buf.getvalue()
assert len(pdf_bytes) > 1000 and pdf_bytes.startswith(b"%PDF"), "Invalid PDF output"
print(f"    => Valid PDF generated ({len(pdf_bytes)} bytes).")

scan_id = insert_scan(user_id, pos_sample["image"])
res_id = insert_segmentation_result(scan_id, model_id, n_px, area, 0.6)
assert not has_report_for_result(res_id)
rep_id = insert_report(res_id, "test_report.pdf", area, n_px, "Segmented region detected")
assert has_report_for_result(res_id)

history = list_history(user_id=user_id)
assert any(h["result_id"] == res_id for h in history)
print("    => History joined successfully and verified.")

print("\nALL VERIFICATIONS PASSED!")
