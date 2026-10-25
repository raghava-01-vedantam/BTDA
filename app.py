import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.inference import load_model, run_inference, threshold_mask
from core.morphology import clean_mask
from core.postprocessing import calculate_tumor_area, create_overlay
from core.preprocessing import decode_image_as_rgb, preprocess_image
from core.regions import analyse_regions, create_outline_overlay
from core.report import _analyse_mask, _classify_tumor, _recommendations, generate_pdf_report
from db.database import init_db
from db.repository import (
    get_or_create_model_version,
    get_or_create_user,
    has_report_for_result,
    insert_report,
    insert_scan,
    insert_segmentation_result,
)
from ui.theme import apply_theme

st.set_page_config(
    page_title="Brain Tumor Detector",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_theme()
init_db()

CHECKPOINT_PATH = PROJECT_ROOT / "model" / "checkpoints" / "best_model.pth"
SAMPLE_INDEX_PATH = PROJECT_ROOT / "data" / "processed" / "test" / "sample_index.json"
ADMIN_PASSWORD = "1234"

# ── Sidebar: profile & role ──────────────────────────────────────────────────
st.sidebar.markdown("### Profile")
user_name = st.sidebar.text_input("Name", value=st.session_state.get("user_name", "Guest"))
role_choice = st.sidebar.selectbox("Role", options=["User", "Administrator"])

# Admin password gate
is_admin = False
if role_choice == "Administrator":
    pwd = st.sidebar.text_input("Admin password", type="password", key="admin_pwd")
    if pwd == ADMIN_PASSWORD:
        is_admin = True
        st.sidebar.success("Admin access granted")
    elif pwd != "":
        st.sidebar.error("Incorrect password")

user_role = "Administrator" if is_admin else "User"
st.session_state["user_name"] = user_name
st.session_state["user_role"] = user_role

user_id = get_or_create_user(user_name, user_role)

if not CHECKPOINT_PATH.exists():
    st.error("Model checkpoint not found. Please train a model first.")
    st.stop()


@st.cache_resource
def get_cached_model(path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return load_model(path, device=device), device


model, device = get_cached_model(str(CHECKPOINT_PATH))

model_version_id = get_or_create_model_version(
    name="UNet-ResNet34",
    backbone="resnet34",
    iou_score=0.7316,
    trained_date=datetime.now().strftime("%Y-%m-%d"),
)

# ── Page header ──────────────────────────────────────────────────────────────
st.markdown(
    '<div class="page-title">Brain Tumor Detector</div>'
    '<div class="page-subtitle">MRI Scan Analysis and Clinical Diagnostic Report</div>',
    unsafe_allow_html=True,
)

# ── Sample index (admin only) ─────────────────────────────────────────────────
sample_items: list[dict] = []
if is_admin and SAMPLE_INDEX_PATH.exists():
    try:
        import json
        with open(SAMPLE_INDEX_PATH, "r", encoding="utf-8") as f:
            sample_items = json.load(f)
    except Exception:
        sample_items = []

sample_options = ["None"]
sample_map: dict = {}
for s in sample_items:
    img_name = Path(s.get("image", "")).name
    label = f"{s.get('label', 'Sample')} ({s.get('pixels', 0)} px) — {img_name}"
    sample_options.append(label)
    sample_map[label] = s

# ── Layout columns ────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 2.2], gap="large")

with col_left:
    uploaded_file = st.file_uploader(
        "Upload MRI Scan",
        type=["png", "jpg", "jpeg", "tif", "tiff"],
    )

    # Sample selector — admin only
    sample_choice = "None"
    if is_admin:
        sample_choice = st.selectbox("Test Sample (Admin)", options=sample_options, index=0)

    opacity = st.slider("Overlay Opacity", min_value=0.0, max_value=1.0, value=0.60, step=0.05)

    # Advanced options — admin only
    overlay_style = "Filled"
    pixel_spacing = 0.115
    options_placeholder = st.empty()

    if is_admin:
        with st.expander("Advanced Options", expanded=False):
            overlay_style = st.selectbox("Overlay style", options=["Filled", "Outline"])
            pixel_spacing = st.number_input(
                "Pixel spacing (cm)",
                min_value=0.001, max_value=1.0, value=0.115, step=0.005, format="%.3f",
            )
            options_placeholder = st.empty()

# ── Resolve source image ──────────────────────────────────────────────────────
source_image_rgb: Optional[np.ndarray] = None
source_filename: str = ""

if uploaded_file is not None:
    source_filename = uploaded_file.name
    try:
        source_image_rgb = decode_image_as_rgb(uploaded_file)
    except Exception as e:
        st.error(f"Could not load image: {e}")
        st.stop()
elif is_admin and sample_choice != "None" and sample_choice in sample_map:
    sample_info = sample_map[sample_choice]
    img_rel = sample_info.get("image", sample_info.get("image_path", ""))
    full_img = PROJECT_ROOT / img_rel if not Path(img_rel).is_absolute() else Path(img_rel)
    if full_img.is_file():
        source_filename = full_img.name
        source_image_rgb = decode_image_as_rgb(full_img)

# ── Inference pipeline ────────────────────────────────────────────────────────
if source_image_rgb is not None:
    input_tensor = preprocess_image(source_image_rgb)
    prob_mask = run_inference(model, input_tensor, device)
    raw_bin_mask = threshold_mask(prob_mask, threshold=0.5)
    clean_bin_mask = clean_mask(raw_bin_mask)

    n_pixels, area_cm2 = calculate_tumor_area(clean_bin_mask, pixel_spacing_cm=pixel_spacing)
    region_stats = analyse_regions(clean_bin_mask)

    if n_pixels == 0:
        overlay_img = source_image_rgb.copy()
    elif overlay_style == "Outline":
        overlay_img = create_outline_overlay(source_image_rgb, clean_bin_mask, color=(255, 80, 80), thickness=2)
    else:
        overlay_img = create_overlay(source_image_rgb, clean_bin_mask, alpha=opacity)

    mask_display = (clean_bin_mask * 255).astype(np.uint8)

    # Region breakdown in admin panel
    if is_admin and region_stats.count > 1:
        with options_placeholder.container():
            st.markdown("**Region Breakdown**")
            rows = []
            for idx, r_px in enumerate(region_stats.per_region_pixels, 1):
                rows.append({"Region": idx, "Pixels": r_px, "Area (cm²)": f"{r_px * pixel_spacing ** 2:.3f}"})
            st.dataframe(pd.DataFrame(rows), hide_index=True)

    # Persist scan (once per unique image)
    scan_key = f"{source_filename}_{source_image_rgb.shape}_{int(source_image_rgb.sum() % 100000)}"
    if st.session_state.get("last_scan_key") != scan_key:
        scan_id = insert_scan(user_id=user_id, filename=source_filename)
        current_result_id = insert_segmentation_result(
            scan_id=scan_id,
            model_id=model_version_id,
            pixel_count=n_pixels,
            area_cm2=area_cm2,
            opacity=opacity,
        )
        st.session_state["last_scan_key"] = scan_key
        st.session_state["current_result_id"] = current_result_id
    else:
        current_result_id = st.session_state.get("current_result_id", 1)

    # ── Right column: images + metrics + clinical impression ─────────────────
    with col_right:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.image(source_image_rgb, caption="Original MRI", use_container_width=True)
        with c2:
            st.image(overlay_img, caption="AI Overlay", use_container_width=True)
        with c3:
            st.image(mask_display, caption="Segmentation Mask", use_container_width=True)

        st.markdown("<div style='margin-top: 0.8rem;'></div>", unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        m1.metric("Lesion Area", f"{area_cm2:.2f} cm²")
        m2.metric("Lesion Pixels", f"{n_pixels:,}")
        m3.metric("Foci Detected", f"{region_stats.count}")

        st.markdown("<div style='margin-top: 1.0rem;'></div>", unsafe_allow_html=True)

        # ── Radiological analysis ─────────────────────────────────────────────
        rf = _analyse_mask(
            binary_mask=clean_bin_mask,
            pixel_spacing_cm=pixel_spacing,
            n_foci=region_stats.count,
            largest_px=region_stats.largest_pixels,
            largest_diameter_px=region_stats.largest_diameter_px,
        )
        severity, _, _ = _classify_tumor(area_cm2, rf)
        badge_class = {
            "No Tumor": "badge-clear",
            "Mild":     "badge-mild",
            "Moderate": "badge-moderate",
            "Critical": "badge-critical",
        }.get(severity, "badge-clear")
        recs = _recommendations(severity, rf)
        rec_html = "".join(f"<li>{r}</li>" for r in recs)

        if severity == "No Tumor":
            imp_html = "No abnormal hyperintense lesion identified. Parenchymal signal appears within normal limits."
        else:
            ci_desc = "regular" if rf.circularity >= 0.70 else "irregular"
            imp_html = (
                f"<b>{rf.n_foci} focus/foci</b> detected &mdash; "
                f"max diameter <b>{rf.max_diameter_mm:.1f} mm</b>, "
                f"RANO: <b>{rf.max_diameter_mm:.1f}&times;{rf.perp_diameter_mm:.1f} mm</b> "
                f"({rf.rano_product_mm2:.0f} mm²). "
                f"Margins: <b>{ci_desc}</b> (CI&nbsp;{rf.circularity:.2f}). "
                f"Brain burden: <b>{rf.brain_burden_pct:.2f}%</b>. "
                f"Radiological profile: <b>{rf.who_grade_suspicion}</b>."
            )

        st.markdown(
            f"""
            <div class="impression-card">
                <h4>Clinical Impression</h4>
                <span class="badge {badge_class}">{severity}</span>
                <p style="margin-top:0.5rem">{imp_html}</p>
                <h4 style="margin-top:0.9rem">Clinical Recommendations</h4>
                <ul style="color:#1a202c;font-size:0.88rem;line-height:1.7;padding-left:1.2rem;margin:0.3rem 0 0 0">
                    {rec_html}
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div style='margin-top: 0.8rem;'></div>", unsafe_allow_html=True)

        pdf_bytes = generate_pdf_report(
            original_img=source_image_rgb,
            overlay_img=overlay_img,
            n_pixels=n_pixels,
            area_cm2=area_cm2,
            user_name=user_name,
            model_name="UNet-ResNet34",
            n_regions=region_stats.count,
            largest_px=region_stats.largest_pixels,
            pixel_spacing_cm=pixel_spacing,
            binary_mask=clean_bin_mask,
            largest_diameter_px=region_stats.largest_diameter_px,
        )

        impression_text = f"{severity} — {area_cm2:.2f} cm²" if n_pixels > 0 else "No tumour detected"
        report_pdf_name = f"report_{Path(source_filename).stem}.pdf"

        def record_report_download():
            if not has_report_for_result(current_result_id):
                insert_report(
                    result_id=current_result_id,
                    filename=report_pdf_name,
                    area_cm2=area_cm2,
                    pixel_count=n_pixels,
                    impression=impression_text,
                )

        st.download_button(
            label="⬇ Download Clinical Report (PDF)",
            data=pdf_bytes,
            file_name=report_pdf_name,
            mime="application/pdf",
            on_click=record_report_download,
        )
