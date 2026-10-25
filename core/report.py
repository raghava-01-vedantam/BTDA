from __future__ import annotations

import io
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import cv2
import numpy as np
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    HRFlowable,
    Image as RLImage,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)



@dataclass
class RadiologicalFeatures:
    """All measurable features a neuroradiologist derives from a 2D MRI slice."""
    total_pixels: int
    area_cm2: float

    max_diameter_mm: float
    perp_diameter_mm: float
    rano_product_mm2: float       

    
    circularity: float           
    aspect_ratio: float          
    perimeter_mm: float

    n_foci: int
    is_multifocal: bool
    largest_focus_area_cm2: float

    brain_burden_pct: float      

    quadrant: str                 
    centroid_x_pct: float        
    centroid_y_pct: float        

    who_grade_suspicion: str      
    severity: str                 
    severity_rationale: str


def _analyse_mask(
    binary_mask: np.ndarray,
    pixel_spacing_cm: float,
    n_foci: int,
    largest_px: int,
    largest_diameter_px: float,
) -> RadiologicalFeatures:
    """
    Derive all radiological features from the binary segmentation mask.
    Mirrors how a neuroradiologist reads a 2D axial slice.
    """
    px_mm = pixel_spacing_cm * 10.0          # cm → mm
    mask_u8 = (binary_mask > 0).astype(np.uint8)
    total_px = int(mask_u8.sum())
    total_brain_px = binary_mask.size        # 256×256 frame used as proxy

    if total_px == 0:
        return RadiologicalFeatures(
            total_pixels=0, area_cm2=0.0,
            max_diameter_mm=0.0, perp_diameter_mm=0.0, rano_product_mm2=0.0,
            circularity=0.0, aspect_ratio=1.0, perimeter_mm=0.0,
            n_foci=0, is_multifocal=False, largest_focus_area_cm2=0.0,
            brain_burden_pct=0.0, quadrant="N/A",
            centroid_x_pct=50.0, centroid_y_pct=50.0,
            who_grade_suspicion="N/A",
            severity="No Tumor",
            severity_rationale="No segmented lesion identified on this slice.",
        )

    area_cm2 = total_px * (pixel_spacing_cm ** 2)
    largest_area_cm2 = largest_px * (pixel_spacing_cm ** 2)

    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)
    largest_cnt = contours_sorted[0]

    rect = cv2.minAreaRect(largest_cnt)        
    box_w_px, box_h_px = rect[1]
    long_px  = max(box_w_px, box_h_px)
    short_px = min(box_w_px, box_h_px)

    max_diameter_mm  = long_px  * px_mm
    perp_diameter_mm = short_px * px_mm
    rano_product_mm2 = max_diameter_mm * perp_diameter_mm

    # Aspect ratio
    aspect_ratio = (long_px / short_px) if short_px > 0 else 1.0

    # Circularity
    # 4π·A / P²  ∈ (0, 1].  < 0.7 = irregular margins (concerning)
    perimeter_px = cv2.arcLength(largest_cnt, closed=True)
    perimeter_mm = perimeter_px * px_mm
    area_largest_px2 = largest_px                        
    if perimeter_px > 0:
        circularity = (4.0 * math.pi * area_largest_px2) / (perimeter_px ** 2)
        circularity = min(circularity, 1.0)
    else:
        circularity = 0.0

    # Centroid & quadrant 
    M = cv2.moments(largest_cnt)
    if M["m00"] > 0:
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
    else:
        cx = cy = binary_mask.shape[0] / 2.0

    h_px, w_px = binary_mask.shape
    cx_pct = (cx / w_px) * 100.0
    cy_pct = (cy / h_px) * 100.0

    lr = "Right" if cx_pct < 50 else "Left"
    if cy_pct < 33:
        fb = "Frontal"
    elif cy_pct < 66:
        fb = "Temporal / Parietal"
    else:
        fb = "Occipital"
    quadrant = f"{lr} {fb}"

    # Brain burden
    brain_burden_pct = (total_px / total_brain_px) * 100.0

    # Multifocality 
    is_multifocal = n_foci > 1

    # Clinical severity scoring (multi-factor, evidence-based)
    
    score = 0
    rationale_parts = []

    if max_diameter_mm < 10:
        score += 0; rationale_parts.append(f"max diameter {max_diameter_mm:.1f} mm (< 10 mm, micro-lesion)")
    elif max_diameter_mm < 20:
        score += 1; rationale_parts.append(f"max diameter {max_diameter_mm:.1f} mm (10–20 mm, small lesion)")
    elif max_diameter_mm < 40:
        score += 2; rationale_parts.append(f"max diameter {max_diameter_mm:.1f} mm (20–40 mm, intermediate)")
    else:
        score += 3; rationale_parts.append(f"max diameter {max_diameter_mm:.1f} mm (> 40 mm, large)")


    if circularity >= 0.75:
        score += 0; rationale_parts.append(f"well-defined margins (CI = {circularity:.2f})")
    elif circularity >= 0.50:
        score += 1; rationale_parts.append(f"mildly irregular margins (CI = {circularity:.2f})")
    else:
        score += 2; rationale_parts.append(f"irregular/lobulated margins (CI = {circularity:.2f}), concerning for aggressive histology")


    if is_multifocal:
        score += 2; rationale_parts.append(f"multifocal disease ({n_foci} foci), raises concern for high-grade glioma or metastatic lesions")
    else:
        rationale_parts.append("unifocal lesion")


    if brain_burden_pct >= 5.0:
        score += 2; rationale_parts.append(f"significant brain burden ({brain_burden_pct:.1f}%), mass effect likely")
    elif brain_burden_pct >= 2.0:
        score += 1; rationale_parts.append(f"moderate brain burden ({brain_burden_pct:.1f}%)")
    else:
        rationale_parts.append(f"low brain burden ({brain_burden_pct:.1f}%)")


    if aspect_ratio > 2.5:
        score += 1; rationale_parts.append(f"elongated morphology (AR = {aspect_ratio:.1f}), possible infiltrative pattern")


    if score <= 1:
        severity = "Mild"
    elif score <= 3:
        severity = "Moderate"
    else:
        severity = "Critical"


    if score <= 2 and circularity >= 0.60 and not is_multifocal:
        who_grade_suspicion = "Low-grade (WHO I–II)"
    else:
        who_grade_suspicion = "High-grade (WHO III–IV)"

    severity_rationale = "; ".join(rationale_parts)

    return RadiologicalFeatures(
        total_pixels=total_px,
        area_cm2=area_cm2,
        max_diameter_mm=max_diameter_mm,
        perp_diameter_mm=perp_diameter_mm,
        rano_product_mm2=rano_product_mm2,
        circularity=circularity,
        aspect_ratio=aspect_ratio,
        perimeter_mm=perimeter_mm,
        n_foci=n_foci,
        is_multifocal=is_multifocal,
        largest_focus_area_cm2=largest_area_cm2,
        brain_burden_pct=brain_burden_pct,
        quadrant=quadrant,
        centroid_x_pct=cx_pct,
        centroid_y_pct=cy_pct,
        who_grade_suspicion=who_grade_suspicion,
        severity=severity,
        severity_rationale=severity_rationale,
    )


def _classify_tumor(area_cm2: float, rf: Optional["RadiologicalFeatures"] = None):
    """
    Public helper — returns (severity, color_hex, short_description).
    Uses full radiological features if available, otherwise falls back to area.
    """
    if rf is not None:
        severity = rf.severity
    else:
        if area_cm2 <= 0:
            severity = "No Tumor"
        elif area_cm2 < 1.0:
            severity = "Mild"
        elif area_cm2 < 4.0:
            severity = "Moderate"
        else:
            severity = "Critical"

    color_map = {
        "No Tumor": "#00c9a7",
        "Mild":     "#23c55e",
        "Moderate": "#e3b341",
        "Critical": "#f85149",
    }
    desc_map = {
        "No Tumor": "No abnormal segmented region detected on this slice.",
        "Mild":     "Small or well-circumscribed lesion; low immediate risk.",
        "Moderate": "Intermediate lesion; specialist evaluation required.",
        "Critical": "Large or complex lesion; urgent specialist review required.",
    }
    return severity, color_map.get(severity, "#8b949e"), desc_map.get(severity, "")


def _recommendations(severity: str, rf: Optional["RadiologicalFeatures"] = None) -> list[str]:
    base = {
        "No Tumor": [
            "No immediate intervention required.",
            "Routine follow-up MRI in 12 months or as clinically indicated.",
            "Advise patient to report any new neurological symptoms promptly.",
        ],
        "Mild": [
            "Clinical correlation with presenting symptoms is advised.",
            "Follow-up MRI with gadolinium contrast in 3–6 months to assess stability.",
            "Neurology outpatient consultation within 4 weeks.",
            "Seizure prophylaxis: assess need for anti-epileptic drug (AED) therapy.",
            "Avoid strenuous activity until further evaluation.",
        ],
        "Moderate": [
            "Urgent neurology referral within 1–2 weeks.",
            "Gadolinium-enhanced MRI ± MR Spectroscopy to characterize lesion biochemistry.",
            "Consider CT-guided or stereotactic biopsy for histopathological grading.",
            "Corticosteroid therapy (Dexamethasone 4 mg QDS) if perilesional edema suspected.",
            "Anti-epileptic prophylaxis (Levetiracetam 500 mg BD) per neurology assessment.",
            "Neuropsychological screening for cognitive function baseline.",
        ],
        "Critical": [
            "URGENT: Neurosurgical consultation within 24–48 hours.",
            "Multi-disciplinary team (MDT) review: neurosurgery, neuro-oncology, radiation oncology.",
            "Full staging workup: whole-brain MRI, PET-CT, systemic evaluation for primary/secondary.",
            "High-dose corticosteroid therapy (Dexamethasone 8–16 mg/day) for cerebral edema.",
            "Anti-epileptic cover: Levetiracetam 500–1000 mg BD; consider valproate if required.",
            "Surgical options: maximal safe resection vs. stereotactic biopsy — discuss with MDT.",
            "Adjuvant Temozolomide chemotherapy + fractionated radiotherapy (Stupp protocol) if GBM confirmed.",
            "Discuss palliative care and supportive measures with patient and family.",
        ],
    }

    recs = list(base.get(severity, []))

    if rf is not None:
        if rf.is_multifocal and severity != "No Tumor":
            recs.insert(1, "Multifocal disease: evaluate for metastatic disease — MRI whole-spine and CT chest/abdomen/pelvis recommended.")
        if rf.circularity < 0.45 and severity != "No Tumor":
            recs.insert(1, "Irregular lesion margins: histopathological correlation essential; ring-enhancing pattern should be excluded with contrast MRI.")
        if rf.brain_burden_pct > 5.0:
            recs.insert(1, f"Significant mass burden ({rf.brain_burden_pct:.1f}% of imaged brain area): assess for midline shift with full MRI series; ICP monitoring may be indicated.")

    return recs



def _numpy_to_rl_image(array: np.ndarray, max_width: float, max_height: float) -> RLImage:
    if array.dtype != np.uint8:
        array = (array * 255).clip(0, 255).astype(np.uint8)
    pil_img = Image.fromarray(array).convert("RGB")
    orig_w, orig_h = pil_img.size
    scale = min(max_width / orig_w, max_height / orig_h)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    return RLImage(buf, width=orig_w * scale, height=orig_h * scale)


# Main Report Generator
def generate_pdf_report(
    original_img: np.ndarray,
    overlay_img: np.ndarray,
    n_pixels: int,
    area_cm2: float,
    timestamp: datetime | None = None,
    user_name: str = "Guest",
    model_name: str = "U-Net / ResNet34",
    n_regions: int = 0,
    largest_px: int = 0,
    pixel_spacing_cm: float = 0.115,
    binary_mask: Optional[np.ndarray] = None,
    largest_diameter_px: float = 0.0,
) -> io.BytesIO:
    if timestamp is None:
        timestamp = datetime.now()


    if binary_mask is not None and n_pixels > 0:
        rf = _analyse_mask(
            binary_mask=binary_mask,
            pixel_spacing_cm=pixel_spacing_cm,
            n_foci=n_regions,
            largest_px=largest_px,
            largest_diameter_px=largest_diameter_px,
        )
    elif n_pixels == 0:
        rf = _analyse_mask(
            binary_mask=np.zeros((256, 256), dtype=np.uint8),
            pixel_spacing_cm=pixel_spacing_cm,
            n_foci=0, largest_px=0, largest_diameter_px=0.0,
        )
    else:
        rf = None

    severity, sev_color, sev_short_desc = _classify_tumor(area_cm2, rf)
    recs = _recommendations(severity, rf)

    # PDF Setup
    pdf_buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        pdf_buffer, pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=1.8 * cm, bottomMargin=2 * cm,
    )
    page_width, _ = A4
    W = page_width - 4 * cm
    base = getSampleStyleSheet()

    NAVY   = colors.HexColor("#0d1117")
    TEAL   = colors.HexColor("#00c9a7")
    SLATE  = colors.HexColor("#1e2530")
    BORDER = colors.HexColor("#2a3340")
    LIGHT  = colors.HexColor("#e6edf3")
    MUTED  = colors.HexColor("#8b949e")
    SEV_C  = colors.HexColor(sev_color)

    def _s(name, **kw):
        defaults = dict(parent=base["Normal"], fontSize=9, textColor=LIGHT, spaceAfter=4, leading=13)
        defaults.update(kw)
        return ParagraphStyle(name, **defaults)

    S = {
        "inst":     _s("Inst",  fontSize=8,  textColor=MUTED, alignment=TA_LEFT,    spaceAfter=0),
        "title":    _s("Ttl",   fontSize=17, textColor=LIGHT, alignment=TA_LEFT,    spaceAfter=2,  fontName="Helvetica-Bold"),
        "meta":     _s("Meta",  fontSize=8,  textColor=MUTED, alignment=TA_LEFT,    spaceAfter=1),
        "sec":      _s("Sec",   fontSize=9.5,textColor=TEAL,  alignment=TA_LEFT,    spaceBefore=10,spaceAfter=5, fontName="Helvetica-Bold"),
        "body":     _s("Body",  fontSize=8.5,textColor=LIGHT, alignment=TA_JUSTIFY, spaceAfter=4,  leading=13),
        "bullet":   _s("Bul",   fontSize=8.5,textColor=LIGHT, leftIndent=10,        spaceAfter=3,  leading=13),
        "sev_lbl":  _s("Slbl",  fontSize=20, textColor=SEV_C, fontName="Helvetica-Bold", spaceAfter=1),
        "sev_sub":  _s("Ssub",  fontSize=8.5,textColor=MUTED, spaceAfter=0),
        "tbl_hdr":  _s("Thdr",  fontSize=8,  textColor=LIGHT, fontName="Helvetica-Bold"),
        "tbl_val":  _s("Tval",  fontSize=8,  textColor=LIGHT),
        "cap":      _s("Cap",   fontSize=7,  textColor=MUTED, alignment=TA_CENTER,  spaceAfter=3),
        "disc":     _s("Disc",  fontSize=7.5,textColor=MUTED, alignment=TA_CENTER,  spaceAfter=0),
        "flag":     _s("Flag",  fontSize=8,  textColor=colors.HexColor("#e3b341"), spaceAfter=3),
    }

    def _tbl_style(has_header=True):
        ts = [
            ("FONTNAME",      (0, 0), (-1, -1), "Helvetica"),
            ("FONTSIZE",      (0, 0), (-1, -1), 8),
            ("TEXTCOLOR",     (0, 0), (-1, -1), LIGHT),
            ("ALIGN",         (1, 0), (-1, -1), "RIGHT"),
            ("ALIGN",         (0, 0), (0, -1),  "LEFT"),
            ("LEFTPADDING",   (0, 0), (-1, -1), 7),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 7),
            ("TOPPADDING",    (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("GRID",          (0, 0), (-1, -1), 0.35, BORDER),
        ]
        if has_header:
            ts += [
                ("BACKGROUND",  (0, 0), (-1, 0),  SLATE),
                ("FONTNAME",    (0, 0), (-1, 0),  "Helvetica-Bold"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [NAVY, colors.HexColor("#111720")]),
            ]
        return TableStyle(ts)

    story = []

    hdr = Table([[
        Paragraph("BRAIN TUMOR DETECTOR  ·  DIAGNOSTIC REPORT", S["inst"]),
        Paragraph(f"Generated: {timestamp.strftime('%d %b %Y, %H:%M')}", S["inst"]),
    ]], colWidths=[W * 0.6, W * 0.4])
    hdr.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), NAVY),
        ("ALIGN",         (1, 0), (1, 0),   "RIGHT"),
        ("LEFTPADDING",   (0, 0), (-1, -1), 0),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 0),
        ("TOPPADDING",    (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(hdr)
    story.append(Paragraph("Cranial MRI — AI-Assisted Diagnostic Report", S["title"]))
    story.append(Paragraph(
        f"Requestor / Patient: <b>{user_name}</b> &nbsp;|&nbsp; "
        f"Analysis engine: {model_name} &nbsp;|&nbsp; "
        f"Modality: MRI (Axial 2D slice)", S["meta"],
    ))
    story.append(Spacer(1, 0.15 * cm))
    story.append(HRFlowable(width="100%", thickness=1.2, color=TEAL, spaceAfter=8))

    story.append(Paragraph("1. CLINICAL IMPRESSION", S["sec"]))

    if severity == "No Tumor":
        impression = (
            "No abnormal hyperintense or hypointense lesion is identified on the submitted axial MRI slice. "
            "The parenchymal signal and sulcal/gyral pattern appear within normal limits for the imaged region. "
            "No mass effect, midline shift, or abnormal enhancement is detected."
        )
    else:
        margin_txt  = "well-defined and regular" if rf and rf.circularity >= 0.70 else "irregular and lobulated"
        shape_txt   = "rounded" if rf and rf.aspect_ratio < 1.5 else ("elongated" if rf and rf.aspect_ratio < 2.5 else "markedly elongated / infiltrative")
        focus_txt   = "unifocal" if not (rf and rf.is_multifocal) else f"multifocal ({n_regions} discrete foci)"
        rano_str    = f"{rf.max_diameter_mm:.1f} × {rf.perp_diameter_mm:.1f} mm (RANO product: {rf.rano_product_mm2:.0f} mm²)" if rf else "N/A"
        burden_str  = f"{rf.brain_burden_pct:.2f}%" if rf else "N/A"
        who_str     = rf.who_grade_suspicion if rf else "Undetermined"

        impression = (
            f"An abnormal lesion is identified on the submitted axial MRI slice. "
            f"The lesion is <b>{focus_txt}</b>, with a <b>{shape_txt}</b>, {margin_txt} morphology. "
            f"AI-derived bidimensional measurement (RANO criteria): <b>{rano_str}</b>. "
            f"Total segmented area: <b>{area_cm2:.2f} cm²</b> representing approximately <b>{burden_str}</b> "
            f"of the imaged brain parenchyma. "
            f"Lesion margin circularity index (CI): <b>{rf.circularity:.2f}</b> (CI &lt; 0.70 indicates irregular margins). "
            f"Based on morphological and dimensional analysis, the radiological profile is most consistent with "
            f"<b>{who_str}</b> neoplasm. Final histopathological grading requires biopsy."
        )

    story.append(Paragraph(impression, S["body"]))

    if rf and severity != "No Tumor" and rf.severity_rationale:
        story.append(Paragraph(
            f"<b>Scoring rationale:</b> {rf.severity_rationale}.",
            S["flag"],
        ))

    # Severity badge table
    sev_left = Paragraph(severity.upper(), S["sev_lbl"])
    sev_right_lines = [f"WHO Grade Suspicion: <b>{rf.who_grade_suspicion if rf else 'N/A'}</b>"]
    if rf and severity != "No Tumor":
        sev_right_lines += [
            f"Max diameter: <b>{rf.max_diameter_mm:.1f} mm</b>",
            f"RANO bidimensional: <b>{rf.rano_product_mm2:.0f} mm²</b>",
            f"Brain burden: <b>{rf.brain_burden_pct:.2f}%</b>",
            f"Foci: <b>{n_regions} ({'multifocal' if rf.is_multifocal else 'unifocal'})</b>",
        ]
    sev_right = Paragraph("<br/>".join(sev_right_lines), S["sev_sub"])

    sev_tbl = Table([[sev_left, sev_right]], colWidths=[W * 0.3, W * 0.7])
    sev_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), SLATE),
        ("BOX",           (0, 0), (-1, -1), 0.8, SEV_C),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 10),
        ("TOPPADDING",    (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LINEAFTER",     (0, 0), (0, -1),  0.6, BORDER),
    ]))
    story.append(sev_tbl)
    story.append(Spacer(1, 0.25 * cm))

    story.append(Paragraph("2. IMAGING — AXIAL MRI SLICE", S["sec"]))
    imw = (W / 2) - 0.3 * cm
    orig_rl = _numpy_to_rl_image(original_img, imw, 6.0 * cm)
    over_rl = _numpy_to_rl_image(overlay_img,  imw, 6.0 * cm)
    img_tbl = Table([[orig_rl, over_rl]], colWidths=[imw + 0.3 * cm, imw + 0.3 * cm])
    img_tbl.setStyle(TableStyle([
        ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("BACKGROUND",    (0, 0), (-1, -1), SLATE),
        ("BOX",           (0, 0), (-1, -1), 0.5, BORDER),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
        ("TOPPADDING",    (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    story.append(img_tbl)
    cap_tbl = Table(
        [[Paragraph("Figure 1: Original MRI slice", S["cap"]),
          Paragraph("Figure 2: AI segmentation overlay (lesion highlighted)", S["cap"])]],
        colWidths=[imw + 0.3 * cm, imw + 0.3 * cm],
    )
    cap_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), SLATE),
        ("BOX",           (0, 0), (-1, -1), 0.5, BORDER),
        ("TOPPADDING",    (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(cap_tbl)
    story.append(Spacer(1, 0.2 * cm))

    story.append(Paragraph("3. QUANTITATIVE RADIOLOGICAL FINDINGS", S["sec"]))

    if rf and severity != "No Tumor":
        ci_interp = "regular (benign morphology)" if rf.circularity >= 0.75 else \
                    ("mildly irregular" if rf.circularity >= 0.50 else "highly irregular (aggressive morphology)")
        ar_interp = "rounded" if rf.aspect_ratio < 1.5 else ("ovoid/lobulated" if rf.aspect_ratio < 2.5 else "elongated/infiltrative")
        loc_interp = rf.quadrant
        rows = [
            ["Measurement", "Value", "Clinical Context"],
            ["Max diameter (RANO long axis)",    f"{rf.max_diameter_mm:.2f} mm",         "< 10 mm micro | 10–20 small | 20–40 inter | > 40 large"],
            ["Perp. diameter (RANO short axis)", f"{rf.perp_diameter_mm:.2f} mm",        "Bidimensional RANO criterion"],
            ["RANO bidimensional product",       f"{rf.rano_product_mm2:.1f} mm²",       "Progression defined as > 25% increase"],
            ["Total lesion area (axial)",        f"{rf.area_cm2:.4f} cm²",              f"({rf.total_pixels:,} pixels at {pixel_spacing_cm:.3f} cm/px)"],
            ["Largest single focus area",        f"{rf.largest_focus_area_cm2:.4f} cm²", "—"],
            ["Circularity index (CI)",           f"{rf.circularity:.3f}",               f"0–1; > 0.70 regular; ≤ 0.50 irregular → {ci_interp}"],
            ["Aspect ratio (L/W)",               f"{rf.aspect_ratio:.2f}",              f"Shape: {ar_interp}"],
            ["Perimeter",                        f"{rf.perimeter_mm:.1f} mm",           "Lesion boundary length"],
            ["Number of foci",                   str(rf.n_foci),                        "Multifocal if > 1"],
            ["Estimated brain burden",           f"{rf.brain_burden_pct:.2f}%",         "> 5% → mass-effect risk"],
            ["Lesion location (approx.)",        loc_interp,                            "Based on centroid of largest focus"],
        ]
    else:
        rows = [
            ["Measurement", "Value", "Note"],
            ["Total lesion area",    "0.00 cm²",  "No lesion detected"],
            ["Lesion pixels",        "0",          "Below detection threshold (0.50)"],
            ["Max diameter (RANO)",  "0.0 mm",     "N/A"],
            ["Brain burden",         "0.00%",      "No lesion"],
        ]

    col_w = [W * 0.33, W * 0.18, W * 0.49]
    qtbl = Table(rows, colWidths=col_w)
    qtbl.setStyle(_tbl_style(has_header=True))
    story.append(qtbl)
    story.append(Spacer(1, 0.2 * cm))

    story.append(Paragraph("4. CLINICAL RECOMMENDATIONS", S["sec"]))
    for rec in recs:
        story.append(Paragraph(f"• {rec}", S["bullet"]))
    story.append(Spacer(1, 0.3 * cm))

    story.append(HRFlowable(width="100%", thickness=0.5, color=BORDER, spaceAfter=5))
    story.append(Paragraph(
        "⚠  This report is produced by an AI-assisted research prototype (Brain Tumor Detection System). "
        "It does not constitute a clinical diagnosis and must not be used as the sole basis for any medical decision. "
        "All findings must be reviewed and confirmed by a qualified neuroradiologist or neurologist. "
        "AI segmentation accuracy: IoU 0.73, Dice 0.85 on held-out test set.",
        S["disc"],
    ))

    doc.build(story)
    pdf_buffer.seek(0)
    return pdf_buffer
