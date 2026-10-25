import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from db.database import init_db
from db.repository import list_model_versions
from ui.theme import apply_theme

st.set_page_config(page_title="Model", layout="wide", initial_sidebar_state="expanded")
apply_theme()
init_db()

# Check user role from session state
user_role = st.session_state.get("user_role", "User")
user_name = st.session_state.get("user_name", "Guest")

st.markdown('<div class="page-title">Model Management</div>', unsafe_allow_html=True)

if user_role != "Administrator":
    st.warning("Administrator access required to view model specifications.")
    st.stop()

# 1. Registered model versions table
st.subheader("Registered Model Versions")
models = list_model_versions()
if models:
    df_models = pd.DataFrame(models)
    rename_map = {
        "model_id": "ID",
        "model_name": "Name",
        "model_backbone": "Backbone",
        "model_iou_score": "IoU Score",
        "model_trained_date": "Trained Date",
    }
    cols = [c for c in rename_map.keys() if c in df_models.columns]
    df_models = df_models[cols].rename(columns=rename_map)
    st.dataframe(df_models, use_container_width=True, hide_index=True)
else:
    st.info("No registered models found in database.")

# 2. Benchmark Test Metrics
st.subheader("Test Set Performance")
eval_results_file = PROJECT_ROOT / "model" / "checkpoints" / "eval_results.md"
per_slice_csv = PROJECT_ROOT / "model" / "checkpoints" / "per_slice_metrics.csv"

if per_slice_csv.exists():
    try:
        df_slices = pd.read_csv(per_slice_csv)
        total_slices = len(df_slices)
        pos_slices = df_slices[df_slices["has_tumor"] == 1]
        n_pos = len(pos_slices)

        metrics_summary = [
            {
                "Subset": f"All Test Slices (N={total_slices})",
                "IoU": f"{df_slices['iou'].mean():.4f}",
                "Dice": f"{df_slices['dice'].mean():.4f}",
                "Precision": f"{df_slices['precision'].mean():.4f}",
                "Recall": f"{df_slices['recall'].mean():.4f}",
            },
            {
                "Subset": f"Tumor-Positive Only (N={n_pos})",
                "IoU": f"{pos_slices['iou'].mean():.4f}" if n_pos > 0 else "N/A",
                "Dice": f"{pos_slices['dice'].mean():.4f}" if n_pos > 0 else "N/A",
                "Precision": f"{pos_slices['precision'].mean():.4f}" if n_pos > 0 else "N/A",
                "Recall": f"{pos_slices['recall'].mean():.4f}" if n_pos > 0 else "N/A",
            },
        ]
        st.dataframe(pd.DataFrame(metrics_summary), use_container_width=True, hide_index=True)
    except Exception as e:
        st.error(f"Error loading metrics summary: {e}")
elif eval_results_file.exists():
    st.markdown(eval_results_file.read_text(encoding="utf-8"))
else:
    st.info("Evaluation metrics will appear here after running model evaluation.")

# 3. Training Curves
st.subheader("Training Curves")
loss_curve_path = PROJECT_ROOT / "model" / "checkpoints" / "loss_curve.png"
if not loss_curve_path.exists():
    # Check runs directory for any loss curve
    runs_dir = PROJECT_ROOT / "model" / "checkpoints" / "runs"
    if runs_dir.exists():
        curves = list(runs_dir.glob("*/loss_curve.png"))
        if curves:
            loss_curve_path = curves[-1]

if loss_curve_path.exists():
    st.image(str(loss_curve_path), caption="Training & Validation Loss / Dice Curves", use_container_width=True)
else:
    st.info("No training curves found.")
