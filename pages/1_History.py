import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from db.database import init_db
from db.repository import list_history, list_users
from ui.theme import apply_theme

st.set_page_config(page_title="History", layout="wide", initial_sidebar_state="expanded")
apply_theme()
init_db()

st.markdown('<div class="page-title">History</div>', unsafe_allow_html=True)

# User Filter
users = list_users()
user_options = {"All Users": None}
for u in users:
    user_options[f"{u['user_name']} ({u['user_role']})"] = u["user_id"]

selected_label = st.selectbox("User", options=list(user_options.keys()), label_visibility="visible")
selected_user_id = user_options[selected_label]

records = list_history(user_id=selected_user_id)

if records:
    # Format records for display
    display_rows = []
    for r in records:
        date_str = r.get("date", "")
        if "T" in date_str:
            date_str = date_str.replace("T", " ")[:19]
        display_rows.append({
            "Date": date_str,
            "User": r.get("user", ""),
            "File": r.get("filename", ""),
            "Area (cm2)": f"{r.get('area_cm2', 0.0):.2f}",
            "Pixels": r.get("pixels", 0),
            "Report": r.get("report", "No"),
        })

    df = pd.DataFrame(display_rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    csv_data = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Export CSV",
        data=csv_data,
        file_name="analysis_history.csv",
        mime="text/csv",
    )
else:
    st.info("No analysis records found.")
