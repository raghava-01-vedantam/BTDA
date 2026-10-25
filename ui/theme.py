import streamlit as st

_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Design Tokens ── */
:root {
    --bg:          #f5f7fa;
    --surface:     #ffffff;
    --surface-2:   #eef2f7;
    --border:      #d1d9e6;
    --border-soft: #e4e9f2;
    --text:        #1a202c;
    --text-muted:  #5a6a82;
    --text-light:  #8294aa;
    --accent:      #2563eb;
    --accent-soft: #eff4ff;
    --accent-glow: rgba(37,99,235,0.15);
    --teal:        #0891b2;
    --teal-soft:   #f0f9ff;
    --success:     #16a34a;
    --warning:     #d97706;
    --danger:      #dc2626;
    --radius:      10px;
    --radius-sm:   7px;
    --shadow-sm:   0 1px 3px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.04);
    --shadow-md:   0 4px 12px rgba(0,0,0,0.08), 0 2px 4px rgba(0,0,0,0.04);
}

/* ── Global Reset ── */
html, body, [class*="css"], .stApp {
    font-family: 'Inter', sans-serif !important;
    background-color: var(--bg) !important;
    color: var(--text) !important;
    -webkit-font-smoothing: antialiased;
}
.stMainBlockContainer {
    background: var(--bg) !important;
    max-width: 1360px;
    padding: 1.5rem 2rem !important;
}

/* ── Page Header ── */
.page-title {
    font-size: 1.65rem;
    font-weight: 700;
    color: var(--text);
    letter-spacing: -0.5px;
    margin: 0 0 0.2rem 0;
    line-height: 1.2;
}
.page-subtitle {
    font-size: 0.9rem;
    color: var(--text-muted);
    font-weight: 400;
    margin-bottom: 1.5rem;
    padding-bottom: 1rem;
    border-bottom: 2px solid var(--border-soft);
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * {
    color: var(--text) !important;
}
[data-testid="stSidebar"] .stMarkdown h3,
[data-testid="stSidebar"] .stMarkdown h4 {
    color: var(--text) !important;
    font-size: 0.78rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    margin-bottom: 0.5rem !important;
}
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
[data-testid="stSidebar"] label {
    color: var(--text-muted) !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
}
/* Sidebar nav links */
[data-testid="stSidebarNav"] a,
[data-testid="stSidebarNav"] span,
[data-testid="stSidebarNavItems"] span {
    color: var(--text-muted) !important;
    font-size: 0.85rem !important;
}
[data-testid="stSidebarNav"] a:hover span {
    color: var(--accent) !important;
}

/* ── Text inputs (sidebar) ── */
[data-testid="stTextInput"] input {
    background: var(--surface) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text) !important;
    font-size: 0.88rem !important;
    padding: 0.45rem 0.75rem !important;
    box-shadow: none !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--accent-glow) !important;
    outline: none !important;
}
[data-testid="stTextInput"] [data-testid="stWidgetLabel"] p,
[data-testid="stTextInput"] label {
    color: var(--text-muted) !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
}

/* ── Select boxes ── */
[data-testid="stSelectbox"] [data-testid="stWidgetLabel"] p,
[data-testid="stSelectbox"] label {
    color: var(--text-muted) !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
}
[data-testid="stSelectbox"] [data-baseweb="select"] {
    background: var(--surface) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    box-shadow: none !important;
}
[data-testid="stSelectbox"] [data-baseweb="select"] > div {
    background: transparent !important;
    border: none !important;
    color: var(--text) !important;
    font-size: 0.88rem !important;
}
/* Dropdown menu */
[data-baseweb="popover"] [data-baseweb="menu"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
}
[data-baseweb="popover"] [role="option"] {
    color: var(--text) !important;
    font-size: 0.88rem !important;
}
[data-baseweb="popover"] [role="option"]:hover,
[data-baseweb="popover"] [aria-selected="true"] {
    background: var(--accent-soft) !important;
    color: var(--accent) !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: var(--surface) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 0.5rem !important;
    box-shadow: var(--shadow-sm) !important;
}
[data-testid="stFileUploader"] section {
    border: none !important;
    background: transparent !important;
}
[data-testid="stFileUploadDropzone"] {
    background: var(--accent-soft) !important;
    border: 2px dashed var(--accent) !important;
    border-radius: var(--radius-sm) !important;
    padding: 1.2rem !important;
}
[data-testid="stFileUploadDropzone"] span,
[data-testid="stFileUploadDropzone"] p,
[data-testid="stFileUploadDropzoneInstructions"] span {
    color: var(--accent) !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
    opacity: 1 !important;
}
[data-testid="stFileUploadDropzone"] small,
[data-testid="stFileUploadDropzoneInstructions"] small {
    color: var(--text-muted) !important;
    font-size: 0.78rem !important;
    opacity: 1 !important;
}
[data-testid="stFileUploader"] [data-testid="stWidgetLabel"] p {
    color: var(--text) !important;
    font-size: 0.9rem !important;
    font-weight: 600 !important;
}

/* ── Slider ── */
[data-testid="stSlider"] {
    background: transparent !important;
    box-shadow: none !important;
}
[data-testid="stSlider"] [data-testid="stWidgetLabel"] p,
[data-testid="stSlider"] label {
    color: var(--text-muted) !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
}
.stSlider [data-baseweb="slider"] [role="slider"] {
    background: var(--accent) !important;
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--accent-glow) !important;
}
.stSlider [data-baseweb="slider"] div[data-baseweb="slider-track-fill"] {
    background: var(--accent) !important;
}
.stSlider [data-baseweb="slider"] div[data-baseweb="slider-track"] {
    background: var(--border) !important;
}
[data-testid="stThumbValue"] {
    color: var(--accent) !important;
    font-weight: 600 !important;
}

/* ── Buttons ── */
.stDownloadButton > button,
.stButton > button {
    background: var(--accent) !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    padding: 0.6rem 1.6rem !important;
    letter-spacing: 0.01em !important;
    cursor: pointer !important;
    box-shadow: var(--shadow-sm) !important;
    transition: background 0.15s ease, transform 0.1s ease, box-shadow 0.15s ease !important;
}
.stDownloadButton > button:hover,
.stButton > button:hover {
    background: #1d4ed8 !important;
    transform: translateY(-1px) !important;
    box-shadow: var(--shadow-md) !important;
}
.stDownloadButton > button:active,
.stButton > button:active {
    transform: translateY(0px) !important;
}

/* ── Metric tiles ── */
[data-testid="metric-container"] {
    background: var(--surface) !important;
    border: 1.5px solid var(--border-soft) !important;
    border-radius: var(--radius) !important;
    padding: 1rem 1.2rem !important;
    box-shadow: var(--shadow-sm) !important;
}
[data-testid="metric-container"] label,
[data-testid="metric-container"] [data-testid="stWidgetLabel"] {
    color: var(--text-muted) !important;
    font-size: 0.72rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: var(--accent) !important;
    font-size: 1.5rem !important;
    font-weight: 700 !important;
}
[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    display: none !important;
}

/* ── Number input ── */
[data-testid="stNumberInput"] input {
    background: var(--surface) !important;
    border: 1.5px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: var(--radius-sm) !important;
    font-size: 0.88rem !important;
}
[data-testid="stNumberInput"] [data-testid="stWidgetLabel"] p,
[data-testid="stNumberInput"] label {
    color: var(--text-muted) !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background: var(--surface) !important;
    border: 1.5px solid var(--border-soft) !important;
    border-radius: var(--radius) !important;
    box-shadow: none !important;
}
[data-testid="stExpander"] summary {
    color: var(--text-muted) !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    padding: 0.55rem 0.75rem !important;
}
[data-testid="stExpander"] summary:hover {
    color: var(--accent) !important;
}

/* ── Images ── */
[data-testid="stImage"] img {
    border-radius: 8px;
    border: 1.5px solid var(--border-soft);
    display: block;
    margin: 0 auto;
    box-shadow: var(--shadow-sm);
}

/* ── Caption ── */
.stCaption,
[data-testid="stCaptionContainer"] p {
    color: var(--text-muted) !important;
    font-size: 0.75rem !important;
    text-align: center;
}

/* ── Dataframe / table ── */
[data-testid="stDataFrame"],
[data-testid="stTable"] {
    background: var(--surface) !important;
    border: 1.5px solid var(--border-soft) !important;
    border-radius: var(--radius) !important;
    overflow: hidden;
    box-shadow: var(--shadow-sm) !important;
}

/* ── Alert / info boxes ── */
[data-testid="stAlert"] {
    background: var(--accent-soft) !important;
    border: 1px solid rgba(37,99,235,0.25) !important;
    border-left: 4px solid var(--accent) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text) !important;
    box-shadow: none !important;
}
.stAlert p { color: var(--text) !important; }

/* ── Spinner ── */
[data-testid="stSpinner"] { color: var(--accent) !important; }

/* ── Tabs ── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: transparent !important;
    gap: 0.2rem !important;
    border-bottom: 2px solid var(--border-soft) !important;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    background: transparent !important;
    box-shadow: none !important;
    border: none !important;
    color: var(--text-muted) !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    padding: 0.4rem 1rem !important;
    border-radius: 0 !important;
    border-bottom: 2px solid transparent !important;
    margin-bottom: -2px !important;
    transition: color 0.15s !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    color: var(--accent) !important;
    font-weight: 600 !important;
    border-bottom: 2px solid var(--accent) !important;
    box-shadow: none !important;
}
[data-testid="stTabs"] [data-baseweb="tab-border"] { display: none !important; }
[data-testid="stTabsContent"] {
    background: transparent !important;
    box-shadow: none !important;
    padding: 0.7rem 0 !important;
    border: none !important;
}

/* ── Hide chrome ── */
header[data-testid="stHeader"] { background: transparent !important; }
#MainMenu, footer { visibility: hidden; }

/* ── Success/warning/error state messages ── */
div[data-testid="stSuccessMessage"],
.element-container .stSuccess {
    background: #f0fdf4 !important;
    border-left: 4px solid var(--success) !important;
    color: #14532d !important;
    border-radius: var(--radius-sm) !important;
}
div[data-testid="stErrorMessage"],
.element-container .stError {
    background: #fef2f2 !important;
    border-left: 4px solid var(--danger) !important;
    color: #7f1d1d !important;
    border-radius: var(--radius-sm) !important;
}

/* ── Severity badges ── */
.badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-size: 0.77rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.badge-mild     { background: #dcfce7; color: #15803d; border: 1.5px solid #86efac; }
.badge-moderate { background: #fef9c3; color: #92400e; border: 1.5px solid #fde047; }
.badge-critical { background: #fee2e2; color: #991b1b; border: 1.5px solid #fca5a5; }
.badge-clear    { background: #e0f2fe; color: #075985; border: 1.5px solid #7dd3fc; }

/* ── Clinical impression card ── */
.impression-card {
    background: var(--surface);
    border: 1.5px solid var(--border-soft);
    border-radius: var(--radius);
    padding: 1.2rem 1.4rem;
    margin-top: 0.8rem;
    box-shadow: var(--shadow-sm);
}
.impression-card h4 {
    color: var(--text-light);
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin: 0 0 0.5rem 0;
}
.impression-card p {
    color: var(--text);
    font-size: 0.92rem;
    line-height: 1.65;
    margin: 0;
}
.impression-card ul li {
    color: var(--text) !important;
}
"""


def apply_theme() -> None:
    st.markdown(f"<style>{_CSS}</style>", unsafe_allow_html=True)
