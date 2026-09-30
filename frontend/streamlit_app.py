"""
frontend/streamlit_app.py — Main Streamlit application entry point.

Launches a multi-page app connecting to the GraphRAG backend API.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="GraphRAG · Multi-Hop QA",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Add project root to path for local imports
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Global CSS — research-oriented dark theme
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    /* ── Root theme ── */
    :root {
        --bg-primary: #0d1117;
        --bg-secondary: #161b22;
        --bg-card: #1c2128;
        --bg-hover: #21262d;
        --border: #30363d;
        --text-primary: #e6edf3;
        --text-secondary: #8b949e;
        --text-muted: #6e7681;
        --accent-blue: #58a6ff;
        --accent-green: #3fb950;
        --accent-orange: #d29922;
        --accent-red: #f85149;
        --accent-purple: #bc8cff;
        --accent-teal: #39d353;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: var(--bg-primary);
        color: var(--text-primary);
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: var(--bg-secondary) !important;
        border-right: 1px solid var(--border);
    }
    section[data-testid="stSidebar"] .stRadio label {
        color: var(--text-secondary) !important;
    }

    /* ── Main content ── */
    .main .block-container {
        padding-top: 2rem;
        max-width: 1400px;
    }

    /* ── Cards ── */
    .grag-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1rem;
    }
    .grag-card-title {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--text-muted);
        margin-bottom: 0.4rem;
    }
    .grag-card-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: var(--text-primary);
        line-height: 1;
    }
    .grag-card-sub {
        font-size: 0.8rem;
        color: var(--text-secondary);
        margin-top: 0.25rem;
    }

    /* ── Status badges ── */
    .badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-green { background: rgba(63,185,80,0.15); color: #3fb950; border: 1px solid rgba(63,185,80,0.3); }
    .badge-red   { background: rgba(248,81,73,0.15);  color: #f85149; border: 1px solid rgba(248,81,73,0.3); }
    .badge-blue  { background: rgba(88,166,255,0.15); color: #58a6ff; border: 1px solid rgba(88,166,255,0.3); }
    .badge-orange{ background: rgba(210,153,34,0.15); color: #d29922; border: 1px solid rgba(210,153,34,0.3); }

    /* ── Section headers ── */
    .grag-section-header {
        font-size: 0.85rem;
        font-weight: 600;
        color: var(--accent-blue);
        text-transform: uppercase;
        letter-spacing: 0.06em;
        border-bottom: 1px solid var(--border);
        padding-bottom: 0.5rem;
        margin: 1.5rem 0 1rem 0;
    }

    /* ── Evidence cards ── */
    .evidence-card {
        background: var(--bg-secondary);
        border: 1px solid var(--border);
        border-left: 3px solid var(--accent-blue);
        border-radius: 6px;
        padding: 0.9rem 1.1rem;
        margin-bottom: 0.75rem;
        font-size: 0.875rem;
    }
    .evidence-source {
        font-size: 0.7rem;
        color: var(--text-muted);
        font-family: 'JetBrains Mono', monospace;
        margin-bottom: 0.4rem;
    }

    /* ── Triple cards ── */
    .triple-card {
        background: var(--bg-secondary);
        border: 1px solid var(--border);
        border-left: 3px solid var(--accent-purple);
        border-radius: 6px;
        padding: 0.6rem 1rem;
        margin-bottom: 0.4rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        color: var(--text-secondary);
    }

    /* ── Answer box ── */
    .answer-box {
        background: linear-gradient(135deg, rgba(88,166,255,0.08), rgba(188,140,255,0.05));
        border: 1px solid rgba(88,166,255,0.3);
        border-radius: 10px;
        padding: 1.25rem 1.5rem;
        font-size: 1rem;
        color: var(--text-primary);
        line-height: 1.6;
        margin: 1rem 0;
    }

    /* ── Metric pills ── */
    .metric-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 0.8rem;
        color: var(--text-secondary);
        margin: 2px 4px;
    }
    .metric-pill-value {
        font-weight: 600;
        color: var(--accent-blue);
    }

    /* ── Stmetric overrides ── */
    [data-testid="stMetric"] {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 12px 16px;
    }
    [data-testid="stMetricLabel"] { color: var(--text-muted) !important; }
    [data-testid="stMetricValue"] { color: var(--text-primary) !important; }

    /* ── Button ── */
    .stButton > button {
        background: var(--accent-blue) !important;
        color: #0d1117 !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 0.5rem 1.5rem !important;
    }
    .stButton > button:hover {
        background: #79c0ff !important;
    }

    /* ── Expander ── */
    .streamlit-expanderHeader {
        background: var(--bg-secondary) !important;
        color: var(--text-secondary) !important;
        border: 1px solid var(--border) !important;
        border-radius: 6px !important;
    }

    /* ── Select boxes / inputs ── */
    .stSelectbox > div > div,
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        background: var(--bg-secondary) !important;
        border: 1px solid var(--border) !important;
        color: var(--text-primary) !important;
    }

    /* ── Tables ── */
    .stDataFrame { border: 1px solid var(--border) !important; }

    /* ── Spinner ── */
    .stSpinner > div { border-color: var(--accent-blue) transparent transparent transparent !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

from dotenv import load_dotenv

# Load .env file
load_dotenv()

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
API_BASE = os.getenv("API_BASE_URL", "http://127.0.0.1:8001")


def sidebar_nav():
    with st.sidebar:
        st.markdown(
            """
            <style>
            [data-testid="stSidebarNav"] {display: none;}
            </style>
            <div style="padding: 0.5rem 0 1.5rem 0;">
                <div style="font-size: 1.3rem; font-weight: 700; color: #58a6ff;">🔬 GraphRAG</div>
                <div style="font-size: 0.75rem; color: #6e7681; margin-top: 4px;">Multi-Hop Question Answering</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        page = st.radio(
            "Navigation",
            options=["Overview", "Ask Questions", "Graph Explorer", "RAG Comparison", "Experiment Logs"],
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.markdown(
            f'<div style="font-size:0.7rem; color:#6e7681;">API: <code>{API_BASE}</code></div>',
            unsafe_allow_html=True,
        )

        # Quick connection status
        try:
            import requests
            r = requests.get(f"{API_BASE}/health", timeout=2)
            if r.status_code == 200:
                data = r.json()
                neo4j_ok = data.get("neo4j_connected", False)
                if neo4j_ok:
                    st.markdown('<div class="badge badge-green" style="margin-top:6px">● Neo4j Connected</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="badge badge-blue" style="margin-top:6px">● Embedded KG (Active)</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="badge badge-red" style="margin-top:6px">● API Unreachable</div>', unsafe_allow_html=True)
        except Exception:
            st.markdown('<div class="badge badge-red" style="margin-top:6px">● API Offline</div>', unsafe_allow_html=True)

    return page


# ---------------------------------------------------------------------------
# Main dispatcher
# ---------------------------------------------------------------------------

def main():
    page = sidebar_nav()

    if page == "Overview":
        from frontend.views.overview import render
    elif page == "Ask Questions":
        from frontend.views.ask import render
    elif page == "Graph Explorer":
        from frontend.views.graph_explorer import render
    elif page == "RAG Comparison":
        from frontend.views.comparison import render
    elif page == "Experiment Logs":
        from frontend.views.experiments import render
    else:
        render = None

    if render:
        render(API_BASE)


if __name__ == "__main__":
    main()
