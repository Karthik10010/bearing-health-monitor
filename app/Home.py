"""Bearing Health Monitor - dashboard entry point (top navigation).

Run from the project folder:  .venv\\Scripts\\streamlit.exe run app\\Home.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st  # noqa: E402

import components.common  # noqa: E402,F401  (puts the project root on sys.path)

st.set_page_config(page_title="Bearing Health Monitor", page_icon="⚙️", layout="wide")

st.html("""
<style>
  .block-container { max-width: 1180px; padding-top: 4.2rem; padding-bottom: 3rem; }
  h1 { font-size: 2rem !important; letter-spacing: -.01em; }
  h4 { margin-top: .2rem !important; }
  [data-testid="stMetricValue"] { font-size: 1.55rem; }
  [data-testid="stVerticalBlockBorderWrapper"] { border-radius: 14px; }
  div[data-testid="stPageLink"] a { border-radius: 10px; }
</style>
""")

pages = [
    st.Page("views/overview.py", title="Home", icon="🏠", default=True),
    st.Page("views/signal.py", title="Signal", icon="📈"),
    st.Page("views/diagnosis.py", title="Diagnosis", icon="🩺"),
    st.Page("views/maintenance.py", title="Maintenance", icon="🛠️"),
    st.Page("views/assistant.py", title="AI Assistant", icon="💬"),
    st.Page("views/edge.py", title="Edge AI", icon="📟"),
]
st.navigation(pages, position="top").run()
