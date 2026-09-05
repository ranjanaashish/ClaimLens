"""
streamlit_app.py — ClaimLens entry-point.
"""
import os
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

# ---------------------------------------------------------------------------
# Page config — must be FIRST Streamlit call
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="ClaimLens — Multimodal Assessment AI",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": "ClaimLens — AI-powered visual assessment.\n\nBuilt with Streamlit, Gemini Vision, and FAISS RAG.",
    },
)

# ---------------------------------------------------------------------------
# Theme & CSS injection
# ---------------------------------------------------------------------------

css_path = Path(__file__).parent / "assets" / "style_chat.css"
if css_path.exists():
    st.html(f"<style>{css_path.read_text(encoding='utf-8')}</style>")

# Load .env if present
env_file = Path(__file__).parent / ".env"
if env_file.exists():
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

# ---------------------------------------------------------------------------
# Global session-state defaults
# ---------------------------------------------------------------------------

defaults = {
    "messages": [],
    "pending_image": None,
    "theme":             "dark",
    "vlm_backend":       os.getenv("VLM_BACKEND", "gemini"),
    "gemini_api_key":    os.getenv("GEMINI_API_KEY", ""),
    "gemini_vlm_model":  os.getenv("GEMINI_VLM_MODEL", "gemini-3.6-flash"),
    "openrouter_vlm_key": "",
    "openrouter_vlm_model": "meta-llama/llama-3.2-11b-vision-instruct:free",
    "openai_vlm_key": "",
    "openai_vlm_model": "gpt-4o-mini",
    "ollama_vlm_model": "llava",
    "vlm_rest_endpoint": os.getenv("VLM_REST_ENDPOINT", "http://localhost:8000/assess"),
    "llm_provider":      os.getenv("LLM_PROVIDER", "gemini"),
    "llm_model":         os.getenv("LLM_MODEL", "gemini-3.6-flash"),
    "llm_api_key":       os.getenv("GEMINI_API_KEY", ""),
    "ollama_base_url":   "http://localhost:11434",
    "chat_history":      [],
    "last_assessment":   None,
    "last_image":        None,
    "batch_results":     [],
    "persona":           "Adjuster",
    "_settings_open":    False,
    "ds_images":         [],
    "ds_metadata":       {},
    "ds_gt_col":         None,
    "ds_columns":        [],
    "ds_img_col":        None,
    "ds_mode":           "base64",
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# Apply light theme class injection
if st.session_state.get("theme", "dark") == "light":
    st.html("""<style>
.stApp, [data-testid="stAppViewContainer"],
html, body,
.cl-navbar, .cl-thread, .cl-input-bar,
.cl-card, .cl-text-card, .cl-thinking,
.cl-kpi-card, .cl-example-card, .cl-msg-user {
  --bg-base: #F4F4F6;
  --bg-surface: #FFFFFF;
  --bg-elevated: #EFEFF2;
  --bg-input: #FFFFFF;
  --bg-user-msg: #E8F0FE;
  --border-faint: rgba(0,0,0,0.05);
  --border-def: rgba(0,0,0,0.10);
  --border-strong: rgba(0,0,0,0.20);
  --border-focus: #2563EB;
  --text-primary: #111111;
  --text-muted: #505060;
  --text-faint: #9090A0;
  --text-user: #1A3A6E;
  --accent: #2563EB;
  --accent-sub: rgba(37,99,235,0.08);
  --accent-glow: rgba(37,99,235,0.16);
  --green: #059669;
  --green-sub: rgba(5,150,105,0.09);
  --amber: #D97706;
  --amber-sub: rgba(217,119,6,0.09);
  --red: #DC2626;
  --red-sub: rgba(220,38,38,0.09);
  --shadow-sm: 0 1px 4px rgba(0,0,0,0.08);
  --shadow-md: 0 4px 20px rgba(0,0,0,0.11);
}
</style>""")

# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------

pages = [
    st.Page("pages/chat.py",         title="Chat",         icon=None, default=True),
    st.Page("pages/3_metrics.py",    title="Eval Metrics", icon=":material/bar_chart:"),
    st.Page("pages/4_batch_view.py", title="Batch Review", icon=":material/table_view:"),
]

pg = st.navigation(pages, position="hidden")
pg.run()