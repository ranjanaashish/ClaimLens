"""
pages/chat.py
=============
ClaimLens — clean, full-screen chat interface.
No emojis; supports dark / light theme toggle.
"""
from __future__ import annotations

import base64
import io
import os
import sys
from pathlib import Path

import streamlit as st
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from core import chat_engine, feedback_store, vlm_adapter
from core.response_renderer import (
    ChatResponse,
    render_card,
    render_thinking,
    render_user_bubble,
)
from core.schema import DAMAGE_CODES, Persona, Severity

# ---------------------------------------------------------------------------
# Session-state initialisation
# ---------------------------------------------------------------------------

def _init():
    defaults = {
        "messages":            [],
        "pending_image":       None,
        "theme":               "dark",
        "vlm_backend":         "gemini",
        "gemini_api_key":      os.getenv("GEMINI_API_KEY", ""),
        "gemini_vlm_model":    "gemini-3.6-flash",
        "openrouter_vlm_key":  "",
        "openrouter_vlm_model": "meta-llama/llama-3.2-11b-vision-instruct:free",
        "openai_vlm_key":      "",
        "openai_vlm_model":    "gpt-4o-mini",
        "ollama_vlm_model":    "llava",
        "llm_provider":        "gemini",
        "llm_model":           "gemini-3.6-flash",
        "llm_api_key":         os.getenv("GEMINI_API_KEY", ""),
        "ollama_base_url":     "http://localhost:11434",
        "vlm_rest_endpoint":   "http://localhost:8000/assess",
        "enable_rag":          True,
        "rag_persona":         Persona.ADJUSTER,
        "_settings_open":      False,
        "ds_images":           [],
        "ds_metadata":         {},
        "ds_gt_col":           None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()

# ---------------------------------------------------------------------------
# Provider / backend config
# ---------------------------------------------------------------------------

_PROVIDERS = {
    "gemini":     ("Google Gemini",   ["gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.5-flash", "gemini-flash-latest"]),
    "stub":       ("Stub (demo)",     ["stub"]),
    "openrouter": ("OpenRouter",      ["meta-llama/llama-3.1-8b-instruct:free", "qwen/qwen-2.5-72b-instruct:free"]),
    "ollama":     ("Ollama (local)",  ["mistral", "llama3", "qwen2.5"]),
    "openai":     ("OpenAI",          ["gpt-4o-mini", "gpt-4o"]),
}

_VLM_BACKENDS = {
    "gemini": ("Google Gemini Vision", [
        "gemini-3.6-flash", "gemini-3.7-flash",
        "gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-flash-latest",
    ]),
    "stub":       ("Stub (instant demo)", ["stub"]),
    "openrouter": ("OpenRouter Vision", [
        "meta-llama/llama-3.2-11b-vision-instruct:free",
        "qwen/qwen-2.5-vl-72b-instruct:free",
        "google/gemma-3-27b-it:free",
        "openai/gpt-4o-mini",
        "anthropic/claude-3-5-sonnet",
    ]),
    "ollama": ("Ollama Vision (local)", ["llava", "llama3.2-vision", "minicpm-v", "bakllava"]),
    "openai": ("OpenAI Vision",         ["gpt-4o-mini", "gpt-4o"]),
    "rest":   ("Local REST endpoint",   ["custom"]),
}

def _is_live() -> bool:
    return (
        st.session_state.vlm_backend != "stub"
        or st.session_state.llm_provider != "stub"
    )

def _settings_label() -> str:
    parts = []
    if st.session_state.vlm_backend != "stub":
        parts.append(f"Vision: {st.session_state.vlm_backend}")
    if st.session_state.llm_provider != "stub":
        parts.append(f"LLM: {st.session_state.llm_provider}")
    return "Settings" + (" — " + " · ".join(parts) if parts else "")

# ---------------------------------------------------------------------------
# Navbar
# ---------------------------------------------------------------------------

theme = st.session_state.get("theme", "dark")
status_class = "live" if _is_live() else "stub"
status_label = "Live" if _is_live() else "Demo"

nav_left, nav_center, nav_right = st.columns([3, 6, 3])

with nav_left:
    st.html(f"""
<div class="cl-navbar" style="position:static;border-bottom:none;padding:10px 4px;">
  <div class="cl-brand">
    <span class="cl-brand-name">ClaimLens</span>
    <span class="cl-brand-sub">Multimodal AI</span>
  </div>
</div>
""")

with nav_right:
    r1, r2 = st.columns([2, 2])
    with r1:
        st.html(f'<div style="padding:10px 0;"><span class="cl-status-pill {status_class}">{status_label}</span></div>')
    with r2:
        toggle_label = "Light Mode" if theme == "dark" else "Dark Mode"
        if st.button(toggle_label, key="theme_toggle"):
            st.session_state.theme = "light" if theme == "dark" else "dark"
            st.rerun()

# Separator line
st.html('<div style="border-bottom:1px solid var(--border-faint);margin-bottom:0;"></div>')

# ---------------------------------------------------------------------------
# Settings panel
# ---------------------------------------------------------------------------

with st.expander(_settings_label(), expanded=st.session_state._settings_open):
    st.session_state._settings_open = True

    col_vlm, col_llm = st.columns(2, gap="large")

    with col_vlm:
        st.markdown("**Vision Model**")
        vlm = st.selectbox(
            "VLM backend",
            list(_VLM_BACKENDS.keys()),
            index=list(_VLM_BACKENDS.keys()).index(st.session_state.vlm_backend)
            if st.session_state.vlm_backend in _VLM_BACKENDS else 0,
            format_func=lambda k: _VLM_BACKENDS[k][0],
            key="_set_vlm",
        )
        st.session_state.vlm_backend = vlm
        vlm_models = _VLM_BACKENDS[vlm][1]

        if vlm == "gemini":
            gk = st.text_input("Gemini API Key", value=st.session_state.gemini_api_key,
                                type="password", placeholder="AIzaSy…", key="_set_gk")
            st.session_state.gemini_api_key = gk
            if st.session_state.llm_provider == "gemini" and gk and not st.session_state.llm_api_key:
                st.session_state.llm_api_key = gk
            cur_vm = st.session_state.get("gemini_vlm_model", "gemini-3.6-flash")
            idx_vm = vlm_models.index(cur_vm) if cur_vm in vlm_models else 0
            v_chosen = st.selectbox("Vision Model", vlm_models, index=idx_vm, key="_set_vlm_model")
            st.session_state.gemini_vlm_model = v_chosen
            st.caption("Free tier — 15 RPM — aistudio.google.com")

        elif vlm == "openrouter":
            ork = st.text_input("OpenRouter API Key",
                                 value=st.session_state.get("openrouter_vlm_key", "") or st.session_state.llm_api_key,
                                 type="password", placeholder="sk-or-…", key="_set_ork")
            st.session_state.openrouter_vlm_key = ork
            cur_orm = st.session_state.get("openrouter_vlm_model", vlm_models[0])
            idx_orm = vlm_models.index(cur_orm) if cur_orm in vlm_models else 0
            or_chosen = st.selectbox("Vision Model", vlm_models, index=idx_orm, key="_set_or_vlm_model")
            st.session_state.openrouter_vlm_model = or_chosen
            st.caption("Free models available with :free suffix — openrouter.ai")

        elif vlm == "ollama":
            ob = st.text_input("Ollama URL", value=st.session_state.ollama_base_url, key="_set_ol_url")
            st.session_state.ollama_base_url = ob
            cur_olm = st.session_state.get("ollama_vlm_model", vlm_models[0])
            idx_olm = vlm_models.index(cur_olm) if cur_olm in vlm_models else 0
            ol_chosen = st.selectbox("Local Vision Model", vlm_models, index=idx_olm, key="_set_ol_vlm_model")
            st.session_state.ollama_vlm_model = ol_chosen
            st.caption("Runs locally — no quotas, no API key required")

        elif vlm == "openai":
            oak = st.text_input("OpenAI API Key",
                                 value=st.session_state.get("openai_vlm_key", "") or st.session_state.llm_api_key,
                                 type="password", placeholder="sk-…", key="_set_oak")
            st.session_state.openai_vlm_key = oak
            cur_oam = st.session_state.get("openai_vlm_model", vlm_models[0])
            idx_oam = vlm_models.index(cur_oam) if cur_oam in vlm_models else 0
            oa_chosen = st.selectbox("Vision Model", vlm_models, index=idx_oam, key="_set_oa_vlm_model")
            st.session_state.openai_vlm_model = oa_chosen

        elif vlm == "rest":
            ep = st.text_input("REST endpoint", value=st.session_state.vlm_rest_endpoint, key="_set_ep")
            st.session_state.vlm_rest_endpoint = ep

    with col_llm:
        st.markdown("**Language Model**")
        prov = st.selectbox(
            "LLM provider",
            list(_PROVIDERS.keys()),
            index=list(_PROVIDERS.keys()).index(st.session_state.llm_provider),
            format_func=lambda k: _PROVIDERS[k][0],
            key="_set_prov",
        )
        st.session_state.llm_provider = prov
        models = _PROVIDERS[prov][1]
        idx = models.index(st.session_state.llm_model) if st.session_state.llm_model in models else 0
        mdl = st.selectbox("Model", models, index=idx, key="_set_mdl")
        st.session_state.llm_model = mdl

        if prov in ("gemini", "openrouter", "openai", "anthropic"):
            default_key = st.session_state.llm_api_key
            if prov == "gemini" and not default_key and st.session_state.gemini_api_key:
                default_key = st.session_state.gemini_api_key
            lk = st.text_input(f"{prov.capitalize()} API Key", value=default_key,
                                type="password", placeholder="API key…", key="_set_lk")
            st.session_state.llm_api_key = lk
            if prov == "gemini":
                st.session_state.gemini_api_key = lk
        elif prov == "ollama":
            ob2 = st.text_input("Ollama URL", value=st.session_state.ollama_base_url, key="_set_ob")
            st.session_state.ollama_base_url = ob2

    st.markdown("---")
    col_rag1, col_rag2 = st.columns(2)
    with col_rag1:
        st.markdown("**Knowledge Base**")
        en_rag = st.toggle("Enable RAG Grounding", value=st.session_state.get("enable_rag", True), key="_set_en_rag")
        st.session_state.enable_rag = en_rag
        st.caption("Sources: taxonomy, policy clauses, severity rubrics, precedents")

    with col_rag2:
        st.markdown("**Assistant Persona**")
        personas = [p.value for p in Persona]
        cur_p = st.session_state.get("rag_persona", Persona.ADJUSTER)
        idx_p = personas.index(cur_p) if cur_p in personas else 0
        chosen_p = st.selectbox("Persona", personas, index=idx_p, key="_set_persona", label_visibility="collapsed")
        st.session_state.rag_persona = chosen_p
        persona_hints = {
            "Adjuster":         "Technical precision, references taxonomy and damage codes",
            "Underwriter":      "Risk exposure, coverage terms, and reserving implications",
            "Customer Service": "Empathetic, clear, and reassuring language",
            "Researcher / Demo": "Benchmark-oriented, cites dataset sources and tables",
        }
        st.caption(persona_hints.get(chosen_p, ""))

    st.markdown("---")
    col_ft1, col_ft2 = st.columns([3, 2])
    with col_ft1:
        st.markdown("**Fine-Tuning Dataset**")
        ft_count = feedback_store.get_fine_tuning_count()
        st.caption(f"{ft_count} human feedback samples recorded in `data/fine_tuning_feedback.jsonl`")
    with col_ft2:
        if ft_count > 0:
            st.download_button(
                "Download Fine-Tuning JSONL",
                data=feedback_store.get_fine_tuning_jsonl_bytes(),
                file_name="fine_tuning_feedback.jsonl",
                mime="application/jsonlines",
                key="dl_ft_data",
            )

    if st.button("Close", key="_close_settings"):
        st.session_state._settings_open = False
        st.rerun()

# ---------------------------------------------------------------------------
# API Key reminder notice (if live backend selected without a key)
# ---------------------------------------------------------------------------

_missing_keys = []
if (st.session_state.vlm_backend == "gemini" or st.session_state.llm_provider == "gemini") and not st.session_state.gemini_api_key:
    _missing_keys.append("Gemini")
if (st.session_state.vlm_backend == "openrouter" or st.session_state.llm_provider == "openrouter") and not (st.session_state.get("openrouter_vlm_key") or st.session_state.llm_api_key):
    _missing_keys.append("OpenRouter")
if (st.session_state.vlm_backend == "openai" or st.session_state.llm_provider == "openai") and not (st.session_state.get("openai_vlm_key") or st.session_state.llm_api_key):
    _missing_keys.append("OpenAI")

if _missing_keys:
    _key_str = " / ".join(_missing_keys)
    st.html(f"""
<div style="max-width:var(--thread-w);margin:8px auto;padding:9px 14px;background:var(--amber-sub);border:1px solid var(--amber);border-radius:var(--r-md);font-size:12.5px;color:var(--amber);">
  <span>Enter your {_key_str} API key in Settings above to enable live AI analysis, or switch to Stub mode.</span>
</div>
""")

# ---------------------------------------------------------------------------
# Chat thread
# ---------------------------------------------------------------------------

st.html("<div class='cl-thread'>")

# Welcome / empty state
if not st.session_state.messages:
    st.html("""
<div class="cl-welcome">
  <div class="cl-welcome-title">ClaimLens Multimodal AI</div>
  <div class="cl-welcome-sub">
    Upload any image — plants, crops, vehicles, or structures — and ask anything.
    Receive structured assessments with figures, tables, and factual summaries.
  </div>
  <div class="cl-examples">
    <div class="cl-example-card">
      <div class="cl-example-title">Plant health analysis</div>
      <div class="cl-example-body">Diagnose crop diseases, blight patterns, and treatment options from leaf or foliage images.</div>
    </div>
    <div class="cl-example-card">
      <div class="cl-example-title">Vehicle damage assessment</div>
      <div class="cl-example-body">Identify collision damage, estimate repair costs, and generate insurance-ready reports.</div>
    </div>
    <div class="cl-example-card">
      <div class="cl-example-title">Affected area breakdown</div>
      <div class="cl-example-body">Get percentage breakdowns, severity metrics, and spatial detections for any inspection image.</div>
    </div>
    <div class="cl-example-card">
      <div class="cl-example-title">Treatment recommendations</div>
      <div class="cl-example-body">Receive step-by-step action plans, cost estimates, and prioritized remediation guidance.</div>
    </div>
  </div>
</div>
""")

# Render conversation history
for idx, msg in enumerate(st.session_state.messages):
    st.html(msg["html"])

    # Persona-specific preference & remarks section after assistant response
    if msg.get("role") == "assistant":
        p_val = msg.get("persona") or st.session_state.get("rag_persona", Persona.ADJUSTER)
        p_str = p_val.value if hasattr(p_val, "value") else str(p_val)
        schema = feedback_store.get_persona_preference_schema(p_str)

        if st.session_state.get(f"fb_done_{idx}"):
            st.html(f"""
<div style="margin:2px 0 16px;padding:8px 14px;background:var(--bg-surface);border:1px solid var(--border-def);border-radius:var(--r-md);font-size:12px;color:var(--green);display:flex;align-items:center;justify-content:space-between;">
  <span>Feedback recorded for {p_str} fine-tuning.</span>
  <span style="color:var(--text-faint);font-size:11px;">Saved to fine_tuning_feedback.jsonl</span>
</div>
""")
        else:
            with st.expander(f"Provide Feedback ({schema['title']})", expanded=False):
                st.caption(schema["description"])

                col_pref, col_tags = st.columns([1, 1], gap="medium")
                with col_pref:
                    pref_choice = st.radio(
                        "Quality Assessment",
                        schema["preferences"],
                        key=f"fb_pref_{idx}",
                    )
                with col_tags:
                    tags_choice = st.multiselect(
                        "Specific Aspects",
                        schema["tags"],
                        key=f"fb_tags_{idx}",
                    )

                user_remarks = st.text_area(
                    "Remarks & Desired Corrections (for model fine-tuning)",
                    placeholder=schema["remarks_placeholder"],
                    key=f"fb_rem_{idx}",
                    height=70,
                )

                col_save, col_info = st.columns([2, 5])
                with col_save:
                    if st.button("Save Feedback", key=f"fb_submit_{idx}", type="secondary"):
                        feedback_store.log_chat_feedback(
                            persona=p_str,
                            user_prompt=msg.get("prompt", ""),
                            response_text=msg.get("text", ""),
                            preference=pref_choice,
                            tags=tags_choice,
                            remarks=user_remarks,
                            domain=msg.get("domain", "general"),
                            has_image=msg.get("has_image", False),
                            models={
                                "vlm": st.session_state.get("vlm_backend", ""),
                                "llm": st.session_state.get("llm_provider", ""),
                            },
                        )
                        st.session_state[f"fb_done_{idx}"] = True
                        st.rerun()
                with col_info:
                    curr_ft = feedback_store.get_fine_tuning_count()
                    st.caption(f"{curr_ft} dataset training pairs logged")

st.html("</div>")   # close cl-thread

# ---------------------------------------------------------------------------
# Input bar
# ---------------------------------------------------------------------------

st.html("<div class='cl-input-bar'><div class='cl-input-inner'>")

with st.container():
    img_col, prompt_col = st.columns([1, 12], gap="small")

    with img_col:
        uploaded = st.file_uploader(
            "Image",
            type=["jpg", "jpeg", "png", "webp"],
            key="chat_image_upload",
            label_visibility="collapsed",
        )
        if uploaded:
            pil = Image.open(io.BytesIO(uploaded.read()))
            st.session_state.pending_image = pil

    with prompt_col:
        if st.session_state.pending_image is not None:
            buf = io.BytesIO()
            thumb = st.session_state.pending_image.copy()
            thumb.thumbnail((80, 80))
            thumb.save(buf, format="JPEG")
            b64 = base64.b64encode(buf.getvalue()).decode()
            st.html(
                f'''<div class="cl-thumb-row">'''
                f'''<img src="data:image/jpeg;base64,{b64}" class="cl-thumb" alt="Pending image"/>'''
                f'''<span class="cl-thumb-label">Image ready — type your prompt and press Enter</span>'''
                f'''</div>'''
            )

        prompt = st.chat_input(
            "Ask anything, or attach an image to start an assessment…",
            key="chat_prompt",
        )

st.html("</div></div>")   # close cl-input-inner, cl-input-bar

# ---------------------------------------------------------------------------
# Process turn on submit
# ---------------------------------------------------------------------------

if prompt:
    image = st.session_state.pending_image

    user_html = render_user_bubble(prompt, image)
    st.session_state.messages.append({
        "role": "user",
        "text": prompt,
        "html": user_html,
    })

    thinking_placeholder = st.empty()
    thinking_placeholder.html(render_thinking())

    settings = {
        "vlm_backend":          st.session_state.vlm_backend,
        "gemini_api_key":       st.session_state.gemini_api_key,
        "gemini_vlm_model":     st.session_state.get("gemini_vlm_model", "gemini-3.6-flash"),
        "openrouter_vlm_key":   st.session_state.get("openrouter_vlm_key", ""),
        "openrouter_vlm_model": st.session_state.get("openrouter_vlm_model", "meta-llama/llama-3.2-11b-vision-instruct:free"),
        "openai_vlm_key":       st.session_state.get("openai_vlm_key", ""),
        "openai_vlm_model":     st.session_state.get("openai_vlm_model", "gpt-4o-mini"),
        "ollama_vlm_model":     st.session_state.get("ollama_vlm_model", "llava"),
        "vlm_rest_endpoint":    st.session_state.vlm_rest_endpoint,
        "llm_provider":         st.session_state.llm_provider,
        "llm_model":            st.session_state.llm_model,
        "llm_api_key":          st.session_state.llm_api_key,
        "ollama_base_url":      st.session_state.ollama_base_url,
        "enable_rag":           st.session_state.get("enable_rag", True),
        "rag_persona":          st.session_state.get("rag_persona", Persona.ADJUSTER),
    }

    history = [
        {"role": m["role"], "text": m.get("text", "")}
        for m in st.session_state.messages[:-1]
    ]
    response: ChatResponse = chat_engine.process_turn(
        prompt=prompt,
        image=image,
        history=history,
        settings=settings,
    )

    thinking_placeholder.empty()
    card_html = render_card(response)
    current_persona = st.session_state.get("rag_persona", Persona.ADJUSTER)
    persona_str = current_persona.value if hasattr(current_persona, "value") else str(current_persona)
    domain_str = getattr(response.result, "domain", "general") if response.result else "general"
    st.session_state.messages.append({
        "role": "assistant",
        "text": response.text,
        "html": card_html,
        "prompt": prompt,
        "persona": persona_str,
        "domain": domain_str,
        "has_image": image is not None,
    })

    st.session_state.pending_image = None
    st.rerun()

# ---------------------------------------------------------------------------
# Dataset import
# ---------------------------------------------------------------------------

with st.expander("Import from Dataset File (TSV / CSV / Excel)", expanded=False):
    st.caption("Load rows from a dataset file — each row will be assessed as a separate image.")

    dataset_file = st.file_uploader(
        "Dataset file",
        type=["tsv", "csv", "xlsx", "xls"],
        key="chat_dataset_upload",
        label_visibility="collapsed",
    )

    if "ds_columns" not in st.session_state:
        st.session_state.ds_columns = []
    if "ds_img_col" not in st.session_state:
        st.session_state.ds_img_col = None
    if "ds_mode" not in st.session_state:
        st.session_state.ds_mode = "base64"

    if dataset_file is not None:
        try:
            from core.tsv_loader import sniff_columns
            buf = io.BytesIO(dataset_file.read())
            buf.name = dataset_file.name
            cols, auto_col, auto_mode = sniff_columns(buf, max_rows=3)
            st.session_state.ds_columns = cols
            st.session_state.ds_img_col = auto_col or (cols[0] if cols else None)
            st.session_state.ds_mode = auto_mode or "base64"
        except Exception as e:
            st.warning(f"Preview failed: {e}")

    if st.session_state.ds_columns:
        c1, c2, c3 = st.columns([2, 2, 1])
        with c1:
            img_col_choice = st.selectbox(
                "Image column",
                st.session_state.ds_columns,
                index=st.session_state.ds_columns.index(st.session_state.ds_img_col)
                if st.session_state.ds_img_col in st.session_state.ds_columns else 0,
            )
        with c2:
            mode_choice = st.selectbox(
                "Encoding",
                ["base64", "url", "path"],
                format_func=lambda m: {"base64": "Base64", "url": "URL", "path": "File path"}[m],
            )
        with c3:
            max_rows = st.number_input("Max rows", 1, 50, 5)

        load_btn = st.button("Load and Assess All Rows", type="primary")
        if load_btn and dataset_file is not None:
            from core.tsv_loader import load_dataset_file
            dataset_file.seek(0)
            buf = io.BytesIO(dataset_file.read())
            buf.name = dataset_file.name
            try:
                rows = load_dataset_file(buf, max_rows=int(max_rows), image_col=img_col_choice, mode=mode_choice)
            except Exception as e:
                st.error(f"Load failed: {e}")
                rows = []

            progress = st.progress(0)
            for i, (label, img, meta) in enumerate(rows):
                meta_str = ", ".join(f"{k}: {v}" for k, v in list(meta.items())[:5] if k != img_col_choice)
                auto_prompt = f"Assess this image. Dataset context: {meta_str}" if meta_str else "Assess this image."

                user_html = render_user_bubble(auto_prompt, img)
                st.session_state.messages.append({"role": "user", "text": auto_prompt, "html": user_html})

                _settings = {
                    "vlm_backend":          st.session_state.vlm_backend,
                    "gemini_api_key":       st.session_state.gemini_api_key,
                    "gemini_vlm_model":     st.session_state.get("gemini_vlm_model", "gemini-3.6-flash"),
                    "openrouter_vlm_key":   st.session_state.get("openrouter_vlm_key", ""),
                    "openrouter_vlm_model": st.session_state.get("openrouter_vlm_model", "meta-llama/llama-3.2-11b-vision-instruct:free"),
                    "openai_vlm_key":       st.session_state.get("openai_vlm_key", ""),
                    "openai_vlm_model":     st.session_state.get("openai_vlm_model", "gpt-4o-mini"),
                    "ollama_vlm_model":     st.session_state.get("ollama_vlm_model", "llava"),
                    "vlm_rest_endpoint":    st.session_state.vlm_rest_endpoint,
                    "llm_provider":         st.session_state.llm_provider,
                    "llm_model":            st.session_state.llm_model,
                    "llm_api_key":          st.session_state.llm_api_key,
                    "ollama_base_url":      st.session_state.ollama_base_url,
                    "enable_rag":           st.session_state.get("enable_rag", True),
                    "rag_persona":          st.session_state.get("rag_persona", Persona.ADJUSTER),
                }
                response = chat_engine.process_turn(
                    prompt=auto_prompt, image=img, history=[], settings=_settings
                )
                card_html = render_card(response)
                current_persona = st.session_state.get("rag_persona", Persona.ADJUSTER)
                persona_str = current_persona.value if hasattr(current_persona, "value") else str(current_persona)
                domain_str = getattr(response.result, "domain", "vehicle") if response.result else "vehicle"
                st.session_state.messages.append({
                    "role": "assistant",
                    "text": response.text,
                    "html": card_html,
                    "prompt": auto_prompt,
                    "persona": persona_str,
                    "domain": domain_str,
                    "has_image": True,
                })
                progress.progress(int((i + 1) / len(rows) * 100))

            st.success(f"{len(rows)} rows assessed and added to the conversation.")
            st.rerun()

# ---------------------------------------------------------------------------
# Clear conversation
# ---------------------------------------------------------------------------

st.html("<div style='text-align:center;padding:10px 0;'>")
if st.session_state.messages:
    if st.button("Clear conversation", key="clear_chat"):
        st.session_state.messages = []
        st.session_state.pending_image = None
        st.rerun()
st.html("</div>")