"""
pages/1_assessment.py — Damage Assessment page.

Features:
- Single or multi-image upload + sample image picker
- VLM backend selector (stub / Gemini / REST) + key input in sidebar
- Results card: damage codes, severity badge, cost range, confidence gauge
- Bounding box overlay on image
- Flag banner for fraud / low-confidence
- Override form (expander): edit codes, severity, cost, reason
- Accept / Edit & Submit / Reject → writes to feedback.db
- Reasoning expander
- Multi-image tab support with parallel inference
"""
import hashlib
import io
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import streamlit as st
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

import importlib
from core import feedback_store, vlm_adapter
importlib.reload(vlm_adapter)
importlib.reload(feedback_store)
from core.schema import (
    DAMAGE_CODES,
    SEVERITY_COLORS,
    AssessmentResult,
    FeedbackRecord,
    Persona,
    Severity,
)

# ---------------------------------------------------------------------------
# Helper: colour chip for damage code
# ---------------------------------------------------------------------------

def _damage_chip(code: str) -> str:
    label = DAMAGE_CODES.get(code, code)
    return (
        f'<span style="display:inline-block;background:#1e3a5f;color:#93c5fd;'
        f'border:1px solid #3b82f6;border-radius:12px;padding:2px 10px;'
        f'margin:2px;font-size:12px">{label}</span>'
    )


def _severity_badge(severity: str) -> str:
    color = SEVERITY_COLORS.get(severity, "#6b7280")
    return (
        f'<span style="background:{color};color:white;border-radius:6px;'
        f'padding:4px 14px;font-weight:700;font-size:15px">{severity}</span>'
    )


def _image_hash(img: Image.Image) -> str:
    return hashlib.sha256(img.tobytes()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Sidebar — VLM configuration
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("###  VLM Backend")
    vlm_backend = st.selectbox(
        "Inference backend",
        options=["stub", "gemini", "rest"],
        index=["stub", "gemini", "rest"].index(st.session_state.vlm_backend),
        format_func=lambda x: {
            "stub": " Stub (instant, demo)",
            "gemini": " Google Gemini Vision (free)",
            "rest": " Local REST endpoint",
        }[x],
        help="Stub returns deterministic demo results. Gemini uses the free Vision API.",
    )
    st.session_state.vlm_backend = vlm_backend

    if vlm_backend == "gemini":
        gemini_key = st.text_input(
            "Gemini API Key",
            value=st.session_state.gemini_api_key,
            type="password",
            placeholder="AIza…",
            help="Get a free key at aistudio.google.com",
        )
        st.session_state.gemini_api_key = gemini_key
        st.caption("Free tier: 15 RPM / 1M tokens/day")

    elif vlm_backend == "rest":
        rest_ep = st.text_input(
            "REST endpoint URL",
            value=st.session_state.vlm_rest_endpoint,
            placeholder="http://localhost:8000/assess",
        )
        st.session_state.vlm_rest_endpoint = rest_ep

    st.divider()
    st.markdown("### Persona")
    persona = st.selectbox(
        "Viewing as",
        options=[p.value for p in Persona],
        index=0,
        help="Adjusts tone of RAG answers on the Knowledge Agent page.",
    )
    st.session_state.persona = persona

    st.divider()
    st.markdown("###  Image Upload")
    uploaded_files = st.file_uploader(
        "Upload vehicle photo(s)",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    # Sample images picker
    sample_dir = Path(__file__).parent.parent / "data" / "sample_images"
    sample_files = list(sample_dir.glob("*.jpg")) + list(sample_dir.glob("*.png"))
    use_sample = None
    if sample_files:
        st.markdown("**Or pick a sample:**")
        sample_names = ["— none —"] + [f.name for f in sample_files]
        chosen = st.selectbox("Sample images", sample_names, label_visibility="collapsed")
        if chosen != "— none —":
            use_sample = sample_dir / chosen

    # ── Dataset file import ─────────────────────────────────────────────
    st.divider()
    with st.expander("Import from TSV / CSV / Excel", expanded=False):
        st.caption(
            "Load images from a dataset file. Supports:\n"
            "• **INS-MMBench TSV** — `image` column contains base64 JPEG\n"
            "• Any CSV/TSV with a URL or local-path image column"
        )
        dataset_file = st.file_uploader(
            "Dataset file",
            type=["tsv", "csv", "xlsx", "xls"],
            key="dataset_upload",
            label_visibility="collapsed",
        )

        # Persist sniffed columns in session
        if "ds_columns" not in st.session_state:
            st.session_state.ds_columns = []
        if "ds_img_col" not in st.session_state:
            st.session_state.ds_img_col = None
        if "ds_mode" not in st.session_state:
            st.session_state.ds_mode = "base64"

        if dataset_file is not None:
            try:
                from core.tsv_loader import sniff_columns
                import io as _io
                raw_bytes = dataset_file.read()
                dataset_file.seek(0)
                buf = _io.BytesIO(raw_bytes)
                buf.name = dataset_file.name
                cols, auto_img_col, auto_mode = sniff_columns(buf, max_rows=3)
                st.session_state.ds_columns = cols
                st.session_state.ds_img_col = auto_img_col or (cols[0] if cols else None)
                st.session_state.ds_mode = auto_mode or "base64"
            except Exception as e:
                st.warning(f"Could not preview file: {e}")

        if st.session_state.ds_columns:
            col1, col2 = st.columns(2)
            with col1:
                img_col_choice = st.selectbox(
                    "Image column",
                    options=st.session_state.ds_columns,
                    index=st.session_state.ds_columns.index(st.session_state.ds_img_col)
                    if st.session_state.ds_img_col in st.session_state.ds_columns else 0,
                    key="ds_img_col_select",
                )
            with col2:
                mode_choice = st.selectbox(
                    "Encoding",
                    options=["base64", "url", "path"],
                    index=["base64", "url", "path"].index(st.session_state.ds_mode)
                    if st.session_state.ds_mode in ["base64", "url", "path"] else 0,
                    format_func=lambda m: {
                        "base64": "Base64 (INS-MMBench)",
                        "url": "HTTP URL",
                        "path": "Local path",
                    }[m],
                )

            max_rows_ds = st.slider(
                "Max rows to load", min_value=1, max_value=50, value=10,
                help="Keep ≤10 for Gemini free tier (15 RPM). No limit for Stub mode.",
            )

            # Let user pick a label / ground-truth column to display
            gt_col_options = ["— none —"] + st.session_state.ds_columns
            gt_col = st.selectbox(
                "Ground-truth column (optional)",
                options=gt_col_options,
                index=next(
                    (i for i, c in enumerate(gt_col_options) if "severity" in c.lower() or "label" in c.lower() or "answer" in c.lower()),
                    0,
                ),
                help="If set, shows the dataset label next to the AI prediction.",
            )
            if gt_col == "— none —":
                gt_col = None

            load_ds_btn = st.button("Load from dataset", use_container_width=True)
            if load_ds_btn and dataset_file is not None:
                try:
                    from core.tsv_loader import load_dataset_file
                    import io as _io
                    dataset_file.seek(0)
                    buf = _io.BytesIO(dataset_file.read())
                    buf.name = dataset_file.name
                    rows = load_dataset_file(
                        buf,
                        max_rows=max_rows_ds,
                        image_col=img_col_choice,
                        mode=mode_choice,
                    )
                    if rows:
                        st.session_state.ds_images = [(lbl, img) for lbl, img, _ in rows]
                        st.session_state.ds_metadata = {lbl: meta for lbl, _, meta in rows}
                        st.session_state.ds_gt_col = gt_col
                        st.success(f"Loaded {len(rows)} image(s) from dataset.")
                    else:
                        st.error("No images could be decoded from the file.")
                except Exception as e:
                    st.error(f"Failed to load dataset: {e}")

    run_btn = st.button("Run Assessment", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Collect images to process
# ---------------------------------------------------------------------------

# Session-stored dataset images take priority over manual upload / sample
if "ds_images" not in st.session_state:
    st.session_state.ds_images = []
if "ds_metadata" not in st.session_state:
    st.session_state.ds_metadata = {}
if "ds_gt_col" not in st.session_state:
    st.session_state.ds_gt_col = None

images: list[tuple[str, Image.Image]] = []
_source = "manual"

if st.session_state.ds_images:
    images = list(st.session_state.ds_images)
    _source = "dataset"
elif uploaded_files:
    for f in uploaded_files:
        img = Image.open(io.BytesIO(f.read()))
        images.append((f.name, img))
elif use_sample:
    img = Image.open(use_sample)
    images.append((use_sample.name, img))

# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------

st.title("Damage Assessment")
st.caption("Upload one or more vehicle photos and run the AI assessment. Review, override, and submit feedback.")

if _source == "dataset":
    st.info(f"**Dataset mode** — {len(images)} image(s) loaded from file. Clear by uploading a new dataset or refreshing.", icon="")

if not images:
    st.info("Upload a vehicle image, select a sample, or import a TSV/CSV dataset from the sidebar to get started.")
    st.stop()

# ---------------------------------------------------------------------------
# Run inference (with caching via session_state)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _run_assessment(img_bytes: bytes, backend: str, gemini_key: str, rest_ep: str) -> dict:
    img = Image.open(io.BytesIO(img_bytes))
    result = vlm_adapter.assess_damage(
        image=img,
        backend=backend,
        gemini_api_key=gemini_key or None,
        rest_endpoint=rest_ep or None,
    )
    return result.to_dict()


results: dict[str, AssessmentResult] = {}

if run_btn or st.session_state.get("_auto_run"):
    progress_bar = st.progress(0, text="Running assessment…")

    def _assess_one(name_img: tuple[str, Image.Image]) -> tuple[str, AssessmentResult]:
        name, img = name_img
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        try:
            data = _run_assessment(
                buf.getvalue(),
                st.session_state.vlm_backend,
                st.session_state.gemini_api_key,
                st.session_state.vlm_rest_endpoint,
            )
            return name, AssessmentResult.from_dict(data)
        except Exception as exc:
            st.error(f"Error during assessment of **{name}**: {exc}")
            return name, AssessmentResult(
                damage_codes=["ASSESSMENT_ERROR"],
                severity=Severity.MODERATE,
                cost_range=(1000, 3000),
                confidence=0.0,
                low_confidence_flag=True,
                reasoning=f"Assessment could not complete due to error: {exc}. Please check your API key or use Stub mode.",
                backend_used=f"{st.session_state.vlm_backend} (error)",
            )

    if len(images) == 1:
        name, result = _assess_one(images[0])
        results[name] = result
        progress_bar.progress(100, text="Done!")
    else:
        with ThreadPoolExecutor(max_workers=4) as ex:
            futures = {ex.submit(_assess_one, ni): ni[0] for ni in images}
            done = 0
            for fut in as_completed(futures):
                name, result = fut.result()
                results[name] = result
                done += 1
                progress_bar.progress(int(done / len(images) * 100), text=f"Processed {done}/{len(images)}…")

    st.session_state.last_assessment = results
    st.session_state.last_image = images
else:
    # Restore from session if available
    if st.session_state.last_assessment:
        results = st.session_state.last_assessment
        images = st.session_state.last_image or images

# ---------------------------------------------------------------------------
# Render per-image results in tabs
# ---------------------------------------------------------------------------

if not results:
    st.warning("Click **Run Assessment** to analyse the uploaded image(s).")
    # Still show image preview
    tabs = st.tabs([name for name, _ in images])
    for tab, (name, img) in zip(tabs, images):
        with tab:
            st.image(img, use_container_width=True)
    st.stop()

tabs = st.tabs([f"{name}" for name, _ in images])

for tab, (name, img) in zip(tabs, images):
    with tab:
        if name not in results:
            st.warning(f"No result for {name}.")
            continue

        result: AssessmentResult = results[name]
        img_hash = _image_hash(img)

        # ── Flag banner ──────────────────────────────────────────────────
        if result.fraud_flag:
            st.error(
                "**FRAUD ALERT**: The damage pattern is inconsistent with the reported incident. "
                "This claim has been flagged for senior adjuster review.",
                icon=None,
            )
        elif result.low_confidence_flag:
            st.warning(
                f"**Low Confidence** ({result.confidence:.0%}): The model is uncertain about this assessment. "
                "Manual review recommended before finalising.",
                icon=None,
            )

        # ── Ground-truth comparison (dataset mode only) ──────────────────
        if _source == "dataset":
            meta = st.session_state.ds_metadata.get(name, {})
            gt_col = st.session_state.ds_gt_col
            if meta:
                with st.expander("Dataset Row — Ground Truth vs Prediction", expanded=True):
                    # Show ground-truth severity if a GT column was chosen
                    if gt_col and gt_col in meta:
                        gt_val = str(meta[gt_col]).strip()
                        gt_sev_map = {
                            "minor": Severity.MINOR, "moderate": Severity.MODERATE,
                            "severe": Severity.SEVERE, "total loss": Severity.TOTAL_LOSS,
                            "total_loss": Severity.TOTAL_LOSS,
                        }
                        gt_norm = gt_sev_map.get(gt_val.lower(), None)
                        pred_norm = result.severity
                        match = gt_norm and gt_norm == pred_norm
                        c_gt, c_pred, c_match = st.columns([1, 1, 0.6])
                        with c_gt:
                            st.markdown("**Ground Truth**")
                            st.markdown(
                                _severity_badge(gt_norm or gt_val),
                                unsafe_allow_html=True,
                            )
                        with c_pred:
                            st.markdown("**Predicted**")
                            st.markdown(
                                _severity_badge(pred_norm),
                                unsafe_allow_html=True,
                            )
                        with c_match:
                            st.markdown("**Match?**")
                            st.markdown("Yes" if match else "No")

                    # Show remaining metadata as a compact table
                    display_meta = {k: v for k, v in meta.items()
                                    if k != gt_col and k not in ("image", "img")
                                    and not str(k).lower().startswith("unnamed")}
                    if display_meta:
                        import pandas as pd
                        meta_df = pd.DataFrame(
                            [(k, v) for k, v in display_meta.items()],
                            columns=["Field", "Value"],
                        )
                        st.dataframe(meta_df, use_container_width=True, hide_index=True)

        # ── Two-column layout: Image | Results card ──────────────────────
        col_img, col_res = st.columns([1.1, 1], gap="large")

        with col_img:
            if result.bounding_boxes:
                annotated = vlm_adapter.draw_bboxes(img, result)
                st.image(annotated, caption="Annotated — AI-identified damage regions", use_container_width=True)
            else:
                st.image(img, caption="Uploaded image", use_container_width=True)
            st.caption(f"Inference: {result.latency_ms:.0f} ms via **{result.backend_used}**")

        with col_res:
            st.markdown("#### Assessment Result")

            # Severity
            st.markdown(
                f"**Severity:** {_severity_badge(result.severity)}",
                unsafe_allow_html=True,
            )
            st.markdown("")

            # Damage codes
            st.markdown("**Damage Codes:**")
            chips_html = " ".join(_damage_chip(c) for c in result.damage_codes)
            st.markdown(chips_html or "_No codes detected_", unsafe_allow_html=True)
            st.markdown("")

            # Cost range
            if isinstance(result.cost_range, (list, tuple)) and len(result.cost_range) >= 2:
                lo, hi = int(result.cost_range[0]), int(result.cost_range[1])
            elif isinstance(result.cost_range, (list, tuple)) and len(result.cost_range) == 1:
                lo, hi = int(result.cost_range[0]), int(result.cost_range[0]) * 2
            else:
                lo, hi = 1500, 5000
            st.markdown(f"**Estimated Repair Cost:** `${lo:,}` – `${hi:,}`")
            mid = (lo + hi) / 2
            max_cost = 60_000
            st.progress(min(int(mid / max_cost * 100), 100))

            # Confidence gauge (Plotly)
            import plotly.graph_objects as go

            conf_pct = max(0.0, min(100.0, float(result.confidence or 0.5) * 100))
            conf_color = "#22c55e" if conf_pct >= 75 else ("#f59e0b" if conf_pct >= 50 else "#ef4444")
            gauge_fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=conf_pct,
                    number={"suffix": "%", "font": {"size": 28}},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": conf_color},
                        "steps": [
                            {"range": [0, 50], "color": "rgba(239, 68, 68, 0.15)"},
                            {"range": [50, 75], "color": "rgba(245, 158, 11, 0.15)"},
                            {"range": [75, 100], "color": "rgba(34, 197, 94, 0.15)"},
                        ],
                    },
                )
            )
            gauge_fig.update_layout(height=200, margin=dict(t=30, b=10, l=20, r=20))
            st.plotly_chart(gauge_fig, use_container_width=True)

        # ── Reasoning expander ───────────────────────────────────────────
        with st.expander("Model Reasoning", expanded=False):
            st.markdown(result.reasoning or "_No reasoning provided._")

        # ── Override form ────────────────────────────────────────────────
        with st.expander("Adjuster Override", expanded=False):
            st.caption("Edit the AI assessment if you disagree. Your corrections feed the fine-tuning pipeline.")

            with st.form(key=f"override_{name}_{img_hash}"):
                valid_defaults = [c for c in result.damage_codes if c in DAMAGE_CODES]
                ov_codes = st.multiselect(
                    "Damage Codes",
                    options=list(DAMAGE_CODES.keys()),
                    default=valid_defaults,
                    format_func=lambda c: DAMAGE_CODES.get(c, c),
                )
                ov_severity = st.selectbox(
                    "Severity",
                    options=[s.value for s in Severity],
                    index=[s.value for s in Severity].index(result.severity)
                    if result.severity in [s.value for s in Severity]
                    else 1,
                )
                col_lo, col_hi = st.columns(2)
                with col_lo:
                    ov_cost_lo = st.number_input(
                        "Cost Low ($)", value=int(lo), step=100, min_value=0
                    )
                with col_hi:
                    ov_cost_hi = st.number_input(
                        "Cost High ($)", value=int(hi), step=100, min_value=0
                    )
                ov_reason = st.text_area(
                    "Reason for override",
                    placeholder="Describe why you are changing the assessment…",
                    height=80,
                )

                c1, c2, c3 = st.columns(3)
                accepted_btn = c1.form_submit_button("Accept as-is", use_container_width=True)
                edit_btn = c2.form_submit_button("Edit & Submit", type="primary", use_container_width=True)
                reject_btn = c3.form_submit_button("Reject", use_container_width=True)

            if accepted_btn or edit_btn or reject_btn:
                record = FeedbackRecord(
                    image_hash=img_hash,
                    original_assessment=result,
                    accepted=accepted_btn,
                    adjuster_damage_codes=ov_codes if edit_btn else [],
                    adjuster_severity=ov_severity if edit_btn else "",
                    adjuster_cost_low=int(ov_cost_lo) if edit_btn else 0,
                    adjuster_cost_high=int(ov_cost_hi) if edit_btn else 0,
                    override_reason=ov_reason,
                    persona=st.session_state.persona,
                )
                row_id = feedback_store.log_feedback(record)
                if accepted_btn:
                    st.success(f"Assessment accepted and logged (ID #{row_id}).")
                elif edit_btn:
                    st.success(f"Override submitted (ID #{row_id}). Thank you!")
                else:
                    st.error(f"Assessment rejected and logged (ID #{row_id}).")

        st.divider()
