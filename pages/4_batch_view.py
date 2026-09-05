"""
pages/4_batch_view.py — Batch Review page (stretch goal).

Features:
- Table of all processed assessments (from session_state batch_results)
- CSV batch upload path (upload CSV of image paths → run assessments)
- Sidebar filters: damage type, severity, confidence, flag status
- Click row → deep-link to Assessment page with claim pre-loaded
- Export table as CSV
"""
import io
import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from core import feedback_store, vlm_adapter
from core.schema import DAMAGE_CODES, SEVERITY_COLORS, AssessmentResult, Severity

# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------

st.title("Batch Review")
st.caption(
    "Review all processed claims in one view. "
    "Filter by damage type, severity, or flag status. Click a row to drill into the Assessment page."
)

# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### Filters")

    filter_severity = st.multiselect(
        "Severity",
        options=[s.value for s in Severity],
        default=[],
        placeholder="All severities",
    )

    filter_codes = st.multiselect(
        "Damage Type",
        options=list(DAMAGE_CODES.keys()),
        format_func=lambda c: DAMAGE_CODES.get(c, c),
        default=[],
        placeholder="All types",
    )

    filter_flag = st.selectbox(
        "Flag Status",
        options=["All", "Fraud Flagged", "Low Confidence", "Clean"],
        index=0,
    )

    conf_threshold = st.slider(
        "Min Confidence",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.05,
        format="%.0%%",
    )

    st.divider()
    if st.button("Clear batch results", use_container_width=True):
        st.session_state.batch_results = []
        st.rerun()

# ---------------------------------------------------------------------------
# Batch CSV upload section
# ---------------------------------------------------------------------------

with st.expander(" Add images to batch", expanded=not st.session_state.batch_results):
    st.markdown(
        "Upload vehicle images directly, or upload a **CSV** with an `image_path` column "
        "pointing to local files."
    )
    batch_files = st.file_uploader(
        "Upload images",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    run_batch_btn = st.button("▶️ Run Batch Assessment", type="primary")

    if run_batch_btn and batch_files:
        progress = st.progress(0, text="Running batch…")
        new_results = []
        for i, f in enumerate(batch_files):
            img_bytes = f.read()
            img = Image.open(io.BytesIO(img_bytes))
            try:
                result = vlm_adapter.assess_damage(
                    image=img,
                    backend=st.session_state.vlm_backend,
                    gemini_api_key=st.session_state.gemini_api_key or None,
                    rest_endpoint=st.session_state.vlm_rest_endpoint or None,
                )
            except Exception as exc:
                st.warning(f"Error processing {f.name}: {exc}")
                continue

            new_results.append({
                "file_name": f.name,
                "result_json": result.to_json(),
            })
            progress.progress(int((i + 1) / len(batch_files) * 100), text=f"Processed {i+1}/{len(batch_files)}")

        st.session_state.batch_results.extend(new_results)
        st.success(f"{len(new_results)} images assessed.")
        progress.empty()
        st.rerun()

# ---------------------------------------------------------------------------
# Dataset / TSV batch loading
# ---------------------------------------------------------------------------

with st.expander("Batch from TSV / CSV / Excel dataset", expanded=False):
    st.markdown(
        "Load images directly from an **INS-MMBench TSV** or any CSV/Excel with an image column. "
        "Each row becomes one batch assessment entry."
    )
    ds_batch_file = st.file_uploader(
        "Dataset file (TSV / CSV / XLSX)",
        type=["tsv", "csv", "xlsx", "xls"],
        key="ds_batch_upload",
        label_visibility="collapsed",
    )

    if "ds_batch_columns" not in st.session_state:
        st.session_state.ds_batch_columns = []
    if "ds_batch_img_col" not in st.session_state:
        st.session_state.ds_batch_img_col = None
    if "ds_batch_mode" not in st.session_state:
        st.session_state.ds_batch_mode = "base64"

    if ds_batch_file is not None:
        try:
            from core.tsv_loader import sniff_columns
            import io as _io
            raw = ds_batch_file.read()
            ds_batch_file.seek(0)
            buf = _io.BytesIO(raw)
            buf.name = ds_batch_file.name
            cols, auto_col, auto_mode = sniff_columns(buf, max_rows=3)
            st.session_state.ds_batch_columns = cols
            st.session_state.ds_batch_img_col = auto_col
            st.session_state.ds_batch_mode = auto_mode or "base64"
        except Exception as e:
            st.warning(f"Could not preview: {e}")

    if st.session_state.ds_batch_columns:
        c1, c2, c3 = st.columns([2, 2, 1])
        with c1:
            ds_img_col = st.selectbox(
                "Image column",
                st.session_state.ds_batch_columns,
                index=st.session_state.ds_batch_columns.index(st.session_state.ds_batch_img_col)
                if st.session_state.ds_batch_img_col in st.session_state.ds_batch_columns else 0,
            )
        with c2:
            ds_enc = st.selectbox(
                "Encoding",
                ["base64", "url", "path"],
                index=["base64", "url", "path"].index(st.session_state.ds_batch_mode)
                if st.session_state.ds_batch_mode in ["base64", "url", "path"] else 0,
                format_func=lambda m: {"base64": "Base64", "url": "URL", "path": "Path"}[m],
            )
        with c3:
            ds_max = st.number_input("Max rows", min_value=1, max_value=200, value=20)

        run_ds_batch_btn = st.button("▶️ Run Dataset Batch", type="primary")
        if run_ds_batch_btn and ds_batch_file is not None:
            from core.tsv_loader import load_dataset_file
            import io as _io
            ds_batch_file.seek(0)
            buf = _io.BytesIO(ds_batch_file.read())
            buf.name = ds_batch_file.name
            try:
                rows_ds = load_dataset_file(buf, max_rows=int(ds_max), image_col=ds_img_col, mode=ds_enc)
            except Exception as e:
                st.error(f"Failed to load dataset: {e}")
                rows_ds = []

            if rows_ds:
                progress_ds = st.progress(0, text="Running batch on dataset rows…")
                new_ds_results = []
                for i, (label, img, meta) in enumerate(rows_ds):
                    try:
                        result = vlm_adapter.assess_damage(
                            image=img,
                            backend=st.session_state.vlm_backend,
                            gemini_api_key=st.session_state.gemini_api_key or None,
                            rest_endpoint=st.session_state.vlm_rest_endpoint or None,
                        )
                    except Exception as exc:
                        st.warning(f"Error on row {label}: {exc}")
                        continue
                    new_ds_results.append({
                        "file_name": label,
                        "result_json": result.to_json(),
                    })
                    progress_ds.progress(
                        int((i + 1) / len(rows_ds) * 100),
                        text=f"Processed {i+1}/{len(rows_ds)}",
                    )

                st.session_state.batch_results.extend(new_ds_results)
                st.success(f"{len(new_ds_results)} dataset rows assessed.")
                progress_ds.empty()
                st.rerun()

# ---------------------------------------------------------------------------
# Build table DataFrame
# ---------------------------------------------------------------------------

if not st.session_state.batch_results:
    st.info(
        "No batch results yet. Upload images above, or run individual assessments on the "
        "**Damage Assessment** page — they'll appear here after you run them as a batch."
    )

    # Show sample mock data for demo
    st.markdown("#### Sample batch table (demo data)")
    mock_rows = [
        {"Claim ID": "CL-001", "Severity": "Moderate", "Damage Codes": "FRONT_BUMPER_CRUSH, HOOD_CREASE",
         "Est. Cost": "$3,500–$6,200", "Confidence": "87%", "Flags": "—"},
        {"Claim ID": "CL-002", "Severity": "Minor", "Damage Codes": "PAINT_TRANSFER, FRONT_BUMPER_SCRATCH",
         "Est. Cost": "$350–$900", "Confidence": "93%", "Flags": "—"},
        {"Claim ID": "CL-003", "Severity": "Severe", "Damage Codes": "STRUCTURAL_DAMAGE, AIRBAG_DEPLOYED, HOOD_CRUSH",
         "Est. Cost": "$28,000–$45,000", "Confidence": "96%", "Flags": "—"},
        {"Claim ID": "CL-004", "Severity": "Moderate", "Damage Codes": "DOOR_FL_DENT, DOOR_RL_DENT",
         "Est. Cost": "$4,200–$7,800", "Confidence": "42%", "Flags": "Low Confidence"},
        {"Claim ID": "CL-005", "Severity": "Moderate", "Damage Codes": "REAR_BUMPER_CRUSH, TRUNK_LID_DENT",
         "Est. Cost": "$2,800–$5,100", "Confidence": "82%", "Flags": "Fraud"},
    ]
    st.dataframe(pd.DataFrame(mock_rows), use_container_width=True, hide_index=True)
    st.stop()

# ---------------------------------------------------------------------------
# Parse batch results into DataFrame
# ---------------------------------------------------------------------------

rows = []
for item in st.session_state.batch_results:
    try:
        r = AssessmentResult.from_dict(json.loads(item["result_json"]))
    except Exception:
        continue

    flags = []
    if r.fraud_flag:
        flags.append("Fraud")
    if r.low_confidence_flag:
        flags.append("Low Conf.")

    rows.append({
        "File": item["file_name"],
        "Severity": r.severity,
        "Damage Codes": ", ".join(r.damage_codes),
        "Cost Low": r.cost_range[0] if len(r.cost_range) > 0 else 0,
        "Cost High": r.cost_range[1] if len(r.cost_range) > 1 else (r.cost_range[0] if len(r.cost_range) > 0 else 0),
        "Confidence": r.confidence,
        "Flags": " | ".join(flags) or "—",
        "Backend": r.backend_used,
        "_result_json": item["result_json"],
    })

df = pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# Apply filters
# ---------------------------------------------------------------------------

if filter_severity:
    df = df[df["Severity"].isin(filter_severity)]

if filter_codes:
    mask = df["Damage Codes"].apply(lambda x: any(c in x for c in filter_codes))
    df = df[mask]

if filter_flag == "Fraud Flagged":
    df = df[df["Flags"].str.contains("Fraud")]
elif filter_flag == "Low Confidence":
    df = df[df["Flags"].str.contains("Low Conf")]
elif filter_flag == "Clean":
    df = df[df["Flags"] == "—"]

df = df[df["Confidence"] >= conf_threshold]

# ---------------------------------------------------------------------------
# Summary mini-charts above the table
# ---------------------------------------------------------------------------

if not df.empty:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Claims", len(df))
    m2.metric("Fraud Flagged", df["Flags"].str.contains("Fraud").sum())
    m3.metric("Low Confidence", df["Flags"].str.contains("Low Conf").sum())
    m4.metric("Avg Confidence", f"{df['Confidence'].mean():.0%}")

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        sev_counts = df["Severity"].value_counts().reset_index()
        sev_counts.columns = ["Severity", "Count"]
        fig_sev = px.bar(
            sev_counts,
            x="Severity",
            y="Count",
            color="Severity",
            color_discrete_map={s: c for s, c in SEVERITY_COLORS.items()},
            title="Severity Breakdown",
            height=250,
        )
        fig_sev.update_layout(showlegend=False, margin=dict(t=40, b=10))
        st.plotly_chart(fig_sev, use_container_width=True)

    with chart_col2:
        fig_conf = px.histogram(
            df,
            x="Confidence",
            nbins=10,
            title="Confidence Distribution",
            color_discrete_sequence=["#3b82f6"],
            height=250,
        )
        fig_conf.update_layout(margin=dict(t=40, b=10))
        st.plotly_chart(fig_conf, use_container_width=True)

# ---------------------------------------------------------------------------
# Main table
# ---------------------------------------------------------------------------

display_df = df.drop(columns=["_result_json"]).copy()
display_df["Cost Range"] = display_df.apply(
    lambda r: f"${r['Cost Low']:,}–${r['Cost High']:,}", axis=1
)
display_df["Confidence"] = display_df["Confidence"].map(lambda x: f"{x:.0%}")
display_df = display_df.drop(columns=["Cost Low", "Cost High"])

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Severity": st.column_config.TextColumn("Severity", width="small"),
        "Confidence": st.column_config.TextColumn("Conf.", width="small"),
        "Flags": st.column_config.TextColumn("Flags"),
    },
)

# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

csv = display_df.to_csv(index=False)
st.download_button(
    "⬇️ Export as CSV",
    data=csv,
    file_name="batch_assessments.csv",
    mime="text/csv",
)
