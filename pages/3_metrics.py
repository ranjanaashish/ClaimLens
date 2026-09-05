"""
pages/3_metrics.py — Evaluation Metrics & Model Performance page.

Visualisations (all Plotly, interactive):
  1. Summary stat cards  (accuracy, FP rate, mIoU, feedback count)
  2. Radar chart         — per-category performance (our model vs baseline vs GPT-4V)
  3. Per-class F1 bar chart — horizontal, sorted descending
  4. Confusion matrix heatmap — severity predicted vs true
  5. INS-MMBench grouped bar chart — category-level comparison
  6. Confidence distribution histogram
  7. Severity distribution pie / donut
  8. Feedback quality panel (live from SQLite)
  9. JSONL download button
"""
import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from core import feedback_store

# ---------------------------------------------------------------------------
# Load eval data
# ---------------------------------------------------------------------------

EVAL_PATH = Path(__file__).parent.parent / "data" / "eval_results.json"

@st.cache_data
def _load_eval() -> dict:
    if EVAL_PATH.exists():
        try:
            return json.loads(EVAL_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

data = _load_eval()

# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------

st.title("Evaluation Metrics")
st.caption(
    "Model performance on the held-out test set. "
    "Comparison includes INS-MMBench auto-insurance subset and a GPT-4V baseline."
)

if not data:
    st.error("No evaluation data found. Place `eval_results.json` in `data/`.")
    st.stop()

overall = data.get("overall", {})

# ---------------------------------------------------------------------------
# 1. Summary stat cards
# ---------------------------------------------------------------------------

feedback_count = feedback_store.get_feedback_count()
acceptance = feedback_store.get_acceptance_stats()
total_decisions = sum(acceptance.values()) or 1

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Accuracy",         f"{overall.get('accuracy', 0):.1%}",    delta=f"+{(overall.get('accuracy',0) - 0.771):.1%} vs baseline")
col2.metric("False Positive Rate", f"{overall.get('false_positive_rate', 0):.1%}", delta=None)
col3.metric("mIoU",              f"{overall.get('mIoU', 0):.2f}",       delta=None)
col4.metric("Feedback Submissions", feedback_count,                     delta=None)
col5.metric("Acceptance Rate",   f"{acceptance['accepted'] / total_decisions:.0%}" if feedback_count else "—", delta=None)

st.divider()

# ---------------------------------------------------------------------------
# Layout: two columns for compact display
# ---------------------------------------------------------------------------

left_col, right_col = st.columns(2, gap="large")

# ---------------------------------------------------------------------------
# 2. Radar chart — INS-MMBench category comparison
# ---------------------------------------------------------------------------

bench = data.get("ins_mmbench", {})
categories = bench.get("categories", [])
our_scores = bench.get("our_model", [])
baseline_scores = bench.get("baseline", [])
gpt4v_scores = bench.get("gpt4v", [])

with left_col:
    st.subheader("Multi-Category Performance (Radar)")

    radar_fig = go.Figure()
    theta = categories + [categories[0]]  # close the polygon

    radar_fig.add_trace(go.Scatterpolar(
        r=our_scores + [our_scores[0]],
        theta=theta,
        fill="toself",
        name="Our Model",
        line_color="#3b82f6",
        fillcolor="rgba(59,130,246,0.15)",
    ))
    radar_fig.add_trace(go.Scatterpolar(
        r=baseline_scores + [baseline_scores[0]],
        theta=theta,
        fill="toself",
        name="INS-MMBench Baseline",
        line_color="#f59e0b",
        fillcolor="rgba(245,158,11,0.10)",
    ))
    radar_fig.add_trace(go.Scatterpolar(
        r=gpt4v_scores + [gpt4v_scores[0]],
        theta=theta,
        fill="toself",
        name="GPT-4V",
        line_color="#a855f7",
        fillcolor="rgba(168,85,247,0.10)",
    ))
    radar_fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0.5, 1.0])),
        legend=dict(orientation="h", y=-0.15),
        margin=dict(t=20, b=60, l=40, r=40),
        height=360,
    )
    st.plotly_chart(radar_fig, use_container_width=True)

# ---------------------------------------------------------------------------
# 3. Confusion matrix heatmap
# ---------------------------------------------------------------------------

with right_col:
    st.subheader("Severity Confusion Matrix")

    cm_data = data.get("severity_confusion_matrix", {})
    labels = cm_data.get("labels", [])
    matrix = cm_data.get("matrix", [])

    if labels and matrix:
        import numpy as np
        mat = np.array(matrix)
        # Normalise by row (true label) to show recall
        row_sums = mat.sum(axis=1, keepdims=True)
        mat_norm = mat / row_sums

        cm_fig = go.Figure(
            go.Heatmap(
                z=mat_norm,
                x=[f"Pred: {l}" for l in labels],
                y=[f"True: {l}" for l in labels],
                colorscale="Blues",
                text=[[f"{mat[i][j]}\n({mat_norm[i][j]:.0%})" for j in range(len(labels))] for i in range(len(labels))],
                texttemplate="%{text}",
                textfont={"size": 12},
                hovertemplate="True: %{y}<br>Pred: %{x}<br>Count: %{text}<extra></extra>",
            )
        )
        cm_fig.update_layout(
            xaxis_title="Predicted Severity",
            yaxis_title="True Severity",
            margin=dict(t=20, b=40, l=100, r=20),
            height=360,
        )
        st.plotly_chart(cm_fig, use_container_width=True)
    else:
        st.info("No confusion matrix data available.")

st.divider()

left2, right2 = st.columns(2, gap="large")

# ---------------------------------------------------------------------------
# 4. Per-class F1 bar chart (horizontal, sorted)
# ---------------------------------------------------------------------------

with left2:
    st.subheader("Per-Damage-Type F1 Score")

    per_class = data.get("per_class", [])
    if per_class:
        df_pc = pd.DataFrame(per_class).sort_values("f1", ascending=True)
        df_pc["short_name"] = df_pc["damage_code"].str.replace("_", " ").str.title()

        bar_fig = px.bar(
            df_pc,
            x="f1",
            y="short_name",
            orientation="h",
            color="f1",
            color_continuous_scale=["#ef4444", "#f59e0b", "#22c55e"],
            range_color=[0.5, 1.0],
            hover_data={"precision": ":.2f", "recall": ":.2f", "support": True},
            labels={"f1": "F1 Score", "short_name": "Damage Type"},
        )
        bar_fig.add_vline(x=overall.get("f1_macro", 0.8), line_dash="dash",
                          line_color="white", annotation_text="Macro avg")
        bar_fig.update_layout(
            coloraxis_showscale=False,
            margin=dict(t=10, b=40, l=180, r=20),
            height=500,
            yaxis=dict(tickfont=dict(size=11)),
        )
        st.plotly_chart(bar_fig, use_container_width=True)

# ---------------------------------------------------------------------------
# 5. INS-MMBench grouped bar
# ---------------------------------------------------------------------------

with right2:
    st.subheader("INS-MMBench Comparison")

    if categories and our_scores:
        bench_df = pd.DataFrame({
            "Category": categories * 3,
            "Score": our_scores + baseline_scores + gpt4v_scores,
            "Model": (
                ["Our Model"] * len(categories)
                + ["Baseline"] * len(categories)
                + ["GPT-4V"] * len(categories)
            ),
        })
        bench_fig = px.bar(
            bench_df,
            x="Category",
            y="Score",
            color="Model",
            barmode="group",
            color_discrete_map={
                "Our Model": "#3b82f6",
                "Baseline": "#f59e0b",
                "GPT-4V": "#a855f7",
            },
            range_y=[0.5, 1.0],
        )
        bench_fig.update_layout(
            legend=dict(orientation="h", y=1.1),
            margin=dict(t=30, b=40),
            height=380,
        )
        st.plotly_chart(bench_fig, use_container_width=True)

st.divider()

left3, right3 = st.columns(2, gap="large")

# ---------------------------------------------------------------------------
# 6. Confidence histogram
# ---------------------------------------------------------------------------

with left3:
    st.subheader("Confidence Distribution")

    conf_data = data.get("confidence_histogram", {})
    if conf_data:
        hist_fig = go.Figure(
            go.Bar(
                x=conf_data["bins"],
                y=conf_data["counts"],
                marker_color=["#ef4444", "#f97316", "#f59e0b", "#84cc16", "#22c55e"],
                text=conf_data["counts"],
                textposition="outside",
            )
        )
        hist_fig.update_layout(
            xaxis_title="Confidence Bucket",
            yaxis_title="# Predictions",
            margin=dict(t=10, b=40),
            height=300,
        )
        st.plotly_chart(hist_fig, use_container_width=True)

# ---------------------------------------------------------------------------
# 7. Severity distribution donut
# ---------------------------------------------------------------------------

with right3:
    st.subheader("Severity Distribution (Test Set)")

    sev_dist = data.get("severity_distribution", {})
    if sev_dist:
        sev_fig = go.Figure(
            go.Pie(
                labels=list(sev_dist.keys()),
                values=list(sev_dist.values()),
                hole=0.45,
                marker_colors=["#22c55e", "#f59e0b", "#ef4444", "#7c3aed"],
            )
        )
        sev_fig.update_layout(
            legend=dict(orientation="h", y=-0.1),
            margin=dict(t=10, b=40),
            height=300,
        )
        st.plotly_chart(sev_fig, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# 8. Live feedback quality panel
# ---------------------------------------------------------------------------

st.subheader("Human Feedback Loop")

col_a, col_b, col_c = st.columns(3)
col_a.metric("Total Submissions", feedback_count)
col_b.metric("Accepted as-is", acceptance.get("accepted", 0))
col_c.metric("Edited", acceptance.get("edited", 0))

override_df = feedback_store.get_override_distribution()
if not override_df.empty:
    st.markdown("**Most-Corrected Damage Codes** (override signal for fine-tuning)")
    ov_fig = px.bar(
        override_df.head(10),
        x="count",
        y="damage_code",
        orientation="h",
        color="count",
        color_continuous_scale="Oranges",
        labels={"count": "Override Count", "damage_code": "Damage Code"},
    )
    ov_fig.update_layout(coloraxis_showscale=False, height=300, margin=dict(t=10, b=40, l=160))
    st.plotly_chart(ov_fig, use_container_width=True)
else:
    st.info("No override data yet. Accept, edit, or reject assessments on the **Damage Assessment** page to populate this panel.")

# ---------------------------------------------------------------------------
# 9. JSONL export
# ---------------------------------------------------------------------------

st.divider()
st.subheader("Fine-Tuning Export")
st.markdown(
    "Export all adjuster feedback as **DPO-aligned JSONL** for fine-tuning the VLM backbone. "
    "Format is compatible with HuggingFace `trl` DPO trainer and Axolotl."
)

col_exp, col_info = st.columns([1, 2])
with col_exp:
    jsonl_content = feedback_store.export_jsonl()
    if jsonl_content:
        st.download_button(
            label=f"⬇️ Download feedback.jsonl ({feedback_count} records)",
            data=jsonl_content,
            file_name="feedback.jsonl",
            mime="application/jsonl",
            type="primary",
            use_container_width=True,
        )
    else:
        st.button(
            "⬇️ Download feedback.jsonl (0 records)",
            disabled=True,
            use_container_width=True,
        )

with col_info:
    st.markdown("""
    **JSONL record schema:**
    ```json
    {
      "prompt": "Assess the vehicle damage…",
      "chosen": "{adjuster-corrected JSON}",
      "rejected": "{model-predicted JSON}",
      "metadata": {"image_hash": "…", "persona": "Adjuster", "timestamp": "…"}
    }
    ```
    """)
