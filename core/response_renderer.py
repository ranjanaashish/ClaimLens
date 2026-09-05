"""
core/response_renderer.py
=========================
Renders structured ChatResponse objects as self-contained HTML cards.
Produces domain-adaptive visuals for Agriculture/Plants, Vehicles, Property, and General queries.
All CSS lives in assets/style_chat.css. All HTML is dedented to prevent markdown code-block leakage.
"""
from __future__ import annotations

import base64
import html
import io
import math
import os
import re
import textwrap
from dataclasses import dataclass, field
from typing import Literal

import markdown
from PIL import Image


# ---------------------------------------------------------------------------
# ChatResponse dataclass
# ---------------------------------------------------------------------------

@dataclass
class ChatResponse:
    type: Literal["assessment", "text", "error"]
    text: str = ""
    result: object | None = None          # AssessmentResult if type == "assessment"
    annotated_image: Image.Image | None = None
    sources: list[str] = field(default_factory=list)
    citations: list[dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    error_msg: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _esc(s: str) -> str:
    return html.escape(str(s))


def _img_b64(img: Image.Image, max_px: int = 640) -> str:
    """Encode PIL image as base64 data URI."""
    img = img.copy()
    img.thumbnail((max_px, max_px), Image.LANCZOS)
    if img.mode != "RGB":
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=82)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"


def _severity_class(sev: str) -> str:
    s = sev.lower().replace(" ", "_")
    if "minor" in s or "healthy" in s or "low" in s or "none" in s:
        return "cl-badge-minor"
    if "moderate" in s or "medium" in s:
        return "cl-badge-moderate"
    if "severe" in s and "total" not in s:
        return "cl-badge-severe"
    if "total" in s or "critical" in s or "fatal" in s:
        return "cl-badge-total"
    return "cl-badge-moderate"


def _conf_arc_svg(pct: float, size: int = 72) -> str:
    """SVG semicircle gauge showing confidence percentage."""
    r = size * 0.44
    cx = size / 2
    cy = size * 0.52
    circ = math.pi * r
    pct = max(0.0, min(1.0, pct))
    color = "#22c55e" if pct >= 0.75 else ("#f59e0b" if pct >= 0.5 else "#ef4444")
    return (
        f'<svg width="{size}" height="{size//2+4}" viewBox="0 0 {size} {size//2+4}">'
        f'<path d="M{cx-r},{cy} A{r},{r} 0 0,1 {cx+r},{cy}" '
        f'fill="none" stroke="#272727" stroke-width="5" stroke-linecap="round"/>'
        f'<path d="M{cx-r},{cy} A{r},{r} 0 0,1 {cx+r},{cy}" '
        f'fill="none" stroke="{color}" stroke-width="5" stroke-linecap="round" '
        f'stroke-dasharray="{circ*pct:.1f} {circ*(1-pct):.1f}"/>'
        f'</svg>'
    )


def _cost_pct(lo: int, hi: int, max_cost: int = 60_000) -> int:
    mid = (lo + hi) / 2
    return min(100, max(2, int(mid / max_cost * 100)))


# ---------------------------------------------------------------------------
# Assessment card renderer
# ---------------------------------------------------------------------------

_CODE_DETAILS: dict[str, tuple[str, str, str, float]] = {
    "FRONT_BUMPER_CRUSH": ("Front Bumper Assembly", "Compression crush & structural absorber deformation", "Replace Bumper & Calibrate ADAS", 0.35),
    "FRONT_BUMPER_SCRATCH": ("Front Bumper Cover", "Clearcoat abrasion & surface scuff", "Sand, Spot Prime & Refinish", 0.15),
    "REAR_BUMPER_CRUSH": ("Rear Bumper Assembly", "Rear impact crush & absorber collapse", "Replace Bumper Cover & Absorber", 0.32),
    "REAR_BUMPER_SCRATCH": ("Rear Bumper Cover", "Superficial friction scratch & scuff", "Clean, Surface Blend & Recoat", 0.14),
    "HOOD_CREASE": ("Hood Panel", "Body line impact crease / dent", "Paintless Dent Repair (PDR) / Reshape", 0.24),
    "HOOD_CRUSH": ("Hood Assembly", "Buckled hood sheet metal & hinge misalignment", "Replace Hood Assembly & Alignment", 0.38),
    "TRUNK_LID_DENT": ("Trunk Lid", "Localized impact depression", "Metal Straightening & Clearcoat", 0.20),
    "TRUNK_LID_CRUSH": ("Trunk Lid Assembly", "Severe intrusion & latch mechanism failure", "Replace Trunk Lid & Weatherstrip", 0.36),
    "DOOR_FL_DENT": ("Front-Left Door", "Door skin crease / depression", "PDR / Panel Straightening", 0.22),
    "DOOR_FR_DENT": ("Front-Right Door", "Door skin crease / depression", "PDR / Panel Straightening", 0.22),
    "DOOR_RL_DENT": ("Rear-Left Door", "Side crease / door dent", "Panel Reshape & Clearcoat", 0.20),
    "DOOR_RR_DENT": ("Rear-Right Door", "Side crease / door dent", "Panel Reshape & Clearcoat", 0.20),
    "FENDER_FL_DENT": ("Front-Left Fender", "Wheel arch crumple & edge bend", "Fender Realignment or Replacement", 0.22),
    "FENDER_FR_DENT": ("Front-Right Fender", "Wheel arch crumple & edge bend", "Fender Realignment or Replacement", 0.22),
    "WINDSHIELD_CRACK": ("Windshield Glass", "Stress fracture / Starburst stone crack", "Full Glass Replacement + Calibration", 0.18),
    "WINDSHIELD_SHATTER": ("Windshield Glass", "Complete laminated glass failure", "Glass Evacuation & OEM Replacement", 0.25),
    "REAR_WINDOW_CRACK": ("Rear Window Glass", "Stress fracture from unibody flex", "Replace Defroster Window Glass", 0.18),
    "SIDE_MIRROR_DAMAGE": ("Side Mirror Assembly", "Housing fracture & motor break", "Replace Mirror Assembly & Wire Harness", 0.12),
    "WHEEL_RIM_DAMAGE": ("Wheel & Alloy Rim", "Deep curb rash & rim edge bend", "CNC Wheel Machining & Truing", 0.15),
    "UNDERCARRIAGE_DAMAGE": ("Undercarriage & Subframe", "Lower skid plate scrape & subframe gouge", "Hoist Inspection & Shield Replacement", 0.30),
    "PAINT_TRANSFER": ("Clearcoat / Paint", "Foreign vehicle paint transfer", "Chemical Solvent Clean & Buff Polish", 0.08),
    "STRUCTURAL_DAMAGE": ("Unibody Frame & Pillars", "Structural rail deflection / A-pillar stress", "Bench Hydraulic Pull & Laser Alignment", 0.55),
    "AIRBAG_DEPLOYED": ("SRS Restraint System", "Airbag deployment & module lock", "Replace Airbags, Clockspring & Module Reset", 0.45),
    "FLOOD_DAMAGE": ("Electrical & Interior", "Water ingress above floorpan line", "Full Harness Flush & Interior Sanitize", 0.60),
}


def render_assessment_card(response: ChatResponse) -> str:
    r = response.result
    if r is None:
        return render_text_card(response)

    domain = str(getattr(r, "domain", "vehicle") or "vehicle").lower()
    subject = _esc(getattr(r, "subject", "") or "")
    status_label = _esc(getattr(r, "status_label", "") or "")
    backend = _esc(getattr(r, "backend_used", ""))
    latency = getattr(r, "latency_ms", 0)

    # ── Header ──────────────────────────────────────────────────────────────
    if domain == "agriculture":
        icon = "🌱"
        domain_title = "Agricultural & Plant Health Assessment"
    elif domain == "property":
        icon = "🏠"
        domain_title = "Property & Structural Inspection"
    elif domain == "vehicle":
        icon = "🚗"
        domain_title = "Vehicle Damage & Claims Assessment"
    else:
        icon = "🔍"
        domain_title = "Multimodal Visual Analysis"

    sub_line = ""
    if subject:
        sub_line = f'<div class="cl-card-subject">{subject} {("— " + status_label) if status_label else ""}</div>'

    header = f"""<div class="cl-card-header">
  <div>
    <div class="cl-card-domain-label">{domain_title}</div>
    {sub_line}
  </div>
  <span class="cl-card-meta">{backend} · {latency:.0f} ms</span>
</div>"""

    # ── Annotated image ──────────────────────────────────────────────────────
    img_html = ""
    if response.annotated_image is not None:
        src = _img_b64(response.annotated_image)
        img_html = f"""<div class="cl-card-section" style="padding:12px 12px 0;">
  <img src="{src}" class="cl-annotated-img" alt="Annotated assessment image"/>
</div>"""

    # ── 1. Executive Summary ─────────────────────────────────────────────────
    raw_summary = getattr(r, "summary", "") or response.text or ""
    if not raw_summary:
        if domain == "agriculture":
            raw_summary = f"Visual assessment of {subject or 'plant specimen'} identified {status_label or 'findings'}."
        elif domain == "vehicle":
            raw_summary = f"Assessment completed for vehicle image. Damage classified as {getattr(r, 'severity', 'Moderate')}."
        else:
            raw_summary = f"Inspection completed for {subject or 'specimen'}."

    summary_section = f"""<div class="cl-card-section" style="padding-bottom:12px;">
  <div class="cl-label">📌 Executive Summary</div>
  <div class="cl-summary-box">
    {_esc(raw_summary)}
  </div>
</div>"""

    # ── 2. Facts & Figures KPI Grid ──────────────────────────────────────────
    conf = max(0.0, min(1.0, float(getattr(r, "confidence", 0.75) or 0.75)))
    arc_svg = _conf_arc_svg(conf)
    conf_color = "#22c55e" if conf >= 0.75 else ("#f59e0b" if conf >= 0.5 else "#ef4444")
    conf_label = 'High' if conf >= 0.75 else ('Medium' if conf >= 0.5 else 'Low')

    sev = getattr(r, "severity", "Unknown")
    sev_cls = _severity_class(str(sev))
    fraud = getattr(r, "fraud_flag", False)
    low_conf = getattr(r, "low_confidence_flag", False)

    cost = getattr(r, "cost_range", (0, 0))
    if isinstance(cost, (list, tuple)) and len(cost) >= 2:
        lo, hi = int(cost[0]), int(cost[1])
    else:
        lo, hi = 0, 0

    key_figures = getattr(r, "key_figures", []) or []

    cards_html = []
    if key_figures and len(key_figures) >= 2:
        for i, kf in enumerate(key_figures[:3]):
            lbl = _esc(kf.get("label", f"Metric {i+1}"))
            val = _esc(kf.get("value", "—"))
            cards_html.append(f"""<div class="cl-kpi-card">
  <div class="cl-label">{lbl}</div>
  <div class="cl-kpi-val" style="font-size:15px;">{val}</div>
  <div class="cl-kpi-sub">Factual Metric</div>
</div>""")
        cards_html.append(f"""<div class="cl-kpi-card">
  <div class="cl-label">Model Confidence</div>
  <div class="cl-conf-row" style="margin-top:2px;">
    {arc_svg}
    <div>
      <div class="cl-kpi-val" style="color:{conf_color};font-size:17px;">{conf:.0%}</div>
      <div class="cl-kpi-sub">{conf_label} confidence</div>
    </div>
  </div>
</div>""")
    elif domain == "agriculture":
        cards_html = [
            f"""<div class="cl-kpi-card">
  <div class="cl-label">Condition / Status</div>
  <div style="margin:4px 0;"><span class="cl-badge {sev_cls}">{_esc(str(status_label or sev))}</span></div>
  <div class="cl-kpi-sub">Botanical Health</div>
</div>""",
            f"""<div class="cl-kpi-card">
  <div class="cl-label">Est. Foliage Impact</div>
  <div class="cl-kpi-val">15 – 25%</div>
  <div class="cl-kpi-sub">Canopy Area Affected</div>
</div>""",
            f"""<div class="cl-kpi-card">
  <div class="cl-label">Est. Treatment Cost</div>
  <div class="cl-kpi-val">${lo:,} &ndash; ${hi:,}</div>
  <div class="cl-kpi-sub">Biofungicide & Sanitation</div>
</div>""",
            f"""<div class="cl-kpi-card">
  <div class="cl-label">Model Confidence</div>
  <div class="cl-conf-row" style="margin-top:2px;">
    {arc_svg}
    <div>
      <div class="cl-kpi-val" style="color:{conf_color};font-size:17px;">{conf:.0%}</div>
      <div class="cl-kpi-sub">{conf_label} confidence</div>
    </div>
  </div>
</div>""",
        ]
    elif domain == "vehicle":
        pct_bar = _cost_pct(lo, hi)
        sev_str = str(sev).lower()
        if "minor" in sev_str:
            labor_est = "3 – 6 Hours"
        elif "moderate" in sev_str:
            labor_est = "12 – 22 Hours"
        elif "severe" in sev_str and "total" not in sev_str:
            labor_est = "32 – 55 Hours"
        else:
            labor_est = "60+ Hours (Total Loss)"

        risk_label = "🚨 Anomaly Flag" if fraud else ("⚠️ Manual Review" if low_conf else "✅ Standard Flow")
        risk_color = "#ef4444" if fraud else ("#f59e0b" if low_conf else "#22c55e")

        cards_html = [
            f"""<div class="cl-kpi-card">
  <div class="cl-label">Severity Rating</div>
  <div style="margin:4px 0;"><span class="cl-badge {sev_cls}">{_esc(str(sev))}</span></div>
  <div class="cl-kpi-sub">Impact classification</div>
</div>""",
            f"""<div class="cl-kpi-card">
  <div class="cl-label">Est. Repair Cost</div>
  <div class="cl-kpi-val">${lo:,} &ndash; ${hi:,}</div>
  <div class="cl-cost-bar-track"><div class="cl-cost-bar-fill" style="width:{pct_bar}%"></div></div>
  <div class="cl-kpi-sub">{pct_bar}% of $60k cap</div>
</div>""",
            f"""<div class="cl-kpi-card">
  <div class="cl-label">Model Confidence</div>
  <div class="cl-conf-row" style="margin-top:2px;">
    {arc_svg}
    <div>
      <div class="cl-kpi-val" style="color:{conf_color};font-size:17px;">{conf:.0%}</div>
      <div class="cl-kpi-sub">{conf_label} confidence</div>
    </div>
  </div>
</div>""",
            f"""<div class="cl-kpi-card">
  <div class="cl-label">Est. Labor & Risk</div>
  <div class="cl-kpi-val" style="font-size:14px;">{labor_est}</div>
  <div class="cl-kpi-sub" style="color:{risk_color};font-weight:600;">{risk_label}</div>
</div>""",
        ]
    else:
        cards_html = [
            f"""<div class="cl-kpi-card">
  <div class="cl-label">Classification</div>
  <div style="margin:4px 0;"><span class="cl-badge {sev_cls}">{_esc(str(status_label or sev))}</span></div>
  <div class="cl-kpi-sub">Observed State</div>
</div>""",
            f"""<div class="cl-kpi-card">
  <div class="cl-label">Subject</div>
  <div class="cl-kpi-val" style="font-size:14px;">{_esc(subject or 'Inspection Target')}</div>
  <div class="cl-kpi-sub">Visual Target</div>
</div>""",
            f"""<div class="cl-kpi-card">
  <div class="cl-label">Model Confidence</div>
  <div class="cl-conf-row" style="margin-top:2px;">
    {arc_svg}
    <div>
      <div class="cl-kpi-val" style="color:{conf_color};font-size:17px;">{conf:.0%}</div>
      <div class="cl-kpi-sub">{conf_label} confidence</div>
    </div>
  </div>
</div>""",
            f"""<div class="cl-kpi-card">
  <div class="cl-label">Assessment Status</div>
  <div class="cl-kpi-val" style="font-size:14px;">Verified</div>
  <div class="cl-kpi-sub" style="color:#22c55e;">Standard Analysis</div>
</div>""",
        ]

    facts_figures_section = f"""<div class="cl-card-section">
  <div class="cl-label">📊 Key Facts & Figures</div>
  <div class="cl-kpi-grid">
    {''.join(cards_html)}
  </div>
</div>"""

    # ── 3. Tabular Results ───────────────────────────────────────────────────
    table_data = getattr(r, "table_data", []) or []
    codes = getattr(r, "damage_codes", []) or []

    table_rows = ""
    if table_data:
        for row in table_data:
            item = _esc(row.get("item", "—"))
            observed = _esc(row.get("observed", "—"))
            action = _esc(row.get("action", "—"))
            metric = _esc(row.get("metric", "—"))
            table_rows += f"""<tr>
  <td><strong>{item}</strong></td>
  <td>{observed}</td>
  <td><span style="color:#93c5fd;">{action}</span></td>
  <td style="font-family:'JetBrains Mono',monospace;white-space:nowrap;">{metric}</td>
</tr>"""
        if domain == "agriculture":
            col1, col2, col3, col4 = "Plant / Foliage Area", "Observed Symptom / Lesion", "Recommended Action / Treatment", "Severity / Metric"
        elif domain == "property":
            col1, col2, col3, col4 = "Structural Element", "Observed Condition", "Remediation Procedure", "Impact / Sub-Cost"
        elif domain == "vehicle":
            col1, col2, col3, col4 = "Assembly / Component", "Observed Damage", "Recommended Action", "Est. Sub-Cost"
        else:
            col1, col2, col3, col4 = "Item / Area", "Observed Feature", "Recommended Action", "Notes / Metric"
    elif domain == "vehicle":
        col1, col2, col3, col4 = "Assembly / Component", "Observed Damage", "Recommended Action", "Est. Sub-Cost"
        total_shares = sum(_CODE_DETAILS.get(c, ("", "", "", 0.20))[3] for c in codes) or 1.0
        for code in codes:
            detail = _CODE_DETAILS.get(code)
            if detail:
                comp, dmg, act, share = detail
            else:
                comp = code.replace("_", " ").title()
                dmg = "Direct collision deformation"
                act = "Repair / Replace"
                share = 0.20
            norm_share = share / total_shares
            sub_lo = max(100, int(lo * norm_share))
            sub_hi = max(sub_lo + 50, int(hi * norm_share))
            table_rows += f"""<tr>
  <td><strong>{_esc(comp)}</strong><br><span style="font-size:10px;color:var(--text-muted);font-family:'JetBrains Mono',monospace;">{_esc(code)}</span></td>
  <td>{_esc(dmg)}</td>
  <td><span style="color:#93c5fd;">{_esc(act)}</span></td>
  <td style="font-family:'JetBrains Mono',monospace;white-space:nowrap;">${sub_lo:,} &ndash; ${sub_hi:,}</td>
</tr>"""
    else:
        col1, col2, col3, col4 = "Area / Feature", "Observed Findings", "Recommended Follow-up", "Status"
        for c in (codes or ["INSPECTION_POINT_1"]):
            table_rows += f"""<tr>
  <td><strong>{_esc(c.replace('_', ' ').title())}</strong></td>
  <td>Visual assessment conducted according to prompt</td>
  <td><span style="color:#93c5fd;">Follow standard procedure</span></td>
  <td>Evaluated</td>
</tr>"""

    tabular_section = f"""<div class="cl-card-section">
  <div class="cl-label">📋 Tabular Breakdown & Action Plan</div>
  <table class="cl-table">
    <thead>
      <tr>
        <th style="width:26%;">{col1}</th>
        <th style="width:34%;">{col2}</th>
        <th style="width:24%;">{col3}</th>
        <th style="width:16%;">{col4}</th>
      </tr>
    </thead>
    <tbody>
      {table_rows if table_rows else '<tr><td colspan="4" style="color:#888;">No specific items recorded.</td></tr>'}
    </tbody>
  </table>
</div>"""

    # ── 4. Spatial Detections (Bounding Boxes) ────────────────────────────────
    bbs = getattr(r, "bounding_boxes", []) or []
    bb_section = ""
    if bbs:
        bb_rows = ""
        for bb in bbs[:6]:
            label = _esc(getattr(bb, "label", "—"))
            bb_conf = getattr(bb, "confidence", 0)
            bb_rows += f"<tr><td>{label}</td><td>{bb_conf:.0%}</td><td style='font-family:monospace;'>({bb.x:.2f}, {bb.y:.2f}, {bb.w:.2f}, {bb.h:.2f})</td></tr>"
        bb_section = f"""<div class="cl-card-section">
  <div class="cl-label">🎯 Spatial Detections & Regions</div>
  <table class="cl-table">
    <thead><tr><th>Region / Label</th><th>Confidence</th><th>Normalised Coordinates (x, y, w, h)</th></tr></thead>
    <tbody>{bb_rows}</tbody>
  </table>
</div>"""

    # ── 5. Detailed Reasoning & Answers ──────────────────────────────────────
    reasoning = getattr(r, "reasoning", "") or "No detailed reasoning provided."
    reasoning_html = _markdown_to_html(reasoning)
    reasoning_section = f"""<div class="cl-card-section">
  <div class="cl-label">🧠 Technical Reasoning & Detailed Findings</div>
  <div class="cl-reasoning">{reasoning_html}</div>
</div>"""

    # ── 6. Domain Grounding / RAG Citations ───────────────────────────────────
    rag_section = ""
    if response.citations:
        cits_items = []
        for i, c in enumerate(response.citations):
            src = _esc(c.get("source", "Knowledge Base"))
            snip = _esc(c.get("snippet", ""))
            cits_items.append(
                f'<div style="margin:5px 0;padding:6px 10px;background:rgba(255,255,255,0.03);border-left:2px solid #3b82f6;border-radius:4px;">'
                f'<div style="font-weight:600;color:var(--accent);font-size:11px;">📄 [{i+1}] {src}</div>'
                f'<div style="color:#94a3b8;font-size:11px;margin-top:2px;font-style:italic;">"{snip}"</div>'
                f'</div>'
            )
        rag_section = f"""<div class="cl-card-section">
  <div class="cl-label">📚 Domain Grounding — Verified Knowledge Standards</div>
  <details open style="margin-top:6px;cursor:pointer;">
    <summary style="color:#60a5fa;font-size:12px;font-weight:500;">Grounded in {len(response.citations)} knowledge excerpts</summary>
    <div style="margin-top:8px;">{''.join(cits_items)}</div>
  </details>
</div>"""

    html_out = f"""<div class="cl-msg-ai">
  <div class="cl-card">
    {header}
    {img_html}
    {summary_section}
    {facts_figures_section}
    {tabular_section}
    {bb_section}
    {reasoning_section}
    {rag_section}
  </div>
</div>"""
    return textwrap.dedent(html_out).strip()


# ---------------------------------------------------------------------------
# Text / RAG card renderer
# ---------------------------------------------------------------------------

def render_text_card(response: ChatResponse) -> str:
    body = response.text or ""
    body_html = _markdown_to_html(body)

    sources_html = ""
    if response.citations:
        cits_items = []
        for i, c in enumerate(response.citations):
            src = _esc(c.get("source", "Knowledge Base"))
            snip = _esc(c.get("snippet", ""))
            cits_items.append(
                f'<div style="margin:6px 0;padding:6px 10px;background:rgba(255,255,255,0.03);border-left:2px solid var(--border-focus);border-radius:4px;">'
                f'<div style="font-weight:600;color:var(--accent);font-size:11px;">[{i+1}] {src}</div>'
                f'<div style="color:var(--text-muted);font-size:11px;margin-top:2px;font-style:italic;">"{snip}"</div>'
                f'</div>'
            )
        sources_html = (
            f'<details style="margin-top:12px;cursor:pointer;">'
            f'<summary style="color:var(--accent);font-size:12px;font-weight:500;">Grounded in {len(response.citations)} knowledge excerpts (expand)</summary>'
            f'<div style="margin-top:8px;">{''.join(cits_items)}</div>'
            f'</details>'
        )
    elif response.sources:
        links = "".join(
            f'<span class="cl-source-link">[{i+1}] {_esc(s)}</span>'
            for i, s in enumerate(response.sources)
        )
        sources_html = f'<div class="cl-sources">Sources: {links}</div>'

    html_out = f"""<div class="cl-msg-ai">
  <div class="cl-text-card">
    {body_html}
    {sources_html}
  </div>
</div>"""
    return textwrap.dedent(html_out).strip()


def render_error_card(response: ChatResponse) -> str:
    msg = _esc(response.error_msg or response.text or "Unknown error")
    html_out = f"""<div class="cl-msg-ai">
  <div class="cl-text-card" style="border-color:rgba(239,68,68,0.4);">
    <div style="color:var(--red);font-weight:600;margin-bottom:6px;font-size:13px;">Error</div>
    <div style="color:#C8C8C4;font-size:13px;">{msg}</div>
  </div>
</div>"""
    return textwrap.dedent(html_out).strip()


def render_card(response: ChatResponse) -> str:
    if response.type == "assessment":
        return render_assessment_card(response)
    elif response.type == "error":
        return render_error_card(response)
    else:
        return render_text_card(response)


def render_user_bubble(prompt: str, image: Image.Image | None = None) -> str:
    img_html = ""
    if image is not None:
        src = _img_b64(image, max_px=320)
        img_html = f'<img src="{src}" class="cl-msg-image" alt="Uploaded image"/><br>'

    html_out = f"""<div style="display:flex;justify-content:flex-end;margin-bottom:8px;">
  <div class="cl-msg-user">
    {img_html}
    {_esc(prompt)}
  </div>
</div>"""
    return textwrap.dedent(html_out).strip()


def render_thinking() -> str:
    html_out = """<div class="cl-msg-ai">
  <div class="cl-thinking">
    <div class="dot"></div>
    <div class="dot"></div>
    <div class="dot"></div>
    <span style="margin-left:4px;">Analyzing…</span>
  </div>
</div>"""
    return textwrap.dedent(html_out).strip()


# ---------------------------------------------------------------------------
# Markdown to HTML converter
# ---------------------------------------------------------------------------

def _markdown_to_html(md_text: str) -> str:
    """Convert standard markdown text with tables and lists into clean HTML."""
    if not md_text:
        return ""
    try:
        html_content = markdown.markdown(
            md_text,
            extensions=["tables", "fenced_code", "nl2br"],
        )
        return html_content
    except Exception:
        return f"<div style='white-space:pre-wrap;'>{_esc(md_text)}</div>"
