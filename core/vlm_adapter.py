"""
core/vlm_adapter.py
VLM inference abstraction.

Backends:
  - stub   : deterministic fake result seeded from image hash (no key, instant)
  - gemini : Google Gemini 1.5 Flash Vision (free tier, ~2-3s latency)
  - rest   : POST to a local REST endpoint (for real LoRA model)

Set VLM_BACKEND env var or pass backend= explicitly.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import random
import re
import time
from typing import Any

import requests
from PIL import Image

from core.schema import (
    AssessmentResult,
    BoundingBox,
    DAMAGE_CODES,
    Severity,
    SEVERITY_COST_RANGES,
)

# ---------------------------------------------------------------------------
# Gemini structured-output prompt
# ---------------------------------------------------------------------------

_VLM_SYSTEM_PROMPT = """You are an advanced visual inspection and multimodal assessment AI specializing in factual analysis across multiple domains (Agriculture/Plants, Vehicles, Property/Structures, and General visual queries).

Analyze the uploaded image carefully. FIRST determine what is actually depicted in the image:
1. **Agriculture & Plants**: Crops, foliage, vegetables, fruit, plant diseases (e.g. blights, rust, powdery mildew, pests), nutrient deficiencies, soil, harvest stages.
2. **Vehicles & Transportation**: Cars, trucks, motorcycles, collision impact, scratches, dents, parts, structural integrity.
3. **Property & Infrastructure**: Houses, roofs, interiors, water leaks, fire, storm or structural damage.
4. **General / Other**: Any other scene, item, document, receipt, or object.

DO NOT hallucinate vehicle parts (such as bumpers, fenders, windshields) if the image is of plants, nature, buildings, or other non-vehicle subjects! Accurately identify and describe what is present.

Return ONLY a valid JSON object matching this schema:
{
  "domain": "agriculture" | "vehicle" | "property" | "general",
  "subject": "Precise name of the plant/crop/vehicle/object (e.g. 'Tomato Leaf (Solanum lycopersicum)', '2019 Honda Civic', 'Residential Asphalt Roof')",
  "status_label": "Direct diagnosis or condition (e.g. 'Early Blight (Alternaria solani)', 'Moderate Front Impact', 'Healthy Vegetation')",
  "severity": "Minor" | "Moderate" | "Severe" | "Total Loss" | "None",
  "confidence": 0.85,
  "summary": "2-3 sentence executive summary answering the user's prompt directly based on visual evidence.",
  "key_figures": [
    {"label": "Affected Foliage / Est. Cost / Severity Metric", "value": "Value with unit"},
    {"label": "Est. Yield Risk / Labor Hours / Impact", "value": "Value with unit"},
    {"label": "Est. Treatment Cost / Repair Cost", "value": "Value with unit"}
  ],
  "cost_range": [low_int, high_int],
  "table_data": [
    {
      "item": "Component, leaf area, or inspected part",
      "observed": "Observed symptom, damage, or visual feature",
      "action": "Recommended treatment, repair procedure, or next step",
      "metric": "Affected %, severity rating, or estimated sub-cost"
    }
  ],
  "damage_codes": ["SHORT_DESCRIPTIVE_CODE1", "SHORT_DESCRIPTIVE_CODE2"],
  "fraud_flag": false,
  "low_confidence_flag": false,
  "bounding_boxes": [
    {"label": "description of spotted symptom, lesion, or damaged region", "x": 0.1, "y": 0.2, "w": 0.3, "h": 0.2, "confidence": 0.9}
  ],
  "reasoning": "In-depth, factually correct answer addressing the user's prompt in detail with scientific or technical explanations, biological/physical mechanisms, exact figures, and actionable instructions."
}

Return ONLY the JSON. No markdown code fences, no extra text outside JSON."""


# ---------------------------------------------------------------------------
# Stub: deterministic demo results (Vehicles & Agriculture)
# ---------------------------------------------------------------------------

_PLANT_STUB_PRESET = AssessmentResult(
    damage_codes=["EARLY_BLIGHT_FOLIAR", "NECROTIC_TARGET_LESION", "CHLOROTIC_HALO"],
    severity=Severity.MODERATE,
    cost_range=(45, 120),
    confidence=0.92,
    fraud_flag=False,
    low_confidence_flag=False,
    domain="agriculture",
    subject="Tomato Foliage (Solanum lycopersicum)",
    status_label="Early Blight (Alternaria solani)",
    summary="The plant foliage exhibits characteristic Early Blight symptoms (Alternaria solani) with dark concentric target-spot lesions and yellow chlorotic halos on mid and lower leaves. Yield loss is estimated at 10–15% if untreated.",
    key_figures=[
        {"label": "Affected Foliage", "value": "~22%"},
        {"label": "Est. Yield Risk", "value": "10 – 15%"},
        {"label": "Est. Treatment Cost", "value": "$45 – $120"},
        {"label": "Pathogen Identified", "value": "Alternaria solani"},
    ],
    table_data=[
        {
            "item": "Lower Leaflets",
            "observed": "Concentric dark brown target lesions with yellow chlorotic halos",
            "action": "Sanitation pruning: remove and dispose of lower 3 leaflets",
            "metric": "High Symptom Density",
        },
        {
            "item": "Mid-Canopy Leaves",
            "observed": "Scattered circular necrotic lesions (3–6mm diameter)",
            "action": "Apply copper-based biofungicide or chlorothalonil spray",
            "metric": "Moderate Severity",
        },
        {
            "item": "Main Stem & Nodes",
            "observed": "Clean, no dark collar cankers observed",
            "action": "Maintain soil-level drip irrigation; keep leaves dry",
            "metric": "Healthy / Intact",
        },
    ],
    bounding_boxes=[
        BoundingBox("Early Blight Target Lesion", 0.15, 0.25, 0.35, 0.30, 0.94),
        BoundingBox("Chlorotic Halo", 0.55, 0.40, 0.30, 0.25, 0.88),
    ],
    reasoning=(
        "Factual Botanical & Agronomic Findings:\n\n"
        "1. **Diagnosis**: Visual evidence indicates *Alternaria solani* (Early Blight), a fungal pathogen common in Solanaceous crops. The distinctive 'target board' concentric rings in the lesions are diagnostic.\n"
        "2. **Progression**: Infection is currently confined to mid-to-lower canopy leaves (approx. 22% total surface area). Stems and fruit clusters remain unaffected.\n"
        "3. **Treatment & Management Protocol**:\n"
        "   - **Cultural Control**: Prune all foliage within 12 inches of soil level to reduce soil-splash reinoculation.\n"
        "   - **Fungicidal Application**: Apply a fixed copper fungicide (e.g. Copper Hydroxide @ 1.5–2 lbs/acre) or Daconil every 7–10 days until dry conditions return.\n"
        "   - **Irrigation**: Cease overhead sprinkling; switch exclusively to ground drip lines to eliminate leaf wetness duration."
    ),
    backend_used="stub",
)

_STUB_PRESETS = [
    AssessmentResult(
        damage_codes=["FRONT_BUMPER_CRUSH", "HOOD_CREASE", "WINDSHIELD_CRACK"],
        severity=Severity.MODERATE,
        cost_range=(3_500, 6_200),
        confidence=0.87,
        domain="vehicle",
        subject="Vehicle Front-End Collision",
        status_label="Front Bumper Crush & Hood Crease",
        summary="Moderate front-end collision damage detected with bumper crush and secondary hood deformation. Frame and mechanical structures remain intact.",
        key_figures=[
            {"label": "Severity Rating", "value": "Moderate"},
            {"label": "Est. Repair Cost", "value": "$3,500 – $6,200"},
            {"label": "Est. Labor Hours", "value": "16 – 22 Hours"},
            {"label": "Structural Risk", "value": "Low / Cosmetic & Bolted"},
        ],
        table_data=[
            {"item": "Front Bumper Fascia", "observed": "Crush deformation with structural foam collapse", "action": "Replace Assembly & Recalibrate Sensors", "metric": "$1,200 – $1,800"},
            {"item": "Hood Panel", "observed": "Secondary crease along leading edge", "action": "Straighten, fill, and refinish", "metric": "$650 – $950"},
            {"item": "Windshield Glass", "observed": "Star-crack pattern from flying debris", "action": "Replace windshield & calibrate ADAS camera", "metric": "$550 – $850"},
        ],
        fraud_flag=False,
        low_confidence_flag=False,
        bounding_boxes=[
            BoundingBox("Front Bumper Crush", 0.05, 0.55, 0.40, 0.30, 0.91),
            BoundingBox("Hood Crease", 0.15, 0.30, 0.55, 0.25, 0.84),
            BoundingBox("Windshield Crack", 0.20, 0.10, 0.60, 0.20, 0.79),
        ],
        reasoning=(
            "The vehicle shows a moderate front-end collision. The front bumper exhibits "
            "significant crush damage consistent with a low-speed impact (~20–30 mph). "
            "The hood has a crease suggesting secondary contact with a fixed object. "
            "The windshield has a star-crack pattern from a flying debris event. "
            "No structural/frame damage detected. Fraud indicators absent."
        ),
        backend_used="stub",
    ),
    AssessmentResult(
        damage_codes=["REAR_BUMPER_CRUSH", "TRUNK_LID_DENT", "REAR_WINDOW_CRACK"],
        severity=Severity.MODERATE,
        cost_range=(2_800, 5_100),
        confidence=0.82,
        fraud_flag=False,
        low_confidence_flag=False,
        bounding_boxes=[
            BoundingBox("Rear Bumper Crush", 0.10, 0.60, 0.80, 0.30, 0.88),
            BoundingBox("Trunk Lid Dent", 0.20, 0.35, 0.60, 0.25, 0.76),
        ],
        reasoning=(
            "Rear-end collision damage. The rear bumper shows compression consistent with "
            "a stationary vehicle impact. The trunk lid has a single-panel dent. "
            "Rear window crack is a stress fracture from frame flex, not direct impact. "
            "Repair scope: bumper replacement, trunk lid PDR, rear glass replacement."
        ),
        backend_used="stub",
    ),
    AssessmentResult(
        damage_codes=["DOOR_FL_DENT", "DOOR_RL_DENT", "SIDE_MIRROR_DAMAGE", "FENDER_FL_DENT"],
        severity=Severity.MODERATE,
        cost_range=(4_200, 7_800),
        confidence=0.79,
        fraud_flag=False,
        low_confidence_flag=False,
        bounding_boxes=[
            BoundingBox("Front-Left Door Dent", 0.05, 0.40, 0.25, 0.35, 0.83),
            BoundingBox("Rear-Left Door Dent", 0.28, 0.40, 0.25, 0.35, 0.80),
            BoundingBox("Side Mirror Damage", 0.03, 0.35, 0.10, 0.15, 0.91),
        ],
        reasoning=(
            "Side-swipe collision on the driver's side. Both left-side doors show denting "
            "consistent with a longitudinal scraping force. The side mirror housing is fractured. "
            "The front-left fender has a crease near the door junction. "
            "No window glass damage. Repair estimate includes PDR on both doors, "
            "mirror assembly replacement, and fender straightening."
        ),
        backend_used="stub",
    ),
    AssessmentResult(
        damage_codes=["FRONT_BUMPER_SCRATCH", "PAINT_TRANSFER"],
        severity=Severity.MINOR,
        cost_range=(350, 900),
        confidence=0.93,
        fraud_flag=False,
        low_confidence_flag=False,
        bounding_boxes=[
            BoundingBox("Scratch + Paint Transfer", 0.10, 0.60, 0.40, 0.20, 0.94),
        ],
        reasoning=(
            "Minor cosmetic damage only. Paint transfer and surface scratching on the front bumper "
            "fascia, consistent with a low-speed parking lot incident. No structural, mechanical, "
            "or glass damage. Repair: wet sanding + paint blend or bumper respray."
        ),
        backend_used="stub",
    ),
    AssessmentResult(
        damage_codes=[
            "STRUCTURAL_DAMAGE", "AIRBAG_DEPLOYED", "HOOD_CRUSH",
            "FRONT_BUMPER_CRUSH", "WINDSHIELD_SHATTER",
        ],
        severity=Severity.TOTAL_LOSS,
        cost_range=(28_000, 45_000),
        confidence=0.96,
        fraud_flag=False,
        low_confidence_flag=False,
        bounding_boxes=[
            BoundingBox("Hood Crush", 0.05, 0.05, 0.90, 0.50, 0.97),
            BoundingBox("Windshield Shatter", 0.10, 0.05, 0.80, 0.25, 0.95),
        ],
        reasoning=(
            "Severe high-speed frontal collision. The engine bay shows significant intrusion. "
            "Hood crush exceeds 40% of hood length. Both front airbags deployed. "
            "Windshield is fully shattered. Frame rail deflection detected in the lower engine bay area. "
            "Estimated repair cost exceeds 80% of ACV — total loss declaration recommended."
        ),
        backend_used="stub",
    ),
]


def _stub_assess(image: Image.Image, latency: float = 0.9, user_prompt: str | None = None) -> AssessmentResult:
    """Return a deterministic preset based on image content and domain heuristics."""
    import copy

    p_lower = (user_prompt or "").lower()
    is_plant = any(w in p_lower for w in ["plant", "leaf", "leaves", "crop", "tree", "flower", "blight", "fungus", "disease", "agriculture", "botanical", "weed", "soil", "vegetation"])
    if not is_plant and image is not None:
        try:
            colors = list(image.resize((32, 32)).convert("RGB").getdata())
            avg_r = sum(c[0] for c in colors) / len(colors)
            avg_g = sum(c[1] for c in colors) / len(colors)
            avg_b = sum(c[2] for c in colors) / len(colors)
            if avg_g > avg_r * 1.12 and avg_g > avg_b * 1.12:
                is_plant = True
        except Exception:
            pass

    if is_plant:
        result = copy.deepcopy(_PLANT_STUB_PRESET)
    else:
        img_hash = hashlib.md5(image.tobytes()).hexdigest()
        idx = int(img_hash[:4], 16) % len(_STUB_PRESETS)
        result = copy.deepcopy(_STUB_PRESETS[idx])

    time.sleep(latency)
    result.latency_ms = latency * 1000

    if user_prompt and user_prompt.strip():
        u_p = user_prompt.strip()
        default_prompts = ("assess this vehicle for damage.", "assess this vehicle", "assess", "assess this image")
        if u_p.lower() not in default_prompts:
            subj = result.subject or ("the plant" if is_plant else "the vehicle")
            result.reasoning = (
                f"**Regarding your query (\"{u_p}\"):**\n\n"
                f"Visual inspection of {subj} indicates **{result.status_label or result.severity}**.\n\n"
                + result.reasoning
            )
    return result


# ---------------------------------------------------------------------------
# Gemini Vision backend
# ---------------------------------------------------------------------------

def _image_to_base64(image: Image.Image, max_size: tuple[int, int] = (1024, 1024)) -> str:
    """Resize and encode image as base64 JPEG for API transmission."""
    image = image.copy()
    image.thumbnail(max_size, Image.LANCZOS)
    if image.mode != "RGB":
        image = image.convert("RGB")
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _is_vision_compatible(name: str) -> bool:
    """Filter out text-to-speech, audio, and embedding models."""
    n = name.lower()
    incompatible = ["tts", "embedding", "aqa", "imagen", "whisper", "audio", "realtime"]
    return not any(inc in n for inc in incompatible)


def _resolve_gemini_models(api_key: str, preferred: str | None = None) -> list[str]:
    """Return an ordered list of vision-capable Gemini models to try, prioritizing stable high-quota free tier models."""
    import google.generativeai as genai
    genai.configure(api_key=api_key)

    # Primary active free-tier vision models
    stable_defaults = [
        "gemini-3.6-flash",
        "gemini-3.7-flash",
        "gemini-3.5-flash",
        "gemini-3.5-flash-lite",
        "gemini-flash-latest",
        "gemini-flash-lite-latest",
        "gemini-2.5-flash",
        "gemini-2.0-flash",
    ]

    candidates = [preferred] if preferred else []
    candidates.extend(stable_defaults)

    dedup: list[str] = []
    for c in candidates:
        if c and _is_vision_compatible(c) and c not in dedup:
            dedup.append(c)

    try:
        models = list(genai.list_models())
        supported = [
            m.name.replace("models/", "")
            for m in models
            if "generateContent" in getattr(m, "supported_generation_methods", [])
            and _is_vision_compatible(m.name)
        ]
        if supported:
            active_list = [c for c in dedup if c in supported]
            for s in supported:
                if s not in active_list:
                    active_list.append(s)
            if active_list:
                return active_list
    except Exception as e:
        print(f"[Gemini] list_models notice: {e}")

    return dedup


def _safe_parse_json(raw: str) -> dict[str, Any]:
    """Robustly extract and parse JSON from LLM output, handling fences, unescaped text, and truncation."""
    text = raw.strip()

    # 1. Look for markdown code fence ```json ... ``` or ``` ... ```
    fence_match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
    if fence_match:
        text = fence_match.group(1).strip()
    else:
        # Find outermost braces
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start : end + 1]
        elif start != -1:
            text = text[start:]

    # 2. Direct JSON load attempt
    try:
        return json.loads(text)
    except Exception:
        pass

    # 3. Clean trailing commas (e.g. [1, 2,] or {"a": 1,})
    cleaned = re.sub(r",\s*([}\]])", r"\1", text)
    try:
        return json.loads(cleaned)
    except Exception:
        pass

    # 4. Repair truncated JSON (unterminated quotes, open brackets, unclosed braces)
    repaired = cleaned
    quotes = len(re.findall(r'(?<!\\)"', repaired))
    if quotes % 2 != 0:
        repaired += '"'
    open_b = repaired.count("[") - repaired.count("]")
    if open_b > 0:
        repaired += "]" * open_b
    open_c = repaired.count("{") - repaired.count("}")
    if open_c > 0:
        repaired += "}" * open_c
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)

    try:
        return json.loads(repaired)
    except Exception:
        pass

    # 5. Regex extraction fallback if JSON structure is heavily damaged
    data: dict[str, Any] = {}
    m_codes = re.search(r'"damage_codes"\s*:\s*\[(.*?)\]', raw, re.DOTALL)
    if m_codes:
        data["damage_codes"] = re.findall(r'"([A-Za-z0-9_]+)"', m_codes.group(1))

    m_sev = re.search(r'"severity"\s*:\s*"([^"]+)"', raw)
    if m_sev:
        data["severity"] = m_sev.group(1).strip()

    m_cost = re.search(r'"cost_range"\s*:\s*\[\s*(\d+)\s*,\s*(\d+)\s*\]', raw)
    if m_cost:
        data["cost_range"] = [int(m_cost.group(1)), int(m_cost.group(2))]

    m_conf = re.search(r'"confidence"\s*:\s*([0-9.]+)', raw)
    if m_conf:
        data["confidence"] = float(m_conf.group(1))

    m_fraud = re.search(r'"fraud_flag"\s*:\s*(true|false)', raw, re.IGNORECASE)
    if m_fraud:
        data["fraud_flag"] = m_fraud.group(1).lower() == "true"

    m_reason = re.search(r'"reasoning"\s*:\s*"([^"]*)', raw)
    if m_reason:
        data["reasoning"] = m_reason.group(1).strip()

    if data:
        return data

    # 6. Absolute safe default
    return {
        "damage_codes": ["UNKNOWN"],
        "severity": "Moderate",
        "cost_range": [1500, 5000],
        "confidence": 0.5,
        "reasoning": raw[:300] if raw else "Assessment completed.",
    }


def _gemini_assess(
    image: Image.Image,
    api_key: str,
    model_name: str | None = None,
    user_prompt: str | None = None,
) -> AssessmentResult:
    """Call Gemini Vision with auto-detection and fallback."""
    try:
        import google.generativeai as genai
    except ImportError as exc:
        raise ImportError(
            "google-generativeai not installed. Run: pip install google-generativeai"
        ) from exc

    genai.configure(api_key=api_key)
    models_to_try = _resolve_gemini_models(api_key, model_name)

    # Resize image before sending
    img_copy = image.copy()
    img_copy.thumbnail((1024, 1024), Image.LANCZOS)
    if img_copy.mode != "RGB":
        img_copy = img_copy.convert("RGB")

    last_exc = None
    response = None
    used_model = models_to_try[0]

    for m_name in models_to_try:
        try:
            model = genai.GenerativeModel(m_name)
            t0 = time.time()
            try:
                gen_cfg = genai.types.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=4096,
                    response_mime_type="application/json",
                )
            except Exception:
                gen_cfg = genai.types.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=4096,
                )

            prompt_parts = [_VLM_SYSTEM_PROMPT]
            if user_prompt and user_prompt.strip():
                prompt_parts.append(
                    f"\n\nUSER PROMPT / SPECIFIC QUERY:\n{user_prompt.strip()}\n"
                    "Make sure your 'reasoning' specifically addresses and answers this query with facts and figures, "
                    "while strictly maintaining the required JSON schema."
                )
            prompt_parts.append(img_copy)

            response = model.generate_content(
                prompt_parts,
                generation_config=gen_cfg,
            )
            latency_ms = (time.time() - t0) * 1000
            used_model = m_name
            break
        except Exception as exc:
            err_str = str(exc).lower()
            last_exc = exc
            is_recoverable = any(k in err_str for k in [
                "not found", "404", "unsupported", "not supported",
                "modality", "image input", "invalidargument", "400",
                "429", "quota", "resource_exhausted", "rate limit",
                "too many requests", "exceeded your current quota",
            ])
            if is_recoverable:
                err_clean = re.sub(r'(?:AQ\.[a-zA-Z0-9_-]{15,}|AIzaSy[a-zA-Z0-9_-]{20,}|sk-[a-zA-Z0-9_-]{20,})', '[REDACTED]', str(exc))
                print(f"[Gemini] Model '{m_name}' unavailable or quota exceeded ({err_clean}), falling back to next model...")
                continue
            raise exc

    if response is None:
        err_clean = re.sub(r'(?:AQ\.[a-zA-Z0-9_-]{15,}|AIzaSy[a-zA-Z0-9_-]{20,}|sk-[a-zA-Z0-9_-]{20,})', '[REDACTED]', str(last_exc))
        raise RuntimeError(f"All Gemini models failed: {err_clean}") from last_exc

    raw = getattr(response, "text", "") or ""
    data = _safe_parse_json(raw)
    return _build_assessment_result(data, f"gemini ({used_model})", latency_ms)


def _build_assessment_result(
    data: dict[str, Any],
    backend_used: str,
    latency_ms: float,
) -> AssessmentResult:
    bbs: list[BoundingBox] = []
    for b in data.get("bounding_boxes", []):
        try:
            if isinstance(b, dict):
                label = str(b.get("label", "damage"))
                conf = float(b.get("confidence", 0.8))
                x = float(b.get("x", 0.0))
                y = float(b.get("y", 0.0))
                w = float(b.get("w", b.get("width", 0.1)))
                h = float(b.get("h", b.get("height", 0.1)))
                bbs.append(BoundingBox(label=label, confidence=conf, x=x, y=y, w=w, h=h))
        except Exception:
            continue

    # Domain detection
    domain = str(data.get("domain", "")).lower().strip()
    full_text = f"{data.get('subject', '')} {data.get('status_label', '')} {data.get('reasoning', '')} {data.get('summary', '')}".lower()
    if not domain:
        if any(w in full_text for w in ["plant", "leaf", "leaves", "crop", "tree", "foliage", "blight", "fungus", "botanical", "vegetation", "chlorosis", "mildew"]):
            domain = "agriculture"
        elif any(w in full_text for w in ["roof", "building", "house", "drywall", "pipe", "structural", "shingle", "plumbing"]):
            domain = "property"
        elif any(w in full_text for w in ["vehicle", "car", "bumper", "hood", "fender", "windshield", "airbag", "collision", "automotive"]):
            domain = "vehicle"
        else:
            domain = "general"

    # Safe cost range
    raw_cost = data.get("cost_range")
    if isinstance(raw_cost, (list, tuple)) and len(raw_cost) >= 2:
        cost_range = (int(raw_cost[0]), int(raw_cost[1]))
    elif isinstance(raw_cost, (list, tuple)) and len(raw_cost) == 1:
        cost_range = (int(raw_cost[0]), int(raw_cost[0]) * 2)
    elif domain == "vehicle":
        cost_range = (1500, 5000)
    elif domain == "agriculture":
        cost_range = (50, 200)
    else:
        cost_range = (0, 0)

    # Safe severity
    sev_str = str(data.get("severity", "Moderate")).strip().title()
    valid_severities = {s.value.title(): s.value for s in Severity}
    severity = valid_severities.get(sev_str, Severity.MODERATE if domain in ("vehicle", "agriculture") else "Normal")

    subject = str(data.get("subject", "")).strip()
    if not subject:
        if domain == "agriculture":
            subject = "Botanical / Foliage Specimen"
        elif domain == "vehicle":
            subject = "Vehicle Inspection"
        elif domain == "property":
            subject = "Property / Structure"
        else:
            subject = "Visual Assessment"

    status_label = str(data.get("status_label", "")).strip() or f"{severity} Condition"
    summary = str(data.get("summary", "")).strip()
    key_figures = list(data.get("key_figures", []))
    table_data = list(data.get("table_data", []))

    return AssessmentResult(
        damage_codes=data.get("damage_codes", []) or [f"{domain.upper()}_INSPECTED"],
        severity=severity,
        cost_range=cost_range,
        confidence=float(data.get("confidence", 0.85)),
        fraud_flag=bool(data.get("fraud_flag", False)),
        low_confidence_flag=bool(data.get("low_confidence_flag", False)),
        bounding_boxes=bbs,
        reasoning=str(data.get("reasoning", "")),
        backend_used=backend_used,
        latency_ms=latency_ms,
        domain=domain,
        subject=subject,
        status_label=status_label,
        summary=summary,
        key_figures=key_figures,
        table_data=table_data,
    )


def _openai_compat_vision_assess(
    image: Image.Image,
    api_key: str,
    model: str,
    base_url: str | None = None,
    extra_headers: dict | None = None,
    user_prompt: str | None = None,
    backend_label: str = "openai",
) -> AssessmentResult:
    """Call an OpenAI-compatible vision endpoint (OpenRouter, OpenAI, etc.)."""
    from openai import OpenAI

    t0 = time.time()
    b64 = _image_to_base64(image)

    prompt = _VLM_SYSTEM_PROMPT
    if user_prompt and user_prompt.strip():
        prompt += (
            f"\n\nUSER PROMPT / SPECIFIC QUERY:\n{user_prompt.strip()}\n"
            "Make sure your 'reasoning' specifically addresses and answers this query with facts and figures, "
            "while strictly maintaining the required JSON schema."
        )

    kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    if extra_headers:
        kwargs["default_headers"] = extra_headers

    client = OpenAI(**kwargs)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                },
            ],
        }
    ]

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.1,
        max_tokens=4096,
    )
    latency_ms = (time.time() - t0) * 1000
    raw = resp.choices[0].message.content or ""
    data = _safe_parse_json(raw)
    return _build_assessment_result(data, f"{backend_label} ({model})", latency_ms)


def _ollama_vision_assess(
    image: Image.Image,
    model: str = "llava",
    base_url: str = "http://localhost:11434",
    user_prompt: str | None = None,
) -> AssessmentResult:
    """Call Ollama local vision model (llava, llama3.2-vision, minicpm-v, etc.)."""
    import requests

    t0 = time.time()
    b64 = _image_to_base64(image)

    prompt = _VLM_SYSTEM_PROMPT
    if user_prompt and user_prompt.strip():
        prompt += (
            f"\n\nUSER PROMPT / SPECIFIC QUERY:\n{user_prompt.strip()}\n"
            "Make sure your 'reasoning' specifically addresses and answers this query with facts and figures, "
            "while strictly maintaining the required JSON schema."
        )

    url = base_url.rstrip("/") + "/api/chat"
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [b64],
            }
        ],
        "stream": False,
        "options": {"temperature": 0.1},
    }

    r = requests.post(url, json=payload, timeout=120)
    r.raise_for_status()
    latency_ms = (time.time() - t0) * 1000
    data_resp = r.json()
    raw = data_resp.get("message", {}).get("content", "")
    data = _safe_parse_json(raw)
    return _build_assessment_result(data, f"ollama ({model})", latency_ms)


# ---------------------------------------------------------------------------
# REST backend (for real LoRA model)
# ---------------------------------------------------------------------------

def _rest_assess(image: Image.Image, endpoint_url: str) -> AssessmentResult:
    """POST image to a local REST endpoint returning AssessmentResult JSON."""
    b64 = _image_to_base64(image)
    t0 = time.time()
    resp = requests.post(
        endpoint_url,
        json={"image_b64": b64},
        timeout=30,
    )
    resp.raise_for_status()
    latency_ms = (time.time() - t0) * 1000
    data = resp.json()
    result = AssessmentResult.from_dict(data)
    result.latency_ms = latency_ms
    result.backend_used = f"rest:{endpoint_url}"
    return result


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def assess_damage(
    image: Image.Image,
    backend: str | None = None,
    gemini_api_key: str | None = None,
    gemini_model: str | None = None,
    openrouter_api_key: str | None = None,
    openrouter_model: str | None = None,
    openai_api_key: str | None = None,
    openai_model: str | None = None,
    ollama_base_url: str | None = None,
    ollama_model: str | None = None,
    rest_endpoint: str | None = None,
    user_prompt: str | None = None,
) -> AssessmentResult:
    """
    Run vehicle damage assessment on a PIL image across multiple providers:
    gemini | openrouter | ollama | openai | rest | stub
    """
    backend = (backend or os.getenv("VLM_BACKEND", "stub")).lower()

    if backend == "stub":
        return _stub_assess(image, user_prompt=user_prompt)

    elif backend == "gemini":
        key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            raise ValueError(
                "Gemini API key required. Pass gemini_api_key= or set GEMINI_API_KEY env var."
            )
        return _gemini_assess(image, key, gemini_model, user_prompt=user_prompt)

    elif backend == "openrouter":
        key = openrouter_api_key or os.getenv("OPENROUTER_API_KEY")
        if not key:
            raise ValueError("OpenRouter API key required for OpenRouter Vision. Get a free key at openrouter.ai")
        m = openrouter_model or "meta-llama/llama-3.2-11b-vision-instruct:free"
        return _openai_compat_vision_assess(
            image=image,
            api_key=key,
            model=m,
            base_url="https://openrouter.ai/api/v1",
            extra_headers={"HTTP-Referer": "https://claimlens.local", "X-Title": "ClaimLens"},
            user_prompt=user_prompt,
            backend_label="openrouter",
        )

    elif backend == "openai":
        key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("OpenAI API key required for OpenAI Vision.")
        m = openai_model or "gpt-4o-mini"
        return _openai_compat_vision_assess(
            image=image,
            api_key=key,
            model=m,
            user_prompt=user_prompt,
            backend_label="openai",
        )

    elif backend == "ollama":
        url = ollama_base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        m = ollama_model or "llava"
        return _ollama_vision_assess(
            image=image,
            model=m,
            base_url=url,
            user_prompt=user_prompt,
        )

    elif backend == "rest":
        url = rest_endpoint or os.getenv("VLM_REST_ENDPOINT", "http://localhost:8000/assess")
        return _rest_assess(image, url)

    else:
        raise ValueError(f"Unknown VLM backend: {backend!r}. Use 'stub', 'gemini', 'openrouter', 'ollama', 'openai', or 'rest'.")


def draw_bboxes(image: Image.Image, result: AssessmentResult) -> Image.Image:
    """
    Draw bounding boxes on a copy of the image.
    Boxes use normalised coordinates (0–1).
    Returns annotated PIL image.
    """
    try:
        from PIL import ImageDraw, ImageFont

        img = image.copy().convert("RGBA")
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        W, H = img.size
        colors = [
            (255, 82, 82, 180),    # red
            (255, 165, 0, 180),    # orange
            (255, 220, 0, 180),    # yellow
            (0, 200, 83, 180),     # green
            (100, 181, 246, 180),  # blue
        ]

        for i, bb in enumerate(result.bounding_boxes):
            bx = bb.x / W if bb.x > 1.0 else bb.x
            by = bb.y / H if bb.y > 1.0 else bb.y
            bw = bb.w / W if bb.w > 1.0 else bb.w
            bh = bb.h / H if bb.h > 1.0 else bb.h

            x1 = max(0, min(W, int(bx * W)))
            y1 = max(0, min(H, int(by * H)))
            x2 = max(0, min(W, int((bx + bw) * W)))
            y2 = max(0, min(H, int((by + bh) * H)))
            color = colors[i % len(colors)]

            # Semi-transparent fill
            draw.rectangle([x1, y1, x2, y2], fill=(*color[:3], 40), outline=color[:3], width=3)

            # Label background
            conf = getattr(bb, "confidence", 0.8)
            label = f"{bb.label} ({conf:.0%})"
            font_size = max(12, W // 60)
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except Exception:
                font = ImageFont.load_default()

            ty = max(0, y1 - font_size - 4)
            bbox_text = draw.textbbox((x1, ty), label, font=font)
            draw.rectangle(
                [bbox_text[0] - 2, bbox_text[1] - 2, bbox_text[2] + 2, bbox_text[3] + 2],
                fill=(*color[:3], 200),
            )
            draw.text((x1, ty), label, fill=(255, 255, 255, 255), font=font)

        result_img = Image.alpha_composite(img, overlay)
        return result_img.convert("RGB")
    except Exception as e:
        print(f"[draw_bboxes warning] {e}")
        return image.convert("RGB")
