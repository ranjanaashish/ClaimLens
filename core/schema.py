"""
core/schema.py
Shared dataclasses and enums used across all pages and modules.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Severity(str, Enum):
    MINOR = "Minor"
    MODERATE = "Moderate"
    SEVERE = "Severe"
    TOTAL_LOSS = "Total Loss"


class VLMBackend(str, Enum):
    STUB = "stub"
    GEMINI = "gemini"
    REST = "rest"


class LLMProvider(str, Enum):
    STUB = "stub"
    GEMINI = "gemini/gemini-2.5-flash"
    GEMINI_2_0 = "gemini/gemini-2.0-flash"
    GEMINI_PRO = "gemini/gemini-1.5-pro"
    OPENROUTER_LLAMA = "openrouter/meta-llama/llama-3.1-8b-instruct:free"
    OPENROUTER_QWEN = "openrouter/qwen/qwen-2.5-72b-instruct:free"
    OPENROUTER_MISTRAL = "openrouter/mistralai/mistral-7b-instruct:free"
    OLLAMA_MISTRAL = "ollama/mistral"
    OLLAMA_LLAMA3 = "ollama/llama3"
    OPENAI_GPT4O_MINI = "openai/gpt-4o-mini"
    ANTHROPIC_HAIKU = "anthropic/claude-haiku-20240307"


class Persona(str, Enum):
    ADJUSTER = "Adjuster"
    UNDERWRITER = "Underwriter"
    CUSTOMER_SERVICE = "Customer Service"
    RESEARCHER = "Researcher / Demo"


# ---------------------------------------------------------------------------
# Damage taxonomy
# ---------------------------------------------------------------------------

DAMAGE_CODES: dict[str, str] = {
    "FRONT_BUMPER_CRUSH": "Front Bumper – Crush",
    "FRONT_BUMPER_SCRATCH": "Front Bumper – Scratch/Scuff",
    "REAR_BUMPER_CRUSH": "Rear Bumper – Crush",
    "REAR_BUMPER_SCRATCH": "Rear Bumper – Scratch/Scuff",
    "HOOD_CREASE": "Hood – Crease/Dent",
    "HOOD_CRUSH": "Hood – Major Crush",
    "TRUNK_LID_DENT": "Trunk Lid – Dent",
    "TRUNK_LID_CRUSH": "Trunk Lid – Major Crush",
    "DOOR_FL_DENT": "Front-Left Door – Dent",
    "DOOR_FR_DENT": "Front-Right Door – Dent",
    "DOOR_RL_DENT": "Rear-Left Door – Dent",
    "DOOR_RR_DENT": "Rear-Right Door – Dent",
    "FENDER_FL_DENT": "Front-Left Fender – Dent",
    "FENDER_FR_DENT": "Front-Right Fender – Dent",
    "WINDSHIELD_CRACK": "Windshield – Crack",
    "WINDSHIELD_SHATTER": "Windshield – Shatter",
    "REAR_WINDOW_CRACK": "Rear Window – Crack",
    "SIDE_MIRROR_DAMAGE": "Side Mirror – Damage",
    "WHEEL_RIM_DAMAGE": "Wheel/Rim – Damage",
    "UNDERCARRIAGE_DAMAGE": "Undercarriage – Damage",
    "PAINT_TRANSFER": "Paint Transfer",
    "STRUCTURAL_DAMAGE": "Structural/Frame Damage",
    "AIRBAG_DEPLOYED": "Airbag Deployed",
    "FLOOD_DAMAGE": "Flood/Water Damage",
}

SEVERITY_COLORS: dict[str, str] = {
    Severity.MINOR: "#22c55e",       # green
    Severity.MODERATE: "#f59e0b",    # amber
    Severity.SEVERE: "#ef4444",      # red
    Severity.TOTAL_LOSS: "#7c3aed",  # purple
}

SEVERITY_COST_RANGES: dict[str, tuple[int, int]] = {
    Severity.MINOR: (200, 1_500),
    Severity.MODERATE: (1_500, 8_000),
    Severity.SEVERE: (8_000, 25_000),
    Severity.TOTAL_LOSS: (20_000, 60_000),
}


# ---------------------------------------------------------------------------
# Core result objects
# ---------------------------------------------------------------------------

@dataclass
class BoundingBox:
    label: str
    x: float   # normalised 0–1
    y: float
    w: float
    h: float
    confidence: float = 1.0


@dataclass
class Citation:
    source: str          # filename
    snippet: str         # ~300-char excerpt
    page: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AssessmentResult:
    damage_codes: list[str]
    severity: str
    cost_range: tuple[int, int]
    confidence: float
    fraud_flag: bool
    low_confidence_flag: bool
    bounding_boxes: list[BoundingBox] = field(default_factory=list)
    reasoning: str = ""
    backend_used: str = "stub"
    latency_ms: float = 0.0
    domain: str = "vehicle"                      # "agriculture", "vehicle", "property", "general"
    subject: str = ""                            # e.g. "Tomato Foliage (Early Blight)", "2020 Honda Civic"
    status_label: str = ""                       # e.g. "Early Blight (Alternaria solani)", "Front-End Collision"
    summary: str = ""                            # Factual summary
    key_figures: list[dict[str, str]] = field(default_factory=list)   # [{"label": "...", "value": "..."}]
    table_data: list[dict[str, str]] = field(default_factory=list)    # [{"item": "...", "observed": "...", "action": "...", "metric": "..."}]

    def to_dict(self) -> dict:
        d = asdict(self)
        # tuples aren't JSON-serialisable as-is
        d["cost_range"] = list(self.cost_range)
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)

    @classmethod
    def from_dict(cls, d: dict) -> "AssessmentResult":
        d = dict(d)
        bbs = [BoundingBox(**b) if isinstance(b, dict) else b for b in d.pop("bounding_boxes", [])]
        cr = d.get("cost_range", [0, 0])
        d["cost_range"] = (cr[0], cr[1]) if isinstance(cr, (list, tuple)) and len(cr) >= 2 else (0, 0)
        d["bounding_boxes"] = bbs
        # Accept only known fields for robustness
        from dataclasses import fields as dc_fields
        valid_keys = {f.name for f in dc_fields(cls)}
        clean_d = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**clean_d)


# ---------------------------------------------------------------------------
# Feedback / override record
# ---------------------------------------------------------------------------

@dataclass
class FeedbackRecord:
    image_hash: str
    original_assessment: AssessmentResult
    accepted: bool
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    adjuster_damage_codes: list[str] = field(default_factory=list)
    adjuster_severity: str = ""
    adjuster_cost_low: int = 0
    adjuster_cost_high: int = 0
    override_reason: str = ""
    persona: str = Persona.ADJUSTER
    claim_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_hash": self.image_hash,
            "original_assessment": self.original_assessment.to_dict(),
            "accepted": self.accepted,
            "timestamp": self.timestamp,
            "adjuster_damage_codes": self.adjuster_damage_codes,
            "adjuster_severity": self.adjuster_severity,
            "adjuster_cost_low": self.adjuster_cost_low,
            "adjuster_cost_high": self.adjuster_cost_high,
            "override_reason": self.override_reason,
            "persona": self.persona,
            "claim_id": self.claim_id,
        }

    def to_dpo_pair(self) -> dict[str, Any]:
        """Export as a DPO preference pair for fine-tuning (trl / axolotl format)."""
        prompt = (
            f"Assess the vehicle damage shown in the image. "
            f"Return a JSON with fields: damage_codes, severity, cost_range, "
            f"confidence, fraud_flag, reasoning."
        )
        chosen = {
            "damage_codes": self.adjuster_damage_codes or self.original_assessment.damage_codes,
            "severity": self.adjuster_severity or self.original_assessment.severity,
            "cost_range": [self.adjuster_cost_low, self.adjuster_cost_high]
            if self.adjuster_cost_low
            else list(self.original_assessment.cost_range),
            "rationale": self.override_reason,
        }
        rejected = self.original_assessment.to_dict() if not self.accepted else None
        return {
            "prompt": prompt,
            "chosen": json.dumps(chosen),
            "rejected": json.dumps(rejected) if rejected else None,
            "metadata": {
                "image_hash": self.image_hash,
                "persona": self.persona,
                "timestamp": self.timestamp,
            },
        }
