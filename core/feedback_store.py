"""
core/feedback_store.py
SQLite-backed store for human-in-the-loop override data.
Used by the Assessment page to log adjuster decisions.
Used by the Metrics page to display live feedback counts and export JSONL.
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from core.schema import AssessmentResult, FeedbackRecord

_DB_PATH = Path(__file__).parent.parent / "data" / "feedback.db"
_FINE_TUNING_PATH = Path(__file__).parent.parent / "data" / "fine_tuning_feedback.jsonl"

# ---------------------------------------------------------------------------
# DB initialisation
# ---------------------------------------------------------------------------

_DDL = """
CREATE TABLE IF NOT EXISTS feedback (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp            TEXT NOT NULL,
    claim_id             TEXT DEFAULT '',
    image_hash           TEXT NOT NULL,
    persona              TEXT DEFAULT 'Adjuster',
    original_assessment  TEXT NOT NULL,       -- JSON
    accepted             INTEGER NOT NULL,     -- 1=accepted, 0=edited, -1=rejected
    adjuster_damage_codes TEXT DEFAULT '[]',  -- JSON list
    adjuster_severity    TEXT DEFAULT '',
    adjuster_cost_low    INTEGER DEFAULT 0,
    adjuster_cost_high   INTEGER DEFAULT 0,
    override_reason      TEXT DEFAULT ''
);
"""


@contextmanager
def _db():
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(_DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute(_DDL)
        conn.commit()
        yield conn
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

def log_feedback(record: FeedbackRecord) -> int:
    """Insert a feedback record and return the new row ID."""
    # accepted: 1=accepted as-is, 0=edited+submitted, -1=rejected
    accepted_int = 1 if record.accepted else (0 if record.adjuster_damage_codes else -1)

    with _db() as conn:
        cur = conn.execute(
            """
            INSERT INTO feedback (
                timestamp, claim_id, image_hash, persona, original_assessment,
                accepted, adjuster_damage_codes, adjuster_severity,
                adjuster_cost_low, adjuster_cost_high, override_reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.timestamp,
                record.claim_id,
                record.image_hash,
                record.persona,
                record.original_assessment.to_json(),
                accepted_int,
                json.dumps(record.adjuster_damage_codes),
                record.adjuster_severity,
                record.adjuster_cost_low,
                record.adjuster_cost_high,
                record.override_reason,
            ),
        )
        conn.commit()
        return cur.lastrowid


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

def get_all_feedback() -> pd.DataFrame:
    """Return all feedback rows as a DataFrame."""
    with _db() as conn:
        rows = conn.execute("SELECT * FROM feedback ORDER BY id DESC").fetchall()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame([dict(r) for r in rows])


def get_feedback_count() -> int:
    """Return total number of feedback entries."""
    with _db() as conn:
        return conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]


def get_acceptance_stats() -> dict[str, int]:
    """Return counts of accepted / edited / rejected."""
    with _db() as conn:
        rows = conn.execute(
            "SELECT accepted, COUNT(*) as cnt FROM feedback GROUP BY accepted"
        ).fetchall()
    mapping = {1: "accepted", 0: "edited", -1: "rejected"}
    result = {"accepted": 0, "edited": 0, "rejected": 0}
    for row in rows:
        key = mapping.get(row["accepted"], "other")
        result[key] = row["cnt"]
    return result


def get_override_distribution() -> pd.DataFrame:
    """
    Return a DataFrame showing which damage codes were most frequently
    corrected by adjusters (useful for retraining signal).
    """
    df = get_all_feedback()
    if df.empty or "adjuster_damage_codes" not in df.columns:
        return pd.DataFrame(columns=["damage_code", "count"])

    edited = df[df["accepted"] == 0].copy()
    if edited.empty:
        return pd.DataFrame(columns=["damage_code", "count"])

    from collections import Counter
    codes: list[str] = []
    for val in edited["adjuster_damage_codes"]:
        try:
            codes.extend(json.loads(val))
        except Exception:
            pass
    if not codes:
        return pd.DataFrame(columns=["damage_code", "count"])

    counter = Counter(codes)
    return pd.DataFrame(counter.most_common(), columns=["damage_code", "count"])


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_jsonl(output_path: str | Path | None = None) -> str:
    """
    Export all feedback records as DPO-aligned JSONL.
    Returns the JSONL string (also writes to output_path if provided).
    """
    df = get_all_feedback()
    if df.empty:
        return ""

    lines = []
    for _, row in df.iterrows():
        try:
            orig = AssessmentResult.from_dict(json.loads(row["original_assessment"]))
        except Exception:
            continue

        record = FeedbackRecord(
            image_hash=row["image_hash"],
            original_assessment=orig,
            accepted=row["accepted"] == 1,
            timestamp=row["timestamp"],
            adjuster_damage_codes=json.loads(row.get("adjuster_damage_codes", "[]")),
            adjuster_severity=row.get("adjuster_severity", ""),
            adjuster_cost_low=int(row.get("adjuster_cost_low", 0)),
            adjuster_cost_high=int(row.get("adjuster_cost_high", 0)),
            override_reason=row.get("override_reason", ""),
            persona=row.get("persona", "Adjuster"),
            claim_id=row.get("claim_id", ""),
        )
        lines.append(json.dumps(record.to_dpo_pair()))

    content = "\n".join(lines)
    if output_path:
        Path(output_path).write_text(content, encoding="utf-8")
    return content


def delete_record(row_id: int) -> None:
    """Delete a single feedback record by ID."""
    with _db() as conn:
        conn.execute("DELETE FROM feedback WHERE id = ?", (row_id,))
        conn.commit()


def clear_all_feedback() -> None:
    """Wipe all feedback records (use with caution)."""
    with _db() as conn:
        conn.execute("DELETE FROM feedback")
        conn.commit()


# ---------------------------------------------------------------------------
# Fine-Tuning & Persona Feedback Functions
# ---------------------------------------------------------------------------

_PERSONA_SCHEMAS: dict[str, dict[str, Any]] = {
    "Adjuster": {
        "title": "Adjuster Technical Review",
        "description": "Evaluate damage severity, part identification, repair methods, and labor realism.",
        "preferences": [
            "Accurate / Aligned",
            "Overestimated Scope or Labor",
            "Underestimated / Missed Damage",
            "Incorrect Damage Codes",
            "Needs Revision",
        ],
        "positive_preferences": {"Accurate / Aligned"},
        "tags": [
            "Accurate damage codes",
            "Fair labor / cost estimation",
            "Missed structural damage",
            "Incorrect panel / part identified",
            "OEM procedure unaligned",
            "PDR suitable candidate",
        ],
        "remarks_placeholder": "Add specific remarks, corrected damage codes, OEM repair steps, or adjusted labor hours for training...",
    },
    "Underwriter": {
        "title": "Underwriter Risk & Policy Review",
        "description": "Evaluate coverage applicability, deductible assignment, total loss threshold, and fraud risks.",
        "preferences": [
            "Acceptable Risk Profile",
            "Policy Clause Ambiguity",
            "Over-conservative / False Fraud Flag",
            "Missed Hazard / Loss Exposure",
            "Needs Revision",
        ],
        "positive_preferences": {"Acceptable Risk Profile"},
        "tags": [
            "Proper policy terms application",
            "Accurate deductible / coverage determination",
            "Total loss threshold miscalculated",
            "Policy clause ambiguity",
            "Fraud / anomaly risk missed",
            "Unwarranted fraud alert",
        ],
        "remarks_placeholder": "Add remarks on policy clause interpretation, coverage rationale, or fraud exposure notes...",
    },
    "Customer Service": {
        "title": "Customer Communication Review",
        "description": "Evaluate clarity, reassurance, policyholder empathy, and ease of understanding.",
        "preferences": [
            "Clear & Empathetic",
            "Too Technical / Jargon Heavy",
            "Lacks Reassurance / Cold Tone",
            "Confusing Next Steps",
            "Needs Revision",
        ],
        "positive_preferences": {"Clear & Empathetic"},
        "tags": [
            "Clear and empathetic language",
            "Actionable next steps",
            "Too technical for policyholder",
            "Unclear repair process explanation",
            "Missing reassurance / cold tone",
            "Helpful guidance",
        ],
        "remarks_placeholder": "Suggest how to rephrase or simplify this response for a clearer, more comforting customer message...",
    },
    "Researcher / Demo": {
        "title": "Benchmark & Research Review",
        "description": "Evaluate benchmark accuracy, citations, factual grounding, and methodology.",
        "preferences": [
            "Rigorous & Factual",
            "Citation / Grounding Inaccuracy",
            "Metric Inconsistency",
            "Hallucination / Fact Error",
            "Needs Revision",
        ],
        "positive_preferences": {"Rigorous & Factual"},
        "tags": [
            "Factual numbers verified",
            "Accurate knowledge citations",
            "Citation mismatch / ungrounded claim",
            "Metric calculation error",
            "Benchmark data discrepancy",
            "Methodology aligned",
        ],
        "remarks_placeholder": "Specify exact benchmark citations to correct, mathematical errors, or factual discrepancies...",
    },
}


def get_persona_preference_schema(persona: str) -> dict[str, Any]:
    """Return the feedback preferences, tags, and guidance for a given persona."""
    return _PERSONA_SCHEMAS.get(persona, _PERSONA_SCHEMAS["Adjuster"])


def get_fine_tuning_count() -> int:
    """Return total count of fine-tuning records stored in JSONL."""
    if not _FINE_TUNING_PATH.exists():
        return 0
    try:
        with open(_FINE_TUNING_PATH, "r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())
    except Exception:
        return 0


def get_fine_tuning_jsonl_bytes() -> bytes:
    """Return raw bytes of the fine-tuning JSONL file for downloading."""
    if not _FINE_TUNING_PATH.exists():
        return b""
    try:
        return _FINE_TUNING_PATH.read_bytes()
    except Exception:
        return b""


def log_chat_feedback(
    persona: str,
    user_prompt: str,
    response_text: str,
    preference: str,
    tags: list[str] | None = None,
    remarks: str = "",
    domain: str = "general",
    has_image: bool = False,
    models: dict | None = None,
) -> dict[str, Any]:
    """
    Log persona-specific feedback on an assistant turn.
    Saves in DPO / SFT training format to data/fine_tuning_feedback.jsonl.
    """
    _FINE_TUNING_PATH.parent.mkdir(parents=True, exist_ok=True)

    tags = tags or []
    models = models or {}
    schema = get_persona_preference_schema(persona)
    is_positive = (
        preference in schema.get("positive_preferences", set())
        or "Accurate" in preference
        or "Aligned" in preference
        or "Acceptable" in preference
        or "Clear" in preference
        or "Rigorous" in preference
    )

    quality_rating = 1.0 if is_positive else 0.0
    rec_id = f"ft_{uuid.uuid4().hex[:10]}"
    timestamp = datetime.now(timezone.utc).isoformat()

    # SFT / DPO data format:
    # If the user provided remarks correcting an imperfect response,
    # the remarks serve as the chosen/preferred answer and original response is rejected.
    # If it was accurate, response is chosen and rejected is None.
    if quality_rating == 1.0:
        chosen = response_text
        rejected = None
    else:
        chosen = remarks.strip() if remarks.strip() else response_text
        rejected = response_text

    record: dict[str, Any] = {
        "id": rec_id,
        "timestamp": timestamp,
        "persona": persona,
        "domain": domain,
        "user_prompt": user_prompt,
        "has_image": has_image,
        "response_text": response_text,
        "preference": preference,
        "tags": tags,
        "remarks": remarks.strip(),
        "models": models,
        "training_data": {
            "instruction": f"Role: {persona}. Query: {user_prompt}",
            "response": response_text,
            "chosen": chosen,
            "rejected": rejected,
            "critique": remarks.strip(),
            "quality_rating": quality_rating,
        },
    }

    with open(_FINE_TUNING_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return record
