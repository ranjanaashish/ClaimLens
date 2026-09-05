"""
tests/test_feedback_integration.py
Tests for persona-based feedback store and fine-tuning dataset generation.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import feedback_store
from core.schema import AssessmentResult, Persona, Severity


def test_fine_tuning_feedback_logging():
    initial_count = feedback_store.get_fine_tuning_count()

    # 1. Test Adjuster feedback
    rec1 = feedback_store.log_chat_feedback(
        persona=Persona.ADJUSTER.value,
        user_prompt="Assess front bumper collision",
        response_text="Front bumper crushed, repair cost $1,200 - $2,500.",
        preference="Accurate / Aligned",
        tags=["Accurate damage codes", "Fair labor / cost estimation"],
        remarks="Codes match OEM guide exactly.",
        domain="vehicle",
        has_image=True,
    )
    assert rec1["persona"] == "Adjuster"
    assert rec1["preference"] == "Accurate / Aligned"
    assert rec1["training_data"]["quality_rating"] == 1.0

    # 2. Test Underwriter critique feedback
    rec2 = feedback_store.log_chat_feedback(
        persona=Persona.UNDERWRITER.value,
        user_prompt="Explain coverage applicability for rear collision",
        response_text="Standard collision deductible applies.",
        preference="Needs Improvement",
        tags=["Policy clause ambiguity"],
        remarks="Should specify comprehensive vs collision clause 4.2 under policy terms.",
        domain="vehicle",
        has_image=False,
    )
    assert rec2["persona"] == "Underwriter"
    assert rec2["preference"] == "Needs Improvement"
    assert rec2["training_data"]["quality_rating"] == 0.0
    assert rec2["training_data"]["rejected"] is not None

    # 3. Check count increment
    new_count = feedback_store.get_fine_tuning_count()
    assert new_count == initial_count + 2

    # 4. Check JSONL export
    data_bytes = feedback_store.get_fine_tuning_jsonl_bytes()
    assert len(data_bytes) > 0
    lines = [json.loads(line) for line in data_bytes.decode("utf-8").splitlines() if line.strip()]
    assert any(line["id"] == rec1["id"] for line in lines)
    assert any(line["id"] == rec2["id"] for line in lines)

    print(f"[feedback_store test OK] Logged 2 records successfully. Total count: {new_count}")


if __name__ == "__main__":
    test_fine_tuning_feedback_logging()
    print("\nALL FEEDBACK INTEGRATION TESTS PASSED")
