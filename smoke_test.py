"""Full smoke test — run via: python smoke_test.py"""
import sys
sys.path.insert(0, ".")

print("=== ClaimLens Full Smoke Test ===\n")

# --- Core schema ---
from core.schema import AssessmentResult, FeedbackRecord, DAMAGE_CODES, Citation, Severity
print("[OK] schema.py")

# --- Feedback store ---
from core.feedback_store import get_feedback_count, export_jsonl, get_acceptance_stats
count = get_feedback_count()
print(f"[OK] feedback_store.py  (rows: {count})")

# --- LLM router (stub) ---
from core.llm_router import chat, stream_chat, PROVIDER_REGISTRY
resp = chat([{"role": "user", "content": "Why is severity Severe?"}], model="stub", provider="stub")
assert len(resp) > 50, "Stub response too short"
print(f"[OK] llm_router.py      providers: {list(PROVIDER_REGISTRY.keys())}")

# --- Streaming stub ---
chunks = list(stream_chat([{"role": "user", "content": "similar past claims"}], model="stub", provider="stub"))
assert len(chunks) > 1
print(f"[OK] llm_router stream  ({len(chunks)} chunks)")

# --- VLM adapter (stub) ---
from PIL import Image
img = Image.new("RGB", (640, 480), color=(120, 80, 60))
from core.vlm_adapter import assess_damage, draw_bboxes
result = assess_damage(img, backend="stub")
assert result.severity in [s.value for s in Severity]
annotated = draw_bboxes(img, result)
assert annotated.size == (640, 480)
print(f"[OK] vlm_adapter.py     severity={result.severity}, conf={result.confidence:.0%}, bboxes={len(result.bounding_boxes)}")
print(f"[OK] draw_bboxes()      output={annotated.size}")

# --- AssessmentResult round-trip ---
d = result.to_dict()
r2 = AssessmentResult.from_dict(d)
assert r2.severity == result.severity
assert r2.confidence == result.confidence
print("[OK] AssessmentResult   round-trip serialisation")

# --- FeedbackRecord DPO export ---
fb = FeedbackRecord(
    image_hash="abc123",
    original_assessment=result,
    accepted=False,
    adjuster_damage_codes=["DOOR_FL_DENT"],
    adjuster_severity="Minor",
    adjuster_cost_low=500,
    adjuster_cost_high=900,
    override_reason="Test override",
    persona="Adjuster",
)
dpo = fb.to_dpo_pair()
assert "chosen" in dpo and "rejected" in dpo
print("[OK] FeedbackRecord     to_dpo_pair()")

# --- Eval data ---
import json
from pathlib import Path
data = json.loads(Path("data/eval_results.json").read_text())
acc = data["overall"]["accuracy"]
n_classes = len(data["per_class"])
print(f"[OK] eval_results.json  accuracy={acc:.1%}, damage_classes={n_classes}")

# --- Library versions ---
import sentence_transformers
import faiss
import openai
import google.generativeai as genai
import plotly, pandas, numpy, streamlit

print(f"[OK] sentence-transformers {sentence_transformers.__version__}")
test_idx = faiss.IndexFlatIP(128)
assert test_idx.ntotal == 0
print(f"[OK] faiss-cpu          IndexFlatIP created")
print(f"[OK] openai             {openai.__version__}")
print(f"[OK] google-generativeai loaded")
print(f"[OK] plotly             {plotly.__version__}")
print(f"[OK] pandas             {pandas.__version__}")
print(f"[OK] streamlit          {streamlit.__version__}")

print()
print("=" * 44)
print("  ALL CHECKS PASSED — ready to launch!")
print("=" * 44)
print()
print("Run:  streamlit run streamlit_app.py")
