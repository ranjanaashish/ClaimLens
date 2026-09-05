"""Quick integration test for chat UI components."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# 1. Compile checks
import py_compile
files = ['streamlit_app.py', 'core/chat_engine.py', 'core/response_renderer.py', 'pages/chat.py']
for f in files:
    py_compile.compile(f, doraise=True)
    print('[compile OK]', f)

# 2. Renderer — assessment card
from core.schema import AssessmentResult, Severity, BoundingBox
from core.response_renderer import ChatResponse, render_card, render_user_bubble, render_thinking

result = AssessmentResult(
    damage_codes=['FRONT_BUMPER_CRUSH', 'HOOD_CREASE'],
    severity=Severity.MODERATE,
    cost_range=(3500, 6200),
    confidence=0.87,
    fraud_flag=False,
    low_confidence_flag=False,
    bounding_boxes=[BoundingBox('Front Bumper', 0.1, 0.1, 0.4, 0.3, 0.91)],
    reasoning='Moderate frontal impact detected.',
    backend_used='stub',
    latency_ms=42.0,
)
resp = ChatResponse(type='assessment', result=result, text='Moderate damage.')
html = render_card(resp)
assert 'cl-card' in html, "Missing cl-card class"
assert 'FRONT_BUMPER_CRUSH' in html, "Missing damage code"
assert '3,500' in html, "Missing cost"
print(f'[render OK] assessment card ({len(html)} chars)')

# 3. Renderer — text card
resp2 = ChatResponse(
    type='text',
    text='# Overview\n\nThe **FRONT_BUMPER_CRUSH** code means significant crush damage.\n\n| Code | Description |\n|---|---|\n| FBC | Front bumper crush |',
    sources=['Policy-2024.pdf'],
)
html2 = render_card(resp2)
assert 'cl-text-card' in html2
print(f'[render OK] text card ({len(html2)} chars)')

# 4. User bubble
ub = render_user_bubble('Assess this car')
assert 'cl-msg-user' in ub
print('[render OK] user bubble')

# 5. Thinking
th = render_thinking()
assert 'cl-thinking' in th
print('[render OK] thinking indicator')

# 6. chat_engine — text turn (no image)
from core.chat_engine import process_turn
settings = {
    'vlm_backend': 'stub', 'gemini_api_key': '', 'vlm_rest_endpoint': '',
    'llm_provider': 'stub', 'llm_model': 'stub', 'llm_api_key': '', 'ollama_base_url': ''
}
r = process_turn('What is FRONT_BUMPER_CRUSH?', None, [], settings)
assert r.type in ('text', 'assessment', 'error'), f"Unexpected type: {r.type}"
print(f'[engine OK] text turn -> type={r.type}, text[:60]={r.text[:60]!r}')

# 7. chat_engine — assessment turn WITH image
from PIL import Image
img = Image.new('RGB', (200, 150), color=(180, 60, 40))
r2 = process_turn('Assess this vehicle', img, [], settings)
assert r2.type == 'assessment', f"Expected assessment, got {r2.type}"
assert r2.result is not None
print(f'[engine OK] assessment turn -> severity={r2.result.severity}')

# 8. Rendered card from engine result is valid HTML
html3 = render_card(r2)
assert 'cl-card' in html3
print(f'[render OK] engine result rendered ({len(html3)} chars)')

print()
print('=' * 48)
print('  ALL CHAT UI INTEGRATION TESTS PASSED')
print('=' * 48)
