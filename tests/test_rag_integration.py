import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from PIL import Image
from core import chat_engine, rag_agent
from core.response_renderer import render_card

def test_rag_retrieval():
    print('[1] Testing rag_agent.retrieve_context...')
    ctx, citations = rag_agent.retrieve_context('total loss threshold and bumper crush', top_k=3)
    assert len(citations) > 0, 'Expected at least 1 citation'
    print(f'    PASS ({len(citations)} citations found)')

def test_chat_engine_rag_text():
    print('[2] Testing chat_engine with RAG prompt in stub mode...')
    resp = chat_engine.process_turn(
        prompt='What is the total loss formula in the policy?',
        image=None,
        history=[],
        settings={'llm_provider': 'stub', 'llm_model': 'stub', 'enable_rag': True, 'rag_persona': 'Adjuster'},
    )
    assert resp.type == 'text'
    assert len(resp.citations) > 0 or len(resp.sources) > 0
    html = render_card(resp)
    assert 'cl-text-card' in html
    print('    PASS (RAG text card rendered with citations)')

def test_chat_engine_rag_vlm():
    print('[3] Testing chat_engine VLM assessment with RAG grounding...')
    img = Image.new('RGB', (200, 200), color=(180, 50, 50))
    resp = chat_engine.process_turn(
        prompt='Assess this car and check policy guidelines.',
        image=img,
        history=[],
        settings={'vlm_backend': 'stub', 'enable_rag': True, 'rag_persona': 'Adjuster'},
    )
    assert resp.type == 'assessment'
    assert resp.result is not None
    html = render_card(resp)
    assert 'Knowledge Sources' in html or 'Domain Grounding' in html
    print('    PASS (Assessment card rendered with structured RAG grounding)')

if __name__ == '__main__':
    test_rag_retrieval()
    test_chat_engine_rag_text()
    test_chat_engine_rag_vlm()
    print('\n=================================================')
    print('  ALL RAG INTEGRATION TESTS PASSED!  ')
    print('=================================================')
