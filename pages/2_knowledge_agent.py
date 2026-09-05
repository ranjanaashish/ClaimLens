"""
pages/2_knowledge_agent.py — RAG Knowledge Agent chat page.

Features:
- Multi-provider LLM selector in sidebar (Gemini, OpenRouter, Ollama, stub)
- API key input per provider (password field, never stored to disk)
- Persona selector (Adjuster / Underwriter / CS / Researcher)
- Suggested prompt chips
- Streaming chat with per-message citation expanders
- Clear chat button
"""
import os
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

import importlib
from core import llm_router, rag_agent
importlib.reload(llm_router)
importlib.reload(rag_agent)
from core.llm_router import PROVIDER_REGISTRY
from core.schema import Citation, Persona

# ---------------------------------------------------------------------------
# Sidebar — LLM configuration
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### LLM Provider")

    provider_key = st.selectbox(
        "Provider",
        options=list(PROVIDER_REGISTRY.keys()),
        index=list(PROVIDER_REGISTRY.keys()).index(st.session_state.llm_provider),
        format_func=lambda k: PROVIDER_REGISTRY[k]["label"],
        label_visibility="collapsed",
    )
    st.session_state.llm_provider = provider_key

    provider_info = PROVIDER_REGISTRY[provider_key]

    if provider_info["free"]:
        st.caption("Free tier available")
    else:
        st.caption("Paid API")

    if provider_info["notes"]:
        st.info(provider_info["notes"], icon="ℹ️")

    # Model selector
    models = provider_info["models"]
    current_model = st.session_state.llm_model
    if current_model not in models:
        current_model = models[0]
    selected_model = st.selectbox(
        "Model",
        options=models,
        index=models.index(current_model),
        format_func=lambda m: m.split("/")[-1] if "/" in m else m,
    )
    st.session_state.llm_model = selected_model

    # API key input
    if provider_info["key_label"]:
        env_key = provider_info.get("key_env", "")
        default_key = st.session_state.llm_api_key or os.getenv(env_key or "", "")
        api_key = st.text_input(
            provider_info["key_label"],
            value=default_key,
            type="password",
            placeholder="Paste your API key…",
        )
        st.session_state.llm_api_key = api_key
    else:
        st.session_state.llm_api_key = ""

    # Ollama URL
    if provider_key == "ollama":
        ollama_url = st.text_input(
            "Ollama base URL",
            value=st.session_state.ollama_base_url,
            placeholder="http://localhost:11434",
        )
        st.session_state.ollama_base_url = ollama_url

    st.divider()

    # Persona selector
    st.markdown("### Persona")
    persona = st.selectbox(
        "Adjust response style",
        options=[p.value for p in Persona],
        index=[p.value for p in Persona].index(st.session_state.persona),
        label_visibility="collapsed",
    )
    st.session_state.persona = persona
    persona_desc = {
        "Adjuster": "Technical precision, references taxonomy codes",
        "Underwriter": "Risk focus, tables and summaries",
        "Customer Service": "Plain language, empathetic, concise",
        "Researcher / Demo": "Factual, metadata-rich, cites sources",
    }
    st.caption(persona_desc.get(persona, ""))

    st.divider()
    if st.button("Clear chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------

st.title("Knowledge Agent")
st.caption(
    "Ask questions about policy rules, damage taxonomy, past claim precedents, and more. "
    "Answers are grounded in the knowledge base with citations."
)

# ---------------------------------------------------------------------------
# Suggested prompt chips
# ---------------------------------------------------------------------------

SUGGESTED_PROMPTS = [
    "Why was this claim rated Severe?",
    "Show similar past claims",
    "Policy on pre-existing damage",
    "What does Total Loss mean?",
    "How are fraud flags triggered?",
    "What is the cost breakdown for Moderate severity?",
]

st.markdown("**Quick questions:**")
chip_cols = st.columns(3)
chip_trigger: str | None = None
for i, prompt in enumerate(SUGGESTED_PROMPTS):
    if chip_cols[i % 3].button(prompt, key=f"chip_{i}", use_container_width=True):
        chip_trigger = prompt

# ---------------------------------------------------------------------------
# Chat history display
# ---------------------------------------------------------------------------

chat_container = st.container()
with chat_container:
    if not st.session_state.chat_history:
        with st.chat_message("assistant"):
            st.markdown(
                f"Hello! I'm your ClaimLens knowledge assistant. "
                f"I'm currently set to **{persona}** mode.\n\n"
                f"Ask me about policy rules, damage taxonomy definitions, "
                f"similar past claims, or anything else related to this claim. "
                f"I'll cite my sources so you can verify."
            )

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("citations"):
                citations: list[dict] = msg["citations"]
                with st.expander(f"Sources ({len(citations)})", expanded=False):
                    for cit in citations:
                        st.markdown(
                            f"**{cit['source']}**"
                            + (f" · page {cit['page']}" if cit.get("page") else "")
                        )
                        st.caption(cit["snippet"])
                        st.divider()

# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------

user_input = st.chat_input("Ask anything about policy, taxonomy, or past claims…")

# Chip click pre-fills the input
if chip_trigger and not user_input:
    user_input = chip_trigger

if user_input:
    # Append user message
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    with chat_container:
        with st.chat_message("user"):
            st.markdown(user_input)

        # Stream assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            final_citations: list[Citation] = []

            try:
                for chunk, citations in rag_agent.stream_query(
                    question=user_input,
                    persona=st.session_state.persona,
                    llm_model=st.session_state.llm_model,
                    llm_provider=st.session_state.llm_provider,
                    llm_api_key=st.session_state.llm_api_key or None,
                    ollama_base_url=st.session_state.ollama_base_url,
                ):
                    if citations is not None:
                        final_citations = citations
                    else:
                        full_response += chunk
                        message_placeholder.markdown(full_response + "▌")

                message_placeholder.markdown(full_response)

                # Show citations inline
                if final_citations:
                    with st.expander(f"Sources ({len(final_citations)})", expanded=False):
                        for cit in final_citations:
                            st.markdown(
                                f"**{cit.source}**"
                                + (f" · page {cit.page}" if cit.page else "")
                            )
                            st.caption(cit.snippet)
                            st.divider()

            except Exception as exc:
                error_msg = str(exc)
                if "api_key" in error_msg.lower() or "authentication" in error_msg.lower():
                    message_placeholder.error(
                        "**API key error.** Please check the key you entered in the sidebar. "
                        "Switch to **Stub** mode for a keyless demo."
                    )
                    full_response = f"[Error: {error_msg}]"
                else:
                    message_placeholder.error(f"**Error:** {error_msg}")
                    full_response = f"[Error: {error_msg}]"
                final_citations = []

    # Save to history
    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": full_response,
            "citations": [c.to_dict() for c in final_citations],
        }
    )
    st.rerun()
