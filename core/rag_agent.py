"""
core/rag_agent.py
RAG Knowledge Agent: FAISS + sentence-transformers + LiteLLM

On first run, indexes the data/knowledge_base/ directory and caches the FAISS
index to data/faiss_index/. Subsequent starts use the cache.

query() returns (answer, citations)
stream_query() returns a generator of (text_chunk, final_citations_or_None) tuples.
"""
from __future__ import annotations

import os
import re
import time
from collections.abc import Generator
from pathlib import Path
from typing import Any

from core.llm_router import _stub_response, chat, stream_chat
from core.schema import Citation, Persona

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_BASE_DIR = Path(__file__).parent.parent
_KB_DIR = _BASE_DIR / "data" / "knowledge_base"
_INDEX_DIR = _BASE_DIR / "data" / "faiss_index"
_CHUNK_SIZE = 400   # characters
_CHUNK_OVERLAP = 80
_TOP_K = 4

# ---------------------------------------------------------------------------
# Persona system prompts
# ---------------------------------------------------------------------------

_PERSONA_PROMPTS: dict[str, str] = {
    Persona.ADJUSTER: (
        "You are a vehicle damage assessment assistant helping a licensed claims adjuster. "
        "Be precise, technical, and reference specific taxonomy codes or policy clauses when relevant. "
        "Use bullet points for structured assessments. Keep responses under 300 words."
    ),
    Persona.UNDERWRITER: (
        "You are an insurance underwriting assistant. Focus on risk exposure, aggregate severity trends, "
        "policy terms, and reserving implications. Use tables or structured summaries. "
        "Avoid jargon where a plain-English explanation suffices."
    ),
    Persona.CUSTOMER_SERVICE: (
        "You are a customer-facing claims assistant. Use simple, empathetic language. "
        "Avoid technical jargon. Explain what will happen next in the claims process. "
        "Keep responses concise and reassuring — under 150 words."
    ),
    Persona.RESEARCHER: (
        "You are a research assistant for an ML team evaluating a vehicle damage VLM system. "
        "Be factual, cite sources precisely, and include relevant metadata (taxonomy version, "
        "benchmark references) when available."
    ),
}


# ---------------------------------------------------------------------------
# Text chunking
# ---------------------------------------------------------------------------

def _chunk_text(text: str, source: str) -> list[dict[str, str]]:
    """Split text into overlapping character chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + _CHUNK_SIZE
        chunk = text[start:end]
        chunks.append({"text": chunk.strip(), "source": source, "start": start})
        start += _CHUNK_SIZE - _CHUNK_OVERLAP
    return [c for c in chunks if len(c["text"]) > 50]


# ---------------------------------------------------------------------------
# Index management & Embeddings
# ---------------------------------------------------------------------------

_EMBED_MODEL = None


def _get_embed_model():
    """Singleton loader for sentence transformer embedding model."""
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
            _EMBED_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        except Exception as exc:
            print(f"[RAG] SentenceTransformer load warning: {exc}")
            return None
    return _EMBED_MODEL


def _load_or_build_index() -> tuple[Any, list[dict[str, str]]]:
    """
    Load FAISS index from cache or build from knowledge base docs.
    Returns (faiss_index, chunks_metadata).
    """
    import pickle

    chunks_path = _INDEX_DIR / "chunks.pkl"
    index_path = _INDEX_DIR / "index.faiss"

    # If KB dir is empty or doesn't exist, return None (stub mode)
    if not _KB_DIR.exists() or not list(_KB_DIR.glob("*.txt")):
        return None, []

    # Load from cache if available and fresh
    if chunks_path.exists() and index_path.exists():
        kb_mtime = max(p.stat().st_mtime for p in _KB_DIR.glob("*.txt"))
        idx_mtime = index_path.stat().st_mtime
        if idx_mtime > kb_mtime:
            try:
                import faiss
                with open(chunks_path, "rb") as f:
                    chunks = pickle.load(f)
                index = faiss.read_index(str(index_path))
                return index, chunks
            except Exception as e:
                print(f"[RAG] Warning reading cached index: {e}")

    # Build fresh chunks from disk
    all_chunks: list[dict[str, str]] = []
    for doc_path in sorted(_KB_DIR.glob("*.txt")):
        text = doc_path.read_text(encoding="utf-8", errors="ignore")
        all_chunks.extend(_chunk_text(text, doc_path.name))

    if not all_chunks:
        return None, []

    # Try building FAISS index
    try:
        model = _get_embed_model()
        if model is not None:
            import faiss
            import numpy as np

            texts = [c["text"] for c in all_chunks]
            embeddings = model.encode(texts, show_progress_bar=False, batch_size=32)
            embeddings = np.array(embeddings, dtype="float32")
            faiss.normalize_L2(embeddings)

            dim = embeddings.shape[1]
            index = faiss.IndexFlatIP(dim)
            index.add(embeddings)

            _INDEX_DIR.mkdir(parents=True, exist_ok=True)
            faiss.write_index(index, str(index_path))
            with open(chunks_path, "wb") as f:
                pickle.dump(all_chunks, f)

            print(f"[RAG] Index built: {len(all_chunks)} chunks from {len(list(_KB_DIR.glob('*.txt')))} docs")
            return index, all_chunks
    except Exception as exc:
        print(f"[RAG] Could not build FAISS index ({exc}). Using in-memory lexical fallback.")

    return None, all_chunks


def _lexical_retrieve(
    question: str,
    chunks: list[dict[str, str]],
    top_k: int = _TOP_K,
) -> list[Citation]:
    """Fast, deterministic keyword/lexical retrieval fallback."""
    q_words = set(re.findall(r"\w+", question.lower()))
    scored: list[tuple[float, dict[str, str]]] = []

    for c in chunks:
        c_text = c["text"].lower()
        c_src = c["source"].lower()
        score = 0.0
        for w in q_words:
            if len(w) <= 2:
                continue
            if w in c_text:
                count = c_text.count(w)
                score += 1.0 + min(count * 0.5, 3.0)
            if w in c_src:
                score += 2.0
        if score > 0:
            scored.append((score, c))

    scored.sort(key=lambda x: x[0], reverse=True)
    results = [s[1] for s in scored[:top_k]]

    if not results and chunks:
        results = chunks[:top_k]

    return [
        Citation(
            source=c["source"],
            snippet=c["text"][:280] + ("…" if len(c["text"]) > 280 else ""),
            page=None,
        )
        for c in results
    ]


def _retrieve(
    question: str,
    index: Any,
    chunks: list[dict[str, str]],
    top_k: int = _TOP_K,
) -> list[Citation]:
    """Retrieve top-k chunks: instant zero-lag lexical retrieval by default, or FAISS if model is warm."""
    if index is not None and _EMBED_MODEL is not None:
        try:
            import numpy as np
            q_emb = _EMBED_MODEL.encode([question], normalize_embeddings=True)
            q_emb = np.array(q_emb, dtype="float32")
            scores, idxs = index.search(q_emb, top_k)
            citations = []
            for idx in idxs[0]:
                if 0 <= idx < len(chunks):
                    chunk = chunks[idx]
                    citations.append(
                        Citation(
                            source=chunk["source"],
                            snippet=chunk["text"][:280] + ("…" if len(chunk["text"]) > 280 else ""),
                            page=None,
                        )
                    )
            if citations:
                return citations
        except Exception as e:
            print(f"[RAG] FAISS search error, falling back to lexical: {e}")

    return _lexical_retrieve(question, chunks, top_k)


def retrieve_context(question: str, top_k: int = _TOP_K) -> tuple[str, list[Citation]]:
    """Retrieve grounded knowledge base context and citations for any query."""
    index, chunks = _load_or_build_index()
    if not chunks:
        return "", []
    citations = _retrieve(question, index, chunks, top_k=top_k)
    context = "\n\n---\n\n".join(
        f"Source: {c.source}\n{c.snippet}" for c in citations
    )
    return context, citations


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def query(
    question: str,
    persona: str = Persona.ADJUSTER,
    llm_model: str = "stub",
    llm_provider: str = "stub",
    llm_api_key: str | None = None,
    ollama_base_url: str = "http://localhost:11434",
) -> tuple[str, list[Citation]]:
    """
    Query the RAG agent.

    Returns:
        (answer_text, citations)
    """
    index, chunks = _load_or_build_index()

    # Pure stub mode if no KB / no index
    if index is None or llm_provider == "stub" or llm_model == "stub":
        answer, source_names = _stub_response(question)
        citations = [
            Citation(
                source=src,
                snippet=f"[Stub citation for {src}. Real content loads when KB docs are present.]",
            )
            for src in source_names
        ]
        return answer, citations

    citations = _retrieve(question, index, chunks)
    context = "\n\n---\n\n".join(
        f"Source: {c.source}\n{c.snippet}" for c in citations
    )

    system_prompt = _PERSONA_PROMPTS.get(persona, _PERSONA_PROMPTS[Persona.ADJUSTER])
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                f"Use the following knowledge base excerpts to answer the question.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {question}\n\n"
                f"Answer based on the context. If the context doesn't cover the question, "
                f"say so clearly and answer from general insurance knowledge."
            ),
        },
    ]

    answer = chat(
        messages=messages,
        model=llm_model,
        provider=llm_provider,
        api_key=llm_api_key,
        temperature=0.3,
        stream=False,
        ollama_base_url=ollama_base_url,
    )
    return answer, citations


def stream_query(
    question: str,
    persona: str = Persona.ADJUSTER,
    llm_model: str = "stub",
    llm_provider: str = "stub",
    llm_api_key: str | None = None,
    ollama_base_url: str = "http://localhost:11434",
) -> Generator[tuple[str, list[Citation] | None], None, None]:
    """
    Streaming version of query().

    Yields:
        (text_chunk, None) during streaming
        ("", citations) as the final item
    """
    index, chunks = _load_or_build_index()

    if index is None or llm_provider == "stub" or llm_model == "stub":
        answer, source_names = _stub_response(question)
        citations = [
            Citation(source=src, snippet=f"[Stub citation for {src}.]")
            for src in source_names
        ]
        for word in answer.split(" "):
            yield word + " ", None
            time.sleep(0.018)
        yield "", citations
        return

    citations = _retrieve(question, index, chunks)
    context = "\n\n---\n\n".join(
        f"Source: {c.source}\n{c.snippet}" for c in citations
    )

    system_prompt = _PERSONA_PROMPTS.get(persona, _PERSONA_PROMPTS[Persona.ADJUSTER])
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                f"Context:\n{context}\n\nQuestion: {question}\n\n"
                f"Answer based on the context. Cite the source name when referencing it."
            ),
        },
    ]

    stream = stream_chat(
        messages=messages,
        model=llm_model,
        provider=llm_provider,
        api_key=llm_api_key,
        temperature=0.3,
        ollama_base_url=ollama_base_url,
    )

    for chunk in stream:
        yield chunk, None
    yield "", citations

