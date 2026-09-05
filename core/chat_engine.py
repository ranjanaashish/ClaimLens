"""
core/chat_engine.py
===================
Routes a user prompt + optional image to the right backend
and returns a ChatResponse.

Routing logic:
  image attached → VLM assessment (Gemini Vision / stub / REST)
  no image + Q about damage/policy/claims → RAG knowledge agent
  no image + general → LLM free-form answer
"""
from __future__ import annotations

import re
import time
from typing import Any

from PIL import Image

from core.response_renderer import ChatResponse


# ---------------------------------------------------------------------------
# Keyword sets for routing
# ---------------------------------------------------------------------------

_DAMAGE_KEYWORDS = {
    "assess", "damage", "repair", "cost", "severity", "dent", "scratch",
    "bumper", "hood", "crash", "collision", "accident", "vehicle", "car",
    "claim", "claim code", "estimate", "insurance", "total loss", "fraud",
}

_POLICY_KEYWORDS = {
    "policy", "coverage", "deductible", "premium", "exclusion", "clause",
    "liability", "comprehensive", "collision coverage", "underwrite",
    "adjuster", "settlement", "subrogation", "indemnity",
}


def _has_keywords(text: str, kw_set: set[str]) -> bool:
    text_lower = text.lower()
    return any(kw in text_lower for kw in kw_set)


# ---------------------------------------------------------------------------
# VLM assessment turn
# ---------------------------------------------------------------------------

def _run_vlm(
    prompt: str,
    image: Image.Image,
    settings: dict[str, Any],
) -> ChatResponse:
    from core import vlm_adapter
    from core.schema import DAMAGE_CODES

    backend = settings.get("vlm_backend", "stub")
    gemini_key = settings.get("gemini_api_key") or None
    gemini_model = settings.get("gemini_vlm_model") or settings.get("gemini_model") or None
    openrouter_key = settings.get("openrouter_vlm_key") or settings.get("llm_api_key") or None
    openrouter_model = settings.get("openrouter_vlm_model") or "meta-llama/llama-3.2-11b-vision-instruct:free"
    openai_key = settings.get("openai_vlm_key") or settings.get("llm_api_key") or None
    openai_model = settings.get("openai_vlm_model") or "gpt-4o-mini"
    ollama_url = settings.get("ollama_base_url", "http://localhost:11434")
    ollama_model = settings.get("ollama_vlm_model") or "llava"
    rest_ep = settings.get("vlm_rest_endpoint") or None

    try:
        result = vlm_adapter.assess_damage(
            image=image,
            backend=backend,
            gemini_api_key=gemini_key,
            gemini_model=gemini_model,
            openrouter_api_key=openrouter_key,
            openrouter_model=openrouter_model,
            openai_api_key=openai_key,
            openai_model=openai_model,
            ollama_base_url=ollama_url,
            ollama_model=ollama_model,
            rest_endpoint=rest_ep,
            user_prompt=prompt,
        )
    except Exception as exc:
        err_msg = str(exc)
        if "429" in err_msg or "quota" in err_msg.lower():
            hint = (
                "Google Gemini free-tier quota reached for this model. "
                "In ⚙ Settings, try switching the Vision Model to **gemini-2.0-flash** or **gemini-1.5-flash-8b**, "
                "or switch backend to **Stub**."
            )
        else:
            hint = "Try switching to Stub mode or checking your API key."
        return ChatResponse(
            type="error",
            error_msg=f"Assessment failed: {exc}. {hint}",
        )

    # Annotate image with bounding boxes
    try:
        annotated = vlm_adapter.draw_bboxes(image, result)
    except Exception:
        annotated = image

    # Build a domain-adaptive summary text
    domain = getattr(result, "domain", "vehicle") or "vehicle"
    cost = result.cost_range
    lo, hi = (int(cost[0]), int(cost[1])) if len(cost) >= 2 else (1500, 5000)

    if result.summary:
        summary = result.summary
    elif domain == "agriculture":
        summary = (
            f"Visual assessment of **{result.subject or 'plant/crop'}** identified "
            f"**{result.status_label or result.severity}**. "
            f"Estimated treatment cost: **${lo:,}–${hi:,}**."
        )
    elif domain == "vehicle":
        codes = result.damage_codes or []
        code_desc = ", ".join(DAMAGE_CODES.get(c, c) for c in codes[:4])
        summary = (
            f"**{result.severity}** damage detected. "
            f"{'Fraud indicators present. ' if result.fraud_flag else ''}"
            f"{'Low confidence — manual review recommended. ' if result.low_confidence_flag else ''}"
            f"Estimated repair: **${lo:,}–${hi:,}**. "
            f"Damage regions: {code_desc or 'None identified'}."
        )
    else:
        summary = f"Visual assessment of **{result.subject or 'specimen'}**: **{result.status_label or result.severity}**."

    # Ground VLM findings with RAG knowledge base ONLY when domain is vehicle
    rag_sources: list[str] = []
    rag_citations: list[dict] = []
    if settings.get("enable_rag", True) and domain == "vehicle":
        try:
            from core.rag_agent import retrieve_context
            damage_str = " ".join(result.damage_codes or [])
            rag_query = f"{damage_str} {result.severity} {prompt}"
            _, cits = retrieve_context(rag_query, top_k=3)
            rag_sources = [c.source for c in cits]
            rag_citations = [c.to_dict() for c in cits]
        except Exception as e:
            print(f"[RAG] VLM grounding error: {e}")

    return ChatResponse(
        type="assessment",
        text=summary,
        result=result,
        annotated_image=annotated,
        sources=rag_sources,
        citations=rag_citations,
    )


# ---------------------------------------------------------------------------
# RAG / LLM text turn
# ---------------------------------------------------------------------------

def _run_llm(
    prompt: str,
    history: list[dict],
    settings: dict[str, Any],
) -> ChatResponse:
    provider = settings.get("llm_provider", "stub")
    model = settings.get("llm_model", "stub")
    api_key = settings.get("llm_api_key") or None
    ollama_url = settings.get("ollama_base_url", "http://localhost:11434")
    enable_rag = settings.get("enable_rag", True)
    persona = settings.get("rag_persona", "Adjuster")

    sources: list[str] = []
    citations_data: list[dict] = []
    context = ""

    # 1. Retrieve domain knowledge from RAG index
    if enable_rag:
        try:
            from core.rag_agent import retrieve_context
            context, cits = retrieve_context(prompt, top_k=4)
            sources = [c.source for c in cits]
            citations_data = [c.to_dict() for c in cits]
        except Exception as exc:
            print(f"[RAG] Context retrieval error: {exc}")

    # In stub mode, use canned response with citations
    if provider == "stub" or model == "stub":
        try:
            from core.rag_agent import query
            answer_rag, stub_cits = query(
                question=prompt,
                persona=persona,
                llm_provider="stub",
                llm_model="stub",
            )
            return ChatResponse(
                type="text",
                text=answer_rag,
                sources=[c.source for c in stub_cits],
                citations=[c.to_dict() for c in stub_cits],
            )
        except Exception:
            pass

    # 2. Build domain-aware structured system prompt
    system = (
        f"You are ClaimLens, an advanced multimodal assessment and analytical AI operating in {persona} mode.\n"
        "You provide factual, precise, structured answers across multiple domains:\n"
        "- Agriculture, crops, botanical health, plant diseases, pathogens & treatments\n"
        "- Vehicles, collision damage, mechanical inspection, repair estimates & insurance claims\n"
        "- Property, structural assessment & insurance policies\n\n"
        "STRUCTURE YOUR RESPONSE WITH THE FOLLOWING SECTIONS:\n"
        "1. 📌 **Executive Summary**: 2–3 clear sentences answering the prompt directly.\n"
        "2. 📊 **Key Facts & Figures**: Concrete numbers, statistics, percentages, costs, or metric ranges.\n"
        "3. 📋 **Tabular Breakdown**: A clean markdown table organizing relevant items, categories, symptoms, actions, or comparisons.\n"
        "4. 🧠 **Detailed Factual Analysis**: In-depth explanation with technical/scientific mechanisms and actionable recommendations.\n\n"
        "Be factual, avoid generic fluff, and cite specific standards or taxonomy when relevant."
    )
    if context and _has_keywords(prompt, _DAMAGE_KEYWORDS | _POLICY_KEYWORDS):
        system += (
            "\n\n--- VERIFIED INSURANCE KNOWLEDGE BASE EXCERPTS ---\n"
            f"{context}\n"
            "--- END KNOWLEDGE BASE EXCERPTS ---\n"
            "Please ground your vehicle/insurance explanation in these verified domain documents and cite the sources."
        )

    messages = []
    messages.append({"role": "system", "content": system})
    for msg in history[-6:]:
        if msg.get("role") in ("user", "assistant"):
            messages.append({"role": msg["role"], "content": msg.get("text", "")})
    messages.append({"role": "user", "content": prompt})

    try:
        from core.llm_router import chat
        full_key = api_key if provider in ("gemini", "openai", "openrouter", "anthropic") else None

        resp = chat(
            messages=messages,
            model=model,
            provider=provider,
            api_key=full_key,
            ollama_base_url=ollama_url,
        )

        if hasattr(resp, "__iter__") and not isinstance(resp, str):
            text = "".join(resp)
        else:
            text = str(resp)

    except Exception as exc:
        text = (
            f"I couldn't reach the LLM ({provider}). "
            f"Error: {exc}\n\n"
            "**Try switching to Stub mode** in the settings panel, "
            "or check your API key."
        )

    return ChatResponse(
        type="text",
        text=text,
        sources=sources,
        citations=citations_data,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def process_turn(
    prompt: str,
    image: Image.Image | None,
    history: list[dict],
    settings: dict[str, Any],
) -> ChatResponse:
    """
    Route a user turn to the correct backend and return a ChatResponse.

    Parameters
    ----------
    prompt   : User's text message
    image    : Uploaded PIL image (or None)
    history  : Previous messages [{"role":..., "text":..., "type":...}]
    settings : dict with vlm_backend, gemini_api_key, llm_provider, etc.
    """
    prompt = (prompt or "").strip()

    if not prompt and image is None:
        return ChatResponse(type="error", error_msg="Please type a message or attach an image.")

    # ── Image attached → always assess ──────────────────────────────────────
    if image is not None:
        if not prompt:
            prompt = "Assess this image and provide a factual breakdown with key figures and summary."
        return _run_vlm(prompt, image, settings)

    # ── No image, damage keywords → ask to attach image ─────────────────────
    if _has_keywords(prompt, _DAMAGE_KEYWORDS) and "what" not in prompt.lower():
        # If user says "assess this" with no image, nudge them
        short_words = len(prompt.split()) < 8
        imperative = any(
            prompt.lower().startswith(v)
            for v in ("assess", "check", "analyse", "analyze", "evaluate")
        )
        if short_words and imperative:
            return ChatResponse(
                type="text",
                text=(
                    "To assess vehicle damage, please **attach an image** using the 📎 button below.\n\n"
                    "You can upload:\n"
                    "- A photo of the damaged vehicle\n"
                    "- A row from the INS-MMBench TSV dataset\n"
                    "- Any JPEG / PNG image"
                ),
            )

    # ── Everything else → LLM / RAG ──────────────────────────────────────────
    return _run_llm(prompt, history, settings)
