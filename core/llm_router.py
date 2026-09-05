"""
core/llm_router.py
Unified LLM interface — direct clients, no litellm dependency.

Supported providers:
  - stub        : canned responses, no key, instant
  - gemini      : google-generativeai (free tier, 15 RPM)
  - openrouter  : openai client + openrouter.ai base_url (many free models)
  - openai      : openai client + openai.com (paid)
  - ollama      : HTTP REST to local Ollama server (free, offline)
  - anthropic   : anthropic client (paid; install: pip install anthropic)

API key injected at call time from st.session_state — never stored to disk.
"""
from __future__ import annotations

import json
import os
import time
from collections.abc import Generator
from typing import Any

# ---------------------------------------------------------------------------
# Provider registry — shown in the Streamlit sidebar
# ---------------------------------------------------------------------------

PROVIDER_REGISTRY: dict[str, dict[str, Any]] = {
    "stub": {
        "label": "🔧 Stub (offline, no key needed)",
        "models": ["stub"],
        "key_env": None,
        "key_label": None,
        "free": True,
        "notes": "Returns canned responses. Perfect for offline demos.",
    },
    "gemini": {
        "label": "✨ Google Gemini (free tier)",
        "models": [
            "gemini-3.6-flash",
            "gemini-3.7-flash",
            "gemini-3.5-flash",
            "gemini-flash-latest",
            "gemini-3.5-flash-lite",
            "gemini-2.5-flash",
        ],
        "key_env": "GEMINI_API_KEY",
        "key_label": "Gemini API Key (aistudio.google.com → free)",
        "free": True,
        "notes": "15 RPM / 1M tok/day free. Recommended for best quality.",
    },
    "openrouter": {
        "label": "🌐 OpenRouter (free models)",
        "models": [
            "meta-llama/llama-3.1-8b-instruct:free",
            "meta-llama/llama-3.2-3b-instruct:free",
            "qwen/qwen-2.5-72b-instruct:free",
            "mistralai/mistral-7b-instruct:free",
            "google/gemma-2-9b-it:free",
        ],
        "key_env": "OPENROUTER_API_KEY",
        "key_label": "OpenRouter API Key (openrouter.ai → free tier)",
        "free": True,
        "notes": "Aggregates 50+ providers. Many free models with :free suffix.",
    },
    "ollama": {
        "label": "🏠 Ollama (local, fully offline)",
        "models": ["mistral", "llama3", "llama3.2", "qwen2.5", "phi3"],
        "key_env": None,
        "key_label": None,
        "free": True,
        "notes": "Runs locally. Install Ollama at ollama.ai then: ollama pull mistral",
    },
    "openai": {
        "label": "🤖 OpenAI",
        "models": ["gpt-4o-mini", "gpt-4o"],
        "key_env": "OPENAI_API_KEY",
        "key_label": "OpenAI API Key",
        "free": False,
        "notes": "Paid. gpt-4o-mini is cheapest.",
    },
    "anthropic": {
        "label": "🧠 Anthropic Claude",
        "models": ["claude-haiku-20240307", "claude-3-5-sonnet-20241022"],
        "key_env": "ANTHROPIC_API_KEY",
        "key_label": "Anthropic API Key",
        "free": False,
        "notes": "Paid. Also run: pip install anthropic",
    },
}

# ---------------------------------------------------------------------------
# Stub canned responses
# ---------------------------------------------------------------------------

_STUB_PATTERNS: list[tuple[list[str], str, list[str]]] = [
    (
        ["severe", "severity", "why severe"],
        r"**Severity: Severe** is assigned when repair costs are estimated between $8,000–$25,000, "
        "or when there is structural/frame damage, multiple panel replacements, or airbag deployment. "
        r"Per Taxonomy §3.2, a vehicle is classified Severe when ≥3 primary damage codes are present "
        "on critical load-bearing components.",
        ["taxonomy_v2.txt", "rubric_severity.txt"],
    ),
    (
        ["minor", "why minor"],
        r"**Severity: Minor** covers cosmetic damage costing $200–$1,500. "
        "This includes single-panel scratches, paint transfer, and small dents where no structural "
        r"component is affected (Taxonomy §2.1). The vehicle is fully drivable post-incident.",
        ["taxonomy_v2.txt"],
    ),
    (
        ["pre-existing", "prior damage", "pre existing"],
        r"Per Policy Clause 7(b), pre-existing damage must be documented at policy inception. "
        "Claims that include pre-existing damage must be split: the pre-existing portion is excluded "
        "and the new damage assessed independently. Adjusters should flag the claim for photo evidence "
        "review if pre-existing damage is suspected.",
        ["policy_clauses.txt", "rubric_severity.txt"],
    ),
    (
        ["similar", "past claims", "precedent", "comparable"],
        "Based on the knowledge base, three comparable precedent claims are:\n"
        r"- **CL-2024-0471**: Front bumper crush + hood crease → Moderate → $4,200 settlement" + "\n"
        r"- **CL-2024-0389**: Rear bumper crush + trunk lid dent → Moderate → $3,800 settlement" + "\n"
        r"- **CL-2023-1102**: Windshield shatter + door dent → Minor → $1,100 settlement" + "\n\n"
        r"These precedents suggest cost range $3,800–$4,500 for similar multi-panel moderate claims.",
        ["claim_precedents.txt"],
    ),
    (
        ["fraud", "suspicious", "flag", "investigate"],
        "Fraud indicators include:\n"
        "1. Damage inconsistent with reported incident (e.g., rear-only damage from a head-on collision)\n"
        "2. Recent policy inception (< 30 days before claim)\n"
        "3. Damage pattern suggesting staged incident (bilateral symmetric damage)\n"
        "4. High-value claim on a vehicle near end of insured life\n\n"
        "When the model flags `fraud_flag=True`, the claim is automatically queued for senior adjuster review.",
        ["policy_clauses.txt"],
    ),
    (
        ["total loss", "totalled", "write off", "write-off"],
        r"**Total Loss** is declared when the estimated repair cost exceeds 75% of the vehicle's "
        "Actual Cash Value (ACV), or when structural/frame damage makes safe repair impractical. "
        "Total loss claims are processed separately: an ACV appraisal is ordered and the settlement "
        r"is ACV minus the salvage value (Policy §12).",
        ["policy_clauses.txt", "taxonomy_v2.txt"],
    ),
]

_STUB_DEFAULT = (
    "I'm running in **stub mode** — no live LLM connected. "
    "This is a placeholder response demonstrating the citation format. "
    "To get real answers, select a provider in the sidebar and enter your API key.",
    ["taxonomy_v2.txt"],
)


def _stub_response(question: str) -> tuple[str, list[str]]:
    q = question.lower()
    for keywords, answer, sources in _STUB_PATTERNS:
        if any(k in q for k in keywords):
            return answer, sources
    return _STUB_DEFAULT


# ---------------------------------------------------------------------------
# Provider dispatch helpers
# ---------------------------------------------------------------------------

def _chat_gemini(
    messages: list[dict[str, str]],
    model: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
    stream: bool,
) -> str | Generator[str, None, None]:
    import google.generativeai as genai

    genai.configure(api_key=api_key)

    if model and ("1.5-flash" in model or "tts" in model or "2.0-flash" in model or "2.5-flash" in model):
        model = "gemini-3.6-flash"

    candidates = [
        model,
        "gemini-3.6-flash",
        "gemini-3.7-flash",
        "gemini-3.5-flash",
        "gemini-flash-latest",
        "gemini-3.5-flash-lite",
    ]
    dedup: list[str] = []
    for c in candidates:
        if c and "tts" not in c.lower() and c not in dedup:
            dedup.append(c)

    sys_inst = next((m["content"] for m in messages if m["role"] == "system"), None)
    history_template = [
        {"role": "user" if m["role"] == "user" else "model", "parts": [m["content"]]}
        for m in messages
        if m["role"] != "system"
    ]
    last_user = history_template.pop()["parts"][0] if history_template else ""
    gen_cfg = genai.types.GenerationConfig(temperature=temperature, max_output_tokens=max_tokens)

    last_exc = None
    for m_name in dedup:
        try:
            gmodel = genai.GenerativeModel(m_name, system_instruction=sys_inst)
            chat_session = gmodel.start_chat(history=list(history_template))
            if stream:
                response = chat_session.send_message(last_user, stream=True, generation_config=gen_cfg)
                def _gen():
                    for chunk in response:
                        if chunk.text:
                            yield chunk.text
                return _gen()
            else:
                response = chat_session.send_message(last_user, generation_config=gen_cfg)
                return response.text
        except Exception as exc:
            err_str = str(exc).lower()
            last_exc = exc
            if any(k in err_str for k in ["not found", "404", "unsupported", "not supported", "429", "quota", "resource_exhausted", "rate limit"]):
                print(f"[Gemini chat] Model '{m_name}' unavailable or quota reached, trying next fallback...")
                continue
            raise exc

    raise RuntimeError(f"All Gemini chat models failed: {last_exc}") from last_exc


def _chat_openai_compat(
    messages: list[dict[str, str]],
    model: str,
    api_key: str,
    base_url: str | None,
    temperature: float,
    max_tokens: int,
    stream: bool,
    extra_headers: dict | None = None,
) -> str | Generator[str, None, None]:
    """Works for OpenAI, OpenRouter, and any OpenAI-compatible endpoint."""
    from openai import OpenAI

    kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    if extra_headers:
        kwargs["default_headers"] = extra_headers

    client = OpenAI(**kwargs)

    if stream:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        def _gen():
            for chunk in response:
                delta = chunk.choices[0].delta.content or ""
                if delta:
                    yield delta
        return _gen()
    else:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""


def _chat_ollama(
    messages: list[dict[str, str]],
    model: str,
    base_url: str,
    temperature: float,
    stream: bool,
) -> str | Generator[str, None, None]:
    """Direct Ollama REST API — no extra package needed."""
    import requests as req

    url = base_url.rstrip("/") + "/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "stream": stream,
        "options": {"temperature": temperature},
    }

    if stream:
        def _gen():
            with req.post(url, json=payload, stream=True, timeout=120) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            content = data.get("message", {}).get("content", "")
                            if content:
                                yield content
                            if data.get("done"):
                                break
                        except Exception:
                            continue
        return _gen()
    else:
        payload["stream"] = False
        r = req.post(url, json=payload, timeout=120)
        r.raise_for_status()
        return r.json()["message"]["content"]


def _chat_anthropic(
    messages: list[dict[str, str]],
    model: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
    stream: bool,
) -> str | Generator[str, None, None]:
    try:
        import anthropic
    except ImportError as exc:
        raise ImportError("Run: pip install anthropic") from exc

    client = anthropic.Anthropic(api_key=api_key)
    system = next((m["content"] for m in messages if m["role"] == "system"), "")
    user_msgs = [m for m in messages if m["role"] != "system"]

    if stream:
        def _gen():
            with client.messages.stream(
                model=model,
                max_tokens=max_tokens,
                system=system,
                messages=user_msgs,
            ) as s:
                for text in s.text_stream:
                    yield text
        return _gen()
    else:
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=user_msgs,
        )
        return response.content[0].text


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def chat(
    messages: list[dict[str, str]],
    model: str,
    provider: str = "stub",
    api_key: str | None = None,
    temperature: float = 0.3,
    max_tokens: int = 1024,
    stream: bool = False,
    ollama_base_url: str = "http://localhost:11434",
) -> str | Generator[str, None, None]:
    """
    Unified chat call dispatched to the right provider.

    Args:
        messages  : OpenAI-format message list [{role, content}]
        model     : Model name (provider-specific, e.g. "gemini-1.5-flash")
        provider  : One of the PROVIDER_REGISTRY keys
        api_key   : Provider API key (injected at runtime, never stored)
        stream    : Return a generator of text chunks if True
    """
    if provider == "stub" or model == "stub":
        user_msg = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        answer, _ = _stub_response(user_msg)
        if stream:
            def _gen():
                for word in answer.split(" "):
                    yield word + " "
                    time.sleep(0.018)
            return _gen()
        return answer

    if provider == "gemini":
        resolved_key = (api_key or "").strip() or os.getenv("GEMINI_API_KEY", "")
        return _chat_gemini(messages, model, resolved_key, temperature, max_tokens, stream)

    if provider == "openrouter":
        resolved_key = (api_key or "").strip() or os.getenv("OPENROUTER_API_KEY", "")
        return _chat_openai_compat(
            messages, model, resolved_key,
            base_url="https://openrouter.ai/api/v1",
            temperature=temperature, max_tokens=max_tokens, stream=stream,
            extra_headers={"HTTP-Referer": "https://claimlens.local", "X-Title": "ClaimLens"},
        )

    if provider == "openai":
        resolved_key = (api_key or "").strip() or os.getenv("OPENAI_API_KEY", "")
        return _chat_openai_compat(
            messages, model, resolved_key,
            base_url=None,
            temperature=temperature, max_tokens=max_tokens, stream=stream,
        )

    if provider == "ollama":
        return _chat_ollama(messages, model, ollama_base_url, temperature, stream)

    if provider == "anthropic":
        resolved_key = (api_key or "").strip() or os.getenv("ANTHROPIC_API_KEY", "")
        return _chat_anthropic(messages, model, resolved_key, temperature, max_tokens, stream)

    raise ValueError(f"Unknown provider: {provider!r}. Choose from: {list(PROVIDER_REGISTRY)}")


def stream_chat(
    messages: list[dict[str, str]],
    model: str,
    provider: str = "stub",
    api_key: str | None = None,
    temperature: float = 0.3,
    max_tokens: int = 1024,
    ollama_base_url: str = "http://localhost:11434",
) -> Generator[str, None, None]:
    """Convenience wrapper — always streams."""
    return chat(
        messages=messages, model=model, provider=provider,
        api_key=api_key, temperature=temperature, max_tokens=max_tokens,
        stream=True, ollama_base_url=ollama_base_url,
    )
