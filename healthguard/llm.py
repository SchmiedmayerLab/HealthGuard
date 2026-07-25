"""Model-access layer: OpenAI, Anthropic, and Ollama backends with chat, chat_json, and embedding helpers."""

from __future__ import annotations

import json
import os
import re

from openai import OpenAI

_client: OpenAI | None = None
_anthropic = None  # lazily-initialised Anthropic client

MODEL = os.environ.get("PIPELINE_MODEL", "gpt-5.4")
EMBED_MODEL = os.environ.get("PIPELINE_EMBED_MODEL", "text-embedding-3-small")
# Output-token ceiling for the Anthropic backend.
CLAUDE_MAX_TOKENS = int(os.environ.get("CLAUDE_MAX_TOKENS", "8192"))


def get_client() -> OpenAI:
    """Return a lazily-initialised OpenAI client (singleton)."""
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


# --- Anthropic backend ---------------------------------------
# Only models whose id starts with "claude" route here.

def _is_claude(model: str | None) -> bool:
    return (model or MODEL).startswith("claude")


def _anthropic_client():
    """Lazily-initialised Anthropic client. Reads ANTHROPIC_API_KEY or ANTHROPY_API_KEY (sic)."""
    global _anthropic
    if _anthropic is None:
        import anthropic
        key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPY_API_KEY")
        _anthropic = anthropic.Anthropic(api_key=key) if key else anthropic.Anthropic()
    return _anthropic


# --- Ollama backend (local open-weights models) --------------
# Models named "ollama/<tag>" (e.g. ollama/gemma2:9b-instruct-q4_K_M) route to a local Ollama
# server via its OpenAI-compatible endpoint.
_ollama = None


def _is_ollama(model: str | None) -> bool:
    return (model or MODEL).startswith("ollama/")


def _ollama_name(model: str | None) -> str:
    return (model or MODEL).split("/", 1)[1]


def _ollama_client() -> OpenAI:
    global _ollama
    if _ollama is None:
        _ollama = OpenAI(base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
                         api_key="ollama", timeout=600.0)
    return _ollama


def _ollama_max_tokens() -> int:
    # Cap generation so verbose small models can't ramble to the context limit (latency guard).
    return int(os.environ.get("OLLAMA_MAX_TOKENS", "1536"))


# Models that reject the temperature parameter.
_TEMP_REJECTING = ("opus-4-7", "opus-4-8", "fable", "mythos")


def _claude_text(system: str, user: str, model: str | None, temperature: float = 0.0) -> str:
    """One Anthropic Messages call returning concatenated text. ``temperature`` is passed
    only to models that accept it."""
    m = model or MODEL
    kwargs = {} if any(x in m for x in _TEMP_REJECTING) else {"temperature": temperature}
    resp = _anthropic_client().messages.create(
        model=m,
        max_tokens=CLAUDE_MAX_TOKENS,
        system=system,
        messages=[{"role": "user", "content": user}],
        **kwargs,
    )
    return "".join(b.text for b in resp.content if getattr(b, "type", None) == "text").strip()


def _loads_lenient(text: str) -> dict:
    """Parse JSON from a model response, tolerating code fences / surrounding prose."""
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.S).strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        a, b = t.find("{"), t.rfind("}")
        if a != -1 and b > a:
            return json.loads(t[a:b + 1])
        raise


def chat(
    system: str,
    user: str,
    model: str | None = None,
    temperature: float = 0.0,
) -> str:
    """Plain-text chat completion."""
    if _is_claude(model):
        return _claude_text(system, user, model, temperature)
    if _is_ollama(model):
        resp = _ollama_client().chat.completions.create(
            model=_ollama_name(model),
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=_ollama_max_tokens(),
        )
        return resp.choices[0].message.content.strip()
    client = get_client()
    resp = client.chat.completions.create(
        model=model or MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
    )
    return resp.choices[0].message.content.strip()


def chat_json(
    system: str,
    user: str,
    model: str | None = None,
    temperature: float = 0.0,
) -> dict:
    """JSON-mode chat completion.  Returns a parsed dict."""
    if _is_claude(model):
        sys = system + "\n\nRespond with ONLY the JSON object --- no prose, no code fences."
        return _loads_lenient(_claude_text(sys, user, model, temperature))
    if _is_ollama(model):
        sys = system + "\n\nRespond with ONLY the JSON object --- no prose, no code fences, no disclaimers."
        resp = _ollama_client().chat.completions.create(
            model=_ollama_name(model),
            messages=[
                {"role": "system", "content": sys},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},
            temperature=temperature,
            max_tokens=_ollama_max_tokens(),
        )
        return _loads_lenient(resp.choices[0].message.content)
    client = get_client()
    resp = client.chat.completions.create(
        model=model or MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_object"},
        temperature=temperature,
    )
    return json.loads(resp.choices[0].message.content)


def embed(texts: list[str], model: str | None = None) -> list[list[float]]:
    """Batched embedding call. Returns one vector per input text, in order."""
    if not texts:
        return []
    client = get_client()
    resp = client.embeddings.create(model=model or EMBED_MODEL, input=texts)
    return [item.embedding for item in resp.data]


def clamp_unit(value) -> float:
    """Coerce to float and clamp into [0.0, 1.0]; returns 0.0 on non-numeric input."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, f))
