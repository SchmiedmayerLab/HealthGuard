"""Cross-vendor judge dispatch: route a judge LLM call by model name."""

from __future__ import annotations

import json
import os
import re

from healthguard import llm  # OpenAI path

_anthropic_client = None


def _anthropic():
    global _anthropic_client
    if _anthropic_client is None:
        import anthropic
        # .env may name the key ANTHROPY_API_KEY (sic) or ANTHROPIC_API_KEY
        key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPY_API_KEY")
        if not key:
            raise RuntimeError("No Anthropic key found (ANTHROPIC_API_KEY / ANTHROPY_API_KEY).")
        _anthropic_client = anthropic.Anthropic(api_key=key)
    return _anthropic_client


def _parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        a, b = text.find("{"), text.rfind("}")
        if a != -1 and b > a:
            return json.loads(text[a:b + 1])
        raise


def chat_json(system: str, user: str, model: str | None = None, temperature: float = 0.0) -> dict:
    """JSON judge call. Claude models -> Anthropic Messages API; otherwise -> OpenAI (pipeline.llm)."""
    if model and model.startswith("claude"):
        resp = _anthropic().messages.create(
            model=model,
            max_tokens=4096,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
        return _parse_json(text)
    return llm.chat_json(system=system, user=user, model=model, temperature=temperature)
