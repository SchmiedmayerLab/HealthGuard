"""Single-LLM critic baseline: one LLM call audits an entire reasoning trace and returns a
soundness score plus a list of flagged flaws."""

from __future__ import annotations

from evaluation._util import load_env

load_env()

import json  # noqa: E402

from healthguard import llm  # noqa: E402

CRITIC_SYSTEM = """\
You are a meticulous clinical reasoning auditor. You are given a chain-of-thought \
diagnostic reasoning trace and the reference case data it should be grounded in. \
You do NOT know the correct final diagnosis and must not assume one.

Audit the reasoning on its own terms. For every claim, consider:
- Factual accuracy (e.g. correct lab thresholds, drug facts, epidemiology).
- Evidential support (is the claim backed by findings actually in the reference data?).
- Logical validity (does each inference follow; are there unsupported leaps, \
contradicted steps, or selective emphasis / misweighting of evidence?).

A trace can reach a plausible-sounding conclusion while resting on an inaccurate \
value, an unsupported claim, or an invalid inference — flag those even when the \
prose is fluent and confident.

Return ONLY a JSON object:
{
  "soundness": <float 0.0-1.0, where 1.0 = every claim accurate, supported, and \
logically connected; 0.0 = pervasively flawed reasoning>,
  "flawed": <true if the trace contains at least one material reasoning flaw>,
  "flaws": [{"claim": "<the problematic statement>", "issue": "<why it is flawed>"}],
  "reasoning": "<2-4 sentence overall assessment>"
}"""


def critique(input_text: str, additional_input: dict, model: str | None = None,
             temperature: float = 0.0) -> dict:
    """Audit a reasoning trace with a single LLM call. Returns the parsed verdict."""
    user_payload = json.dumps(
        {"reasoning_trace": input_text, "reference_case_data": additional_input},
        ensure_ascii=False,
    )
    raw = llm.chat_json(system=CRITIC_SYSTEM, user=user_payload, model=model, temperature=temperature)

    flaws = raw.get("flaws")
    if not isinstance(flaws, list):
        flaws = []
    return {
        "soundness": llm.clamp_unit(raw.get("soundness", 0.0)),
        "flawed": bool(raw.get("flawed", False)),
        "flaws": flaws,
        "reasoning": str(raw.get("reasoning", "")),
    }
