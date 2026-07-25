"""Generate a clinical response to a LiveMedBench case (the task under evaluation).

Zero-shot, temperature 0, neutral physician-assistant framing, deliberately NOT
tuned to the rubric (the model never sees the rubric), matching the benchmark's
zero-shot protocol.
"""
from __future__ import annotations

from healthguard import llm

RESPOND_SYSTEM = """\
You are an experienced physician answering a patient's medical question. \
Given the case narrative and the patient's specific request, write a clear, \
accurate, and clinically sound response that directly addresses what the patient \
asked. Be complete but concise, note important safety considerations and red flags, \
and recommend appropriate follow-up or specialist referral where warranted. \
Do not fabricate findings that are not in the case."""


def respond(case: dict, model: str, temperature: float = 0.0) -> str:
    user = (
        f"PATIENT CASE:\n{case.get('narrative', '')}\n\n"
        f"PATIENT'S REQUEST:\n{case.get('core_request', '')}\n\n"
        f"Write your response to the patient."
    )
    return llm.chat(system=RESPOND_SYSTEM, user=user, model=model, temperature=temperature)
