"""Rubric-based grader, faithful to LiveMedBench's automated evaluation.

An LLM grader (default GPT-4.1, as in the paper) is shown the case, the candidate
response, and the numbered rubric criteria, and returns one boolean per criterion
(met / not-met) in order. We then apply the (hidden-from-grader) weighted bipolar
points ourselves:

    earned      = sum(points_i for criteria the response MET)
                  (positive criteria add on a hit; negative criteria subtract on a hit)
    max_positive = sum(points_j for all positive criteria)
    score        = earned / max_positive          (1.0 = perfect; can go < 0)

The grader judges only the criterion text; it never sees the points, so it cannot
be biased by them.
"""
from __future__ import annotations

import json

from healthguard import llm

GRADER = "gpt-4.1"

GRADER_SYSTEM = """\
You are a strict, literal medical evaluator. You are given a patient case (narrative \
and the patient's specific request), a candidate response written by an AI assistant, \
and a numbered list of rubric criteria. For EACH criterion, decide whether the \
candidate response satisfies it, judging ONLY what the response literally says.

Criteria come in two kinds:
- Desirable content the response SHOULD contain (mark met=true only if the response \
actually states it).
- Undesirable content or errors (mark met=true only if the response actually commits \
that error / makes that harmful or incorrect statement).

Do not reward vague or missing content. Do not infer beyond what is written.

Return ONLY a JSON object: {"met": [true/false, ...]} with EXACTLY one boolean per \
criterion, in the SAME order as given."""


def grade(case: dict, response: str, model: str = GRADER) -> dict:
    items = case.get("rubric_items", []) or []
    criteria = [it.get("criterion", "") for it in items]
    points = [float(it.get("points", 0)) for it in items]
    max_positive = sum(p for p in points if p > 0)

    numbered = "\n".join(f"{i}. {c}" for i, c in enumerate(criteria))
    user = (
        f"PATIENT CASE:\n{case.get('narrative', '')}\n\n"
        f"PATIENT'S REQUEST:\n{case.get('core_request', '')}\n\n"
        f"CANDIDATE RESPONSE:\n{response}\n\n"
        f"RUBRIC CRITERIA ({len(criteria)}):\n{numbered}\n\n"
        f'Return {{"met": [...]}} with exactly {len(criteria)} booleans, in order.'
    )
    raw = llm.chat_json(system=GRADER_SYSTEM, user=user, model=model)
    met = raw.get("met", []) or []
    # pad/truncate defensively so scoring never crashes on a short array
    met = [bool(x) for x in met][: len(criteria)]
    met += [False] * (len(criteria) - len(met))

    earned = sum(points[i] for i in range(len(criteria)) if met[i])
    score = earned / max_positive if max_positive else 0.0
    return {
        "met": met,
        "n_criteria": len(criteria),
        "n_positive_met": sum(1 for i in range(len(criteria)) if met[i] and points[i] > 0),
        "n_negative_hit": sum(1 for i in range(len(criteria)) if met[i] and points[i] < 0),
        "earned": round(earned, 3),
        "max_positive": max_positive,
        "score": round(score, 4),
    }
