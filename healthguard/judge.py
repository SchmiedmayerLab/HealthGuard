"""Clinical-equivalence judge: decide whether two diagnosis strings refer to the same entity.

LLM-as-judge with a 3-way verdict (match/related/mismatch), optionally reporting
whether the ground truth appears in a differentials list and its 1-based rank.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Literal

from . import llm
from . import judge_cache

# Bump when JUDGE_SYSTEM_PROMPT changes to invalidate cached verdicts.
JUDGE_PROMPT_VERSION = "v1"

JUDGE_MODEL = os.environ.get("HEALTHGUARD_JUDGE_MODEL", llm.MODEL)

JUDGE_SYSTEM_PROMPT = """\
You are a clinical equivalence judge. Given a ground-truth diagnosis and a \
candidate diagnosis, decide whether they refer to the same clinical entity.

Rules:
- Ignore laterality ("left/right"), severity ("mild/moderate/severe"), and \
chronicity qualifiers UNLESS they identify a distinct ICD entity (e.g. acute \
vs chronic bronchitis).
- Ignore wrapper phrases ("most likely X", "X with Y", "pain due to X").
- "match" = same clinical entity after stripping those qualifiers/wrappers.
- "related" = same organ system or overlapping differential but distinct \
entity (e.g., tension-type headache vs migraine).
- "mismatch" = clinically distinct entity.

If a differentials list is provided, also report whether the ground truth \
appears in it (by clinical synonymy, not string match) and its 1-based rank. \
If no differentials are provided, return null for both fields.

Return a JSON object:
{
  "verdict": "match" | "related" | "mismatch",
  "confidence": 0.0-1.0,
  "reasoning": "brief explanation",
  "in_differential": true | false | null,
  "differential_rank": <1-based int> | null
}"""


@dataclass
class JudgeVerdict:
    verdict: Literal["match", "related", "mismatch"]
    confidence: float
    reasoning: str
    in_differential: bool | None
    differential_rank: int | None

    def to_dict(self) -> dict:
        return asdict(self)


_VALID_VERDICTS = ("match", "related", "mismatch")


def judge_diagnosis(
    ground_truth: str,
    candidate: str,
    differentials: list[dict] | None = None,
    model: str | None = None,
    use_cache: bool = True,
) -> JudgeVerdict:
    """Compare ``candidate`` to ``ground_truth`` and return a 3-way verdict."""
    effective_model = model or JUDGE_MODEL

    key = judge_cache.cache_key(
        model=effective_model,
        prompt_version=JUDGE_PROMPT_VERSION,
        ground_truth=ground_truth,
        candidate=candidate,
        differentials=differentials,
    )

    if use_cache:
        cached = judge_cache.get(key)
        if cached is not None:
            return cached

    payload: dict = {"ground_truth": ground_truth, "candidate": candidate}
    if differentials:
        payload["differentials"] = differentials

    raw = llm.chat_json(
        system=JUDGE_SYSTEM_PROMPT,
        user=json.dumps(payload, ensure_ascii=False),
        model=effective_model,
    )

    verdict_str = raw.get("verdict")
    if verdict_str not in _VALID_VERDICTS:
        verdict_str = "mismatch"

    rank = raw.get("differential_rank")
    if rank is not None:
        try:
            rank = int(rank)
        except (TypeError, ValueError):
            rank = None

    in_diff = raw.get("in_differential")
    if in_diff is not None and not isinstance(in_diff, bool):
        in_diff = bool(in_diff)

    verdict = JudgeVerdict(
        verdict=verdict_str,
        confidence=llm.clamp_unit(raw.get("confidence", 0.0)),
        reasoning=str(raw.get("reasoning", "")),
        in_differential=in_diff,
        differential_rank=rank,
    )

    if use_cache:
        judge_cache.put(key, verdict)
    return verdict
