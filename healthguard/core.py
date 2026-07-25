"""HealthGuard generate/audit/revise wrapper for the LiveMedBench response task.

Applies the verifier to a clinical *response* (rather than a diagnostic trace). The wrapper is
deliberately additive:

  1. COVERAGE  - from the CASE ALONE, enumerate the clinically essential points a complete
                 answer must address (specific management, needed work-up, relevant consultations,
                 mechanisms, and any genuine red-flag/referral to add); check which the draft
                 covers to find gaps to fill.
  2. GROUNDING - decompose the draft into claims; flag only those unsupported by, or contradicting,
                 the case (targets the negative criteria of the scoring rubric).
  3. REVISE    - the same base model rewrites the draft to close the gaps and drop flagged claims,
                 without softening correct, specific recommendations.

Ground-truth-free by construction: the auditor sees only the case narrative, the patient's
request, and the draft, never the rubric. The base model is identical to the baseline's, so the
only thing under test versus the baseline is the audit/revise structure. A `self_refine` control
(one generic "improve it" pass, no structure) isolates the structure from mere second-pass effort.

An optional safety/commission stage (`safety=True`) scans the revised response for clearly
incorrect or unsafe statements and corrects only those. It is off by default because it can push
the reviser to hedge away decisive, correct recommendations; genuine red-flags and referrals are
captured additively inside COVERAGE instead.
"""
from __future__ import annotations

import json

from . import llm

# ---- audit stage prompts (ground-truth-free: case + draft only) ----

COVERAGE_SYSTEM = """\
You are a clinical completeness auditor. You are given ONLY a patient case (narrative + the \
patient's specific request) and a draft response. You do NOT have an answer key.

First, from the CASE ALONE, enumerate EXHAUSTIVELY the clinically essential points that a thorough \
specialist answer to THIS specific patient should contain. Be comprehensive, not minimal — a \
complete expert answer usually covers MANY distinct points (typically 8-14). Do not stop after a \
few. Systematically consider every relevant dimension for this patient:
- a direct, specific answer to the exact question asked (with concrete specifics/numbers);
- the most likely cause(s) / diagnostic reasoning for this patient;
- the necessary work-up — name concrete tests and investigations;
- specific management / treatment options (name drugs, procedures, or interventions) and their tradeoffs;
- how this patient's particular factors (age, comorbidities, pregnancy, findings) change the answer;
- prognosis and what to expect over time;
- genuine red-flags and when to seek urgent care; which specialist to involve;
- monitoring, follow-up, and self-management / lifestyle guidance.

Then, for each essential point, decide whether the DRAFT already addresses it. Mark "covered": true \
ONLY if the draft EXPLICITLY and SPECIFICALLY states that point — a vague, partial, or generic \
mention does NOT count as covered.

Return ONLY JSON: {"essential": [{"point": "<short, specific to this patient>", "covered": true/false}]}. \
Each point must be concrete and specific to this patient; avoid generic filler, but be thorough."""

GROUNDING_SYSTEM = """\
You are a clinical grounding checker. Given the patient case and a draft response, decompose the \
draft into atomic claims and recommendations. Classify each against the case:
- "supported": follows from the case or is standard, correct medical knowledge for it.
- "unsupported": asserts a specific finding/number/diagnosis not derivable from the case.
- "incorrect": clinically wrong, unsafe, or contradicts the case.

Return ONLY JSON: {"claims": [{"text": "<short>", "verdict": "supported|unsupported|incorrect"}]}. \
Only flag genuine factual problems; do NOT flag a recommendation merely for being decisive."""

REVISE_SYSTEM = """\
You are revising your own draft response to a patient after a structured clinical audit. Produce \
an improved response that ADDS every MISSING essential point (with specific, actionable clinical \
content) and drops or corrects any claim flagged as unsupported or incorrect.

Keep everything in the draft that was already correct. Stay DECISIVE and SPECIFIC — do NOT soften, \
hedge, or replace concrete correct recommendations with vague "consult your doctor" language, and \
do not add blanket disclaimers. Do NOT invent findings not in the case. \
Output only the revised response to the patient."""

SELF_REFINE_SYSTEM = """\
You are an experienced physician. Here is a draft response to a patient. Review it and produce an \
improved version that is more accurate, complete, and safe. Output only the improved response."""

RECHECK_SYSTEM = """\
You are checking whether a revised clinical response now covers a list of essential points. For EACH \
point, decide whether the response EXPLICITLY and SPECIFICALLY states it (a vague, partial, or generic \
mention does NOT count). Return ONLY JSON: {"covered": [true/false, ...]} with one boolean per point, \
in the same order."""

SAFETY_SYS = """\
You are a final clinical safety check on a response about to go to a patient. Given the case and the \
response, identify statements that are clearly INCORRECT (factual error, wrong dose/mechanism/association), \
UNSAFE (contraindicated or harmful advice for THIS patient), FALSE REASSURANCE, or that CONTRADICT the \
case. List only genuine, clear errors — do NOT flag correct, decisive, standard-of-care advice, and do \
NOT ask to hedge or soften. Return ONLY JSON: {"issues": ["<the wrong/unsafe statement + why>", ...]} \
(empty list if none)."""

FIX_SYS = """\
You are correcting your response after a safety check. For EACH listed issue, correct or remove the \
specific incorrect/unsafe statement. Change NOTHING else — keep all other content and its specificity \
intact. Do not hedge or add disclaimers. Output only the corrected response."""


# Targeted coverage lenses: the single general coverage pass under-surfaces management and
# prognosis gaps, so these add two focused enumerations (management, then prognosis) and union
# their gaps with the general pass.

MGMT_SYSTEM = """\
You are the treating specialist for this patient. From the CASE ALONE (no answer key), list the \
SPECIFIC clinical management this patient's answer should recommend: name concrete drugs (with class \
and rationale), procedures, interventions, dose/monitoring considerations, and their sequencing or \
tradeoffs for THIS patient. Be specific and complete — a real specialist names actual treatments, not \
"appropriate therapy". Then, for each, mark whether the DRAFT explicitly and specifically states it \
(a vague or generic mention does NOT count).

Return ONLY JSON: {"essential": [{"point": "<specific management point>", "covered": true/false}]}."""

PROGNOSIS_SYSTEM = """\
You are the treating specialist. From the CASE ALONE, list what the answer should tell THIS patient \
about prognosis and expected course: likely outcomes, timeframes, chances of recovery/recurrence, \
and what worsens or improves it. Be specific to this patient. Then mark whether the DRAFT explicitly \
states each point.

Return ONLY JSON: {"essential": [{"point": "<specific prognosis point>", "covered": true/false}]}."""

# A work-up lens (diagnostic work-up is a common omission), paired with a stricter grounding check
# that catches harmful or incorrect statements the base grounding tends to miss.
WORKUP_SYSTEM = """\
You are the treating specialist. From the CASE ALONE, list the specific diagnostic WORK-UP this \
patient's answer should recommend — name concrete tests and investigations (labs, imaging, biopsies, \
functional tests, referrals for tests) needed to establish or exclude the likely diagnoses. Then mark \
whether the DRAFT explicitly recommends each.

Return ONLY JSON: {"essential": [{"point": "<specific test/investigation>", "covered": true/false}]}."""

GROUNDING_SYSTEM_STRICT = """\
You are a clinical accuracy-and-safety checker. Decompose the draft into atomic claims and \
recommendations and scrutinize EACH against the case and standard medical knowledge:
- "supported": correct and appropriate.
- "unsupported": asserts a specific finding/number/diagnosis not derivable from the case.
- "incorrect": a factual error (wrong dose/mechanism/association), an unsafe or contraindicated \
recommendation, false reassurance, or a statement that contradicts the case.

Be thorough — check every factual claim and every recommendation for safety. But do NOT flag a \
recommendation merely for being decisive or specific, and do NOT flag correct standard-of-care advice.

Return ONLY JSON: {"claims": [{"text": "<short>", "verdict": "supported|unsupported|incorrect"}]}."""

# A dedicated lens for dimensions the others have no targeted lens for: diagnosis, mechanism,
# monitoring, and patient-specific factors.
DIMENSIONS_SYSTEM = """\
You are the treating specialist. From the CASE ALONE, list essential points a thorough answer should \
include for THIS patient across these often-missed dimensions: the most likely DIAGNOSIS or cause and \
how it is identified; the underlying MECHANISM worth explaining; the MONITORING and follow-up plan; and \
PATIENT-SPECIFIC factors (age, comorbidities, pregnancy, preferences) that should shape the advice. \
Then mark whether the DRAFT explicitly covers each.

Return ONLY JSON: {"essential": [{"point": "<specific point>", "covered": true/false}]}."""


def _json(system: str, user: str, model: str) -> dict:
    try:
        return llm.chat_json(system=system, user=user, model=model)
    except Exception:  # noqa: BLE001 - a failed audit sub-call just yields no findings
        return {}


def _gaps_of(result: dict) -> list[str]:
    return [e.get("point", "") for e in (result.get("essential") or []) if not e.get("covered", False)]


def _dedup(gaps: list[str]) -> list[str]:
    seen, out = set(), []
    for g in gaps:
        k = " ".join((g or "").lower().split())
        if k and k not in seen:
            seen.add(k); out.append(g)
    return out


def _case_block(case: dict, draft: str) -> str:
    return (
        f"PATIENT CASE:\n{case.get('narrative', '')}\n\n"
        f"PATIENT'S REQUEST:\n{case.get('core_request', '')}\n\n"
        f"DRAFT RESPONSE:\n{draft}\n"
    )


def audit(case: dict, draft: str, model: str, coverage: str = "base") -> dict:
    block = _case_block(case, draft)
    cov = _json(COVERAGE_SYSTEM, block, model)
    gaps = _gaps_of(cov)
    if coverage in ("lenses", "workup", "full"):
        # targeted lenses for known blind spots; union the gaps
        lenses = [MGMT_SYSTEM, PROGNOSIS_SYSTEM]
        if coverage in ("workup", "full"):
            lenses.append(WORKUP_SYSTEM)
        if coverage == "full":
            lenses.append(DIMENSIONS_SYSTEM)   # diagnosis/mechanism/monitoring/patient-factors
        for lens in lenses:
            gaps += _gaps_of(_json(lens, block, model))
        gaps = _dedup(gaps)
    strict = coverage in ("workup", "full")
    grd = _json(GROUNDING_SYSTEM_STRICT if strict else GROUNDING_SYSTEM, block, model)
    bad = [c for c in (grd.get("claims") or []) if c.get("verdict") in ("unsupported", "incorrect")]
    return {
        "essential": cov.get("essential") or [],
        "gaps": gaps,
        "flagged_claims": bad,
        "n_gaps": len(gaps), "n_flagged": len(bad),
    }


def _critique_text(a: dict) -> str:
    parts = []
    if a["gaps"]:
        parts.append("MISSING essential points to add (with specific content):\n"
                     + "\n".join(f"- {g}" for g in a["gaps"]))
    if a["flagged_claims"]:
        parts.append("Claims to remove/correct (only genuine errors):\n" + "\n".join(
            f"- ({c.get('verdict')}) {c.get('text','')}" for c in a["flagged_claims"]))
    return "\n\n".join(parts) if parts else "No missing essentials or bad claims found; ensure completeness."


def revise(case: dict, draft: str, critique: str, model: str) -> str:
    user = _case_block(case, draft) + f"\nAUDIT FINDINGS:\n{critique}\n\nWrite the revised response."
    return llm.chat(system=REVISE_SYSTEM, user=user, model=model)


def _recheck(case: dict, response: str, points: list[str], model: str) -> list[bool]:
    """Which of `points` does the REVISED response now explicitly cover? (landing check)"""
    if not points:
        return []
    payload = {
        "patient_request": case.get("core_request", ""),
        "revised_response": response,
        "points_to_check": [{"index": i, "text": p} for i, p in enumerate(points)],
    }
    cov = (_json(RECHECK_SYSTEM, json.dumps(payload, ensure_ascii=False), model).get("covered") or [])
    cov = [bool(x) for x in cov][: len(points)]
    cov += [True] * (len(points) - len(cov))   # parse-short -> assume covered (don't over-repair)
    return cov


def _safety_scan(case: dict, response: str, model: str) -> list[str]:
    block = (f"PATIENT CASE:\n{case.get('narrative','')}\n\nPATIENT'S REQUEST:\n"
             f"{case.get('core_request','')}\n\nRESPONSE:\n{response}\n")
    return [x for x in (_json(SAFETY_SYS, block, model).get("issues") or []) if x]


def healthguard_response(case: dict, draft: str, model: str,
                         coverage: str = "base", revise_mode: str = "single",
                         safety: bool = False) -> dict:
    """Full audit->revise. revise_mode="loop" adds a verify-and-repair pass (re-check unlanded gaps,
    targeted second pass). safety=True adds a final commission check: scan the response for clearly
    incorrect/unsafe statements and correct exactly those."""
    a = audit(case, draft, model, coverage=coverage)
    critique = _critique_text(a)
    revised = revise(case, draft, critique, model)
    if revise_mode == "loop" and a["gaps"]:
        covered = _recheck(case, revised, a["gaps"], model)
        still = [g for g, c in zip(a["gaps"], covered) if not c]
        a["n_unlanded_after_pass1"] = len(still)
        if still:
            crit2 = ("These essential points are STILL missing from your response — add EACH one "
                     "explicitly with concrete, specific clinical content, and keep everything already "
                     "present:\n" + "\n".join(f"- {g}" for g in still))
            revised = revise(case, revised, crit2, model)
    if safety:
        issues = _safety_scan(case, revised, model)
        a["n_safety_issues"] = len(issues)
        if issues:
            user = _case_block(case, revised) + "\nISSUES TO FIX:\n" + "\n".join(f"- {x}" for x in issues)
            revised = llm.chat(system=FIX_SYS, user=user, model=model)
    return {"response": revised, "audit": a, "critique": critique}


def self_refine(case: dict, draft: str, model: str) -> str:
    """Control: one generic improvement pass, no structured audit."""
    user = _case_block(case, draft) + "\nProduce the improved response."
    return llm.chat(system=SELF_REFINE_SYSTEM, user=user, model=model)
