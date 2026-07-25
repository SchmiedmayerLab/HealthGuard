"""The flexible verifier. See the package docstring for an overview."""
from __future__ import annotations

from . import core as hg

# Soundness weight per grounding verdict (higher => more sound). Tunable; used for the flag/detection
# output. "incorrect" (a factual/safety error) counts fully against soundness; "unsupported"
# (ungrounded but not necessarily wrong) counts partially.
WEIGHTS = {"supported": 1.0, "unsupported": 0.35, "incorrect": 0.0}


def ground(case: dict, output: str, model: str | None = None, strict: bool = True) -> list[dict]:
    """Decompose OUTPUT into atomic claims and verify each against the case. Returns all claims
    with verdicts (supported/unsupported/incorrect)."""
    block = hg._case_block(case, output)
    sysp = hg.GROUNDING_SYSTEM_STRICT if strict else hg.GROUNDING_SYSTEM
    res = hg._json(sysp, block, model)
    return [c for c in (res.get("claims") or []) if c.get("text")]


def soundness(claims: list[dict], weights: dict | None = None) -> float:
    """Aggregate per-claim grounding verdicts into a 0..1 soundness score (higher = sounder)."""
    w = weights or WEIGHTS
    if not claims:
        return 1.0
    return sum(w.get(c.get("verdict"), 0.5) for c in claims) / len(claims)


def verify(case: dict, output: str, model: str | None = None, mode: str = "both",
           coverage: str = "workup", revise_mode: str = "loop",
           checks: tuple = ("grounding", "coverage"), strict: bool = True,
           safety: bool = False, weights: dict | None = None) -> dict:
    """One flexible ground-truth-free verification of `output` against `case`.

    mode:     "flag" (findings + soundness), "revise" (rewrite), or "both".
    checks:   which checks to run: "grounding" (per-claim soundness) and/or "coverage" (gaps).
    coverage: coverage config for the revise path (base/lenses/workup/full), passed to the revise pipeline.
    Everything is a knob so the same object serves detection, escalation, and amplification.
    """
    findings: dict = {}
    claims = ground(case, output, model, strict=strict) if "grounding" in checks else []
    flagged = [c for c in claims if c.get("verdict") in ("unsupported", "incorrect")]
    findings.update(claims=claims, flagged=flagged, n_claims=len(claims), n_flagged=len(flagged),
                    n_incorrect=sum(1 for c in claims if c.get("verdict") == "incorrect"),
                    soundness=soundness(claims, weights))

    if "coverage" in checks and mode in ("revise", "both"):
        a = hg.audit(case, output, model, coverage=coverage)
        findings["gaps"] = a["gaps"]
        findings["essential"] = a["essential"]

    out: dict = {"findings": findings, "mode": mode}
    if mode in ("revise", "both"):
        hr = hg.healthguard_response(case, output, model, coverage=coverage,
                                     revise_mode=revise_mode, safety=safety)
        out["revised"] = hr["response"]
        out["audit"] = hr["audit"]
    if mode in ("flag", "both"):
        out["flags"] = {"unsound_claims": flagged, "coverage_gaps": findings.get("gaps", [])}
        out["soundness"] = findings["soundness"]
    return out
