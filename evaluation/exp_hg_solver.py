"""HealthGuard as a diagnostic solver on MedCaseReasoning: does the verification panel raise accuracy?

HealthGuard is a verifier, not a solver; but its v4 conclusion-plausibility panel independently
RE-DERIVES the diagnosis from the case evidence. Used as a solver, its answer = the panel consensus.
This measures whether that panel beats a single bare diagnosis (i.e. whether HealthGuard adds
diagnostic ACCURACY, the metric the MedCaseReasoning paper reports at ~38-65%). Prediction from the
judgment-ceiling finding: panel ≈ base (self-consistency only), no HealthGuard-specific lift.

Both conditions use the SAME model on the SAME case evidence (rawInput, no diagnosis leak); scored
against ground truth by the pipeline judge (match / related / mismatch).

    ./venv/bin/python -m evaluation.exp_hg_solver [--n 80] [--k 5]
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from evaluation._util import REPO_ROOT, load_env, run_dir

load_env()

from healthguard import judge, llm  # noqa: E402
from healthguard.context import PipelineContext  # noqa: E402

MODEL = "gpt-5.4-mini"
RATIONALE_DIR = REPO_ROOT / "testdata/streamlined/medcasereasoning/rationale"

DIAG_SYS = """\
You are an expert clinician. Given the patient case data (presentation, history, exam, investigations), \
give the single most likely FINAL diagnosis. Be specific. Return ONLY JSON: {"diagnosis": "<diagnosis>"}."""

AGG_SYS = """\
You are an expert clinician. Given the patient case data and several independent expert diagnoses for \
it, decide the single most likely FINAL diagnosis — the best-supported consensus (not necessarily the \
most frequent). Return ONLY JSON: {"diagnosis": "<diagnosis>"}."""


def _diagnose(case_ref: dict, temp: float) -> str:
    try:
        return (llm.chat_json(system=DIAG_SYS, user=json.dumps(case_ref, ensure_ascii=False)[:6000],
                              model=MODEL, temperature=temp).get("diagnosis") or "").strip()
    except Exception:  # noqa: BLE001
        return ""


def _panel(case_ref: dict, k: int) -> str:
    dxs = [_diagnose(case_ref, 0.8) for _ in range(k)]
    dxs = [d for d in dxs if d]
    if not dxs:
        return ""
    payload = {"case": json.dumps(case_ref, ensure_ascii=False)[:5000], "expert_diagnoses": dxs}
    try:
        return (llm.chat_json(system=AGG_SYS, user=json.dumps(payload, ensure_ascii=False),
                              model=MODEL).get("diagnosis") or dxs[0]).strip()
    except Exception:  # noqa: BLE001
        return dxs[0]


def _correct(gt: str, dx: str) -> tuple[bool, bool]:
    """(match, match-or-related) vs ground truth, via the pipeline judge."""
    if not dx:
        return False, False
    v = judge.judge_diagnosis(ground_truth=gt, candidate=dx, differentials=[])
    verdict = getattr(v, "verdict", None) or (v.get("verdict") if isinstance(v, dict) else None) or "mismatch"
    return verdict == "match", verdict in ("match", "related")


def one(path, k: int) -> dict | None:
    env = json.load(open(path))
    gt = (env["input"].get("groundTruthDiagnosis") or "").strip()
    if not gt:
        return None
    case_ref = PipelineContext.from_file(path).additional_input
    base = _diagnose(case_ref, 0.0)
    pan = _panel(case_ref, k)
    bm, br = _correct(gt, base)
    pm, pr = _correct(gt, pan)
    return {"gt": gt, "base": base, "panel": pan,
            "base_match": bm, "base_rel": br, "panel_match": pm, "panel_rel": pr}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    files = sorted(RATIONALE_DIR.glob("*.json"))[: args.n]
    print(f"HealthGuard-as-solver: {len(files)} representative cases, panel K={args.k}, model={MODEL}")
    out = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(one, p, args.k) for p in files]
        for n, f in enumerate(as_completed(futs), 1):
            try:
                r = f.result()
            except Exception as exc:  # noqa: BLE001
                print(f"  ! {exc}"); r = None
            if r:
                out.append(r)
            if n % 20 == 0:
                print(f"  {n}/{len(files)}")

    (run_dir("mcr_natwrong") / "hg_solver.json").write_text(json.dumps(out, indent=2))
    n = len(out)

    def pct(key):
        return 100 * sum(r[key] for r in out) / n

    print(f"\n=== Diagnostic accuracy on MedCaseReasoning (n={n}, model {MODEL}) ===")
    print(f"  {'':22} {'strict match':>13} {'match+related':>14}")
    print(f"  {'base (single-shot)':22} {pct('base_match'):>12.1f}% {pct('base_rel'):>13.1f}%")
    print(f"  {'HealthGuard panel':22} {pct('panel_match'):>12.1f}% {pct('panel_rel'):>13.1f}%")
    print(f"  {'delta (panel - base)':22} {pct('panel_match')-pct('base_match'):>+12.1f}pp {pct('panel_rel')-pct('base_rel'):>+12.1f}pp")
    print(f"\n  context: reported diagnostic accuracy across frontier models is roughly 38-65%; "
          f"the model used here is not among them, so the internally-valid comparison "
          f"is base vs panel (does HealthGuard lift accuracy?).")
    print(f"  -> {run_dir('mcr_natwrong')/'hg_solver.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
