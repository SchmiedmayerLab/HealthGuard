"""Decompose the rubric-quality lift (HealthGuard minus baseline) by rubric-criterion TYPE, to show
how much of the gain is diagnostic-reasoning vs management/coverage. Answers: is the LiveMedBench
boost about getting the DIAGNOSIS right, or about response completeness?

For each held-out (case, model), every rubric criterion whose met-status CHANGED contributes
  points * (hg_met - base_met) / max_positive_points_of_case
to that case's normalized score lift (so the sum over criteria = the case's lift). We classify each
changed criterion by type and average the per-type contribution across all case*model -> each type's
share of the pooled lift (in rubric points, summing to the total lift). Grading uses the stored
per-criterion GPT-4.1 met arrays (the benchmark's official grader). Classifier = gpt-5.4-mini.

    ./venv/bin/python -m evaluation.livemedbench.decompose_lift lmb_50c lmb_50d lmb_50e
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from evaluation._util import load_env, run_dir

load_env()

from healthguard import llm  # noqa: E402

TYPES = ("diagnosis", "mechanism", "management", "workup", "red_flag_referral",
         "prognosis", "monitoring", "patient_factor", "communication", "other")
DIAGNOSTIC = {"diagnosis", "mechanism"}   # "did it identify the right diagnosis/cause"

SYS = """\
Classify each clinical rubric criterion into exactly one TYPE:
- diagnosis: states or identifies the correct diagnosis / most-likely cause for the patient.
- mechanism: explains the underlying pathophysiology or reason (why).
- management: a specific treatment, drug, procedure, dose, or actionable clinical advice.
- workup: a diagnostic test or investigation to order.
- red_flag_referral: an urgent warning sign or a referral to seek care / a specialist.
- prognosis: expected outcome, course, timeframe, or recurrence.
- monitoring: follow-up, monitoring, or self-management over time.
- patient_factor: a patient-specific consideration (age, comorbidity, pregnancy) that shapes the answer.
- communication: how to communicate / reassure / set expectations.
- other: none of the above.
Return ONLY JSON: {"types": ["<type>", ...]} with one entry per criterion, in order."""


def classify(texts: list[str]) -> list[str]:
    if not texts:
        return []
    payload = {"criteria": [{"index": i, "text": t} for i, t in enumerate(texts)]}
    try:
        out = llm.chat_json(system=SYS, user=json.dumps(payload, ensure_ascii=False), model="gpt-5.4-mini")
        ts = out.get("types", []) or []
    except Exception:  # noqa: BLE001
        ts = []
    ts = [t if t in TYPES else "other" for t in ts][: len(texts)]
    ts += ["other"] * (len(texts) - len(ts))
    return ts


def main() -> int:
    rids = sys.argv[1:] or ["lmb_50c", "lmb_50d", "lmb_50e"]
    records = []   # one per (case, model) with >=1 changed criterion
    for rid in rids:
        rd = run_dir(rid)
        cases = {c["case_id"]: c for c in json.loads((rd / "manifest.json").read_text())["cases"]}
        base = {(b["case_id"], b["model"]): b["met"] for b in json.loads((rd / "baseline_results.json").read_text())}
        for p in json.loads((rd / "phase1_results.json").read_text()):
            k = (p["case_id"], p["model"])
            if k not in base:
                continue
            items = cases[p["case_id"]]["rubric_items"]
            bmet, hmet = base[k], p["healthguard"]["met"]
            maxpos = sum(it["points"] for it in items if it["points"] > 0) or 1
            changed = []
            for i, it in enumerate(items):
                b = 1 if (i < len(bmet) and bmet[i]) else 0
                h = 1 if (i < len(hmet) and hmet[i]) else 0
                if h != b:
                    changed.append({"text": it["criterion"], "contrib": it["points"] * (h - b) / maxpos})
            if changed:
                records.append(changed)

    # total case*model (incl. unchanged) for correct averaging
    total_cm = 0
    for rid in rids:
        total_cm += len(json.loads((run_dir(rid) / "phase1_results.json").read_text()))

    print(f"classifying changed criteria over {len(records)} changed case*model "
          f"({total_cm} total) from {rids} ...")
    with ThreadPoolExecutor(max_workers=8) as pool:
        futs = {pool.submit(classify, [c["text"] for c in rec]): rec for rec in records}
        for f in as_completed(futs):
            for c, t in zip(futs[f], f.result()):
                c["type"] = t

    contrib = defaultdict(float)   # type -> summed contribution (fraction)
    gained = defaultdict(int); lost = defaultdict(int)
    for rec in records:
        for c in rec:
            contrib[c["type"]] += c["contrib"]
            (gained if c["contrib"] > 0 else lost)[c["type"]] += 1
    # mean contribution per case*model, in rubric points (x100)
    total = sum(contrib.values()) / total_cm * 100
    print(f"\n=== Lift decomposition by criterion type (GPT-4.1 grader, held-out n={total_cm//2}) ===")
    print(f"  total lift reconstructed: {total:+.1f} rubric pts\n")
    print(f"  {'type':18} {'contribution':>13}  {'% of lift':>9}   (gained/lost criteria)")
    rows = sorted(contrib.items(), key=lambda kv: -kv[1])
    for t, s in rows:
        pts = s / total_cm * 100
        print(f"  {t:18} {pts:+11.1f}    {100*pts/total:>7.0f}%   ({gained[t]}/{lost[t]})")
    diag = sum(contrib[t] for t in DIAGNOSTIC) / total_cm * 100
    cover = total - diag
    print(f"\n  DIAGNOSTIC reasoning (diagnosis+mechanism): {diag:+.1f} pts  ({100*diag/total:.0f}% of lift)")
    print(f"  COMPLETENESS/coverage (everything else):    {cover:+.1f} pts  ({100*cover/total:.0f}% of lift)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
