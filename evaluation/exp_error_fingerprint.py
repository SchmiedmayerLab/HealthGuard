"""Error fingerprint: how do capable models fail to reason like a doctor?

A1 (LiveMedBench, held-out base drafts): the base model's OMISSIONS (positive rubric criteria it
   missed, by clinical type) and COMMISSIONS (negative criteria it triggered = harmful/incorrect
   statements, by harm type). Uses the stored GPT-4.1 per-criterion met arrays.
A2 (MedCaseReasoning, wrong-conclusion traces): a taxonomy of the diagnostic REASONING errors that
   led capable models to the wrong diagnosis.

Models are capable but skip elements of clinical reasoning. Analyst = gpt-5.4-mini.

    ./venv/bin/python -m evaluation.exp_error_fingerprint
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from evaluation._util import load_env, run_dir

load_env()

from healthguard import llm  # noqa: E402
from healthguard.context import PipelineContext  # noqa: E402

MODEL = "gpt-5.4-mini"
HELDOUT = ["lmb_50c", "lmb_50d", "lmb_50e"]

OMIT_TYPES = ("diagnosis", "mechanism", "management", "workup", "red_flag_referral",
              "prognosis", "monitoring", "patient_factor", "communication", "other")
HARM_TYPES = ("unsafe_recommendation", "incorrect_fact", "false_reassurance",
              "overstated_certainty", "contradicts_case", "other")

A1_SYS = f"""\
You classify rubric criteria that a clinical answer failed on. You get OMISSIONS (points the answer \
should have covered but omitted) and COMMISSIONS (harmful/incorrect statements the answer made). \
Classify each OMISSION into one clinical type: {", ".join(OMIT_TYPES)}. Classify each COMMISSION into \
one harm type: {", ".join(HARM_TYPES)}. Return ONLY JSON: \
{{"omissions": ["<type>", ...], "commissions": ["<harm>", ...]}} — one entry per item, in order."""

REASON_ERRORS = ("premature_closure", "ignored_key_finding", "misweighted_evidence",
                 "anchoring", "incomplete_differential", "faulty_inference",
                 "insufficient_information", "other")

A2_SYS = f"""\
You analyze a clinical reasoning trace that reached the WRONG final diagnosis. Given the reasoning, the \
model's (wrong) diagnosis, and the correct diagnosis, name the SINGLE primary diagnostic-reasoning \
error, one of: {", ".join(REASON_ERRORS)}.
- premature_closure: settled on a diagnosis too early without considering alternatives.
- ignored_key_finding: overlooked/dismissed a finding that pointed elsewhere.
- misweighted_evidence: over/under-weighted findings.
- anchoring: fixated on an initial impression.
- incomplete_differential: never considered the correct diagnosis.
- faulty_inference: a logically flawed deduction from the findings.
- insufficient_information: the case genuinely under-determines the answer.
Return ONLY JSON: {{"error": "<type>", "note": "<short reason>"}}."""


def _a1_case(rec: dict) -> dict:
    if not rec["omit"] and not rec["comm"]:
        return {"omissions": [], "commissions": []}
    payload = {"omissions": rec["omit"], "commissions": rec["comm"]}
    try:
        r = llm.chat_json(system=A1_SYS, user=json.dumps(payload, ensure_ascii=False), model=MODEL)
    except Exception:  # noqa: BLE001
        r = {}
    om = [t if t in OMIT_TYPES else "other" for t in (r.get("omissions") or [])][: len(rec["omit"])]
    cm = [t if t in HARM_TYPES else "other" for t in (r.get("commissions") or [])][: len(rec["comm"])]
    return {"omissions": om, "commissions": cm}


def a1() -> None:
    recs, n_cm = [], 0
    for rid in HELDOUT:
        rd = run_dir(rid)
        cases = {c["case_id"]: c for c in json.loads((rd / "manifest.json").read_text())["cases"]}
        for b in json.loads((rd / "baseline_results.json").read_text()):
            items = cases[b["case_id"]]["rubric_items"]
            met = b["met"]
            omit = [items[i]["criterion"] for i in range(len(items))
                    if items[i]["points"] > 0 and not (met[i] if i < len(met) else False)]
            comm = [items[i]["criterion"] for i in range(len(items))
                    if items[i]["points"] < 0 and (met[i] if i < len(met) else False)]
            n_cm += len(comm)
            recs.append({"omit": omit, "comm": comm})
    print(f"A1: {len(recs)} base drafts | {sum(len(r['omit']) for r in recs)} omissions, {n_cm} commissions")
    with ThreadPoolExecutor(max_workers=8) as pool:
        for r, out in zip(recs, pool.map(_a1_case, recs)):
            r.update(out)

    n = len(recs)
    om_all = Counter(t for r in recs for t in r["omissions"])
    om_cases = Counter(t for r in recs for t in set(r["omissions"]))   # cases missing >=1 of type
    total_om = sum(om_all.values())
    print(f"\n=== A1 OMISSIONS — what capable models leave OUT (n={n} answers) ===")
    print(f"  {'clinical element':18} {'omitted':>8} {'% of omissions':>15} {'% of answers ≥1':>16}")
    for t, c in om_all.most_common():
        print(f"  {t:18} {c:>8} {100*c/total_om:>14.0f}% {100*om_cases[t]/n:>15.0f}%")
    hm = Counter(t for r in recs for t in r["commissions"])
    any_harm = sum(1 for r in recs if r["commissions"])
    print(f"\n=== A1 COMMISSIONS — harmful/incorrect statements ({any_harm}/{n} = {100*any_harm/n:.0f}% of answers) ===")
    for t, c in hm.most_common():
        print(f"  {t:22} {c:>4}")


def _a2_trace(path: str) -> dict | None:
    env = json.load(open(path))
    inp = env["input"]
    gt = (inp.get("groundTruthDiagnosis") or "").strip()
    wrong = (inp.get("cotDiagnosis", {}).get("finalDiagnosis") or "").strip()
    reasoning = PipelineContext.from_file(path).input_text[:6000]
    if not gt or not wrong:
        return None
    payload = {"reasoning": reasoning, "model_diagnosis": wrong, "correct_diagnosis": gt}
    try:
        r = llm.chat_json(system=A2_SYS, user=json.dumps(payload, ensure_ascii=False), model=MODEL)
    except Exception:  # noqa: BLE001
        return None
    e = r.get("error")
    return {"error": e if e in REASON_ERRORS else "other", "note": r.get("note", "")}


def a2() -> None:
    rows = [r for r in csv.DictReader(open(run_dir("mcr_natwrong") / "manifest.csv"))
            if r["label"] == "flawed" and r["included"].lower() == "true"]
    print(f"\nA2: analysing {len(rows)} wrong-conclusion traces ...")
    out = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futs = [pool.submit(_a2_trace, r["path"]) for r in rows]
        for f in as_completed(futs):
            r = f.result()
            if r:
                out.append(r)
    (run_dir("mcr_natwrong") / "reasoning_errors.json").write_text(json.dumps(out, indent=2))
    c = Counter(r["error"] for r in out)
    n = len(out)
    print(f"\n=== A2 REASONING ERRORS — why capable models reach the wrong diagnosis (n={n}) ===")
    for t, k in c.most_common():
        print(f"  {t:24} {k:>4}  {100*k/n:>4.0f}%")


def main() -> int:
    a1()
    a2()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
