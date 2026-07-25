"""Help-versus-harm profile of autonomous revision: false-positive and harm analysis.

Part I  (LiveMedBench held-out, lmb_50c/d/e): for the audit->revise wrapper:
  - harm/benefit rates: P(help), P(harm), P(no-change), and the magnitude asymmetry
    (mean gain | helped vs mean loss | harmed) -> net expected value, both graders.
  - regression MECHANISM: when a case regresses, is it over-editing (positive criteria LOST) or
    hallucination (negative criteria NEWLY TRIGGERED)? Uses per-criterion GPT-4.1 met arrays.
Part II (MedCaseReasoning detection, mcr_full2): for the trace verifier (Pillar 1):
  - the clinician-facing FALSE-ALARM rate: at operating points on hg_score, what fraction of SOUND
    traces get flagged flawed (FP), and what fraction of flawed traces slip through (FN)?

Pure data; no new generation.

    ./venv/bin/python -m evaluation.exp_harm_profile
"""
from __future__ import annotations

import csv
import json
import sys

import numpy as np
from sklearn.metrics import roc_curve

from evaluation._util import run_dir

# held-out run-ids (override via CLI args, e.g. the tightened-reviser `*_t` runs)
HELDOUT = [a for a in sys.argv[1:] if not a.startswith("-")] or ["lmb_50c", "lmb_50d", "lmb_50e"]
EPS = 0.005   # score-change deadband (fractions)


# ---------- Part I: LiveMedBench harm/benefit + regression mechanism ----------
def part1() -> None:
    print("=" * 70)
    print("PART I — audit→revise: does it make responses worse? (held-out n=150)")
    print("=" * 70)

    deltas = {"GPT-4.1": [], "Claude": []}
    mech = []   # per (case,model) GPT-4.1 point decomposition
    for rid in HELDOUT:
        rd = run_dir(rid)
        cases = {c["case_id"]: c for c in json.loads((rd / "manifest.json").read_text())["cases"]}
        base = {(b["case_id"], b["model"]): b for b in json.loads((rd / "baseline_results.json").read_text())}
        cg = {(r["case_id"], r["model"], r["condition"]): r["claude_score"]
              for r in json.loads((rd / "cross_grade.json").read_text())}
        for p in json.loads((rd / "phase1_results.json").read_text()):
            k = (p["case_id"], p["model"])
            if k not in base:
                continue
            deltas["GPT-4.1"].append(p["healthguard"]["score"] - base[k]["score"])
            bk, hk = (*k, "baseline"), (*k, "healthguard")
            if bk in cg and hk in cg:
                deltas["Claude"].append(cg[hk] - cg[bk])
            # per-criterion mechanism (GPT-4.1 met arrays)
            items = cases[p["case_id"]]["rubric_items"]
            bmet, hmet = base[k]["met"], p["healthguard"]["met"]
            maxpos = sum(it["points"] for it in items if it["points"] > 0) or 1
            d = {"pos_gain": 0.0, "pos_loss": 0.0, "neg_new": 0.0, "neg_fix": 0.0}
            for i, it in enumerate(items):
                pts = it["points"]
                bb = 1 if (i < len(bmet) and bmet[i]) else 0
                hh = 1 if (i < len(hmet) and hmet[i]) else 0
                if hh == bb:
                    continue
                c = pts * (hh - bb) / maxpos
                if pts > 0 and hh > bb:
                    d["pos_gain"] += c
                elif pts > 0 and hh < bb:
                    d["pos_loss"] += -c
                elif pts < 0 and hh > bb:
                    d["neg_new"] += -c        # positive magnitude of harm from a new hallucination
                elif pts < 0 and hh < bb:
                    d["neg_fix"] += c
            d["delta"] = d["pos_gain"] - d["pos_loss"] + d["neg_fix"] - d["neg_new"]
            mech.append(d)

    for g, ds in deltas.items():
        ds = np.array(ds)
        n = len(ds)
        helped = ds > EPS; harmed = ds < -EPS
        gain = 100 * ds[helped].mean() if helped.any() else 0
        loss = 100 * ds[harmed].mean() if harmed.any() else 0
        print(f"\n[{g}] n={n}")
        print(f"  helped {helped.sum():>3} ({100*helped.mean():.0f}%)   "
              f"harmed {harmed.sum():>3} ({100*harmed.mean():.0f}%)   "
              f"unchanged {n-helped.sum()-harmed.sum():>3}")
        print(f"  mean gain | helped: {gain:+.1f} pts   mean loss | harmed: {loss:+.1f} pts   "
              f"net EV: {100*ds.mean():+.1f} pts")

    harmed = [d for d in mech if d["delta"] < -EPS]
    helped = [d for d in mech if d["delta"] > EPS]
    print(f"\n[regression mechanism, GPT-4.1 met, harmed cases n={len(harmed)}]")
    if harmed:
        pl = 100 * np.mean([d["pos_loss"] for d in harmed])
        nn = 100 * np.mean([d["neg_new"] for d in harmed])
        print(f"  per harmed case: {pl:.1f} pts lost to dropped/over-hedged CORRECT content, "
              f"{nn:.1f} pts to NEW hallucinations")
        print(f"  -> harm is {100*pl/(pl+nn):.0f}% over-editing vs {100*nn/(pl+nn):.0f}% hallucination")
    print(f"[benefit, helped cases n={len(helped)}] per helped case: "
          f"{100*np.mean([d['pos_gain'] for d in helped]):.1f} pts gained (new correct content)")
    # hallucination base rate overall
    any_new = sum(1 for d in mech if d["neg_new"] > 0)
    print(f"[hallucination base rate] cases that triggered ≥1 NEW negative criterion: "
          f"{any_new}/{len(mech)} ({100*any_new/len(mech):.0f}%)")


# ---------- Part II: verifier false-alarm operating point ----------
def part2() -> None:
    print("\n" + "=" * 70)
    print("PART II — trace verifier (Pillar 1): false-alarm rate at operating points")
    print("=" * 70)
    rows = [r for r in csv.DictReader(open(run_dir("mcr_full2") / "results.csv"))
            if r.get("hg_score") and r["label"] in ("sound", "flawed")]
    y = np.array([1 if r["label"] == "flawed" else 0 for r in rows])        # positive = flawed
    score = np.array([1 - float(r["hg_score"]) for r in rows])              # higher -> more flawed
    n_s, n_f = int((y == 0).sum()), int((y == 1).sum())
    fpr, tpr, thr = roc_curve(y, score)
    youden = int(np.argmax(tpr - fpr))
    print(f"  n = {n_s} sound + {n_f} flawed")
    print(f"  {'operating point':28} {'sensitivity':>11} {'FP rate (sound flagged)':>24} {'FN rate':>8}")

    def report(name, i):
        print(f"  {name:28} {tpr[i]:>10.0%} {fpr[i]:>23.0%} {1-tpr[i]:>8.0%}")

    report("Youden-J (balanced)", youden)
    for target in (0.80, 0.90, 0.95):   # threshold that catches ≥target of flawed traces
        i = int(np.argmax(tpr >= target))
        report(f"catch {target:.0%} of flawed", i)
    # high-precision end: threshold with FP <= 10%
    lowfp = np.where(fpr <= 0.10)[0]
    if len(lowfp):
        report("FP ≤ 10% (few false alarms)", int(lowfp[-1]))
    print("  (predictor = 1 − hg_score; a clinician sees the FP-rate column as false alarms on good reasoning)")


def main() -> int:
    part1()
    part2()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
