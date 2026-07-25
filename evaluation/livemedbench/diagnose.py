"""Diagnose how the audit->revise wrapper changes the rubric score, and localize where any
change comes from.

Pure-data analysis (no LLM calls): align each response's per-criterion `met` array against
the baseline and classify every criterion transition. This localizes the bottleneck:

  - If positive criteria GAINED >> LOST and negatives stay flat -> the wrapper helps; the flat
    net score is just noise / small n.
  - If GAINED ~= LOST -> the wrapper CHURNS (fixes some criteria, breaks others) -> net zero.
  - If GAINED is tiny -> the audit isn't surfacing the missed criteria (coverage bottleneck) OR
    the revision can't produce them (capability bottleneck). The audit gap-count vs score-delta
    correlation separates those: many gaps but no gain = revision can't land them.

    ./venv/bin/python -m evaluation.livemedbench.diagnose --run-id lmb_50
"""
from __future__ import annotations

import argparse
import json

from evaluation._util import run_dir


def transitions(cases, base_met, cond_met):
    """Aggregate per-criterion transitions of cond vs baseline over all (case, model)."""
    pos = {"gained": 0, "lost": 0, "stable_hit": 0, "stable_miss": 0}
    neg = {"triggered": 0, "fixed": 0, "stable_ok": 0, "stable_bad": 0}
    pts = {"gained": 0.0, "lost": 0.0, "neg_triggered": 0.0, "neg_fixed": 0.0}
    per_case_delta = []
    for key, cmet in cond_met.items():
        if key not in base_met:
            continue
        cid = key[0]
        items = cases[cid]["rubric_items"]
        bmet = base_met[key]
        d = 0.0
        for i, it in enumerate(items):
            p = float(it["points"])
            b = bool(bmet[i]) if i < len(bmet) else False
            c = bool(cmet[i]) if i < len(cmet) else False
            if p > 0:
                if not b and c:
                    pos["gained"] += 1; pts["gained"] += p; d += p
                elif b and not c:
                    pos["lost"] += 1; pts["lost"] += p; d -= p
                elif b and c:
                    pos["stable_hit"] += 1
                else:
                    pos["stable_miss"] += 1
            elif p < 0:
                if not b and c:
                    neg["triggered"] += 1; pts["neg_triggered"] += p; d += p  # p<0 -> subtracts
                elif b and not c:
                    neg["fixed"] += 1; pts["neg_fixed"] += -p; d += -p
                elif b and c:
                    neg["stable_bad"] += 1
                else:
                    neg["stable_ok"] += 1
        per_case_delta.append((key, d))
    return pos, neg, pts, per_case_delta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="lmb_50")
    args = ap.parse_args()
    rd = run_dir(args.run_id)

    cases = {c["case_id"]: c for c in json.loads((rd / "manifest.json").read_text())["cases"]}
    baseline = json.loads((rd / "baseline_results.json").read_text())
    phase1 = json.loads((rd / "phase1_results.json").read_text())

    base_met = {(b["case_id"], b["model"]): b["met"] for b in baseline}
    base_score = {(b["case_id"], b["model"]): b["score"] for b in baseline}
    sr_met = {(p["case_id"], p["model"]): p["self_refine"]["met"] for p in phase1}
    hg_met = {(p["case_id"], p["model"]): p["healthguard"]["met"] for p in phase1}
    audit = {(p["case_id"], p["model"]): p["healthguard"]["audit"] for p in phase1}
    hg_score = {(p["case_id"], p["model"]): p["healthguard"]["score"] for p in phase1}

    for label, cond_met in [("self_refine", sr_met), ("healthguard", hg_met)]:
        pos, neg, pts, deltas = transitions(cases, base_met, cond_met)
        n = len(deltas)
        net_pts = pts["gained"] - pts["lost"] + pts["neg_fixed"] + pts["neg_triggered"]
        print(f"\n=== {label} vs baseline  (n={n} case*model) ===")
        print(f"  POSITIVE criteria: gained {pos['gained']}  lost {pos['lost']}  "
              f"stable-hit {pos['stable_hit']}  stable-miss {pos['stable_miss']}")
        print(f"    points: +{pts['gained']:.0f} gained  -{pts['lost']:.0f} lost")
        print(f"  NEGATIVE criteria: newly-triggered {neg['triggered']}  fixed {neg['fixed']}  "
              f"stable-ok {neg['stable_ok']}  stable-bad {neg['stable_bad']}")
        print(f"    points: {pts['neg_triggered']:.0f} from new hallucinations  +{pts['neg_fixed']:.0f} fixed")
        changed = pos["gained"] + pos["lost"] + neg["triggered"] + neg["fixed"]
        total = sum(pos.values()) + sum(neg.values())
        print(f"  CHURN: {changed}/{total} criteria changed ({100*changed/total:.0f}%); "
              f"net point change {net_pts:+.0f}")

    # audit efficacy: do more gaps => more score gain?
    print(f"\n=== HealthGuard audit efficacy ===")
    rows = []
    for k, a in audit.items():
        if k in base_score:
            rows.append((a.get("n_gaps", 0), a.get("n_flagged", 0), hg_score[k] - base_score[k]))
    ng = [r[0] for r in rows]
    gained_cases = sum(1 for r in rows if r[2] > 0.01)
    hurt_cases = sum(1 for r in rows if r[2] < -0.01)
    flat_cases = sum(1 for r in rows if abs(r[2]) <= 0.01)
    print(f"  mean gaps/case {sum(ng)/len(ng):.1f} (range {min(ng)}-{max(ng)}); "
          f"mean flagged claims/case {sum(r[1] for r in rows)/len(rows):.1f}")
    print(f"  cases improved {gained_cases}  hurt {hurt_cases}  unchanged {flat_cases}  (of {len(rows)})")
    with_gaps = [r for r in rows if r[0] > 0]
    no_gaps = [r for r in rows if r[0] == 0]
    if with_gaps:
        print(f"  when audit FOUND gaps (n={len(with_gaps)}): mean score delta "
              f"{100*sum(r[2] for r in with_gaps)/len(with_gaps):+.1f}")
    if no_gaps:
        print(f"  when audit found NO gaps (n={len(no_gaps)}): mean score delta "
              f"{100*sum(r[2] for r in no_gaps)/len(no_gaps):+.1f}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
