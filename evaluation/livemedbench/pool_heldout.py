"""Pool one or more held-out runs (all produced with the same config: lenses coverage + loop revise)
into a single HealthGuard-vs-baseline estimate, to track whether the lift holds and the CI tightens
as n grows. Each run needs baseline_results.json + phase1_results.json + cross_grade.json.

    ./venv/bin/python -m evaluation.livemedbench.pool_heldout lmb_50c lmb_50d
"""
from __future__ import annotations

import json
import sys

import numpy as np
from scipy import stats

from evaluation._util import run_dir


def load(rid: str) -> list[dict]:
    rd = run_dir(rid)
    base = {(b["case_id"], b["model"]): b["score"] for b in json.loads((rd / "baseline_results.json").read_text())}
    cg = {(r["case_id"], r["model"], r["condition"]): r["claude_score"]
          for r in json.loads((rd / "cross_grade.json").read_text())}
    p1 = json.loads((rd / "phase1_results.json").read_text())
    rows = []
    for p in p1:
        k = (p["case_id"], p["model"])
        hk, bk = (*k, "healthguard"), (*k, "baseline")
        base_g4, hg_g4 = base[k], p["healthguard"]["score"]
        base_cl, hg_cl = cg.get(bk), cg.get(hk)
        cl = (hg_cl - base_cl) if (base_cl is not None and hg_cl is not None) else None
        rows.append({"run": rid, "model": p["model"], "g4": hg_g4 - base_g4, "cl": cl,
                     "base_g4": base_g4, "hg_g4": hg_g4, "base_cl": base_cl, "hg_cl": hg_cl})
    return rows


def boot(d, n=5000):
    d = np.array(d, float)
    rng = np.random.default_rng(0)
    m = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return d.mean() * 100, np.percentile(m, 2.5) * 100, np.percentile(m, 97.5) * 100


def main() -> int:
    rids = sys.argv[1:] or ["lmb_50c"]
    rows = [r for rid in rids for r in load(rid)]
    print(f"pooled held-out (final config) over {rids} = {len(rows)} case*model\n")
    print(f"  {'group':8} {'grader':7} {'n':>4}  {'lift':>7}  {'95% CI':>16}  {'up/dn':>7}  {'sign p':>8}")
    for lab, pred in [("POOLED", lambda r: True),
                      ("gpt-5.4", lambda r: r["model"] == "gpt-5.4"),
                      ("mini", lambda r: r["model"] == "gpt-5.4-mini")]:
        for key, nm in [("g4", "GPT-4.1"), ("cl", "Claude")]:
            d = [r[key] for r in rows if pred(r) and r[key] is not None]
            if not d:
                continue
            m, lo, hi = boot(d)
            up = sum(1 for x in d if x > 0.005); dn = sum(1 for x in d if x < -0.005)
            pv = stats.binomtest(up, up + dn, 0.5).pvalue if up + dn else 1.0
            print(f"  {lab:8} {nm:7} {len(d):>4}  {m:+6.1f}   [{lo:+5.1f},{hi:+5.1f}]  {up:>3}/{dn:<3}  {pv:>8.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
