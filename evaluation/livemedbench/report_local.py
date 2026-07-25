"""Report absolute baseline / self-refine / HealthGuard rubric means (+ paired lift with
bootstrap CI) for a single run under both graders. Used for the local open-weights model
runs, where the base model's absolute score matters (not just the lift) to place it against
the frontier models on a capability-vs-lift plot.

    ./venv/bin/python -m evaluation.livemedbench.report_local lmb_local
"""
from __future__ import annotations

import json
import sys

import numpy as np
from scipy import stats

from evaluation._util import run_dir


def boot(d, n=5000):
    d = np.array(d, float)
    rng = np.random.default_rng(0)
    m = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return d.mean() * 100, np.percentile(m, 2.5) * 100, np.percentile(m, 97.5) * 100


def load(rid: str) -> tuple[list[dict], list[str]]:
    rd = run_dir(rid)
    base = {(b["case_id"], b["model"]): b["score"] for b in json.loads((rd / "baseline_results.json").read_text())}
    p1 = json.loads((rd / "phase1_results.json").read_text())
    cg_path = rd / "cross_grade.json"
    cg = {}
    if cg_path.exists():
        cg = {(r["case_id"], r["model"], r["condition"]): r["claude_score"]
              for r in json.loads(cg_path.read_text())}
    rows = []
    for p in p1:
        cid, m = p["case_id"], p["model"]
        rows.append({
            "model": m,
            "base_g4": base[(cid, m)],
            "sr_g4": p["self_refine"]["score"],
            "hg_g4": p["healthguard"]["score"],
            "base_cl": cg.get((cid, m, "baseline")),
            "sr_cl": cg.get((cid, m, "self_refine")),
            "hg_cl": cg.get((cid, m, "healthguard")),
        })
    models = sorted({r["model"] for r in rows})
    return rows, models


def main() -> int:
    rid = sys.argv[1] if len(sys.argv) > 1 else "lmb_local"
    rows, models = load(rid)
    print(f"run={rid}  n={len(rows)} case*model  models={models}\n")
    for m in models:
        rm = [r for r in rows if r["model"] == m]
        print(f"model = {m}   (n={len(rm)})")
        for gname, bk, sk, hk in [("GPT-4.1", "base_g4", "sr_g4", "hg_g4"),
                                  ("Claude", "base_cl", "sr_cl", "hg_cl")]:
            b = [r[bk] for r in rm if r[bk] is not None]
            h = [(r[hk], r[bk]) for r in rm if r[hk] is not None and r[bk] is not None]
            s = [(r[sk], r[bk]) for r in rm if r[sk] is not None and r[bk] is not None]
            if not b:
                continue
            base_m = 100 * np.mean(b)
            hg_d = [x - y for x, y in h]
            sr_d = [x - y for x, y in s]
            hg_mean = 100 * np.mean([x for x, _ in h])
            sr_mean = 100 * np.mean([x for x, _ in s]) if s else float("nan")
            lift, lo, hi = boot(hg_d)
            up = sum(1 for x in hg_d if x > 0.005); dn = sum(1 for x in hg_d if x < -0.005)
            pv = stats.binomtest(up, up + dn, 0.5).pvalue if up + dn else 1.0
            srl = 100 * np.mean(sr_d) if sr_d else float("nan")
            print(f"  {gname:7}  base={base_m:5.1f}  self_refine={sr_mean:5.1f} ({srl:+.1f})  "
                  f"healthguard={hg_mean:5.1f}  lift={lift:+.1f} [{lo:+.1f},{hi:+.1f}]  "
                  f"up/dn={up}/{dn}  sign_p={pv:.4f}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
