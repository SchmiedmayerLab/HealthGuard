"""Cross-family grader integrity check: re-grade every condition with a Claude grader.

The primary numbers use GPT-4.1 as the rubric grader (as in the paper). A lift could
be an artifact of gaming GPT-4.1's grading style (e.g. rewarding longer answers). Here we
re-grade the SAME stored responses (baseline / self_refine / healthguard) with a different
vendor family (claude-sonnet-4-6, which accepts temperature=0) and check whether the deltas
survive. If self_refine's lift shrinks or flips under the Claude grader, it was partly gaming.

    ./venv/bin/python -m evaluation.livemedbench.cross_grade
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from evaluation._util import load_env, run_dir

load_env()

from evaluation.livemedbench.grade import grade  # noqa: E402

CLAUDE_GRADER = "claude-sonnet-4-6"   # cross-vendor; accepts temperature=0 (unlike sonnet-5 / opus-4.8)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="lmb_pilot")
    ap.add_argument("--grader", default=CLAUDE_GRADER)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    rd = run_dir(args.run_id)
    cases = {c["case_id"]: c for c in json.loads((rd / "manifest.json").read_text())["cases"]}
    baseline = json.loads((rd / "baseline_results.json").read_text())
    phase1 = json.loads((rd / "phase1_results.json").read_text())

    # assemble (case_id, model, condition, response), keeping the GPT-4.1 score for comparison
    items = []
    for b in baseline:
        items.append((b["case_id"], b["model"], "baseline", b["response"], b["score"]))
    for p in phase1:
        for cond in ("self_refine", "healthguard"):
            items.append((p["case_id"], p["model"], cond, p[cond]["response"], p[cond]["score"]))

    print(f"cross-grading {len(items)} responses with {args.grader} ...")

    def rj(it):
        cid, model, cond, resp, g4 = it
        cg = grade(cases[cid], resp, model=args.grader)
        return {"case_id": cid, "model": model, "condition": cond,
                "gpt41_score": g4, "claude_score": cg["score"]}

    out = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(rj, it) for it in items]
        for n, fut in enumerate(as_completed(futs), 1):
            try:
                out.append(fut.result())
            except Exception as exc:  # noqa: BLE001
                print(f"  ! {exc}")
            if n % 15 == 0:
                print(f"  {n}/{len(items)}")

    (rd / "cross_grade.json").write_text(json.dumps(out, indent=2))

    models = sorted({r["model"] for r in out})
    conds = ["baseline", "self_refine", "healthguard"]
    print(f"\n=== GPT-4.1 vs Claude grader (mean rubric %, n per cell) ===")
    print(f"{'model':>13} | {'condition':>12} | {'GPT-4.1':>8} | {'Claude':>8}")
    summary = {}
    for m in models:
        base_g4 = base_cl = None
        for cond in conds:
            rows = [r for r in out if r["model"] == m and r["condition"] == cond]
            if not rows:
                continue
            g4 = 100 * sum(r["gpt41_score"] for r in rows) / len(rows)
            cl = 100 * sum(r["claude_score"] for r in rows) / len(rows)
            if cond == "baseline":
                base_g4, base_cl = g4, cl
                d4 = dcl = 0.0
            else:
                d4, dcl = g4 - base_g4, cl - base_cl
            summary[f"{m}/{cond}"] = {"gpt41": round(g4, 1), "claude": round(cl, 1),
                                      "delta_gpt41": round(d4, 1), "delta_claude": round(dcl, 1)}
            tag = "" if cond == "baseline" else f"  (Δ {d4:+.1f} / {dcl:+.1f})"
            print(f"{m:>13} | {cond:>12} | {g4:7.1f} | {cl:7.1f}{tag}")

    (rd / "cross_grade_summary.json").write_text(json.dumps(summary, indent=2))
    # agreement: Pearson-free directional check
    both = [(r["gpt41_score"], r["claude_score"]) for r in out]
    import statistics
    g4v = [a for a, _ in both]; clv = [b for _, b in both]
    print(f"\n  per-response grader means: GPT-4.1 {100*statistics.mean(g4v):.1f}%  "
          f"Claude {100*statistics.mean(clv):.1f}%  (n={len(both)})")
    print(f"  -> {rd/'cross_grade_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
