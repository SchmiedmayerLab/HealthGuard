"""Baseline: bare gpt-5.4 and gpt-5.4-mini on the pilot cases.

For each case x model: generate a clinical response (zero-shot) and grade it with the
GPT-4.1 rubric grader. Writes per-case detail + responses and prints the mean rubric
score per model (comparable to the LiveMedBench leaderboard, which reports %).

    ./venv/bin/python -m evaluation.livemedbench.run_baseline
    ./venv/bin/python -m evaluation.livemedbench.run_baseline --models gpt-5.4-mini --limit 2
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from evaluation._util import load_env, run_dir

load_env()

from evaluation.livemedbench.grade import GRADER, grade  # noqa: E402
from evaluation.livemedbench.respond import respond  # noqa: E402

MODELS = ["gpt-5.4", "gpt-5.4-mini"]


def one(case: dict, model: str, grader: str) -> dict:
    text = respond(case, model=model)
    g = grade(case, text, model=grader)
    return {"case_id": case["case_id"], "model": model, "response": text, **g}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="lmb_pilot")
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--grader", default=GRADER)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    rd = run_dir(args.run_id)
    manifest = json.loads((rd / "manifest.json").read_text(encoding="utf-8"))
    cases = manifest["cases"][: args.limit] if args.limit else manifest["cases"]

    jobs = [(c, m) for m in args.models for c in cases]
    print(f"grading {len(cases)} cases x {len(args.models)} models = {len(jobs)} responses "
          f"(grader={args.grader})")

    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(one, c, m, args.grader): (c["case_id"], m) for c, m in jobs}
        for n, fut in enumerate(as_completed(futs), 1):
            cid, m = futs[fut]
            try:
                results.append(fut.result())
            except Exception as exc:  # noqa: BLE001
                print(f"  ! case {cid} / {m} failed: {exc}")
            if n % 5 == 0:
                print(f"  {n}/{len(jobs)}")

    (rd / "baseline_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== BASELINE rubric scores (mean over cases, %) ===")
    summary = {}
    for m in args.models:
        rows = [r for r in results if r["model"] == m]
        if not rows:
            continue
        mean = 100 * sum(r["score"] for r in rows) / len(rows)
        clipped = 100 * sum(max(0.0, r["score"]) for r in rows) / len(rows)
        summary[m] = {"n": len(rows), "mean_pct": round(mean, 2), "mean_pct_clip0": round(clipped, 2),
                      "per_case": {r["case_id"]: r["score"] for r in sorted(rows, key=lambda r: r["case_id"])}}
        print(f"  {m:14} n={len(rows):2}  mean={mean:5.1f}%   (clip0 {clipped:5.1f}%)")

    (rd / "baseline_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n{'case_id':>8} | " + " | ".join(f"{m:>13}" for m in args.models))
    for c in cases:
        cid = c["case_id"]
        cells = []
        for m in args.models:
            r = next((x for x in results if x["case_id"] == cid and x["model"] == m), None)
            cells.append(f"{r['score']*100:6.1f}% ({r['n_positive_met']}/{sum(1 for it in c['rubric_items'] if it['points']>0)})"
                         if r else "   --   ")
        print(f"{cid:>8} | " + " | ".join(f"{c:>13}" for c in cells))
    print(f"\n-> {rd/'baseline_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
