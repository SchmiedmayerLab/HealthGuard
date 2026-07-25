"""HealthGuard audit->revise wrapper: does it push rubric scores beyond the bare model?

Per case x base-model, starting from the SAME baseline draft (so structure is the only variable):
  - baseline    : the bare draft            (loaded from the baseline run)
  - self_refine : one generic improve pass  (control)
  - healthguard : structured audit -> revise
All graded by the GPT-4.1 rubric grader (the auditor never sees the rubric).

    ./venv/bin/python -m evaluation.livemedbench.run_healthguard
    ./venv/bin/python -m evaluation.livemedbench.run_healthguard --models gpt-5.4-mini --limit 2
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from evaluation._util import load_env, run_dir

load_env()

from evaluation.livemedbench.grade import GRADER, grade  # noqa: E402
from healthguard import core as hg  # noqa: E402

MODELS = ["gpt-5.4", "gpt-5.4-mini"]


def process(case: dict, model: str, draft: str, grader: str, coverage: str, revise_mode: str,
            safety: bool) -> dict:
    sr_text = hg.self_refine(case, draft, model)
    hgr = hg.healthguard_response(case, draft, model, coverage=coverage, revise_mode=revise_mode,
                                  safety=safety)
    return {
        "case_id": case["case_id"], "model": model,
        "self_refine": {"response": sr_text, **grade(case, sr_text, model=grader)},
        "healthguard": {"response": hgr["response"], "audit": hgr["audit"],
                        "critique": hgr["critique"], **grade(case, hgr["response"], model=grader)},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="lmb_pilot")
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--grader", default=GRADER)
    ap.add_argument("--coverage", default="base", choices=["base", "lenses", "workup", "full"],
                    help="lenses=+management/prognosis; workup=+workup+strict grounding; full=+diagnosis/mechanism/monitoring/patient lens")
    ap.add_argument("--revise", default="single", choices=["single", "loop"],
                    help="loop = verify-and-repair: re-check unlanded gaps, targeted second revise pass")
    ap.add_argument("--safety", action="store_true",
                    help="final safety pass: scan the response for incorrect/unsafe statements and correct them")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    rd = run_dir(args.run_id)
    cases = json.loads((rd / "manifest.json").read_text(encoding="utf-8"))["cases"]
    cases = cases[: args.limit] if args.limit else cases
    baseline = json.loads((rd / "baseline_results.json").read_text(encoding="utf-8"))
    draft_of = {(b["case_id"], b["model"]): b for b in baseline}

    jobs = [(c, m) for m in args.models for c in cases
            if (c["case_id"], m) in draft_of]
    print(f"phase1: {len(cases)} cases x {len(args.models)} models = {len(jobs)} "
          f"(self_refine + healthguard, grader={args.grader})")

    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(process, c, m, draft_of[(c["case_id"], m)]["response"],
                            args.grader, args.coverage, args.revise, args.safety):
                (c["case_id"], m) for c, m in jobs}
        for n, fut in enumerate(as_completed(futs), 1):
            cid, m = futs[fut]
            try:
                results.append(fut.result())
            except Exception as exc:  # noqa: BLE001
                print(f"  ! case {cid} / {m} failed: {exc}")
            if n % 5 == 0:
                print(f"  {n}/{len(jobs)}")

    (rd / "phase1_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    # summary: mean per condition per model, with delta vs baseline
    print("\n=== PHASE 1 rubric scores (mean over cases, %) ===")
    summary = {}
    for m in args.models:
        rows = [r for r in results if r["model"] == m]
        if not rows:
            continue
        base = {b["case_id"]: b["score"] for b in baseline if b["model"] == m}
        n = len(rows)
        base_mean = 100 * sum(base[r["case_id"]] for r in rows) / n
        sr_mean = 100 * sum(r["self_refine"]["score"] for r in rows) / n
        hg_mean = 100 * sum(r["healthguard"]["score"] for r in rows) / n
        summary[m] = {"n": n, "baseline_pct": round(base_mean, 2),
                      "self_refine_pct": round(sr_mean, 2), "healthguard_pct": round(hg_mean, 2),
                      "delta_selfrefine": round(sr_mean - base_mean, 2),
                      "delta_healthguard": round(hg_mean - base_mean, 2),
                      "delta_hg_vs_sr": round(hg_mean - sr_mean, 2)}
        print(f"  {m:14} n={n:2}  baseline={base_mean:5.1f}  self_refine={sr_mean:5.1f} "
              f"({sr_mean-base_mean:+.1f})  healthguard={hg_mean:5.1f} ({hg_mean-base_mean:+.1f})")

    (rd / "phase1_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n{'case_id':>8} | {'model':>13} | base | self | hlth")
    for m in args.models:
        base = {b["case_id"]: b["score"] for b in baseline if b["model"] == m}
        for r in sorted([r for r in results if r["model"] == m], key=lambda r: r["case_id"]):
            cid = r["case_id"]
            print(f"{cid:>8} | {m:>13} | {base[cid]*100:4.0f} | "
                  f"{r['self_refine']['score']*100:4.0f} | {r['healthguard']['score']*100:4.0f}")
    print(f"\n-> {rd/'phase1_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
