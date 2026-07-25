"""Compute-matched localization baseline: run the critic k times, union its flags,
and compare its recall/precision to HealthGuard's.

    ./venv/bin/python -m evaluation.compute_matched_loc --run-id mcr_v2flaw --k 10 --critic-model gpt-5.4-mini --judge-model gpt-5.4
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from evaluation._util import REPO_ROOT, load_env, run_dir

load_env()

from healthguard.context import PipelineContext  # noqa: E402
from evaluation import baseline_critic  # noqa: E402
from evaluation.localization import align  # noqa: E402 (reuse the alignment judge)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("compute_matched")

K_LEVELS = [1, 3, 5, 10]


def dedup(flaws: list[dict]) -> list[dict]:
    seen, out = set(), []
    for f in flaws:
        key = (f.get("claim") or "").strip().lower()
        if key and key not in seen:
            seen.add(key); out.append(f)
    return out


def process(row: dict, k: int, critic_model: str, judge_model: str) -> dict | None:
    env = json.loads(Path(row["path"]).read_text())
    injected = (env.get("input", {}).get("trick", {}) or {}).get("injected_errors", []) or []
    if not injected:
        return None
    ctx = PipelineContext.from_file(row["path"])
    # k diverse critic samples
    runs = []
    for _ in range(k):
        v = baseline_critic.critique(ctx.input_text, ctx.additional_input, model=critic_model, temperature=0.7)
        runs.append(v.get("flaws", []) or [])
    out = {"trace_id": row["trace_id"], "n_errors": len(injected), "by_k": {}}
    for kk in K_LEVELS:
        if kk > k:
            continue
        union = dedup([f for r in runs[:kk] for f in r])
        judged = align(injected, [], union, judge_model)  # hg empty; score critic union
        caught = sum(1 for j in judged if j.get("critic_caught"))
        out["by_k"][kk] = {"caught": caught, "flags": len(union)}
    return out


def boot(caught, total, n_boot=2000):
    rng = np.random.default_rng(0); c = np.array(caught, float); t = np.array(total, float); n = len(c)
    v = [c[i].sum() / t[i].sum() for i in (rng.integers(0, n, n) for _ in range(n_boot)) if t[i].sum() > 0]
    return round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="mcr_v2flaw")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--critic-model", default="gpt-5.4-mini")
    ap.add_argument("--judge-model", default="gpt-5.4")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    flawed = [r for r in csv.DictReader(open(run_dir(args.run_id) / "manifest.csv"))
              if r["label"] == "flawed" and r["included"].lower() == "true"]
    if args.limit:
        flawed = flawed[: args.limit]
    log.info("Compute-matched critic over %d flawed traces, k=%d (critic=%s, judge=%s)",
             len(flawed), args.k, args.critic_model, args.judge_model)

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(process, r, args.k, args.critic_model, args.judge_model) for r in flawed]
        for n, fut in enumerate(as_completed(futs), 1):
            try:
                res = fut.result()
            except Exception as exc:  # noqa: BLE001
                log.warning("trace failed: %s", exc); continue
            if res:
                results.append(res)
            if n % 25 == 0:
                log.info("  %d/%d", n, len(flawed))

    summary = {}
    for kk in K_LEVELS:
        if kk > args.k:
            continue
        rows = [r for r in results if kk in r["by_k"]]
        errs = sum(r["n_errors"] for r in rows)
        caught = sum(r["by_k"][kk]["caught"] for r in rows)
        flags = sum(r["by_k"][kk]["flags"] for r in rows)
        summary[kk] = {
            "recall": round(caught / errs, 3), "recall_ci95": boot([r["by_k"][kk]["caught"] for r in rows],
                                                                    [r["n_errors"] for r in rows]),
            "precision": round(caught / flags, 3) if flags else 0,
            "avg_flags": round(flags / len(rows), 2),
        }
    out = run_dir(args.run_id) / "compute_matched_loc.json"
    out.write_text(json.dumps({"summary": summary, "per_trace": results, "n_traces": len(results)}, indent=2))

    log.info("=== COMPUTE-MATCHED CRITIC (union of k samples) vs HealthGuard ===")
    for kk, s in summary.items():
        log.info("  critic k=%-2d : recall=%.3f CI%s  precision=%.3f  (%.1f flags/trace)",
                 kk, s["recall"], s["recall_ci95"], s["precision"], s["avg_flags"])
    log.info("  -> %s", out.relative_to(REPO_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
