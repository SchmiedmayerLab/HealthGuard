"""Compute-matched detection baseline: critic AUC at k self-consistency samples.

Runs the critic k times per trace, averages soundness over the first {1,3,5,10}
samples, and computes AUC at each k with a case-level (clustered) bootstrap.

    ./venv/bin/python -m evaluation.compute_matched_det --run-id mcr_full2 --k 10 --critic-model gpt-5.4-mini
"""
from __future__ import annotations
import argparse
import csv
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from sklearn.metrics import roc_auc_score

from evaluation._util import REPO_ROOT, load_env, run_dir

load_env()
from healthguard.context import PipelineContext  # noqa: E402
from evaluation import baseline_critic  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("cm_detection")
K_LEVELS = [1, 3, 5, 10]


def sample(row: dict, k: int, model: str) -> dict:
    ctx = PipelineContext.from_file(row["path"])
    scores = [baseline_critic.critique(ctx.input_text, ctx.additional_input, model=model, temperature=0.7)["soundness"]
              for _ in range(k)]
    return {"trace_id": row["trace_id"], "y": 1 if row["label"] == "flawed" else 0, "scores": scores}


def clustered_auc_ci(recs, k, n_boot=2000):
    # group rows by trace_id (each id has a sound + flawed row -> matched cluster)
    ids = sorted({r["trace_id"] for r in recs})
    by_id = {i: [r for r in recs if r["trace_id"] == i] for i in ids}
    rng = np.random.default_rng(0); aucs = []
    for _ in range(n_boot):
        pick = rng.choice(ids, len(ids), replace=True)
        rows = [r for i in pick for r in by_id[i]]
        y = [r["y"] for r in rows]; fl = [1 - np.mean(r["scores"][:k]) for r in rows]
        if len(set(y)) == 2:
            aucs.append(roc_auc_score(y, fl))
    if not aucs:
        return (float("nan"), float("nan"))
    return round(float(np.percentile(aucs, 2.5)), 3), round(float(np.percentile(aucs, 97.5)), 3)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="mcr_full2")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--critic-model", default="gpt-5.4-mini")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    rows = [r for r in csv.DictReader(open(run_dir(args.run_id) / "manifest.csv"))
            if r["label"] in ("sound", "flawed") and r["included"].lower() == "true"]
    if args.limit:
        rows = rows[: args.limit]
    log.info("Detection compute-match over %d traces, k=%d (critic=%s)", len(rows), args.k, args.critic_model)

    recs = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(sample, r, args.k, args.critic_model) for r in rows]
        for n, fut in enumerate(as_completed(futs), 1):
            try:
                recs.append(fut.result())
            except Exception as exc:  # noqa: BLE001
                log.warning("trace failed: %s", exc)
            if n % 50 == 0:
                log.info("  %d/%d", n, len(rows))

    summary = {}
    for k in K_LEVELS:
        if k > args.k:
            continue
        y = [r["y"] for r in recs]; fl = [1 - np.mean(r["scores"][:k]) for r in recs]
        summary[k] = {"auc": round(roc_auc_score(y, fl), 3), "auc_ci95": clustered_auc_ci(recs, k)}

    # references from the existing run
    res = list(csv.DictReader(open(run_dir(args.run_id) / "results.csv")))
    rr = [r for r in res if r.get("hg_score") and r["label"] in ("sound", "flawed")]
    yy = [1 if r["label"] == "flawed" else 0 for r in rr]
    hg_auc = round(roc_auc_score(yy, [1 - float(r["hg_score"]) for r in rr]), 3)

    out = run_dir(args.run_id) / "compute_matched_det.json"
    out.write_text(json.dumps({"critic_by_k": summary, "healthguard_auc": hg_auc, "n": len(recs)}, indent=2))
    log.info("=== DETECTION compute-match (critic self-consistency) ===")
    for k, s in summary.items():
        log.info("  critic k=%-2d : AUC=%.3f CI%s", k, s["auc"], s["auc_ci95"])
    log.info("  [ref] HealthGuard grounding AUC=%.3f", hg_auc)
    log.info("  -> %s", out.relative_to(REPO_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
