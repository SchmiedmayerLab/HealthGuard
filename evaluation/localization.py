"""Localization eval: per-error recall of HealthGuard's flags vs the single critic's
flags against the known injected errors, aligned by an independent judge.

    ./venv/bin/python -m evaluation.localization --run-id mcr_full2 --critic-model gpt-5.4-mini --judge-model gpt-5.4
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from evaluation._util import REPO_ROOT, load_env, run_dir

load_env()

from healthguard.context import PipelineContext  # noqa: E402
from evaluation import baseline_critic  # noqa: E402
from evaluation import _llm_judge as judge  # noqa: E402  (cross-vendor judge dispatch)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("localization")

ALIGN_SYSTEM = """\
You are auditing whether two error-detection methods caught specific INJECTED errors in a \
clinical reasoning trace. You are given (1) a numbered list of INJECTED ERRORS — each with \
the wrong value the trace stated, the correct value, and why it is wrong; (2) HEALTHGUARD's \
flagged statements; (3) the CRITIC's flagged claims.

For EACH injected error, decide independently whether HealthGuard's flags identify it and \
whether the Critic's flags identify it. A method "catches" an error ONLY if one of its flags \
points to the SAME specific wrong fact/value (or its correct counterpart) — not merely the \
same general topic or organ system. Be strict.

Return JSON:
{"results": [{"error_index": <int>, "healthguard_caught": true/false, "critic_caught": true/false,
"note": "<which flag matched, if any>"}]}"""


def hg_flags(run_id: str, trace_id: str) -> list[str]:
    matches = glob.glob(str(run_dir(run_id) / "state" / f"injected_hard__{trace_id}" / "*expert_verify.json"))
    if not matches:
        return []
    data = json.loads(Path(matches[0]).read_text())["data"]
    return [v["text"] for v in data["verifications"] if v.get("verdict") == "contradicted"]


def align(injected: list[dict], hg: list[str], critic: list[dict], judge_model: str) -> list[dict]:
    payload = {
        "injected_errors": [
            {"error_index": i, "stated_value": e.get("stated_value"),
             "correct_value": e.get("correct_value"), "why_wrong": e.get("why_wrong")}
            for i, e in enumerate(injected)
        ],
        "healthguard_flags": hg,
        "critic_flags": [{"claim": f.get("claim"), "issue": f.get("issue")} for f in critic],
    }
    raw = judge.chat_json(system=ALIGN_SYSTEM, user=json.dumps(payload, ensure_ascii=False), model=judge_model)
    by_idx = {}
    for r in raw.get("results", []) or []:
        try:
            by_idx[int(r["error_index"])] = r
        except (TypeError, ValueError, KeyError):
            continue
    return [by_idx.get(i, {"healthguard_caught": False, "critic_caught": False, "note": "no judgment"})
            for i in range(len(injected))]


def process(row: dict, run_id: str, critic_model: str, judge_model: str) -> dict | None:
    tid = row["trace_id"]
    env = json.loads(Path(row["path"]).read_text())
    injected = (env.get("input", {}).get("trick", {}) or {}).get("injected_errors", []) or []
    if not injected:
        return None
    hg = hg_flags(run_id, tid)
    ctx = PipelineContext.from_file(row["path"])
    critic = baseline_critic.critique(ctx.input_text, ctx.additional_input, model=critic_model).get("flaws", [])
    judged = align(injected, hg, critic, judge_model)
    return {
        "trace_id": tid, "n_errors": len(injected),
        "hg_flag_count": len(hg), "critic_flag_count": len(critic),
        "hg_caught": sum(1 for j in judged if j.get("healthguard_caught")),
        "critic_caught": sum(1 for j in judged if j.get("critic_caught")),
        "per_error": judged,
    }


def boot_ci(per_trace_caught, per_trace_total, n_boot=2000):
    rng = np.random.default_rng(0)
    c = np.array(per_trace_caught, float); t = np.array(per_trace_total, float); n = len(c)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        denom = t[i].sum()
        if denom > 0:
            vals.append(c[i].sum() / denom)
    return round(float(np.percentile(vals, 2.5)), 3), round(float(np.percentile(vals, 97.5)), 3)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="mcr_full2", help="Run whose HealthGuard state holds the verifier flags.")
    ap.add_argument("--critic-model", default="gpt-5.4-mini")
    ap.add_argument("--judge-model", default="gpt-5.4")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    manifest = run_dir(args.run_id) / "manifest.csv"
    flawed = [r for r in csv.DictReader(open(manifest)) if r["label"] == "flawed" and r["included"].lower() == "true"]
    if args.limit:
        flawed = flawed[: args.limit]
    log.info("Localization over %d flawed traces (HG flags from %s; critic=%s; judge=%s)",
             len(flawed), args.run_id, args.critic_model, args.judge_model)

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(process, r, args.run_id, args.critic_model, args.judge_model): r for r in flawed}
        for k, fut in enumerate(as_completed(futs), 1):
            try:
                res = fut.result()
            except Exception as exc:  # noqa: BLE001
                log.warning("trace failed: %s", exc); continue
            if res:
                results.append(res)
            if k % 25 == 0:
                log.info("  %d/%d", k, len(flawed))

    tot = sum(r["n_errors"] for r in results)
    hg_caught = sum(r["hg_caught"] for r in results)
    cr_caught = sum(r["critic_caught"] for r in results)
    hg_ci = boot_ci([r["hg_caught"] for r in results], [r["n_errors"] for r in results])
    cr_ci = boot_ci([r["critic_caught"] for r in results], [r["n_errors"] for r in results])

    out = run_dir(args.run_id) / "localization.json"
    summary = {
        "n_traces": len(results), "n_injected_errors": tot,
        "healthguard": {"localized": hg_caught, "recall": round(hg_caught / tot, 3), "recall_ci95": hg_ci,
                        "avg_flags": round(np.mean([r["hg_flag_count"] for r in results]), 2)},
        "critic": {"localized": cr_caught, "recall": round(cr_caught / tot, 3), "recall_ci95": cr_ci,
                   "avg_flags": round(np.mean([r["critic_flag_count"] for r in results]), 2)},
    }
    out.write_text(json.dumps({"summary": summary, "per_trace": results}, indent=2))
    # per-error CSV for inspection
    with open(run_dir(args.run_id) / "localization_per_error.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["trace_id", "error_index", "stated", "hg_caught", "critic_caught", "note"])
        for r in results:
            for i, j in enumerate(r["per_error"]):
                w.writerow([r["trace_id"], i, "", j.get("healthguard_caught"), j.get("critic_caught"), j.get("note", "")[:120]])

    log.info("=== LOCALIZATION RECALL (of %d injected errors across %d traces) ===", tot, len(results))
    log.info("  HealthGuard: %.3f  CI%s   (avg %.1f flags/trace)", summary["healthguard"]["recall"], hg_ci, summary["healthguard"]["avg_flags"])
    log.info("  LLM critic : %.3f  CI%s   (avg %.1f flags/trace)", summary["critic"]["recall"], cr_ci, summary["critic"]["avg_flags"])
    log.info("  -> %s", out.relative_to(REPO_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
