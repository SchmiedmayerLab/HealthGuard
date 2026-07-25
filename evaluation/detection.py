"""Detection evaluation for the unified verifier's flag mode on MedCaseReasoning traces.

Runs the verifier's grounding-soundness pass over labeled sound/flawed reasoning traces and
computes detection AUC. For comparison it also reports the AUC of the per-trace grounding
score (hg_score) already stored for the same traces.

    ./venv/bin/python -m evaluation.detection 50      # 50 sound/flawed pairs
"""
from __future__ import annotations

import csv
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from sklearn.metrics import roc_auc_score

from evaluation._util import load_env, run_dir

load_env()

from healthguard import verify  # noqa: E402

MODEL = sys.argv[2] if len(sys.argv) > 2 else "claude-sonnet-4-6"
MCR = "testdata/streamlined/medcasereasoning"


def load_trace(trace_id: str, source: str):
    path = (f"{MCR}/rationale/{trace_id}.json" if source == "rationale"
            else f"{MCR}/tricked/injected_hard/{trace_id}.json")
    d = json.load(open(path))["input"]
    raw = d.get("rawInput", {})
    narrative = raw.get("complaint", "")
    meta = raw.get("metadata")
    if meta:
        narrative += "\n\nReference data:\n" + json.dumps(meta, ensure_ascii=False)[:2500]
    cot = d.get("cotDiagnosis", {})
    output = (cot.get("reasoning", "") + "\n\nFinal diagnosis: " + str(cot.get("finalDiagnosis", "")))
    case = {"narrative": narrative,
            "core_request": "Give the single most likely final diagnosis with reasoning."}
    return case, output


def main() -> int:
    n_pairs = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    rows = list(csv.DictReader(open(run_dir("mcr_full2") / "results.csv")))
    by_id: dict = {}
    for r in rows:
        by_id.setdefault(r["trace_id"], {})[r["label"]] = r
    pairs = [(tid, d) for tid, d in by_id.items() if "sound" in d and "flawed" in d][:n_pairs]
    items = [(tid, lab, d[lab]["source"], float(d[lab]["hg_score"]))
             for tid, d in pairs for lab in ("sound", "flawed")]
    print(f"detection: {len(pairs)} pairs, {len(items)} traces (model={MODEL})", flush=True)

    def score(item):
        tid, label, source, hgA = item
        case, output = load_trace(tid, source)
        f = verify(case, output, model=MODEL, mode="flag", checks=("grounding",))["findings"]
        return {"trace_id": tid, "label": label, "y": 1.0 if label == "sound" else 0.0,
                "soundness": f["soundness"], "n_flagged": f["n_flagged"],
                "n_incorrect": f["n_incorrect"], "n_claims": f["n_claims"], "hgA": hgA}

    results = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        futs = [pool.submit(score, it) for it in items]
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                results.append(fut.result())
            except Exception as e:  # noqa: BLE001
                print("  ! ", e, flush=True)
            if i % 20 == 0:
                print(f"  {i}/{len(items)}", flush=True)

    y = np.array([r["y"] for r in results])

    def auc(pred, higher_is_sound=True):
        p = np.array([r[pred] for r in results], float)
        return roc_auc_score(y, p if higher_is_sound else -p)

    print("\n=== HealthGuard detection on MedCaseReasoning traces ===")
    print(f"  n={len(results)}  sound={int(y.sum())}  flawed={int((1 - y).sum())}")
    print(f"  soundness          AUC = {auc('soundness'):.3f}")
    print(f"  -n_flagged         AUC = {auc('n_flagged', False):.3f}")
    print(f"  -n_incorrect       AUC = {auc('n_incorrect', False):.3f}")
    print(f"  reference hg_score AUC = {auc('hgA'):.3f}")
    tag = MODEL.replace("/", "_").replace(":", "_").replace(".", "")
    (run_dir("mcr_full2") / f"detection_{tag}.json").write_text(json.dumps(results, indent=2))
    print(f"  -> evaluation/runs/mcr_full2/detection_{tag}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
