"""Detect WRONG CONCLUSIONS on real cases: split the model's own rationale CoT traces
by whether the final diagnosis was right (match vs mismatch), and report detection
ROC-AUC for HealthGuard's aggregate score and the single critic's score.

  python -m evaluation.exp_natural_wrong build
  PIPELINE_MODEL=gpt-5.4-mini python -m evaluation.run_harness --run-id mcr_natwrong --workers 6
  python -m evaluation.exp_natural_wrong score
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from evaluation._util import REPO_ROOT, load_env, run_dir

load_env()
from healthguard import judge  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("natural_wrong")

RATIONALE_DIR = REPO_ROOT / "testdata/streamlined/medcasereasoning/rationale"
SOUND_RUN = "mcr_full2"      # holds the 'sound' (match) rationale scores
WRONG_RUN = "mcr_natwrong"   # this experiment's run for the wrong (mismatch) traces


def _fields(path: Path):
    d = json.loads(path.read_text())
    inp = d.get("input", {})
    cot = inp.get("cotDiagnosis", {}) or {}
    return (inp.get("groundTruthDiagnosis"), cot.get("finalDiagnosis"),
            cot.get("differentialDiagnoses"), inp.get("resourceType", "clinicalCaseDiagnosis"))


def _judge_one(path: Path):
    gt, final, diffs, rtype = _fields(path)
    if not gt or not final:
        return (path.stem, "skip", path, rtype)
    v = judge.judge_diagnosis(ground_truth=gt, candidate=final, differentials=diffs)
    return (path.stem, v.verdict, path, rtype)


def build() -> int:
    files = sorted(RATIONALE_DIR.glob("*.json"))
    log.info("Judging %d rationale traces (cached where possible)...", len(files))
    counts = {"match": 0, "related": 0, "mismatch": 0, "skip": 0}
    wrong = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for tid, verdict, path, rtype in pool.map(_judge_one, files):
            counts[verdict] = counts.get(verdict, 0) + 1
            if verdict == "mismatch":
                wrong.append((tid, path, rtype))
    log.info("verdicts: %s", counts)
    out = run_dir(WRONG_RUN); out.mkdir(parents=True, exist_ok=True)
    with open(out / "manifest.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["trace_id", "path", "resourceType", "label", "source", "judge_verdict", "included"])
        for tid, path, rtype in wrong:
            w.writerow([tid, str(path), rtype, "flawed", "rationale_wrong", "mismatch", "true"])
    log.info("wrote %d wrong-conclusion traces -> %s/manifest.csv", len(wrong), out.relative_to(REPO_ROOT))
    log.info("(correct-conclusion 'sound' set reused from %s: %d 'match')", SOUND_RUN, counts["match"])
    return 0


def _scores(run_id: str, want_label: str):
    rows = []
    for r in csv.DictReader(open(run_dir(run_id) / "results.csv")):
        if r.get("label") == want_label and r.get("hg_score"):
            rows.append((float(r["hg_score"]), float(r["base_score"]) if r.get("base_score") else None))
    return rows


def _auc_ci(y, pred, nb=2000):
    y = np.array(y); pred = np.array(pred); rng = np.random.default_rng(0); n = len(y); a = []
    for _ in range(nb):
        i = rng.integers(0, n, n)
        if len(set(y[i])) == 2:
            a.append(roc_auc_score(y[i], pred[i]))
    return round(roc_auc_score(y, pred), 3), (round(float(np.percentile(a, 2.5)), 3), round(float(np.percentile(a, 97.5)), 3))


def score() -> int:
    sound = _scores(SOUND_RUN, "sound")          # y=0 (correct conclusion)
    wrong = _scores(WRONG_RUN, "flawed")          # y=1 (wrong conclusion)
    log.info("correct-conclusion (sound): %d | wrong-conclusion (natural): %d", len(sound), len(wrong))
    y, hg, cr = [], [], []
    for s, b in sound:
        y.append(0); hg.append(1 - s); cr.append(1 - b if b is not None else None)
    for s, b in wrong:
        y.append(1); hg.append(1 - s); cr.append(1 - b if b is not None else None)
    hg_auc, hg_ci = _auc_ci(y, hg)
    cr_pairs = [(yy, c) for yy, c in zip(y, cr) if c is not None]
    cr_auc, cr_ci = _auc_ci([p[0] for p in cr_pairs], [p[1] for p in cr_pairs])
    summary = {"n_correct": len(sound), "n_wrong": len(wrong),
               "healthguard": {"auc": hg_auc, "ci95": hg_ci},
               "critic": {"auc": cr_auc, "ci95": cr_ci}}
    (run_dir(WRONG_RUN) / "natural_wrong_auc.json").write_text(json.dumps(summary, indent=2))
    log.info("=== WRONG-CONCLUSION DETECTION (natural, non-injected) ===")
    log.info("  HealthGuard: AUC=%.3f CI%s", hg_auc, hg_ci)
    log.info("  LLM critic : AUC=%.3f CI%s", cr_auc, cr_ci)
    log.info("  (n=%d correct vs %d wrong; predictor = 1 - soundness score)", len(sound), len(wrong))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "score"])
    args = ap.parse_args()
    return build() if args.cmd == "build" else score()


if __name__ == "__main__":
    raise SystemExit(main())
