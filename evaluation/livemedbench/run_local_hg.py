"""Resumable, incrementally-checkpointing HealthGuard runner for slow local (Ollama) models,
where a full run takes long enough to be interrupted. Identical output schema to
run_healthguard.py's phase1_results.json, but:
  - writes after every completed case (atomic tmp+rename), so a kill loses at most one case,
  - skips cases already present on restart (just re-launch to resume).

    OLLAMA_MAX_TOKENS=1536 ./venv/bin/python -m evaluation.livemedbench.run_local_hg \
        --run-id lmb_local --model ollama/gemma2:9b-instruct-q4_K_M --coverage workup --revise loop --limit 24
"""
from __future__ import annotations

import argparse
import json
import os
import time

from evaluation._util import load_env, run_dir

load_env()

from evaluation.livemedbench.grade import GRADER, grade  # noqa: E402
from healthguard import core as hg  # noqa: E402


def _write_atomic(path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="lmb_local")
    ap.add_argument("--model", required=True)
    ap.add_argument("--grader", default=GRADER)
    ap.add_argument("--coverage", default="workup", choices=["base", "lenses", "workup", "full"])
    ap.add_argument("--revise", default="loop", choices=["single", "loop"])
    ap.add_argument("--safety", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    rd = run_dir(args.run_id)
    cases = json.loads((rd / "manifest.json").read_text(encoding="utf-8"))["cases"]
    cases = cases[: args.limit] if args.limit else cases
    baseline = json.loads((rd / "baseline_results.json").read_text(encoding="utf-8"))
    draft_of = {(b["case_id"], b["model"]): b for b in baseline}

    out_path = rd / "phase1_results.json"
    results = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else []
    done = {(r["case_id"], r["model"]) for r in results}

    todo = [c for c in cases if (c["case_id"], args.model) in draft_of
            and (c["case_id"], args.model) not in done]
    print(f"run_local_hg: {len(cases)} cases, {len(done)} already done, {len(todo)} to do "
          f"(model={args.model}, coverage={args.coverage}, revise={args.revise}, grader={args.grader})",
          flush=True)

    for i, c in enumerate(todo, 1):
        cid = c["case_id"]
        draft = draft_of[(cid, args.model)]["response"]
        t = time.time()
        try:
            sr_text = hg.self_refine(c, draft, args.model)
            hgr = hg.healthguard_response(c, draft, args.model, coverage=args.coverage,
                                          revise_mode=args.revise, safety=args.safety)
            row = {
                "case_id": cid, "model": args.model,
                "self_refine": {"response": sr_text, **grade(c, sr_text, model=args.grader)},
                "healthguard": {"response": hgr["response"], "audit": hgr["audit"],
                                "critique": hgr["critique"], **grade(c, hgr["response"], model=args.grader)},
            }
        except Exception as exc:  # noqa: BLE001
            print(f"  ! {cid} failed: {exc}", flush=True)
            continue
        results.append(row)
        _write_atomic(out_path, results)
        b = draft_of[(cid, args.model)]["score"]
        print(f"  [{i}/{len(todo)}] {cid}  base={b*100:5.1f}  sr={row['self_refine']['score']*100:5.1f}"
              f"  hg={row['healthguard']['score']*100:5.1f}  ({time.time()-t:.0f}s)", flush=True)

    print(f"done: {len(results)} rows -> {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
