"""Resumable, append-only baseline generation for a local model (for extending an existing run
without overwriting drafts that already have paired HealthGuard revisions). Per-case atomic
checkpoint; skips cases already present for this model on restart.

    OLLAMA_MAX_TOKENS=1536 ./venv/bin/python -m evaluation.livemedbench.run_local_baseline \
        --run-id lmb_local --model ollama/gemma2:9b-instruct-q4_K_M --limit 50
"""
from __future__ import annotations

import argparse
import json
import os
import time

from evaluation._util import load_env, run_dir

load_env()

from evaluation.livemedbench.grade import GRADER, grade  # noqa: E402
from evaluation.livemedbench.respond import respond  # noqa: E402


def _write_atomic(path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--grader", default=GRADER)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    rd = run_dir(args.run_id)
    cases = json.loads((rd / "manifest.json").read_text(encoding="utf-8"))["cases"]
    cases = cases[: args.limit] if args.limit else cases

    out_path = rd / "baseline_results.json"
    results = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else []
    done = {(r["case_id"], r["model"]) for r in results}
    todo = [c for c in cases if (c["case_id"], args.model) not in done]
    print(f"local baseline: {len(cases)} cases, {len(done)} rows present, {len(todo)} to do "
          f"(model={args.model}, grader={args.grader})", flush=True)

    for i, c in enumerate(todo, 1):
        t = time.time()
        try:
            text = respond(c, model=args.model)
            g = grade(c, text, model=args.grader)
            results.append({"case_id": c["case_id"], "model": args.model, "response": text, **g})
            _write_atomic(out_path, results)
            print(f"  [{i}/{len(todo)}] {c['case_id']}  score={g['score']*100:5.1f}  ({time.time()-t:.0f}s)",
                  flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  ! {c['case_id']} failed: {exc}", flush=True)

    print(f"done: {len(results)} rows -> {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
