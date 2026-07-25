"""Select a reproducible pilot of English, post-cutoff LiveMedBench cases.

Selection is entirely content-blind (no model output involved), so it cannot favour
any method: English cases from the latest snapshot month (post model training cutoff),
with a usable rubric (4-15 criteria, both positive and negative points present), then a
deterministic even spread by case_id for topic diversity.

    ./venv/bin/python -m evaluation.livemedbench.select --n 10
"""
from __future__ import annotations

import argparse
import json

from evaluation._util import run_dir
from evaluation.livemedbench.data import is_english, load_cached, rubric_points

POST_CUTOFF = "2026-04-01"   # latest snapshot month; safely after gpt-5.4 training cutoff
MIN_CRIT, MAX_CRIT = 4, 15


def eligible(cases: list[dict]) -> list[dict]:
    out = []
    for c in cases:
        if not is_english(c):
            continue
        if (c.get("post_time", "") or "")[:10] < POST_CUTOFF:
            continue
        n = len(c.get("rubric_items", []) or [])
        if not (MIN_CRIT <= n <= MAX_CRIT):
            continue
        pos, neg = rubric_points(c)
        if pos <= 0 or neg <= 0:   # require both reward and penalty criteria
            continue
        out.append(c)
    return out


def pick(pool: list[dict], n: int) -> list[dict]:
    pool = sorted(pool, key=lambda c: c["case_id"])
    if len(pool) <= n:
        return pool
    step = len(pool) / n
    return [pool[int(i * step)] for i in range(n)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--run-id", default="lmb_pilot")
    args = ap.parse_args()

    cases = load_cached()
    pool = eligible(cases)
    chosen = pick(pool, args.n)

    manifest = {
        "dataset": "JuelieYann/LiveMedBench (snapshot v202604)",
        "selection": {
            "language": "english", "post_time_min": POST_CUTOFF,
            "criteria_range": [MIN_CRIT, MAX_CRIT], "require_pos_and_neg": True,
            "method": "deterministic even spread by case_id (content-blind)",
            "pool_size": len(pool), "n": len(chosen),
        },
        "cases": chosen,
    }
    out = run_dir(args.run_id) / "manifest.json"
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"eligible pool: {len(pool)} | selected: {len(chosen)}")
    print(f"{'case_id':>8} | {'post_time':10} | {'#crit':>5} | +pts/-pts | core_request")
    for c in chosen:
        pos, neg = rubric_points(c)
        cr = (c.get("core_request", "") or "").replace("\n", " ")[:64]
        print(f"{c['case_id']:>8} | {(c.get('post_time','') or '')[:10]:10} | "
              f"{len(c.get('rubric_items',[])):>5} | +{pos:.0f}/-{neg:.0f} | {cr}")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
