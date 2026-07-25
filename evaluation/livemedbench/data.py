"""Download + cache the LiveMedBench dataset, detect language, and inspect it.

The HF datasets-server auto-conversion is broken for this dataset (an empty
``post_time`` fails Arrow's timestamp parse), so we pull the raw monthly JSON
snapshots the repo ships (``LiveMedBench_v2026MM.json``) via curl, take the latest
(cumulative) one, dedup by ``case_id``, and cache to ``cases_all.json``.

    ./venv/bin/python -m evaluation.livemedbench.data              # download + summarise
    ./venv/bin/python -m evaluation.livemedbench.data --explore    # summarise cached
"""
from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter

from evaluation._util import RUNS_DIR

DATASET = "JuelieYann/LiveMedBench"
CACHE_DIR = RUNS_DIR / "lmb_cache"
RAW_DIR = CACHE_DIR / "raw"
CACHE_FILE = CACHE_DIR / "cases_all.json"
RESOLVE = f"https://huggingface.co/datasets/{DATASET}/resolve/main"
# Monthly snapshots shipped by the repo; the latest is cumulative.
SNAPSHOTS = ["202601", "202602", "202603", "202604"]
LATEST = SNAPSHOTS[-1]


def _curl(url: str, out) -> None:
    subprocess.run(["curl", "-sSL", "--max-time", "120", url, "-o", str(out)], check=True)


def download() -> list[dict]:
    """Fetch the raw snapshots, use the latest (cumulative), dedup by case_id, cache."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for v in SNAPSHOTS:
        _curl(f"{RESOLVE}/LiveMedBench_v{v}.json", RAW_DIR / f"LiveMedBench_v{v}.json")
    raw = json.loads((RAW_DIR / f"LiveMedBench_v{LATEST}.json").read_text(encoding="utf-8"))
    seen, rows = set(), []
    for c in raw:
        cid = c.get("case_id")
        if cid in seen:
            continue
        seen.add(cid)
        rows.append(c)
    CACHE_FILE.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return rows


def load_cached() -> list[dict]:
    return json.loads(CACHE_FILE.read_text(encoding="utf-8"))


def is_english(case: dict) -> bool:
    """Heuristic: a case is Chinese if its narrative has a non-trivial share of CJK
    characters. LiveMedBench is bilingual with no explicit language field."""
    text = (case.get("narrative", "") or "") + (case.get("core_request", "") or "")
    if not text:
        return False
    cjk = sum(1 for ch in text if "一" <= ch <= "鿿")
    return cjk / max(1, len(text)) < 0.02


def rubric_points(case: dict) -> tuple[float, float]:
    """Return (sum of positive points, sum of |negative points|) for a case."""
    pos = neg = 0.0
    for it in case.get("rubric_items", []) or []:
        p = float(it.get("points", 0))
        if p >= 0:
            pos += p
        else:
            neg += -p
    return pos, neg


def summarise(rows: list[dict]) -> None:
    en = [c for c in rows if is_english(c)]
    zh = [c for c in rows if not is_english(c)]
    times = sorted((c.get("post_time", "") or "")[:10] for c in rows if c.get("post_time"))
    print(f"total cases        : {len(rows)}")
    print(f"  english (heur.)  : {len(en)}")
    print(f"  chinese (heur.)  : {len(zh)}")
    if times:
        print(f"  post_time range  : {times[0]}  ->  {times[-1]}")
        yr = Counter(t[:4] for t in times)
        print(f"  by year          : {dict(sorted(yr.items()))}")
    ncrit = [len(c.get("rubric_items", []) or []) for c in rows]
    print(f"  criteria/case    : min={min(ncrit)} max={max(ncrit)} mean={sum(ncrit)/len(ncrit):.2f}")
    # english, recent cases we could pick from
    en_recent = sorted(en, key=lambda c: c.get("post_time", ""), reverse=True)
    print("\n  most-recent ENGLISH cases (id | post_time | #crit | +pts/-pts | core_request):")
    for c in en_recent[:15]:
        pos, neg = rubric_points(c)
        cr = (c.get("core_request", "") or "").replace("\n", " ")[:70]
        print(f"    {c['case_id']:>5} | {(c.get('post_time','') or '')[:10]} | "
              f"{len(c.get('rubric_items',[])):>2} | +{pos:.0f}/-{neg:.0f} | {cr}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--explore", action="store_true", help="summarise the cached file (no download)")
    args = ap.parse_args()
    if args.explore and CACHE_FILE.exists():
        rows = load_cached()
        print(f"(cached) {CACHE_FILE}")
    else:
        print(f"downloading {DATASET} via datasets-server ...")
        rows = download()
        print(f"cached -> {CACHE_FILE}")
    summarise(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
