"""File-backed cache for ``judge.judge_diagnosis`` verdicts.

Cache key is a sha256 hex digest of (model, prompt_version, ground_truth,
candidate, differentials). Layout: ``pipeline_state/judge_cache/<sha[:2]>/<sha>.json``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .judge import JudgeVerdict

CACHE_ROOT = Path("pipeline_state") / "judge_cache"


def cache_key(
    *,
    model: str,
    prompt_version: str,
    ground_truth: str,
    candidate: str,
    differentials: list[dict] | None,
) -> str:
    payload = json.dumps(
        {
            "model": model,
            "prompt_version": prompt_version,
            "ground_truth": ground_truth,
            "candidate": candidate,
            "differentials": differentials or [],
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _path_for(key: str) -> Path:
    return CACHE_ROOT / key[:2] / f"{key}.json"


def get(key: str) -> "JudgeVerdict | None":
    from .judge import JudgeVerdict  # late import to avoid cycle

    path = _path_for(key)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return JudgeVerdict(**data)


def put(key: str, verdict: "JudgeVerdict") -> None:
    path = _path_for(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(verdict.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
