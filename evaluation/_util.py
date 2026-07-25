"""Shared helpers for the evaluation harness: env loading, run directories, and IO."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

RUNS_DIR = REPO_ROOT / "evaluation" / "runs"


def load_env() -> None:
    """Load KEY=VALUE pairs from the repo-root .env into os.environ."""
    env_path = REPO_ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


def run_dir(run_id: str) -> Path:
    """Return (and create) the directory for a given run id."""
    d = RUNS_DIR / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def extract_cot_fields(envelope: dict) -> dict[str, Any]:
    """Pull the diagnosis fields out of a clinicalCaseDiagnosis envelope.

    Returns ``{ground_truth, final_diagnosis, differentials}``. Empty strings /
    list if absent.
    """
    inner = envelope.get("input", {}) or {}
    cot = inner.get("cotDiagnosis", {}) or {}
    return {
        "ground_truth": (inner.get("groundTruthDiagnosis") or "").strip(),
        "final_diagnosis": (cot.get("finalDiagnosis") or "").strip(),
        "differentials": cot.get("differentialDiagnoses") or [],
    }
