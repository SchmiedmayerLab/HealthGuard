"""Pipeline context: carries data between steps and handles state persistence."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .step import StepResult


@dataclass
class PipelineContext:
    """Shared state passed through all pipeline steps; persisted to disk after each step."""

    input_file: str
    input_data: dict[str, Any]
    input_text: str
    additional_input: dict[str, Any]
    results: dict[str, StepResult] = field(default_factory=dict)
    state_dir: Path | None = None

    # -- Construction -----------------------------------------------------

    @classmethod
    def from_file(cls, path: str | Path) -> PipelineContext:
        path = Path(path)
        raw = json.loads(path.read_text(encoding="utf-8"))
        input_text, additional_input = _extract_inputs(raw)
        return cls(
            input_file=str(path),
            input_data=raw,
            input_text=input_text,
            additional_input=additional_input,
        )

    # -- State persistence ------------------------------------------------

    def save(self, state_dir: Path) -> None:
        """Write current pipeline state to *state_dir* for later inspection or resume."""
        state_dir.mkdir(parents=True, exist_ok=True)

        manifest = {
            "input_file": self.input_file,
            "input_text": self.input_text,
            "completed_steps": list(self.results.keys()),
        }
        _write_json(state_dir / "manifest.json", manifest)
        _write_json(state_dir / "input.json", self.input_data)

        for idx, (name, result) in enumerate(self.results.items(), 1):
            _write_json(state_dir / f"step_{idx:02d}_{name}.json", result.to_dict())

    @classmethod
    def load(cls, state_dir: Path) -> PipelineContext:
        """Restore a previous pipeline run from *state_dir*."""
        manifest = json.loads((state_dir / "manifest.json").read_text(encoding="utf-8"))
        input_data = json.loads((state_dir / "input.json").read_text(encoding="utf-8"))

        _, additional_input = _extract_inputs(input_data)
        ctx = cls(
            input_file=manifest["input_file"],
            input_data=input_data,
            input_text=manifest["input_text"],
            additional_input=additional_input,
        )

        for idx, step_name in enumerate(manifest["completed_steps"], 1):
            step_file = state_dir / f"step_{idx:02d}_{step_name}.json"
            if step_file.exists():
                ctx.results[step_name] = StepResult.from_dict(
                    json.loads(step_file.read_text(encoding="utf-8"))
                )
        return ctx


def _write_json(path: Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def _extract_inputs(raw: dict) -> tuple[str, dict]:
    """Map a case envelope to ``(input_text, additional_input)``.

    Supports two envelope shapes: ``clinicalCase`` (input text under
    ``input.input.text``, reference metadata under ``input.additional-input.text``)
    and ``clinicalCaseDiagnosis`` (the CoT reasoning trace becomes the input text,
    ``rawInput.metadata`` is the reference data).
    """
    wrapper = raw["input"]
    resource = wrapper.get("resourceType")

    if resource == "clinicalCaseDiagnosis":
        cot = wrapper.get("cotDiagnosis", {}) or {}
        parts: list[str] = []
        reasoning = (cot.get("reasoning") or "").strip()
        if reasoning:
            parts.append(reasoning)
        diffs = cot.get("differentialDiagnoses") or []
        if diffs:
            parts.append("Differential diagnoses:")
            for d in diffs:
                condition = d.get("condition", "")
                likelihood = d.get("likelihood", "?")
                evidence = d.get("supportingEvidence", "")
                parts.append(f"- {condition} ({likelihood}): {evidence}")
        final = (cot.get("finalDiagnosis") or "").strip()
        if final:
            parts.append(f"Final diagnosis: {final}")
        input_text = "\n".join(parts).strip()
        metadata = (wrapper.get("rawInput") or {}).get("metadata") or {}
        return input_text, metadata

    # Default: clinicalCase envelope.
    text_fields = wrapper["input"]["text"]
    input_text = " ".join(str(v) for v in text_fields.values())
    additional_input = wrapper["additional-input"]["text"]
    return input_text, additional_input
