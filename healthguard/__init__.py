"""HealthGuard: a ground-truth-free verifier for clinical language model outputs.

Given a patient case and a model's answer, the verifier grounds each claim against the case and
audits coverage of the essential clinical points, then flags unsound content, revises the answer,
or both. The entry point is verify.
"""
from .verify import verify, ground, soundness  # noqa: F401
