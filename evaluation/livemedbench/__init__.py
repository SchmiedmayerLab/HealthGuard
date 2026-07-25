"""LiveMedBench integration: reproduce the rubric-based benchmark and evaluate a
HealthGuard generate->audit->revise wrapper on top of it.

Dataset: JuelieYann/LiveMedBench (MIT). Each case = narrative + core_request +
doctor_advice + weighted bipolar rubric_items. Scoring = an LLM grader checks each
criterion met/not, points are summed and normalised by the max positive points.
"""
