"""Select the cardiovascular subset of LiveMedBench for the specialty-matched clinician study.

Our clinician raters are cardiologists, so the validation cases must be ones a cardiologist is
the right reference standard for (see CLINICIAN_STUDY.md). We therefore restrict to cases whose
*chief complaint* is cardiovascular, drawn only from the same contamination-free eligible pool the
main study uses (English, post-model-cutoff snapshot, usable bipolar rubric; see select.py).

Curation is content-blind w.r.t. any model output: cases were screened by a cardiovascular keyword
match on the patient's question (``core_request``) over the eligible pool, then each candidate was
read once and kept only if its primary clinical question is cardiac, vascular, hypertension,
thrombosis, or lipid management. The frozen list + per-case tag below records that judgement so the
selection is reproducible and auditable. Incidental mentions (e.g. hypertension as a comorbidity in
a nephrology case) were excluded.

The same manifest is written into three run dirs (frontier + two local models) so all arms of the
clinician form are drawn from an identical cardiovascular case set.

    ./venv/bin/python -m evaluation.livemedbench.select_cardio \
        --runs lmb_cardio_f lmb_cardio_9b lmb_cardio_3b
"""
from __future__ import annotations

import argparse
import json

from evaluation._util import run_dir
from evaluation.livemedbench.data import load_cached, rubric_points
from evaluation.livemedbench.select import POST_CUTOFF, MIN_CRIT, MAX_CRIT, eligible

# Frozen cardiovascular set: case_id -> (chief-complaint tag, one-line reason).
# tags: arrhythmia | ischemia/chest-pain | valve/structural | heart-failure | hypertension |
#       thrombosis/vascular | lipids | cardio-oncology
CARDIO = {
    4908: ("arrhythmia",          "SVT ablation decision with mitral/tricuspid regurgitation"),
    4909: ("hypertension",        "is 140/90 in an athlete true hypertension"),
    4910: ("arrhythmia",          "atrial fibrillation management in pregnancy"),
    4912: ("valve/structural",    "risk of repeated open-heart (valve) surgery"),
    4914: ("arrhythmia",          "management of premature ventricular contractions"),
    4915: ("ischemia/chest-pain", "chest pain work-up in a young underweight patient"),
    4916: ("ischemia/chest-pain", "chest tightness/burning — angina vs non-cardiac"),
    4917: ("arrhythmia",          "athletic bradycardia — normal vs pathological"),
    4918: ("arrhythmia",          "episodic tachycardia work-up"),
    4919: ("thrombosis/vascular", "recurrent idiopathic pulmonary embolism management"),
    4920: ("arrhythmia",          "sudden HR 195 bpm — danger and management"),
    4921: ("ischemia/chest-pain", "chest pain in a known cardiac patient"),
    4922: ("lipids",              "severe hypertriglyceridemia (850 mg/dL) management"),
    4924: ("hypertension",        "hypertension management in the elderly"),
    4948: ("ischemia/chest-pain", "chest pain differential in a 28-year-old"),
    4951: ("lipids",              "raising low HDL / cardiovascular lipid risk"),
    4963: ("thrombosis/vascular", "anticoagulation (rivaroxaban) for DVT with factor V Leiden"),
    5079: ("cardio-oncology",     "cardiotoxic oncology therapy safety in heart failure"),
    5124: ("heart-failure",       "right-sided heart failure with cardiorenal syndrome"),
    5133: ("hypertension",        "management of fluctuating blood pressure"),
    5177: ("thrombosis/vascular", "exertional leg heaviness — claudication / PAD"),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+",
                    default=["lmb_cardio_f", "lmb_cardio_9b", "lmb_cardio_3b"],
                    help="run dirs to receive the identical cardiovascular manifest")
    args = ap.parse_args()

    pool = {c["case_id"]: c for c in eligible(load_cached())}
    missing = [cid for cid in CARDIO if cid not in pool]
    if missing:
        raise SystemExit(f"curated ids not in eligible pool (contamination/rubric filter): {missing}")

    chosen = [pool[cid] for cid in CARDIO]
    for c in chosen:                                   # annotate for downstream inspection
        tag, reason = CARDIO[c["case_id"]]
        c["cardio_tag"], c["cardio_reason"] = tag, reason

    manifest = {
        "dataset": "JuelieYann/LiveMedBench (snapshot v202604)",
        "selection": {
            "specialty": "cardiovascular",
            "language": "english", "post_time_min": POST_CUTOFF,
            "criteria_range": [MIN_CRIT, MAX_CRIT], "require_pos_and_neg": True,
            "method": "cardiovascular chief-complaint curation over the contamination-free "
                      "eligible pool (keyword screen on the patient question + manual "
                      "confirmation); content-blind w.r.t. model output",
            "eligible_pool_size": len(pool), "n": len(chosen),
            "rater_specialty_match": "cardiology",
        },
        "cases": chosen,
    }
    for rid in args.runs:
        (run_dir(rid) / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    from collections import Counter
    tags = Counter(t for t, _ in CARDIO.values())
    print(f"eligible pool: {len(pool)} | cardiovascular selected: {len(chosen)}")
    print("chief-complaint mix:", dict(tags))
    print(f"{'case_id':>8} | {'#crit':>5} | +pts/-pts | tag                 | question")
    for c in chosen:
        pos, neg = rubric_points(c)
        cr = (c.get("core_request", "") or "").replace("\n", " ")[:52]
        print(f"{c['case_id']:>8} | {len(c.get('rubric_items',[])):>5} | +{pos:.0f}/-{neg:.0f} "
              f"| {c['cardio_tag']:<19} | {cr}")
    print(f"-> wrote manifest into: {', '.join(args.runs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
