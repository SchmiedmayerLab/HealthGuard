> [!IMPORTANT]
> **This repository is archived.** Work on HealthGuard continues at
> [github.com/SchmiedmayerLab/HealthGuard](https://github.com/SchmiedmayerLab/HealthGuard).

# HealthGuard

HealthGuard is a ground-truth-free agentic verifier for clinical language models. Given a
patient case and a base model's answer, and using only that same model and the case (no
reference answer), it decomposes the answer into atomic claims, grounds each claim against the
case to produce a soundness score, and audits coverage of essential management, work-up, and
prognosis. From these checks it produces either an autonomously revised answer that is more
complete and safer, or a set of flags (unsound or unsafe claims and coverage gaps) that route a
case to a clinician, or both.

This repository contains the verifier itself, the evaluation harness used in the accompanying (wip) paper,
the figure-generation code, and the clinician rating form. The run artifacts behind the results are not included in this archived repository; the source benchmark corpora are not redistributed and can be obtained from their original sources (see Datasets).

## Repository layout

```
healthguard/              The verifier
    verify.py                 Flexible entry point: grounding + coverage, with a flag / revise / both output
    core.py                   Grounding and coverage-audit prompts, and the revise-and-repair loop
    llm.py                    Unified client for OpenAI, Anthropic, and Ollama (OpenAI-compatible) backends
    context.py                Loads a patient-case envelope into case text and reference data
    judge.py, judge_cache.py  LLM judge for diagnosis correctness, with an on-disk cache
evaluation/               The evaluation harness
    livemedbench/             Autonomous-mode evaluation on LiveMedBench and local open-weights models
        data.py, select.py        Load the benchmark and build the held-out split
        respond.py                Base-model answer generation
        grade.py, cross_grade.py  LLM grading against the bipolar rubric, with a second-vendor grader
        run_baseline.py           Generate and grade base-model answers
        run_healthguard.py        Generate, revise with HealthGuard, then grade
        pool_heldout.py           Pool held-out cases into a paired lift estimate with a bootstrap CI
        run_local_baseline.py     Resumable baseline run on a local model
        run_local_hg.py           Resumable HealthGuard run on a local model
        report_local.py           Per-run lift report with bootstrap CI under both graders
        decompose_lift.py         Split the rubric-quality lift into diagnosis and coverage components
        diagnose.py               Diagnostic-answer extraction and scoring
        paper_figures.py          Generate the paper figures as vector PDFs (written to figures/)
        make_clinician_form.py    Generate the self-contained clinician rating form
    detection.py              Grounding-based flaw detection on labelled reasoning traces (AUC)
    localization.py           Align verifier flags to known injected errors (precision and recall)
    baseline_critic.py        Single-LLM critic baseline (critique to soundness and flaws)
    compute_matched_det.py    Compute-matched critic detection AUC at k self-consistency samples
    compute_matched_loc.py    Compute-matched critic localization
    exp_error_fingerprint.py  Taxonomy of omissions and errors in base-model answers
    exp_harm_profile.py       Help-versus-harm profile of autonomous revision
    exp_hg_solver.py          Judgment-ceiling test: verification panel used as a diagnostic solver
    exp_natural_wrong.py      Detection AUC split by whether the final diagnosis was naturally right or wrong
    _util.py, _llm_judge.py   Shared helpers (environment, run directories, judge dispatch)
docs/
    index.html                Self-contained clinician rating form for hosting on GitHub Pages
```

## The verifier

The verifier is a single function with a flexible output. A minimal call:

```python
from healthguard import verify

result = verify(
    case,            # the patient-case dict (narrative and reference data)
    output,          # the base model's answer, as a string
    model="gpt-5.4",
    mode="both",     # "flag", "revise", or "both"
)

result["soundness"]        # aggregate grounding score in [0, 1]
result["flags"]            # unsound or unsafe claims and coverage gaps
result["revised"]          # the revised answer (when mode is "revise" or "both")
```

The `coverage` argument selects how thoroughly the coverage audit enumerates essential points
(`base`, `lenses`, `workup`, or `full`, each adding more targeted lenses). The same model that
produced the answer is used for verification, and no reference answer or grading rubric is ever
shown to it. Because there is no external dependency, the verifier runs wherever the base model
runs, including on a local on-premises model.

## Datasets

The paper evaluates on two external, publicly available benchmarks that are not redistributed
here. Obtain them from their original sources and place them under `testdata/` (git-ignored):

- LiveMedBench (Yan et al., 2026, arXiv:2602.10367): a contamination-free, rubric-scored
  benchmark of real patient cases. The autonomous-mode data loader downloads the monthly
  snapshots from the dataset's Hugging Face repository using the system `curl` command.
- MedCaseReasoning (Wu et al., 2025, arXiv:2505.11733): diagnostic-reasoning traces from
  clinical case reports, used for flaw detection and localization.

The flaw-detection and localization experiments use labelled reasoning traces (sound traces and
traces carrying known injected errors at known locations). The construction of the injected-error
traces is described in the paper. The clinical-case materials used during earlier development are
not part of this release.

## Setup

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # then fill in OPENAI_API_KEY and ANTHROPIC_API_KEY
```

The system `curl` command is required to download the LiveMedBench snapshots. To run the local
open-weights experiments, install Ollama, pull the target models, and set `OLLAMA_BASE_URL`.

## Reproducing the paper

Every script is a runnable module. Pass `--help` or read the module docstring for the full
argument list; representative commands are shown below. Run artifacts are written under
`evaluation/runs/<run-id>/`.

Autonomous mode (completeness lift on LiveMedBench):

```
python -m evaluation.livemedbench.data
python -m evaluation.livemedbench.select
python -m evaluation.livemedbench.run_baseline    --run-id lmb_heldout --model gpt-5.4 --limit 100
python -m evaluation.livemedbench.run_healthguard --run-id lmb_heldout --model gpt-5.4 --limit 100
python -m evaluation.livemedbench.cross_grade     --run-id lmb_heldout
python -m evaluation.livemedbench.pool_heldout
python -m evaluation.livemedbench.decompose_lift
```

Competence threshold (local open-weights models):

```
python -m evaluation.livemedbench.run_local_baseline --run-id lmb_local --model "ollama/gemma2:9b-instruct-q4_K_M" --limit 50
python -m evaluation.livemedbench.run_local_hg       --run-id lmb_local --model "ollama/gemma2:9b-instruct-q4_K_M" --coverage workup --revise loop --grader gpt-4.1 --limit 50
python -m evaluation.livemedbench.cross_grade        --run-id lmb_local
python -m evaluation.livemedbench.report_local lmb_local
```

Flag-and-escalate (detection and localization on MedCaseReasoning):

```
python -m evaluation.detection
python -m evaluation.compute_matched_det --run-id mcr_full2 --k 10 --critic-model gpt-5.4-mini
python -m evaluation.localization        --run-id mcr_full2
python -m evaluation.compute_matched_loc --run-id mcr_full2
```

Safety envelope and error analysis:

```
python -m evaluation.exp_error_fingerprint
python -m evaluation.exp_harm_profile
python -m evaluation.exp_hg_solver
python -m evaluation.exp_natural_wrong
```

Figures (written to `figures/`):

```
python -m evaluation.livemedbench.paper_figures
```

## Clinician rating form

`docs/index.html` is a self-contained rating instrument for the blinded clinician validation. It
uses only in-browser storage and exports each rater's responses as a local JSON download; it makes
no network requests and needs no backend. They are served using Github pages.

Regenerate the html using:

```
python -m evaluation.livemedbench.make_clinician_form
```

## Contributing

Contributions to this project are welcome. Please make sure to read the [contribution guide](https://github.com/SchmiedmayerLab/.github/blob/main/CONTRIBUTING.md) and the [Contributor Covenant Code of Conduct](https://github.com/SchmiedmayerLab/.github/blob/main/CODE_OF_CONDUCT.md) first.


## License

This project is licensed under the MIT License. See [Licenses](LICENSES) and [Contributors](CONTRIBUTORS.md) for more information.


## Our Research

For more information, visit the [Schmiedmayer Lab GitHub organization](https://github.com/SchmiedmayerLab).

![Stanford and Stanford Medicine logos](https://raw.githubusercontent.com/SchmiedmayerLab/.github/main/assets/stanford-footer-light.png#gh-light-mode-only)
![Stanford and Stanford Medicine logos](https://raw.githubusercontent.com/SchmiedmayerLab/.github/main/assets/stanford-footer-dark.png#gh-dark-mode-only)
