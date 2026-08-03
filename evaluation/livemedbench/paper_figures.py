"""Publication figures for HealthGuard, written to figures/ as vector PDFs (with PNG proxies).
Rendered at print size for a single-column layout.
All numbers are read directly from the run artifacts in evaluation/runs/.
Result-figure uncertainty uses patient-case- or trace-clustered bootstraps.

    ./venv/bin/python -m evaluation.livemedbench.paper_figures
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evaluation._util import run_dir

OUT = Path("figures")

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "xtick.labelsize": 8.3, "ytick.labelsize": 8.3, "legend.fontsize": 8,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "savefig.dpi": 200, "pdf.fonttype": 42, "ps.fonttype": 42,
    "figure.constrained_layout.use": False,
})

BASE = "#9AA7B0"; HG = "#2E6B43"; SR = "#C29B45"
G4 = "#2A4D77"; CL = "#B06C3C"
POS = "#2E6B43"; NEG = "#B23A3A"; MUT = "#5b6570"


def _save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=190, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> figures/{name}.pdf (+png)")


def boot_clustered(cids, vals, n=5000):
    """Patient-case-clustered bootstrap of a mean. Resamples unique case ids with replacement,
    carrying every record for a resampled case. Returns mean and 95% CI in percentage points."""
    cids = np.asarray(cids); vals = np.asarray(vals, float)
    uniq = np.unique(cids)
    groups = {c: vals[cids == c] for c in uniq}
    rng = np.random.default_rng(0)
    reps = [np.concatenate([groups[c] for c in rng.choice(uniq, len(uniq), replace=True)]).mean()
            for _ in range(n)]
    return vals.mean() * 100, np.percentile(reps, 2.5) * 100, np.percentile(reps, 97.5) * 100


def auc_clustered(y, s, tids, n=2000):
    """Trace-clustered bootstrap of AUC. Resamples matched pairs (by trace id) with replacement."""
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y); s = np.asarray(s); tids = np.asarray(tids)
    uniq = np.unique(tids); idx = {c: np.where(tids == c)[0] for c in uniq}
    rng = np.random.default_rng(0)
    reps = []
    for _ in range(n):
        ii = np.concatenate([idx[c] for c in rng.choice(uniq, len(uniq), replace=True)])
        if len(np.unique(y[ii])) == 2:
            reps.append(roc_auc_score(y[ii], s[ii]))
    return roc_auc_score(y, s), float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))


def load_records(run_ids, model):
    """Per-case records for a model across run_ids: case id, baseline/self-refine/HealthGuard
    scores under both graders."""
    recs = []
    for rid in run_ids:
        rd = run_dir(rid)
        base = {(b["case_id"], b["model"]): b["score"] for b in json.loads((rd / "baseline_results.json").read_text())}
        cg = {(r["case_id"], r["model"], r["condition"]): r["claude_score"]
              for r in json.loads((rd / "cross_grade.json").read_text())} if (rd / "cross_grade.json").exists() else {}
        for p in json.loads((rd / "phase1_results.json").read_text()):
            if p["model"] != model:
                continue
            cid = p["case_id"]; b = base[(cid, model)]
            rec = {"cid": cid, "base_g4": b, "hg_g4": p["healthguard"]["score"], "sr_g4": p["self_refine"]["score"]}
            bcl = cg.get((cid, model, "baseline")); hcl = cg.get((cid, model, "healthguard")); scl = cg.get((cid, model, "self_refine"))
            if bcl is not None:
                rec.update(base_cl=bcl, hg_cl=hcl if hcl is not None else b, sr_cl=scl if scl is not None else b)
            recs.append(rec)
    return recs


# ---------------------------------------------------------------- Fig: amplifier
def fig_amplifier():
    FRONT = ["lmb_50f", "lmb_50g"]
    models = [("gpt-5.4", "GPT-5.4"), ("gpt-5.4-mini", "GPT-5.4-mini")]
    R = {m: load_records(FRONT, m) for m, _ in models}

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.0, 3.0))
    fig.subplots_adjust(wspace=0.42)

    # Panel A: baseline vs HealthGuard mean rubric score, GPT-4.1 grader.
    x = np.arange(len(models)); w = 0.36
    for i, (m, _) in enumerate(models):
        b = 100 * np.mean([r["base_g4"] for r in R[m]]); h = 100 * np.mean([r["hg_g4"] for r in R[m]])
        axA.bar(i - w/2, b, w, color=BASE, edgecolor="white")
        axA.bar(i + w/2, h, w, color=HG, edgecolor="white")
        axA.annotate("", xy=(i + w/2, h), xytext=(i - w/2, b), arrowprops=dict(arrowstyle="->", color=MUT, lw=0.9))
        axA.text(i, max(b, h) + 3, f"+{h-b:.1f}", ha="center", fontsize=8.5, color=HG, fontweight="bold")
        axA.text(i - w/2, b - 5, f"{b:.0f}", ha="center", va="top", fontsize=7.6, color="white")
    axA.set_xticks(x); axA.set_xticklabels([lab for _, lab in models])
    axA.set_ylabel("study-defined rubric score (%)")
    axA.set_ylim(0, 84)
    axA.set_title("(a) Rubric score before and after revision", fontsize=9)
    axA.legend([plt.Rectangle((0, 0), 1, 1, color=BASE), plt.Rectangle((0, 0), 1, 1, color=HG)],
               ["baseline", "+ HealthGuard"], frameon=False, loc="upper right",
               ncol=1, handlelength=1.1, bbox_to_anchor=(1.0, 1.02))

    # Panel B: paired lift with patient-case-clustered CIs, one row per condition, both graders.
    def lift(recs, cond, gk):
        rr = [r for r in recs if f"base_{gk}" in r]
        return boot_clustered([r["cid"] for r in rr], [r[f"{cond}_{gk}"] - r[f"base_{gk}"] for r in rr])
    pooled = R["gpt-5.4"] + R["gpt-5.4-mini"]
    rows = [(2, lambda gk: lift(R["gpt-5.4"], "hg", gk)),
            (1, lambda gk: lift(R["gpt-5.4-mini"], "hg", gk)),
            (0, lambda gk: lift(pooled, "sr", gk))]
    for y, fn in rows:
        for gk, col, off in [("g4", G4, 0.16), ("cl", CL, -0.16)]:
            mid, lo, hi = fn(gk)
            axB.plot([lo, hi], [y + off, y + off], color=col, lw=1.5)
            axB.plot(mid, y + off, "o", color=col, ms=5)
            axB.text(hi + 0.5, y + off, f"{mid:+.1f}", va="center", fontsize=7.2, color=col)
    axB.axhspan(-0.45, 0.45, color="#f0f0f0", zorder=0)
    axB.axvline(0, color="#333", lw=0.9)
    axB.set_yticks([0, 1, 2]); axB.set_yticklabels(["self-refine\n(control)", "GPT-5.4-mini", "GPT-5.4"])
    axB.set_ylim(-0.6, 2.6); axB.set_xlim(-3, 18)
    axB.set_xlabel("HealthGuard minus baseline (rubric points)")
    axB.set_title("(b) Paired revision lift, both graders", fontsize=9)
    axB.legend([plt.Line2D([0], [0], color=G4, marker="o"), plt.Line2D([0], [0], color=CL, marker="o")],
               ["GPT-4.1 grader", "Claude grader"], frameon=False, loc="lower right", fontsize=7, handlelength=1.2)
    _save(fig, "fig_amplifier")


# ---------------------------------------------------------------- Fig: configuration comparison
def stats_for(run_ids, model):
    d_g4, d_cl, cids, bases = [], [], [], []
    cids_cl = []
    for rid in run_ids:
        rd = run_dir(rid)
        base = {(b["case_id"], b["model"]): b["score"] for b in json.loads((rd / "baseline_results.json").read_text())}
        cg = {(r["case_id"], r["model"], r["condition"]): r["claude_score"]
              for r in json.loads((rd / "cross_grade.json").read_text())} if (rd / "cross_grade.json").exists() else {}
        for p in json.loads((rd / "phase1_results.json").read_text()):
            if p["model"] != model:
                continue
            cid = p["case_id"]; b = base[(cid, model)]; bases.append(b)
            d_g4.append(p["healthguard"]["score"] - b); cids.append(cid)
            h = cg.get((cid, model, "healthguard")); bb = cg.get((cid, model, "baseline"))
            if h is not None and bb is not None:
                d_cl.append(h - bb); cids_cl.append(cid)
    lg, log, hig = boot_clustered(cids, d_g4)
    lc, loc, hic = boot_clustered(cids_cl, d_cl) if d_cl else (np.nan, np.nan, np.nan)
    help_ = sum(1 for x in d_g4 if x > 0.005); harm = sum(1 for x in d_g4 if x < -0.005)
    return dict(base=100*np.mean(bases), lg=lg, log=log, hig=hig, lc=lc, loc=loc, hic=hic,
                help=help_, harm=harm, n=len(d_g4))


def fig_capability():
    pts = [(["lmb_local3b"], "ollama/llama3.2:3b", "Llama-3.2\n3B"),
           (["lmb_local"], "ollama/gemma2:9b-instruct-q4_K_M", "Gemma-2\n9B"),
           (["lmb_50f", "lmb_50g"], "gpt-5.4-mini", "GPT-5.4\nmini"),
           (["lmb_50f", "lmb_50g"], "gpt-5.4", "GPT-5.4")]
    D = [(lab, stats_for(r, m)) for r, m, lab in pts]
    x = np.arange(len(D))

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.0, 3.0))
    fig.subplots_adjust(wspace=0.42)

    # Panel A: mean change per configuration, both graders, clustered CIs, categorical (no trend line).
    for gk, col, off, name in [("g", G4, -0.11, "GPT-4.1"), ("c", CL, 0.11, "Claude")]:
        mids = [d["l" + gk] for _, d in D]
        yl = [d["l" + gk] - d["lo" + gk] for _, d in D]; yh = [d["hi" + gk] - d["l" + gk] for _, d in D]
        axA.errorbar(x + off, mids, yerr=[yl, yh], fmt="o", color=col, capsize=3, ms=5, lw=1.3, label=name)
    axA.axhline(0, color="#333", lw=0.9)
    axA.set_xticks(x); axA.set_xticklabels([f"{lab}\nbase {d['base']:.0f}%" for lab, d in D], fontsize=6.4)
    axA.set_ylabel("HealthGuard minus baseline (rubric points)")
    axA.set_title("(a) Revision change by configuration", fontsize=9)
    axA.legend(frameon=False, loc="upper left", handlelength=1.2)

    # Panel B: responses improved vs worsened (>0.5 pt deadband), GPT-4.1 grader, categorical.
    w = 0.38
    for i, (lab, d) in enumerate(D):
        axB.bar(i - w/2, d["help"], w, color=POS, edgecolor="white")
        axB.bar(i + w/2, d["harm"], w, color=NEG, edgecolor="white")
        axB.text(i - w/2, d["help"] + 0.6, str(d["help"]), ha="center", fontsize=6.9, color=POS)
        axB.text(i + w/2, d["harm"] + 0.6, str(d["harm"]), ha="center", fontsize=6.9, color=NEG)
    axB.set_xticks(x); axB.set_xticklabels([lab for lab, _ in D], fontsize=7.0)
    axB.set_ylabel("responses (GPT-4.1 grader)")
    axB.set_title("(b) Responses improved vs worsened (>0.5 pt)", fontsize=9)
    axB.legend([plt.Rectangle((0, 0), 1, 1, color=POS), plt.Rectangle((0, 0), 1, 1, color=NEG)],
               ["improved", "worsened"], frameon=False, loc="upper left", handlelength=1.1)
    _save(fig, "fig_capability")


# ---------------------------------------------------------------- Fig: flag & escalate
def fig_flagescalate():
    from sklearn.metrics import roc_curve

    def load_det(fname):
        d = json.loads((run_dir("mcr_full2") / fname).read_text())
        return (np.array([r["y"] for r in d]), np.array([r["soundness"] for r in d]),
                np.array([r["trace_id"] for r in d]))

    yL, sL, tL = load_det("detection_gpt-54-mini.json")
    yC, sC, tC = load_det("detection_claude-sonnet-4-6.json")
    aL, loL, hiL = auc_clustered(yL, sL, tL)
    aC, loC, hiC = auc_clustered(yC, sC, tC)
    cm = json.loads((run_dir("mcr_full2") / "compute_matched_det.json").read_text())
    k10 = cm["critic_by_k"]["10"]; aK, loK, hiK = k10["auc"], k10["auc_ci95"][0], k10["auc_ci95"][1]

    dets = [("critic\nk=10\nn=207", aK, loK, hiK, BASE),
            ("GPT-5.4-mini\ngrounding\nn=207", aL, loL, hiL, HG),
            ("Claude\ngrounding\nn=40", aC, loC, hiC, HG)]
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.0, 3.0))
    fig.subplots_adjust(wspace=0.34)
    for i, (lab, m, lo, hi, col) in enumerate(dets):
        axA.bar(i, m, 0.6, color=col, edgecolor="white")
        axA.errorbar(i, m, yerr=[[max(0, m - lo)], [max(0, hi - m)]], fmt="none", ecolor="#333", lw=1, capsize=2.5)
        axA.text(i, min(hi + 0.012, 0.985), f"{m:.2f}", ha="center", fontsize=7.4)
    axA.axhline(0.5, color="#999", ls=":", lw=0.8)
    axA.set_xticks(range(len(dets))); axA.set_xticklabels([d[0] for d in dets])
    axA.set_ylabel("original-vs-corrupted detection (AUC)")
    axA.set_ylim(0.5, 1.0)
    axA.set_title("(a) Grounding-based trace discrimination", fontsize=9)

    # Panel B: in-sample operating curve from the GPT-5.4-mini grounding score (positive class = corrupted).
    fpr, tpr, _ = roc_curve(1 - yL, 1 - sL)
    axB.plot(fpr * 100, tpr * 100, color=G4, lw=1.9)
    axB.plot([0, 100], [0, 100], ls=":", color="#999", lw=0.8)
    j = int(np.argmax(tpr - fpr))

    def catch(th):
        idx = np.where(tpr >= th)[0]
        return float(fpr[idx[0]]) if len(idx) else 1.0
    yj, fj = tpr[j] * 100, fpr[j] * 100
    c90 = catch(0.90) * 100
    print(f"  [flag fig] mini AUC={aL:.3f} [{loL:.3f},{hiL:.3f}] n=207 | "
          f"Claude AUC={aC:.3f} [{loC:.3f},{hiC:.3f}] n=40 | critic k10={aK:.3f} | "
          f"Youden {yj:.0f}%@{fj:.0f}%FP | catch90 @{c90:.0f}%FP")
    for fx, ty, lab, dxy in [(fj, yj, f"Youden\n{yj:.0f}% @ {fj:.0f}% FP", (8, -14)),
                             (c90, 90, f"catch 90%\n@ {c90:.0f}% FP", (4, -18))]:
        axB.plot(fx, ty, "o", color=NEG, ms=5)
        axB.annotate(lab, (fx, ty), textcoords="offset points", xytext=dxy, fontsize=6.6, color="#333")
    axB.set_xlabel("false-flag rate: sound traces escalated (%)")
    axB.set_ylabel("corrupted traces caught (%)")
    axB.set_title("(b) Flag-and-escalate operating curve (in-sample)", fontsize=9)
    axB.set_xlim(0, 100); axB.set_ylim(0, 101)
    _save(fig, "fig_flagescalate")


# ---------------------------------------------------------------- Fig: architecture
def fig_pipeline():
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    fig, ax = plt.subplots(figsize=(7.1, 2.75))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")

    def box(x, y, w, h, text, fc, ec, fs=7.6, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12",
                                    linewidth=1.1, facecolor=fc, edgecolor=ec))
        ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs,
                fontweight="bold" if bold else "normal", color="#1c2430")

    def arrow(p, q, col="#444"):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=11,
                                     lw=1.2, color=col, shrinkA=1, shrinkB=1))

    box(0.15, 2.45, 1.7, 1.1, "Patient\ncase", "#eef1f4", "#9AA7B0")
    box(2.25, 2.45, 1.8, 1.1, "Base model\ndraft answer", "#e7edf3", "#2A4D77")
    box(4.35, 1.75, 2.2, 2.5, "Ground-truth-free\naudit\n\ncoverage lenses\n(mgmt / work-up /\nprognosis)\n+ grounding / safety",
        "#eaf3ee", "#2E6B43", fs=6.8)
    box(6.95, 3.35, 2.9, 1.35, "Revise + verify-repair loop\nimproved answer", "#dcebe1", "#2E6B43", fs=7.4, bold=True)
    box(6.95, 1.25, 2.9, 1.35, "Flag unsound / unsafe claims\nclinician review", "#f6ead6", "#B06C3C", fs=7.4, bold=True)

    arrow((1.85, 3.0), (2.25, 3.0))
    arrow((4.05, 3.0), (4.35, 3.0))
    arrow((6.55, 3.4), (6.95, 4.0), "#2E6B43")
    arrow((6.55, 2.6), (6.95, 1.95), "#B06C3C")

    ax.text(5.45, 1.5, "same base model  /  no answer key", ha="center", fontsize=6.6, color=MUT, style="italic")
    ax.text(8.4, 4.9, "AUTONOMOUS  (above the competence floor)", ha="center", fontsize=6.6, color="#2E6B43", fontweight="bold")
    ax.text(8.4, 0.75, "HUMAN-IN-THE-LOOP  (judgment)", ha="center", fontsize=6.6, color="#B06C3C", fontweight="bold")
    _save(fig, "fig_pipeline")


# ---------------------------------------------------------------- Fig: experimental design
def fig_design():
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    fig, ax = plt.subplots(figsize=(5.9, 6.4))
    ax.set_xlim(0, 10); ax.set_ylim(0, 12); ax.axis("off")

    def box(x, y, w, h, text, fc, ec, fs=6.6, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.10",
                                    linewidth=1.0, facecolor=fc, edgecolor=ec))
        ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs,
                fontweight="bold" if bold else "normal", color="#1c2430")

    def arrow(p, q, col="#555", lw=1.1):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=9, lw=lw,
                                     color=col, shrinkA=1, shrinkB=1))

    box(2.8, 10.85, 4.4, 0.9, "Patient case\n(narrative + question)", "#eef1f4", "#9AA7B0", fs=6.8)
    box(3.2, 9.35, 3.6, 0.9, "Base model\ndraft answer", "#e7edf3", "#2A4D77", fs=6.8)
    arrow((5.0, 10.85), (5.0, 10.25))

    arrow((4.1, 9.35), (1.55, 8.72), BASE)
    arrow((5.9, 9.35), (8.45, 8.72), SR)
    arrow((5.0, 9.35), (5.0, 8.37), HG)

    box(0.15, 7.9, 2.8, 0.8, "Baseline\n(draft as-is)", "#eef1f4", "#9AA7B0", fs=6.3)
    box(7.05, 7.9, 2.8, 0.8, "Self-refine control\n(one generic pass)", "#f6efdc", "#C29B45", fs=6.3)

    cx, sw = 5.0, 3.5
    steps = [("Coverage audit\n(mgmt / work-up / prognosis)", 7.55),
             ("Grounding audit", 6.30),
             ("Revise", 5.05),
             ("Verify-repair loop", 3.80),
             ("Improved answer", 2.55)]
    prev_y = None
    for txt, y in steps:
        box(cx - sw/2, y, sw, 0.80, txt, "#eaf3ee", "#2E6B43", fs=6.2)
        if prev_y is not None:
            arrow((cx, prev_y), (cx, y + 0.80), HG)
        prev_y = y
    ax.text(2.98, 5.45, "HealthGuard", fontsize=7.2, color=HG, fontweight="bold",
            rotation=90, ha="center", va="center")

    box(2.4, 0.7, 5.2, 1.0, "GPT-4.1  +  Claude graders\n(rubric score)", "#e7edf3", "#2A4D77", fs=6.8)
    arrow((cx, 2.55), (cx, 1.72), HG)
    arrow((1.55, 7.9), (3.4, 1.72), BASE)
    arrow((8.45, 7.9), (6.6, 1.72), SR)

    ax.text(5.0, 0.18, "the same fixed draft feeds all three arms, so structure is the only variable",
            ha="center", fontsize=6.2, color=MUT, style="italic")
    _save(fig, "fig_design")


def main():
    # The procedural diagrams (fig_pipeline, fig_design) are produced by
    # figures/generate_manuscript_figures.py; this module owns the result figures.
    OUT.mkdir(exist_ok=True)
    print("building result figures ->")
    fig_amplifier()
    fig_capability()
    fig_flagescalate()
    print("done.")


if __name__ == "__main__":
    main()
