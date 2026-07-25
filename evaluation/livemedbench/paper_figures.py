"""Publication figures for HealthGuard, written to figures/ as vector PDFs (with PNG proxies).
Rendered at print size for a single-column layout.
All numbers are read directly from the run artifacts in evaluation/runs/.

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


def boot(d, n=5000):
    d = np.array(d, float)
    rng = np.random.default_rng(0)
    m = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return d.mean() * 100, np.percentile(m, 2.5) * 100, np.percentile(m, 97.5) * 100


def load_pooled(run_ids, model):
    """Pool run_ids; return per-condition case-level score lists for a model, both graders."""
    base_g4, hg_g4, sr_g4 = [], [], []
    base_cl, hg_cl, sr_cl = [], [], []
    for rid in run_ids:
        rd = run_dir(rid)
        base = {(b["case_id"], b["model"]): b["score"] for b in json.loads((rd / "baseline_results.json").read_text())}
        cg = {(r["case_id"], r["model"], r["condition"]): r["claude_score"]
              for r in json.loads((rd / "cross_grade.json").read_text())} if (rd / "cross_grade.json").exists() else {}
        for p in json.loads((rd / "phase1_results.json").read_text()):
            if p["model"] != model:
                continue
            cid = p["case_id"]; b = base[(cid, model)]
            base_g4.append(b); hg_g4.append(p["healthguard"]["score"]); sr_g4.append(p["self_refine"]["score"])
            bcl = cg.get((cid, model, "baseline")); hcl = cg.get((cid, model, "healthguard")); scl = cg.get((cid, model, "self_refine"))
            if bcl is not None:
                base_cl.append(bcl); hg_cl.append(hcl if hcl is not None else b); sr_cl.append(scl if scl is not None else b)
    return dict(base_g4=base_g4, hg_g4=hg_g4, sr_g4=sr_g4, base_cl=base_cl, hg_cl=hg_cl, sr_cl=sr_cl)


# ---------------------------------------------------------------- Fig: amplifier
def fig_amplifier():
    FRONT = ["lmb_50f", "lmb_50g"]
    models = [("gpt-5.4", "GPT-5.4\n(frontier)"), ("gpt-5.4-mini", "GPT-5.4-mini\n(lightweight)")]
    d = {m: load_pooled(FRONT, m) for m, _ in models}

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.0, 3.0))
    fig.subplots_adjust(wspace=0.42)

    # Panel A: absolute levels, GPT-4.1 grader, base -> HealthGuard
    x = np.arange(len(models)); w = 0.36
    for i, (m, _) in enumerate(models):
        b = 100 * np.mean(d[m]["base_g4"]); h = 100 * np.mean(d[m]["hg_g4"])
        axA.bar(i - w/2, b, w, color=BASE, edgecolor="white")
        axA.bar(i + w/2, h, w, color=HG, edgecolor="white")
        axA.annotate("", xy=(i + w/2, h), xytext=(i - w/2, b),
                     arrowprops=dict(arrowstyle="->", color=MUT, lw=0.9))
        axA.text(i, max(b, h) + 3, f"+{h-b:.1f}", ha="center", fontsize=8.5, color=HG, fontweight="bold")
        axA.text(i - w/2, b - 5, f"{b:.0f}", ha="center", va="top", fontsize=7.6, color="white")
    axA.set_xticks(x); axA.set_xticklabels([lab for _, lab in models])
    axA.set_ylabel("LiveMedBench rubric score (%)")
    axA.set_ylim(0, 84)
    axA.set_title("(a) Already-good models, made better", fontsize=9)
    axA.legend([plt.Rectangle((0, 0), 1, 1, color=BASE), plt.Rectangle((0, 0), 1, 1, color=HG)],
               ["bare model", "+ HealthGuard"], frameon=False, loc="upper right",
               ncol=1, handlelength=1.1, bbox_to_anchor=(1.0, 1.02))

    # Panel B: lift, one row per {gpt-5.4, mini, self-refine}, both graders per row
    def lift(m, kind, gk):
        if kind == "hg":
            return boot([h - b for h, b in zip(d[m][f"hg_{gk}"], d[m][f"base_{gk}"])])
        dd = []
        for mm, _ in models:
            dd += [s - b for s, b in zip(d[mm][f"sr_{gk}"], d[mm][f"base_{gk}"])]
        return boot(dd)
    rowdefs = [(2, "GPT-5.4", "gpt-5.4", "hg"), (1, "GPT-5.4-mini", "gpt-5.4-mini", "hg"),
               (0, "self-refine\n(control)", None, "sr")]
    for y, lab, m, kind in rowdefs:
        for gk, col, off in [("g4", G4, 0.16), ("cl", CL, -0.16)]:
            mid, lo, hi = lift(m, kind, gk)
            axB.plot([lo, hi], [y + off, y + off], color=col, lw=1.5)
            axB.plot(mid, y + off, "o", color=col, ms=5)
            axB.text(hi + 0.5, y + off, f"{mid:+.1f}", va="center", fontsize=7.2, color=col)
    axB.axhspan(-0.45, 0.45, color="#f0f0f0", zorder=0)
    axB.axvline(0, color="#333", lw=0.9)
    axB.set_yticks([0, 1, 2]); axB.set_yticklabels(["self-refine\n(control)", "GPT-5.4-mini", "GPT-5.4"])
    axB.set_ylim(-0.6, 2.6); axB.set_xlim(-3, 17)
    axB.set_xlabel("HealthGuard - bare model (rubric pts)")
    axB.set_title("(b) Held-out lift, both graders", fontsize=9)
    axB.legend([plt.Line2D([0], [0], color=G4, marker="o"), plt.Line2D([0], [0], color=CL, marker="o")],
               ["GPT-4.1 grader", "Claude grader"], frameon=False, loc="lower right", fontsize=7, handlelength=1.2)
    _save(fig, "fig_amplifier")


# ---------------------------------------------------------------- Fig: capability curve
def stats_for(run_ids, model):
    d_g4, d_cl, bases = [], [], []
    for rid in run_ids:
        rd = run_dir(rid)
        base = {(b["case_id"], b["model"]): b["score"] for b in json.loads((rd / "baseline_results.json").read_text())}
        cg = {(r["case_id"], r["model"], r["condition"]): r["claude_score"]
              for r in json.loads((rd / "cross_grade.json").read_text())} if (rd / "cross_grade.json").exists() else {}
        for p in json.loads((rd / "phase1_results.json").read_text()):
            if p["model"] != model:
                continue
            b = base[(p["case_id"], model)]; bases.append(b)
            d_g4.append(p["healthguard"]["score"] - b)
            h = cg.get((p["case_id"], model, "healthguard")); bb = cg.get((p["case_id"], model, "baseline"))
            if h is not None and bb is not None:
                d_cl.append(h - bb)
    lg, log, hig = boot(d_g4); lc, loc, hic = boot(d_cl) if d_cl else (np.nan,)*3
    help_ = sum(1 for x in d_g4 if x > 0.005); harm = sum(1 for x in d_g4 if x < -0.005)
    return dict(base=100*np.mean(bases), lg=lg, log=log, hig=hig, lc=lc, loc=loc, hic=hic,
                ratio=help_/max(1, harm), help=help_, harm=harm)


def fig_capability():
    pts = [(["lmb_local3b"], "ollama/llama3.2:3b", "Llama-3.2\n3B"),
           (["lmb_local"], "ollama/gemma2:9b-instruct-q4_K_M", "Gemma-2\n9B"),
           (["lmb_50f", "lmb_50g"], "gpt-5.4-mini", "GPT-5.4\nmini"),
           (["lmb_50f", "lmb_50g"], "gpt-5.4", "GPT-5.4")]
    D = [(lab, stats_for(r, m)) for r, m, lab in pts]
    xs = [d["base"] for _, d in D]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(6.5, 2.9))

    axA.fill_between([-3, 70], 0, 24, color=POS, alpha=0.05)
    axA.fill_between([-3, 70], -34, 0, color=NEG, alpha=0.05)
    for key, col, name, dx in [("lg", G4, "GPT-4.1", -0.7), ("lc", CL, "Claude", 0.7)]:
        lo = "lo" + key[1]; hi = "hi" + key[1]
        ys = [d[key] for _, d in D]
        yl = [d[key] - d[lo] for _, d in D]; yh = [d[hi] - d[key] for _, d in D]
        axA.errorbar([x+dx for x in xs], ys, yerr=[yl, yh], fmt="o-", color=col, lw=1.5, capsize=3, ms=5, label=name)
    axA.axhline(0, color="#333", lw=0.9)
    offs = {"Llama-3.2\n3B": (4, 3, "left"), "Gemma-2\n9B": (-15, -4, "right"),
            "GPT-5.4\nmini": (-4, -21, "right"), "GPT-5.4": (3, 9, "left")}
    for lab, d in D:
        dx, dy, ha = offs[lab]
        axA.annotate(lab, (d["base"], (d["lg"] + d["lc"]) / 2), textcoords="offset points",
                     xytext=(dx, dy), ha=ha, fontsize=6.8, color="#333")
    axA.set_xlim(-3, 70); axA.set_ylim(-34, 24)
    axA.set_xlabel("base model competence (bare score, %)")
    axA.set_ylabel("HealthGuard - bare (rubric pts)")
    axA.set_title("(a) The lift flips sign at low competence", fontsize=9)
    axA.legend(frameon=False, loc="lower right", handlelength=1.3)

    ratios = [d["ratio"] for _, d in D]
    cols = [POS if r >= 1 else NEG for r in ratios]
    axB.plot(xs, ratios, "-", color="#999", lw=1.3, zorder=1)
    axB.scatter(xs, ratios, c=cols, s=55, zorder=2, edgecolor="white", linewidth=0.8)
    axB.axhline(1.0, ls="--", color="#333", lw=0.9)
    axB.text(68, 1.08, "break-even", ha="right", fontsize=7.2, color=MUT)
    for lab, d in D:
        axB.annotate(f"{d['help']}:{d['harm']}", (d["base"], d["ratio"]), textcoords="offset points",
                     xytext=(0, 8), ha="center", fontsize=7.4, color="#333")
    axB.set_xlim(-3, 70); axB.set_ylim(0, max(ratios) + 1.1)
    axB.set_xlabel("base model competence (bare score, %)")
    axB.set_ylabel("help : harm  (cases helped / hurt)")
    axB.set_title("(b) Reliability rises with capability", fontsize=9)
    _save(fig, "fig_capability")


# ---------------------------------------------------------------- Fig: flag & escalate
def fig_flagescalate():
    from sklearn.metrics import roc_curve, roc_auc_score

    def load_detection(fname):
        d = json.loads((run_dir("mcr_full2") / fname).read_text())
        return np.array([r["y"] for r in d]), np.array([r["soundness"] for r in d])

    def auc_ci(y, s, n=2000):
        rng = np.random.default_rng(0)
        b = []
        for _ in range(n):
            idx = rng.integers(0, len(y), len(y))
            if len(np.unique(y[idx])) == 2:
                b.append(roc_auc_score(y[idx], s[idx]))
        return roc_auc_score(y, s), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))

    yL, sL = load_detection("detection_gpt-54-mini.json")
    yC, sC = load_detection("detection_claude-sonnet-4-6.json")
    aL, loL, hiL = auc_ci(yL, sL)
    aC, loC, hiC = auc_ci(yC, sC)

    dets = [("light\nsingle", 0.668, 0.635, 0.701, BASE),
            ("critic\nk=10", 0.696, 0.666, 0.727, BASE),
            ("big\nsingle", 0.742, 0.712, 0.776, BASE),
            ("HG\nlight", aL, loL, hiL, HG),
            ("HG\nstrong", aC, loC, hiC, HG)]
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.0, 3.0))
    fig.subplots_adjust(wspace=0.34)
    for i, (lab, m, lo, hi, col) in enumerate(dets):
        axA.bar(i, m, 0.64, color=col, edgecolor="white")
        axA.errorbar(i, m, yerr=[[max(0, m - lo)], [max(0, hi - m)]], fmt="none", ecolor="#333", lw=1, capsize=2.5)
        axA.text(i, min(hi + 0.012, 0.985), f"{m:.2f}", ha="center", fontsize=7.1)
    axA.axhline(0.5, color="#999", ls=":", lw=0.8)
    axA.set_xticks(range(len(dets))); axA.set_xticklabels([d[0] for d in dets])
    axA.set_ylabel("flawed-reasoning detection (AUC)")
    axA.set_ylim(0.5, 1.0)
    axA.set_title("(a) One grounding pass, structural detection", fontsize=9)
    axA.text(1.0, 0.535, "single call /\nself-consistency", ha="center", fontsize=6.4, color=MUT)

    # Panel B: operating curve from HealthGuard lightweight-verifier soundness (positive class = flawed)
    fpr, tpr, _ = roc_curve(1 - yL, 1 - sL)
    axB.plot(fpr * 100, tpr * 100, color=G4, lw=1.9)
    axB.plot([0, 100], [0, 100], ls=":", color="#999", lw=0.8)
    j = int(np.argmax(tpr - fpr))

    def catch(th):
        idx = np.where(tpr >= th)[0]
        return float(fpr[idx[0]]) if len(idx) else 1.0
    yj, fj = tpr[j] * 100, fpr[j] * 100
    c90 = catch(0.90) * 100
    print(f"  [flag fig] HealthGuard AUC light={aL:.3f} strong={aC:.3f} | Youden {yj:.0f}%@{fj:.0f}%FP | catch90 @{c90:.0f}%FP")
    for fx, ty, lab, dxy in [(fj, yj, f"Youden\n{yj:.0f}% @ {fj:.0f}% FP", (8, -14)),
                             (c90, 90, f"catch 90%\n@ {c90:.0f}% FP", (4, -18))]:
        axB.plot(fx, ty, "o", color=NEG, ms=5)
        axB.annotate(lab, (fx, ty), textcoords="offset points", xytext=dxy, fontsize=6.6, color="#333")
    axB.set_xlabel("false-flag rate: sound traces escalated (%)")
    axB.set_ylabel("flawed reasoning caught (%)")
    axB.set_title("(b) Flag-and-escalate operating curve", fontsize=9)
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

    # shared input (top)
    box(2.8, 10.85, 4.4, 0.9, "Patient case\n(narrative + question)", "#eef1f4", "#9AA7B0", fs=6.8)
    box(3.2, 9.35, 3.6, 0.9, "Base model\ndraft answer", "#e7edf3", "#2A4D77", fs=6.8)
    arrow((5.0, 10.85), (5.0, 10.25))

    # branch to three arms (all start at the bottom edge of the draft box, y=9.35)
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

    # graders (bottom)
    box(2.4, 0.7, 5.2, 1.0, "GPT-4.1  +  Claude graders\n(rubric score)", "#e7edf3", "#2A4D77", fs=6.8)
    arrow((cx, 2.55), (cx, 1.72), HG)
    arrow((1.55, 7.9), (3.4, 1.72), BASE)
    arrow((8.45, 7.9), (6.6, 1.72), SR)

    ax.text(5.0, 0.18, "the same fixed draft feeds all three arms, so structure is the only variable",
            ha="center", fontsize=6.2, color=MUT, style="italic")
    _save(fig, "fig_design")


def main():
    OUT.mkdir(exist_ok=True)
    print("building paper figures ->")
    fig_pipeline()
    fig_design()
    fig_amplifier()
    fig_capability()
    fig_flagescalate()
    print("done.")


if __name__ == "__main__":
    main()
