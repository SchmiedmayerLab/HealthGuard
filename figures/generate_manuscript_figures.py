"""Generate the two procedural figures used by healthguard.tex.

The diagrams describe verified control flow only. Numerical result figures are
intentionally excluded because the archived run artifacts are not available.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUT = Path(__file__).resolve().parent
BLUE = "#2A4D77"
GREEN = "#2E6B43"
ORANGE = "#B06C3C"
GRAY = "#8A949D"
TEXT = "#1C2430"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def add_box(ax, x, y, width, height, text, face, edge, size=7.2, bold=False):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.04,rounding_size=0.10",
            linewidth=1.05,
            facecolor=face,
            edgecolor=edge,
        )
    )
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=size,
        color=TEXT,
        fontweight="bold" if bold else "normal",
    )


def add_arrow(ax, start, end, color=GRAY, width=1.1):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=width,
            color=color,
            shrinkA=1,
            shrinkB=1,
        )
    )


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def pipeline_figure():
    fig, ax = plt.subplots(figsize=(7.1, 3.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")

    add_box(ax, 0.10, 2.8, 1.65, 1.2, "Patient case\n+ request", "#EEF1F4", GRAY)
    add_box(ax, 2.00, 2.8, 1.60, 1.2, "Initial model\ndraft", "#E7EDF3", BLUE)

    add_box(
        ax,
        3.95,
        3.55,
        2.55,
        1.65,
        "Claim grounding (all modes)\nDecompose the draft\nClassify each claim as\nsupported, unsupported,\nor incorrect",
        "#EAF3EE",
        GREEN,
        size=6.0,
        bold=True,
    )

    add_box(
        ax,
        3.95,
        1.45,
        2.55,
        1.45,
        "Coverage audit (revise or both)\nGeneral audit plus\noptional focused prompts",
        "#EAF3EE",
        GREEN,
        size=6.0,
    )

    add_box(
        ax,
        7.05,
        4.30,
        2.20,
        1.45,
        "Structured revision\nUse gaps + claim flags\nOptional landing check",
        "#DCEBE1",
        GREEN,
        size=6.1,
        bold=True,
    )

    add_box(
        ax,
        9.65,
        4.30,
        2.15,
        1.45,
        "Revised response",
        "#DCEBE1",
        GREEN,
        size=6.8,
        bold=True,
    )

    add_box(
        ax,
        7.05,
        1.05,
        2.20,
        1.45,
        "Flag output\nUnsupported / incorrect\nclaims + uncalibrated\nsoundness",
        "#F6EAD6",
        ORANGE,
        size=5.9,
    )

    add_box(
        ax,
        9.65,
        1.05,
        2.15,
        1.45,
        "Caller-defined\nreview or other\napplication action",
        "#F6EAD6",
        ORANGE,
        size=6.8,
        bold=True,
    )

    add_arrow(ax, (1.75, 3.4), (2.00, 3.4))
    add_arrow(ax, (3.60, 3.4), (3.95, 4.25))
    add_arrow(ax, (3.60, 3.4), (3.95, 2.15))
    add_arrow(ax, (6.50, 4.35), (7.05, 5.02), GREEN)
    add_arrow(ax, (6.50, 2.15), (7.05, 4.70), GREEN)
    add_arrow(ax, (9.25, 5.02), (9.65, 5.02), GREEN)
    add_arrow(ax, (6.50, 4.10), (7.05, 1.80), ORANGE)
    add_arrow(ax, (9.25, 1.80), (9.65, 1.80), ORANGE)

    ax.text(
        6.0,
        6.55,
        "The caller selects flag, revise, or both; HealthGuard does not automatically route cases.",
        ha="center",
        va="center",
        fontsize=7.0,
        color=TEXT,
    )
    save(fig, "fig_pipeline")


def design_figure():
    fig, ax = plt.subplots(figsize=(7.1, 3.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis("off")

    add_box(ax, 3.8, 6.65, 4.4, 0.85, "Patient case + one temperature-zero draft", "#E7EDF3", BLUE, bold=True)

    centers = [2.0, 6.0, 10.0]
    titles = [
        ("Baseline", "Stored draft\nwithout revision", GRAY, "#EEF1F4"),
        ("Self-Refine", "One generic\ncritique-and-revision call", "#C29B45", "#F6EFDC"),
        ("HealthGuard", "Grounding + coverage\naudits and revision\n(multiple targeted calls)", GREEN, "#EAF3EE"),
    ]
    for center, (title, body, edge, face) in zip(centers, titles):
        add_arrow(ax, (6.0, 6.65), (center, 5.55), edge)
        add_box(ax, center - 1.55, 4.05, 3.1, 1.45, f"{title}\n\n{body}", face, edge, size=6.9, bold=False)
        add_arrow(ax, (center, 4.05), (center, 2.75), edge)

    add_box(
        ax,
        2.15,
        1.35,
        7.7,
        1.35,
        "Criterion-level automated grading\nGPT-4.1 primary grader + Claude Sonnet 4.6 sensitivity grader",
        "#E7EDF3",
        BLUE,
        size=7.2,
        bold=True,
    )

    ax.text(
        6.0,
        0.55,
        "The fixed draft controls initial generation; the revision arms are not matched for calls, prompt specificity, or response length.",
        ha="center",
        fontsize=6.9,
        color=TEXT,
    )
    save(fig, "fig_design")


if __name__ == "__main__":
    pipeline_figure()
    design_figure()
