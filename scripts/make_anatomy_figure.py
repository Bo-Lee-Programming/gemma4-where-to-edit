"""Outcome figure: outcome of every task in every run, as stacked bars.  Reads paper/results/anatomy.json; writes
paper/figures/fig_outcomes.png and fig_outcomes_data.json (the plotted counts).

Usage: python -I paper/scripts/make_anatomy_figure.py
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PAPER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(PAPER, "figures")
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
PARTS = [("resolved", "Resolved", "#2e9d6a"), ("covered", "Every gold unit edited, tests fail", "#2a78d6"),
         ("partial", "Some gold units edited", "#8fbbe8"), ("wrong_place", "Wrong file or unit", "#eb6834"),
         ("no_patch", "No patch", "#b9b8b3")]
LABELS = {"e4b_base": "E4B baseline", "e4b_base2": "E4B baseline, repeat", "e4b_ctrl": "E4B control", "e4b_skill": "E4B skill",
          "b31_base": "31B baseline", "b31_skill": "31B skill", "b31_gold": "31B gold locations"}


def main():
    with open(os.path.join(PAPER, "results", "anatomy.json"), encoding="utf-8") as f:
        A = json.load(f)
    names = [n for n in A["order"] if n in LABELS]
    data = {}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, ax = plt.subplots(figsize=(8.0, 0.52 * len(names) + 1.3), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for i, n in enumerate(names):
        o = dict(A["runs"][n]["outcomes"])
        o["wrong_place"] = o.pop("wrong_file") + o.pop("wrong_unit")
        data[n] = {k: o[k] for k, _, _ in PARTS}
        left = 0
        for k, lab, col in PARTS:
            ax.barh(i, o[k], left=left, color=col, edgecolor=SURFACE, linewidth=2, height=0.72, label=lab if i == 0 else None, zorder=2)
            if o[k] >= 5:
                ax.text(left + o[k] / 2.0, i, str(o[k]), ha="center", va="center", fontsize=9,
                        color="#ffffff" if k in ("resolved", "covered", "wrong_place") else INK)
            left += o[k]
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([LABELS[n] for n in names], color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, A["n"])
    ax.set_xlabel("Tasks (of %d)" % A["n"], color=INK2)
    ax.tick_params(colors=INK2, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    for sp in ("top", "right", "left", "bottom"):
        ax.spines[sp].set_visible(False)
    ax.legend(loc="upper left", bbox_to_anchor=(-0.02, -0.16), ncol=3, frameon=False, labelcolor=INK, fontsize=9,
              handlelength=1.2, columnspacing=1.2)
    fig.tight_layout()
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, "fig_outcomes.png"), dpi=200, facecolor=SURFACE, bbox_inches="tight")
    with open(os.path.join(OUT, "fig_outcomes_data.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, indent=1, sort_keys=True)
    print("wrote", os.path.join(OUT, "fig_outcomes.png"))


if __name__ == "__main__":
    main()
