"""Figure 1: function Hit@10 of grep vs BM25 + graph on both benchmarks, split by whether the task text
names an edited unit.  Reads paper/results/baselines.json and paper/swebench/results/baselines.json; writes
paper/figures/fig1_named_vs_not.png and fig1_data.json (the plotted values).

Usage: python -I paper/scripts/make_figure.py
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PAPER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(PAPER, "figures")

# reference palette (dataviz skill, light mode): slot 1 blue, slot 2 orange
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = [("grep", "Grep (names in the task)", "#eb6834"),
          ("graph", "BM25 + graph (skill)", "#2a78d6")]
BENCH = [("competition", "Competition public tasks", ("results", "baselines.json")),
         ("swebench_verified", "SWE-bench Verified", ("swebench", "results", "baselines.json"))]
STRATA = [("symbol_named", "Text names\nan edited unit"), ("symbol_not_named", "Text names\nnone")]


def main():
    data = {}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.4), sharey=True, facecolor=SURFACE)
    w = 0.34
    for ax, (bench, title, rel) in zip(axes, BENCH):
        with open(os.path.join(PAPER, *rel), encoding="utf-8") as f:
            R = json.load(f)
        ax.set_facecolor(SURFACE)
        ax.set_title(title + ": function Hit@10", color=INK, fontsize=10, loc="left")
        ticks = []
        for i, (stratum, slabel) in enumerate(STRATA):
            S = R["strata"][stratum]["func"]
            ticks.append("%s (%d tasks)" % (slabel, S["n"]))
            for j, (m, label, color) in enumerate(SERIES):
                v = S[m]["hit@10"]
                x = i + (j - 0.5) * (w + 0.04)
                ax.bar(x, v["mean"], width=w, color=color, edgecolor=SURFACE, linewidth=2,
                       label=label if i == 0 else None, zorder=2)
                ax.errorbar(x, v["mean"], yerr=[[v["mean"] - v["ci95"][0]], [v["ci95"][1] - v["mean"]]],
                            fmt="none", ecolor=INK2, elinewidth=1, capsize=3, zorder=3)
                ax.text(x, v["ci95"][1] + 0.025, "%.2f" % v["mean"], ha="center", va="bottom",
                        color=INK, fontsize=9)
                data["%s/%s/func/hit@10/%s" % (bench, stratum, m)] = {"mean": v["mean"], "ci95": v["ci95"], "n": S["n"]}
        ax.set_xticks(range(len(STRATA)))
        ax.set_xticklabels(ticks, color=INK2)
        ax.set_ylim(0, 1.12)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.tick_params(colors=INK2, length=0)
        ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
    axes[0].set_ylabel("Share of tasks (95% CI)", color=INK2)
    axes[0].legend(loc="upper left", bbox_to_anchor=(0.0, -0.3), ncol=2, frameon=False,
                   labelcolor=INK)
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.06)
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, "fig1_named_vs_not.png"), dpi=200, facecolor=SURFACE)
    with open(os.path.join(OUT, "fig1_data.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, indent=1, sort_keys=True)
    print("wrote", os.path.join(OUT, "fig1_named_vs_not.png"))


if __name__ == "__main__":
    main()
