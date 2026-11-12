"""Figure 2: one real task where the text names the API and the fix lives behind it (rich_3521).

All values are read from the labels, the base-commit AST and results/per_task.jsonl.
Usage: python -I paper/scripts/make_example_figure.py -> paper/figures/fig2_example.png, fig2_data.json
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wte_common import PAPER, Repo, load_labels, load_tasks, locate  # noqa: E402

TASK = "rich_3521"
NAMED = "Segment.split_cells"
SURFACE, INK, INK2, LINE = "#fcfcfb", "#0b0b0b", "#52514e", "#c3c2b7"
ORANGE, BLUE = "#eb6834", "#2a78d6"


def main():
    t = {x["instance_id"]: x for x in load_tasks()}[TASK]
    lab = {x["instance_id"]: x for x in load_labels()}[TASK]
    per = {}
    with open(os.path.join(PAPER, "results", "per_task.jsonl"), encoding="utf-8") as f:
        for l in f:
            r = json.loads(l)
            per[r["instance_id"]] = r
    gold = lab["gold_symbols"][0]
    path = gold["path"]
    files, _ = Repo(t["repo"]).snapshot(t["base_commit"])
    sc = {s.qualname: s for s in locate.extract_scopes(path, files[path])}
    named = sc[NAMED]
    data = {
        "task": TASK, "title": t["problem_statement"].split("\n")[0],
        "named": NAMED, "named_lines": [named.start, named.end],
        "named_calls_gold": gold["qualname"].split(".")[-1] in named.calls,
        "gold": gold["qualname"], "gold_lines": [gold["start"], gold["end"]], "path": path,
        "n_gold_units": len(lab["gold_symbols"]),
        "rank_grep": per[TASK]["issue_hints_func"]["grep"]["first_rank"],
        "rank_skill": per[TASK]["issue_hints_func"]["graph"]["first_rank"],
    }
    assert data["named_calls_gold"] and data["n_gold_units"] == 1

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, ax = plt.subplots(figsize=(8.0, 2.7), facecolor=SURFACE)
    ax.set_xlim(0, 101)
    ax.set_ylim(0, 36)
    ax.axis("off")

    def box(x, w, title, lines, edge):
        ax.add_patch(FancyBboxPatch((x, 9), w, 22, boxstyle="round,pad=0.4,rounding_size=1.5",
                                    facecolor=SURFACE, edgecolor=edge, linewidth=1.6))
        ax.text(x + 1.5, 28.2, title, color=INK2, fontsize=9, va="top")
        for i, (txt, kw) in enumerate(lines):
            ax.text(x + 1.5, 23.2 - 4.6 * i, txt, color=INK, va="top", **kw)

    box(0.5, 27, "Task text (title)", [("“%s”" % data["title"], {"fontsize": 9.5})], LINE)
    box(32.5, 30, "What the text names", [(data["named"] + "()", {"family": "DejaVu Sans Mono", "fontsize": 9.5}),
                                        ("%s:%d–%d" % (path, named.start, named.end), {"fontsize": 9})], ORANGE)
    box(68.5, 32, "What the fix edits", [(data["gold"] + "()", {"family": "DejaVu Sans Mono", "fontsize": 9.5}),
                                       ("%s:%d–%d" % (path, gold["start"], gold["end"]), {"fontsize": 9})], BLUE)
    for x0, x1, label in ((28.3, 31.7, ""), (63.3, 67.7, "calls")):
        ax.add_patch(FancyArrowPatch((x0, 20), (x1, 20), arrowstyle="-|>", mutation_scale=12, color=INK2, lw=1.2))
        if label:
            ax.text((x0 + x1) / 2, 21.5, label, ha="center", color=INK2, fontsize=8.5)
    ax.text(50, 3.2, "Rank of the edited function:   grep on the named identifiers  #%d      "
            "BM25 + call-graph hop (skill)  #%d" % (data["rank_grep"], data["rank_skill"]),
            ha="center", color=INK, fontsize=9.5)
    out = os.path.join(PAPER, "figures")
    os.makedirs(out, exist_ok=True)
    fig.savefig(os.path.join(out, "fig2_example.png"), dpi=200, facecolor=SURFACE, bbox_inches="tight")
    with open(os.path.join(out, "fig2_data.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, indent=1, sort_keys=True)
    print(json.dumps(data))


if __name__ == "__main__":
    main()
