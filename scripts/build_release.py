"""Assemble paper/release/where-to-edit-v1/ and a deterministic zip of it.

Usage: python -I paper/scripts/build_release.py
"""
import hashlib
import os
import shutil
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)
OUT = os.path.join(PAPER, "release", "where-to-edit-v1")

FILES = {
    "README.md": "release_README.md",
    "writeup.md": "writeup.md",
    "REPRODUCE_REPORT.md": "REPRODUCE_REPORT.md",
    "data/where_to_edit.jsonl": "data/where_to_edit.jsonl",
    "data/hard_subset.txt": "data/hard_subset.txt",
    "results/baselines.json": "results/baselines.json",
    "results/per_task.jsonl": "results/per_task.jsonl",
    "results/sensitivity.json": "results/sensitivity.json",
    "results/parity.json": "results/parity.json",
    "skill/where-to-edit/SKILL.md": "skill/where-to-edit/SKILL.md",
    "figures/fig1_named_vs_not.png": "figures/fig1_named_vs_not.png",
    "figures/fig1_data.json": "figures/fig1_data.json",
    "skill/where-to-edit/scripts/locate.py": "skill/where-to-edit/scripts/locate.py",
}
for p in ("data/hard_subset_swebench_verified.txt", "swebench/where_to_edit_swebench_verified.jsonl",
          "swebench/results/baselines.json", "swebench/results/per_task.jsonl",
          "results/graph_embed.json", "results/graph_embed_per_task.jsonl", "results/seed_reach.json",
          "results/seed_reach_per_task.jsonl", "results/e2e.json", "results/e2e_per_task.jsonl",
          "results/adk_wrapper.json", "skill/prompt_paragraph.md", "examples/predictions_skill.jsonl",
          "examples/predictions_grep.jsonl", "figures/fig2_example.png", "figures/fig2_data.json", "subtitle.txt",
          "results/anatomy.json", "results/anatomy_per_task.jsonl", "results/e2e_31b.json", "results/e2e_31b_per_task.jsonl",
          "results/oracle_build.json", "figures/fig_outcomes.png", "figures/fig_outcomes_data.json"):
    FILES[p] = p
for s in ("wte_common.py", "build_labels.py", "run_baselines.py", "sensitivity.py",
          "parity_check.py", "make_figure.py", "reproduce_numbers.py", "build_release.py",
          "run_swebench.py", "run_graph_embed.py", "run_seed_reach.py", "analyze_e2e.py", "adk_wrapper_check.py",
          "score.py", "export_predictions.py", "make_example_figure.py", "analyze_failures.py", "make_anatomy_figure.py",
          "build_oracle_skill.py"):
    FILES["scripts/" + s] = "scripts/" + s


def main():
    shutil.rmtree(OUT, ignore_errors=True)
    for dst, src in sorted(FILES.items()):
        d = os.path.join(OUT, dst)
        os.makedirs(os.path.dirname(d), exist_ok=True)
        shutil.copyfile(os.path.join(PAPER, src), d)
    zp = OUT + ".zip"
    with zipfile.ZipFile(zp, "w") as zf:
        for dst in sorted(FILES):
            info = zipfile.ZipInfo("where-to-edit-v1/" + dst, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            with open(os.path.join(OUT, dst), "rb") as f:
                zf.writestr(info, f.read())
    with open(zp, "rb") as f:
        h = hashlib.sha256(f.read()).hexdigest()
    # guard: the zip must ship the current skill, and the writeup must have no unfilled placeholder
    with open(os.path.join(PAPER, "skill", "where-to-edit", "scripts", "locate.py"), "rb") as f:
        src = f.read()
    with zipfile.ZipFile(zp) as zf:
        assert zf.read("where-to-edit-v1/skill/where-to-edit/scripts/locate.py") == src, "stale locate.py in release"
    with open(os.path.join(PAPER, "writeup.md"), encoding="utf-8") as f:
        if "<<" in f.read():
            print("WARNING: writeup.md still contains a <<...>> placeholder; not a final release")
    print(zp, len(FILES), "files", os.path.getsize(zp), "bytes", h)
    print("locate.py", hashlib.sha256(src).hexdigest())


if __name__ == "__main__":
    sys.exit(main())
