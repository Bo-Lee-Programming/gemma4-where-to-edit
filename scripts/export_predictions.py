"""Write example prediction files in the format scripts/score.py reads.

One JSON object per task: {"instance_id": ..., "files": [ranked paths], "functions": [ranked "path::qualname"]}.
Exports the literal-name grep baseline and the skill's ranking (BM25 over functions + one call-graph hop) for the
129 competition tasks, with full rankings so score.py reproduces results/baselines.json exactly.
Usage: python -I paper/scripts/export_predictions.py  ->  paper/examples/predictions_{grep,skill}.jsonl
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wte_common import PAPER, Repo, load_labels, load_tasks, locate, query_text  # noqa: E402

METHODS = {"grep": "grep", "skill": "graph"}


def main():
    tasks = {t["instance_id"]: t for t in load_tasks()}
    out = {k: [] for k in METHODS}
    key, idx = None, None
    for lab in load_labels():
        t = tasks[lab["instance_id"]]
        if (t["repo"], t["base_commit"]) != key:
            key = (t["repo"], t["base_commit"])
            files, _ = Repo(t["repo"]).snapshot(t["base_commit"])
            idx = locate.Index(files)
        res, _ = idx.run_all(query_text(t))
        for name, m in METHODS.items():
            out[name].append({"instance_id": lab["instance_id"], "files": res[m][0], "functions": res[m][1]})
    d = os.path.join(PAPER, "examples")
    os.makedirs(d, exist_ok=True)
    for name, rows in out.items():
        with open(os.path.join(d, "predictions_%s.jsonl" % name), "w", encoding="utf-8", newline="\n") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    print("wrote", len(out["grep"]), "tasks x", list(METHODS))


if __name__ == "__main__":
    main()
