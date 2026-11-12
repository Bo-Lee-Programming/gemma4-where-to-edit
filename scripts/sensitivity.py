"""Hyper-parameter sensitivity and leave-one-repo-out selection for the
BM25+graph localizer, plus the 'distractor name' analysis.

Usage: python -I paper/scripts/sensitivity.py
Writes paper/results/sensitivity.json
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_baselines import FILE_K, FUNC_K, boot_ci, mean, rank_metrics  # noqa: E402
from wte_common import PAPER, Repo, load_labels, load_tasks, locate, query_text  # noqa: E402

BS = [0.5, 0.75, 1.0]
LAMS = [0.0, 0.25, 0.5, 1.0]


def main():
    tasks = {t["instance_id"]: t for t in load_tasks()}
    labels = load_labels()
    repos = {}
    rows = []
    distract = []
    cache_key, cache = None, None
    for lab in labels:
        t = tasks[lab["instance_id"]]
        key = (t["repo"], t["base_commit"])
        if key != cache_key:
            files, _ = repos.setdefault(t["repo"], Repo(t["repo"])).snapshot(t["base_commit"])
            cache_key, cache = key, {b: locate.Index(files, b=b) for b in BS}
        q = query_text(t)
        gold_f = lab["gold_files"]
        gold_s = [g["sid"] for g in lab["gold_symbols"]]
        row = {"id": lab["instance_id"], "repo": lab["repo"], "res": {}}
        for b in BS:
            idx = cache[b]
            _, bs, bscore = idx.bm25(q)
            for lam in LAMS:
                gr = idx.graph(bscore, lam)
                r = {}
                if gold_f:
                    r["file"] = rank_metrics(locate.files_from_scopes(gr, idx.sid2path), gold_f, FILE_K)
                if gold_s:
                    r["func"] = rank_metrics(gr, gold_s, FUNC_K)
                row["res"]["b=%s,lam=%s" % (b, lam)] = r
        rows.append(row)
        # distractors: definitions named in the issue that are not edit targets
        idx = cache[0.75]
        ids = locate.explicit_identifiers(q)
        named_defs = sorted({s.sid for s in idx.scopes if s.name in ids})
        if gold_s:
            distract.append({"id": lab["instance_id"], "n_named_defs": len(named_defs),
                             "named_gold": sum(1 for s in named_defs if s in gold_s)})

    grid = {}
    for cfg in rows[0]["res"]:
        f = [r["res"][cfg]["file"]["hit@1"] for r in rows if "file" in r["res"][cfg]]
        s = [r["res"][cfg]["func"]["mrr"] for r in rows if "func" in r["res"][cfg]]
        s10 = [r["res"][cfg]["func"]["hit@10"] for r in rows if "func" in r["res"][cfg]]
        grid[cfg] = {"file_hit@1": mean(f), "func_mrr": mean(s), "func_hit@10": mean(s10)}
    # leave-one-repo-out: pick cfg by func MRR on the other repos, score held-out repo
    loro_vals, picks = {"func_mrr": [], "func_hit@10": [], "file_hit@1": []}, {}
    for held in sorted({r["repo"] for r in rows}):
        train = [r for r in rows if r["repo"] != held]
        best = max(grid, key=lambda c: (mean([r["res"][c]["func"]["mrr"] for r in train if "func" in r["res"][c]]), c))
        picks[held] = best
        for r in rows:
            if r["repo"] == held:
                if "func" in r["res"][best]:
                    loro_vals["func_mrr"].append(r["res"][best]["func"]["mrr"])
                    loro_vals["func_hit@10"].append(r["res"][best]["func"]["hit@10"])
                if "file" in r["res"][best]:
                    loro_vals["file_hit@1"].append(r["res"][best]["file"]["hit@1"])
    loro = {k: {"mean": mean(v), "ci95": boot_ci(v), "n": len(v)} for k, v in loro_vals.items()}
    with_named = [d for d in distract if d["n_named_defs"] > 0]
    dist = {
        "tasks_with_func_target": len(distract),
        "issue_names_a_defined_symbol": len(with_named),
        "of_which_none_is_gold": sum(1 for d in with_named if d["named_gold"] == 0),
        "named_defs_total": sum(d["n_named_defs"] for d in with_named),
        "named_defs_gold": sum(d["named_gold"] for d in with_named),
    }
    out = {"grid": grid, "loro": loro, "loro_picks": picks, "default": "b=0.75,lam=0.5",
           "distractors": dist}
    with open(os.path.join(PAPER, "results", "sensitivity.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    for c, v in sorted(grid.items()):
        print("%-16s file_hit@1 %.3f  func_mrr %.3f  func_hit@10 %.3f" % (c, v["file_hit@1"], v["func_mrr"], v["func_hit@10"]))
    print("LORO", json.dumps(loro), picks)
    print("distractors", dist)


if __name__ == "__main__":
    main()
