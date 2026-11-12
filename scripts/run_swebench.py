"""Where-to-Edit on SWE-bench Verified: same labeling code, same localizers, issue text only.

Labels 500 tasks from 12 repositories (problem_statement as the query; hints_text is dropped because it holds
comments written after the issue) and scores the seven CPU localizers of locate.py.
Usage:  WTE_REPOS=paper/swebench/repos python -I paper/scripts/run_swebench.py [workers]
Reads   paper/swebench/tasks.jsonl
Writes  paper/swebench/where_to_edit_swebench_verified.jsonl, paper/swebench/results/{baselines.json,per_task.jsonl}
"""
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
SWE = os.path.join(os.path.dirname(HERE), "swebench")
os.environ.setdefault("WTE_REPOS", os.path.join(SWE, "repos"))

import run_baselines as rb  # noqa: E402
from build_labels import label_one  # noqa: E402
from wte_common import Repo, locate, query_text  # noqa: E402


def one(t):
    t0 = time.time()
    t = dict(t, hints_text="")
    repo = Repo(t["repo"])
    try:
        lab = label_one(t, repo)
        files, _ = repo.snapshot(t["base_commit"])
        idx = locate.Index(files)
    except Exception as e:  # keep going; report at the end
        return {"instance_id": t["instance_id"], "error": repr(e)[-300:]}
    fcost = {p: idx.src[p].count("\n") + 1 for p in idx.paths}
    scost = {s.sid: s.end - s.start + 1 for s in idx.scopes}
    res, ids = idx.run_all(query_text(t, False))
    gold_f = [p for p in lab["gold_files"] if p in fcost]
    gold_s = [g["sid"] for g in lab["gold_symbols"] if g["sid"] in scost]
    row = {"instance_id": lab["instance_id"], "repo": lab["repo"], "features": lab["features"],
           "n_files": len(idx.paths), "n_scopes": len(idx.sids), "secs": 0}
    row["issue_hints_file"] = ({m: rb.rank_metrics(res[m][0], gold_f, rb.FILE_K, fcost) for m in rb.METHODS}
                               if gold_f else None)
    row["issue_hints_func"] = ({m: rb.rank_metrics(res[m][1], gold_s, rb.FUNC_K, scost)
                                for m in rb.METHODS if m != "bm25_file"} if gold_s else None)
    if row["issue_hints_func"] is not None:
        flat = [s.sid for p in res["bm25_file"][0] for s in idx.scopes if s.path == p]
        row["issue_hints_func"]["bm25_file"] = rb.rank_metrics(flat, gold_s, rb.FUNC_K, scost)
    # named definitions in the text: how many, and how many are edited (Section 5.3 statistic)
    named = [s.sid for s in idx.scopes if s.name in ids]
    row["n_named_defs"] = len(named)
    row["n_named_defs_edited"] = len(set(named) & set(gold_s))
    row["secs"] = round(time.time() - t0, 1)
    return {"label": lab, "row": row}


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    with open(os.path.join(SWE, "tasks.jsonl"), encoding="utf-8") as f:
        tasks = [json.loads(l) for l in f if l.strip()]
    t0 = time.time()
    with ProcessPoolExecutor(workers) as ex:
        out = list(ex.map(one, tasks, chunksize=2))
    errs = [o for o in out if "error" in o]
    good = [o for o in out if "row" in o]
    per_task = [o["row"] for o in good]
    os.makedirs(os.path.join(SWE, "results"), exist_ok=True)
    with open(os.path.join(SWE, "where_to_edit_swebench_verified.jsonl"), "w", encoding="utf-8", newline="\n") as f:
        for o in good:
            f.write(json.dumps(o["label"], sort_keys=True) + "\n")
    with open(os.path.join(SWE, "results", "per_task.jsonl"), "w", encoding="utf-8", newline="\n") as f:
        for r in per_task:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    fm = ["mrr"] + ["hit@%d" % k for k in rb.FILE_K]
    sm = ["mrr"] + ["hit@%d" % k for k in rb.FUNC_K]
    res = {"n_tasks": len(tasks), "errors": errs, "elapsed_sec": round(time.time() - t0, 1),
           "file": rb.evaluate(per_task, "issue_hints_file", fm),
           "func": rb.evaluate(per_task, "issue_hints_func", sm), "strata": {}, "paired": {}}
    strata = {"symbol_named": lambda r: r["features"]["symbol_named"],
              "symbol_not_named": lambda r: not r["features"]["symbol_named"]}
    for name, fn in strata.items():
        res["strata"][name] = {"file": rb.evaluate(per_task, "issue_hints_file", ["mrr", "hit@1", "hit@5"], fn),
                               "func": rb.evaluate(per_task, "issue_hints_func", ["mrr", "hit@1", "hit@10"], fn)}
    for level, mn in (("file", "hit@1"), ("file", "mrr"), ("func", "hit@10"), ("func", "mrr")):
        rows = [r for r in per_task if r["issue_hints_" + level] is not None]
        for a, b in (("graph", "grep"), ("bm25_func", "grep"), ("bm25_func", "bm25_file"), ("graph", "bm25_func"),
                     ("cascade", "graph"), ("rrf_all", "bm25_func")):
            res["paired"]["%s:%s:%s-%s" % (level, mn, a, b)] = rb.paired_diff(
                [r["issue_hints_" + level][a][mn] for r in rows], [r["issue_hints_" + level][b][mn] for r in rows])
    hf = [r for r in per_task if r["issue_hints_file"] is not None]
    hs = [r for r in per_task if r["issue_hints_func"] is not None]
    nd = [r for r in hs if r["n_named_defs"] > 0]
    res["stats"] = {
        "with_file_target": len(hf), "with_unit_target": len(hs),
        "symbol_named": sum(r["features"]["symbol_named"] for r in hs),
        "file_path_named": sum(r["features"]["file_path_named"] for r in hf),
        "has_traceback": sum(r["features"]["has_traceback"] for r in per_task),
        "tasks_naming_a_definition": len(nd),
        "tasks_naming_only_unedited_definitions": sum(1 for r in nd if r["n_named_defs_edited"] == 0),
        "named_defs": sum(r["n_named_defs"] for r in hs), "named_defs_edited": sum(r["n_named_defs_edited"] for r in hs),
        "grep_no_unit_found": sum(1 for r in hs if r["issue_hints_func"]["grep"]["first_rank"] is None),
        "any_method_func_hit10": sum(1 for r in hs if any(r["issue_hints_func"][m]["hit@10"] for m in rb.METHODS)),
        "median_issue_tokens": sorted(r["features"]["issue_tokens"] for r in per_task)[len(per_task) // 2],
        "per_repo": {rp: sum(1 for r in per_task if r["repo"] == rp) for rp in sorted({r["repo"] for r in per_task})},
    }
    with open(os.path.join(SWE, "results", "baselines.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=1, sort_keys=True)
    print(json.dumps(res["stats"], indent=1))
    for level, cols in (("file", ["hit@1", "hit@5", "mrr"]), ("func", ["hit@1", "hit@10", "mrr"])):
        print(level, res[level]["n"])
        for m in rb.METHODS:
            print("  %-11s" % m + "".join("%7.3f [%.2f,%.2f]" % (res[level][m][c]["mean"], *res[level][m][c]["ci95"]) for c in cols))
    for k, v in res["paired"].items():
        print("  %-34s %+.3f [%+.3f,%+.3f]" % (k, v["mean"], v["ci95"][0], v["ci95"][1]))
    print("errors", len(errs), "elapsed", res["elapsed_sec"])


if __name__ == "__main__":
    main()
