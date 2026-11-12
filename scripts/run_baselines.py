"""Run CPU localization baselines on Where-to-Edit and write metrics with
bootstrap 95% CIs.

Usage: python -I paper/scripts/run_baselines.py
Writes paper/results/baselines.json and paper/results/per_task.jsonl
"""
import json
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wte_common import PAPER, Repo, load_labels, load_tasks, locate, query_text  # noqa: E402

METHODS = ["size_prior", "grep", "bm25_file", "bm25_func", "graph", "rrf_all", "cascade"]
FILE_K = [1, 3, 5, 10]
FUNC_K = [1, 5, 10, 20]
SEED = 20261108
B = 10000


def rank_metrics(ranked, gold, ks, cost=None):
    pos = {}
    for i, k in enumerate(ranked):  # first occurrence: a qualname can repeat (overloads, TYPE_CHECKING branches)
        pos.setdefault(k, i)
    ranks = sorted(pos[g] + 1 for g in gold if g in pos)
    m = {"mrr": 1.0 / ranks[0] if ranks else 0.0, "first_rank": ranks[0] if ranks else None}
    for k in ks:
        n_in = sum(1 for r in ranks if r <= k)
        m["hit@%d" % k] = 1.0 if n_in else 0.0
        m["recall@%d" % k] = n_in / float(len(gold))
        m["all@%d" % k] = 1.0 if n_in == len(gold) else 0.0
    if cost is not None:
        # lines an agent would read, in ranked order, before reaching the first gold unit
        if ranks:
            m["lines_to_first"] = sum(cost[x] for x in ranked[:ranks[0]])
        else:
            m["lines_to_first"] = None
    return m


def mean(xs):
    return sum(xs) / float(len(xs)) if xs else float("nan")


_IDX = {}


def _boot_idx(n, b):
    if (n, b) not in _IDX:
        _IDX[(n, b)] = np.random.default_rng(SEED + n).integers(0, n, size=(b, n))
    return _IDX[(n, b)]


def boot_ci(vals, rng=None, b=B):
    n = len(vals)
    if n == 0:
        return [float("nan")] * 2
    st = np.asarray(vals, dtype=float)[_boot_idx(n, b)].mean(axis=1)
    lo, hi = np.percentile(st, [2.5, 97.5])
    return [float(lo), float(hi)]


def paired_diff(a, b_, rng=None, b=B):
    d = np.asarray(a, dtype=float) - np.asarray(b_, dtype=float)
    st = d[_boot_idx(len(d), b)].mean(axis=1)
    lo, hi = np.percentile(st, [2.5, 97.5])
    return {"mean": float(d.mean()), "ci95": [float(lo), float(hi)],
            "p_boot_le0": float((st <= 0).mean())}


def evaluate(per_task, level, metric_names, subset=None):
    rows = [r for r in per_task if r[level] is not None and (subset is None or subset(r))]
    out = {"n": len(rows)}
    rng = random.Random(SEED)
    for meth in METHODS:
        out[meth] = {}
        for mn in metric_names:
            vals = [r[level][meth][mn] for r in rows]
            out[meth][mn] = {"mean": mean(vals), "ci95": boot_ci(vals, rng, 2000 if subset else B)}
        lt = sorted(r[level][meth]["lines_to_first"] for r in rows
                    if r[level][meth]["lines_to_first"] is not None)
        out[meth]["median_lines_to_first"] = lt[len(lt) // 2] if lt else None
        out[meth]["found_any"] = sum(1 for r in rows if r[level][meth]["first_rank"] is not None)
    return out


def main():
    tasks = {t["instance_id"]: t for t in load_tasks()}
    labels = load_labels()
    repos, snaps = {}, {}
    per_task = []
    t0 = time.time()
    for lab in labels:
        t = tasks[lab["instance_id"]]
        key = (t["repo"], t["base_commit"])
        if key not in snaps:
            repo = repos.setdefault(t["repo"], Repo(t["repo"]))
            files, _ = repo.snapshot(t["base_commit"])
            snaps.clear()  # keep memory small; tasks are grouped by repo anyway
            snaps[key] = locate.Index(files)
        idx = snaps[key]
        fcost = {p: idx.src[p].count("\n") + 1 for p in idx.paths}
        scost = {s.sid: s.end - s.start + 1 for s in idx.scopes}
        row = {"instance_id": lab["instance_id"], "repo": lab["repo"],
               "features": lab["features"], "n_files": len(idx.paths), "n_scopes": len(idx.sids)}
        for setting, hints in (("issue_hints", True),):  # hints_text is empty for all 129 public tasks
            res, ids = idx.run_all(query_text(t, hints))
            gold_f = [p for p in lab["gold_files"] if p in fcost]
            gold_s = [g["sid"] for g in lab["gold_symbols"] if g["sid"] in scost]
            row[setting + "_file"] = ({m: rank_metrics(res[m][0], gold_f, FILE_K, fcost) for m in METHODS}
                                      if gold_f else None)
            row[setting + "_func"] = ({m: rank_metrics(res[m][1], gold_s, FUNC_K, scost)
                                       for m in METHODS if m != "bm25_file"} if gold_s else None)
            if row[setting + "_func"] is not None:
                # bm25_file has no function ranking: expand file ranking to its scopes in file order
                flat = [s.sid for p in res["bm25_file"][0] for s in idx.scopes if s.path == p]
                row[setting + "_func"]["bm25_file"] = rank_metrics(flat, gold_s, FUNC_K, scost)
            if setting == "issue_hints":
                row["top5_files_rrf_all"] = res["rrf_all"][0][:5]
                row["n_explicit_ids_in_repo"] = sum(1 for x in ids if any(x in fi for fi in idx.file_ids))
        per_task.append(row)
    elapsed = time.time() - t0

    fm = ["mrr"] + ["hit@%d" % k for k in FILE_K] + ["recall@%d" % k for k in FILE_K] + ["all@%d" % k for k in FILE_K]
    sm = ["mrr"] + ["hit@%d" % k for k in FUNC_K] + ["recall@%d" % k for k in FUNC_K]
    results = {"seed": SEED, "bootstrap": B, "elapsed_sec": round(elapsed, 1), "methods": METHODS}
    for setting in ("issue_hints",):
        results[setting] = {
            "file": evaluate(per_task, setting + "_file", fm),
            "func": evaluate(per_task, setting + "_func", sm),
        }
    # strata (issue+hints)
    strata = {
        "symbol_named": lambda r: r["features"]["symbol_named"],
        "symbol_not_named": lambda r: not r["features"]["symbol_named"],
        "single_file": lambda r: r["features"]["n_gold_files"] == 1,
        "multi_file": lambda r: r["features"]["n_gold_files"] > 1,
    }
    for repo in sorted({r["repo"] for r in per_task}):
        strata["repo=" + repo] = (lambda rp: (lambda r: r["repo"] == rp))(repo)
    results["strata"] = {}
    for name, f in strata.items():
        results["strata"][name] = {
            "file": evaluate(per_task, "issue_hints_file", ["mrr", "hit@1", "hit@5"], f),
            "func": evaluate(per_task, "issue_hints_func", ["mrr", "hit@1", "hit@10"], f),
        }
    # paired comparisons
    rng = random.Random(SEED + 1)
    comps = {}
    for level, mn in (("file", "hit@1"), ("file", "mrr"), ("func", "hit@10"), ("func", "mrr")):
        rows = [r for r in per_task if r["issue_hints_" + level] is not None]
        for a, b_ in (("graph", "grep"), ("cascade", "graph"), ("cascade", "grep"), ("rrf_all", "bm25_func"),
                      ("graph", "bm25_func"), ("bm25_func", "grep"), ("bm25_func", "bm25_file")):
            comps["%s:%s:%s-%s" % (level, mn, a, b_)] = paired_diff(
                [r["issue_hints_" + level][a][mn] for r in rows],
                [r["issue_hints_" + level][b_][mn] for r in rows], rng)
    results["paired"] = comps

    os.makedirs(os.path.join(PAPER, "results"), exist_ok=True)
    with open(os.path.join(PAPER, "results", "baselines.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(results, f, indent=1, sort_keys=True)
    with open(os.path.join(PAPER, "results", "per_task.jsonl"), "w", encoding="utf-8", newline="\n") as f:
        for r in per_task:
            f.write(json.dumps(r, sort_keys=True) + "\n")

    # hard subset: no method has an edited unit in its top 10
    hard = [r["instance_id"] for r in per_task if r["issue_hints_func"] is not None
            and not any(r["issue_hints_func"][m]["hit@10"] for m in METHODS)]
    with open(os.path.join(PAPER, "data", "hard_subset.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(hard) + "\n")

    # console table
    for setting in ("issue_hints",):
        for level, cols in (("file", ["hit@1", "hit@5", "recall@10", "mrr"]),
                            ("func", ["hit@1", "hit@10", "recall@20", "mrr"])):
            r = results[setting][level]
            print("\n[%s / %s] n=%d" % (setting, level, r["n"]))
            print("%-11s" % "method" + "".join("%18s" % c for c in cols) + "  med_lines")
            for m in METHODS:
                cells = "".join("%7.3f [%.2f,%.2f]" % (r[m][c]["mean"], r[m][c]["ci95"][0], r[m][c]["ci95"][1])
                                for c in cols)
                print("%-11s%s  %s" % (m, cells, r[m]["median_lines_to_first"]))
    print("\npaired:")
    for k, v in comps.items():
        print("  %-36s %+.3f [%+.3f,%+.3f] p<=0:%.4f" % (k, v["mean"], v["ci95"][0], v["ci95"][1], v["p_boot_le0"]))
    print("elapsed", round(elapsed, 1), "s")


if __name__ == "__main__":
    main()
