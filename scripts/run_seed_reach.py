"""Seed reachability: which seeds make the competition's graph tools land on an edited definition?

get_code_neighbors and search_similar_code both start from a symbol the agent already holds.  For each public task
we take a seed set and ask whether an edited definition lies in
  seeds          the seeds themselves
  +calls         seeds and their neighbours in the organisers' call graph      (get_code_neighbors)
  +embed         seeds and each seed's top-10 embedding neighbours             (search_similar_code, k = 10)
  +both          the union
Seed sets: the definitions the task text names (what a prompt-following agent has), and the top-1/3/5 definitions of
function-level BM25 (what the skill prints).  The reach-set size is the number of definitions the agent would read;
bm25_same_size is the hit rate of reading that many definitions straight down the BM25 list.
Usage: python -I paper/scripts/run_seed_reach.py  ->  paper/results/seed_reach.json, seed_reach_per_task.jsonl
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_baselines as rb  # noqa: E402
from run_graph_embed import DL, dotted  # noqa: E402
from wte_common import PAPER, Repo, load_labels, load_tasks, locate, query_text  # noqa: E402

K = 10
SEEDS = ["named", "bm25_top1", "bm25_top3", "bm25_top5"]
REACH = ["seeds", "+calls", "+embed", "+both"]


def main():
    tasks = {t["instance_id"]: t for t in load_tasks()}
    rows, repos = [], {}
    for lab in load_labels():
        t = tasks[lab["instance_id"]]
        tag = "%s_%s" % (t["repo"].split("/")[1], t["base_commit"])
        gp, ep = os.path.join(DL, "graphs", tag + ".json"), os.path.join(DL, "embeddings", tag + ".npz")
        if not (os.path.exists(gp) and os.path.exists(ep)):
            continue
        repo = repos.setdefault(t["repo"], Repo(t["repo"]))
        files, _ = repo.snapshot(t["base_commit"])
        idx = locate.Index(files)
        n = len(idx.sids)
        gold = {idx.sids.index(x["sid"]) for x in lab["gold_symbols"] if x["sid"] in idx.sid2scope}
        if not gold:
            continue
        g = json.load(open(gp, encoding="utf-8"))
        z = np.load(ep)
        keys = set(z.files)
        name2i = {dotted(s.path, s.qualname): i for i, s in enumerate(idx.scopes)}
        E = np.zeros((n, 256), dtype=np.float32)
        has = np.zeros(n, dtype=bool)
        for nm, i in name2i.items():
            if nm in keys:
                v = z[nm].astype(np.float32)
                nv = float(np.linalg.norm(v))
                if nv > 0:
                    E[i], has[i] = v / nv, True
        nbr = [set() for _ in range(n)]
        for e in g["edges"]:
            a, b = name2i.get(e["source"]), name2i.get(e["target"])
            if a is not None and b is not None and a != b:
                nbr[a].add(b)
                nbr[b].add(a)
        q = query_text(t)
        _, _, _, ids = idx.grep(q)
        _, _, bscore = idx.bm25(q)
        bscore = np.asarray(bscore, dtype=float)
        order = [i for i in sorted(range(n), key=lambda i: (-bscore[i], idx.sids[i])) if bscore[i] > 0]
        seedsets = {"named": [i for i, s in enumerate(idx.scopes) if s.name in ids],
                    "bm25_top1": order[:1], "bm25_top3": order[:3], "bm25_top5": order[:5]}
        row = {"instance_id": lab["instance_id"], "repo": lab["repo"], "symbol_named": lab["features"]["symbol_named"]}
        for sn, seeds in seedsets.items():
            calls, emb = set(), set()
            for i in seeds:
                calls |= nbr[i]
                if has[i]:
                    sim = E @ E[i]
                    sim[i] = -1
                    emb |= {int(j) for j in np.argsort(-sim, kind="stable")[:K] if has[j]}
            sets = {"seeds": set(seeds), "+calls": set(seeds) | calls, "+embed": set(seeds) | emb,
                    "+both": set(seeds) | calls | emb}
            row[sn] = {r: {"hit": int(bool(sets[r] & gold)), "size": len(sets[r]),
                           "bm25_same_size": int(bool(set(order[:len(sets[r])]) & gold))} for r in REACH}
        rows.append(row)

    out = {"n": len(rows), "k": K, "reach": {}, "size_median": {}, "paired": {}, "strata": {}}
    for sn in SEEDS:
        out["reach"][sn] = {r: {"mean": rb.mean([x[sn][r]["hit"] for x in rows]), "ci95": rb.boot_ci([x[sn][r]["hit"] for x in rows]),
                                "count": sum(x[sn][r]["hit"] for x in rows)} for r in REACH}
        out.setdefault("bm25_same_size", {})[sn] = {r: rb.mean([x[sn][r]["bm25_same_size"] for x in rows]) for r in REACH}
        out.setdefault("paired_vs_bm25_same_size", {})[sn] = {r: rb.paired_diff([x[sn][r]["hit"] for x in rows], [x[sn][r]["bm25_same_size"] for x in rows]) for r in REACH}
        out["size_median"][sn] = {r: sorted(x[sn][r]["size"] for x in rows)[len(rows) // 2] for r in REACH}
    for a in ("bm25_top1", "bm25_top3", "bm25_top5"):
        for r in REACH:
            out["paired"]["%s:%s-named" % (r, a)] = rb.paired_diff([x[a][r]["hit"] for x in rows], [x["named"][r]["hit"] for x in rows])
    for r in ("+calls", "+embed", "+both"):
        out["paired"]["bm25_top3:%s-seeds" % r] = rb.paired_diff([x["bm25_top3"][r]["hit"] for x in rows], [x["bm25_top3"]["seeds"]["hit"] for x in rows])
    for name, flag in (("symbol_named", True), ("symbol_not_named", False)):
        sub = [x for x in rows if x["symbol_named"] == flag]
        out["strata"][name] = {"n": len(sub), **{sn: {r: rb.mean([x[sn][r]["hit"] for x in sub]) for r in REACH} for sn in SEEDS}}
    with open(os.path.join(PAPER, "results", "seed_reach.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    with open(os.path.join(PAPER, "results", "seed_reach_per_task.jsonl"), "w", encoding="utf-8", newline="\n") as f:
        for x in rows:
            f.write(json.dumps(x, sort_keys=True) + "\n")
    print("n", len(rows))
    for sn in SEEDS:
        print("  %-10s" % sn + "".join("  %s %.3f (size %d)" % (r, out["reach"][sn][r]["mean"], out["size_median"][sn][r]) for r in REACH))
    for k, v in out["paired"].items():
        print("  %-30s %+.3f [%+.3f,%+.3f]" % (k, v["mean"], v["ci95"][0], v["ci95"][1]))
    for sn in SEEDS:
        print("  same-size BM25 %-10s" % sn + "".join("  %s %.3f (d %+.3f [%+.3f,%+.3f])" % (r, out["bm25_same_size"][sn][r], out["paired_vs_bm25_same_size"][sn][r]["mean"], *out["paired_vs_bm25_same_size"][sn][r]["ci95"]) for r in REACH))
    print(json.dumps(out["strata"]))


if __name__ == "__main__":
    main()
