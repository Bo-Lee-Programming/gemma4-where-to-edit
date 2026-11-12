"""Do the competition's pre-computed code graphs and node embeddings help localization, and how should they be seeded?

For each public task we load the organisers' graph (data/graphs/<repo>_<commit>.json: nodes keyed by dotted symbol
path, 'calls' edges) and embeddings (data/embeddings/<repo>_<commit>.npz: one 256-d vector per node), align nodes
with our definition units, and rank units with:
  named_seed   what search_similar_code offers today: seed = symbols named in the task text, then their top-k
               embedding neighbours (seeds first, neighbours by best cosine)
  bm25_func    lexical baseline (same as run_baselines)
  ast_graph    BM25 + one hop over our name-resolved AST call graph (the skill)
  off_graph    BM25 + one hop over the organisers' call graph
  bm25_embed   BM25-seeded embedding expansion: base(u) + mu * max_{v in top-m BM25} base(v) * cos(u, v)
  graph_embed  ast_graph score + the same embedding term
Parameters fixed before the first run: lam = 0.5, mu = 0.5, m = 5, k = 10.  A grid over (mu, m) is reported
with leave-one-repository-out selection.
Usage: python -I paper/scripts/run_graph_embed.py   ->  paper/results/graph_embed.json, graph_embed_per_task.jsonl
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_baselines as rb  # noqa: E402
from wte_common import PAPER, ROOT, Repo, load_labels, load_tasks, locate, query_text  # noqa: E402

DL = os.environ.get("WTE_COMP_DATA") or os.path.join(ROOT, "data", "comp_dl")
METHODS = ["named_seed", "bm25_func", "ast_graph", "off_graph", "bm25_embed", "graph_embed"]
LAM, MU, M, K = 0.5, 0.5, 5, 10
GRID = [(mu, m) for mu in (0.25, 0.5, 1.0) for m in (1, 5, 20)]


def dotted(path, qualname):
    mod = path[:-3]
    for pre in ("src/", "lib/"):
        if mod.startswith(pre):
            mod = mod[len(pre):]
    if mod.endswith("/__init__"):
        mod = mod[:-9]
    return mod.replace("/", ".") + "." + qualname


def rank(sids, scores):
    order = sorted(range(len(sids)), key=lambda i: (-scores[i], sids[i]))
    return [sids[i] for i in order if scores[i] > 0]


def embed_term(base, E, has, m):
    """max over the top-m BM25 units v (with a vector) of base(v) * cos(u, v); 0 for units without a vector."""
    seeds = [i for i in np.argsort(-base, kind="stable")[:m] if has[i] and base[i] > 0]
    if not seeds:
        return np.zeros(len(base))
    sim = E @ E[seeds].T  # rows are unit-normalised (zero rows for units without a vector)
    sim[seeds, range(len(seeds))] = 0.0  # a seed does not vote for itself
    return np.clip((sim * base[seeds]).max(axis=1), 0.0, None)


def main():
    tasks = {t["instance_id"]: t for t in load_tasks()}
    rows, cover = [], {"units": 0, "units_with_node": 0, "gold": 0, "gold_with_node": 0, "nodes": 0, "edges": 0}
    repos = {}
    for lab in load_labels():
        t = tasks[lab["instance_id"]]
        tag = "%s_%s" % (t["repo"].split("/")[1], t["base_commit"])
        gp, ep = os.path.join(DL, "graphs", tag + ".json"), os.path.join(DL, "embeddings", tag + ".npz")
        if not (os.path.exists(gp) and os.path.exists(ep)):
            rows.append({"instance_id": lab["instance_id"], "repo": lab["repo"], "missing": True})
            continue
        repo = repos.setdefault(t["repo"], Repo(t["repo"]))
        files, _ = repo.snapshot(t["base_commit"])
        idx = locate.Index(files)
        n = len(idx.sids)
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
        off_nbr = [set() for _ in range(n)]
        for e in g["edges"]:
            a, b = name2i.get(e["source"]), name2i.get(e["target"])
            if a is not None and b is not None and a != b:
                off_nbr[a].add(b)
                off_nbr[b].add(a)
        gold = [x["sid"] for x in lab["gold_symbols"] if x["sid"] in idx.sid2scope]
        gi = [idx.sids.index(s) for s in gold]
        cover["units"] += n
        cover["units_with_node"] += int(has.sum())
        cover["gold"] += len(gi)
        cover["gold_with_node"] += int(sum(has[i] for i in gi))
        cover["nodes"] += len(g["nodes"])
        cover["edges"] += len(g["edges"])
        row = {"instance_id": lab["instance_id"], "repo": lab["repo"], "features": lab["features"],
               "n_units": n, "n_units_with_node": int(has.sum())}
        if not gold:
            row["func"] = None
            rows.append(row)
            continue
        q = query_text(t)
        _, _, gscore, ids = idx.grep(q)
        _, bs, bscore = idx.bm25(q)
        bscore = np.asarray(bscore, dtype=float)
        base = bscore / bscore.max() if bscore.max() > 0 else bscore
        ast_hop = np.array([max([base[j] for j in idx.nbr[i]] or [0.0]) for i in range(n)])
        off_hop = np.array([max([base[j] for j in off_nbr[i]] or [0.0]) for i in range(n)])
        res = {"bm25_func": rank(idx.sids, base), "ast_graph": rank(idx.sids, base + LAM * ast_hop),
               "off_graph": rank(idx.sids, base + LAM * off_hop)}
        et = embed_term(base, E, has, M)
        res["bm25_embed"] = rank(idx.sids, base + MU * et)
        res["graph_embed"] = rank(idx.sids, base + LAM * ast_hop + MU * et)
        # the tool as documented: seeds are the definitions the text names
        seeds = [i for i, s in enumerate(idx.scopes) if s.name in ids]
        sc = np.zeros(n)
        for i in seeds:
            sc[i] = 2.0
        for i in seeds:
            if has[i]:
                sim = E @ E[i]
                sim[i] = -1
                for j in np.argsort(-sim, kind="stable")[:K]:
                    if sc[j] < 2.0:
                        sc[j] = max(sc[j], float(sim[j]))
        res["named_seed"] = rank(idx.sids, sc)
        row["n_seeds"] = len(seeds)
        scost = {s.sid: s.end - s.start + 1 for s in idx.scopes}
        row["func"] = {m: rb.rank_metrics(res[m], gold, rb.FUNC_K, scost) for m in METHODS}
        row["grid"] = {"%g_%d" % (mu, m): rb.rank_metrics(rank(idx.sids, base + LAM * ast_hop + mu * embed_term(base, E, has, m)),
                                                         gold, rb.FUNC_K, scost)["mrr"] for mu, m in GRID}
        row["grid"]["0_0"] = row["func"]["ast_graph"]["mrr"]
        rows.append(row)

    ok = [r for r in rows if r.get("func")]
    out = {"n_tasks": len(rows), "missing": [r["instance_id"] for r in rows if r.get("missing")], "n_func": len(ok),
           "coverage": cover, "params": {"lam": LAM, "mu": MU, "m": M, "k": K}, "methods": {}, "paired": {}, "strata": {}}
    for m in METHODS:
        out["methods"][m] = {mn: {"mean": rb.mean([r["func"][m][mn] for r in ok]), "ci95": rb.boot_ci([r["func"][m][mn] for r in ok])}
                             for mn in ("hit@1", "hit@5", "hit@10", "mrr")}
        out["methods"][m]["found_any"] = sum(1 for r in ok if r["func"][m]["first_rank"] is not None)
    for a, b in (("off_graph", "ast_graph"), ("off_graph", "bm25_func"), ("bm25_embed", "bm25_func"), ("graph_embed", "ast_graph"),
                 ("named_seed", "bm25_func"), ("graph_embed", "named_seed")):
        for mn in ("hit@10", "mrr", "hit@1"):
            out["paired"]["%s:%s-%s" % (mn, a, b)] = rb.paired_diff([r["func"][a][mn] for r in ok], [r["func"][b][mn] for r in ok])
    for name, fn in (("symbol_named", lambda r: r["features"]["symbol_named"]), ("symbol_not_named", lambda r: not r["features"]["symbol_named"])):
        sub = [r for r in ok if fn(r)]
        out["strata"][name] = {"n": len(sub), **{m: {mn: rb.mean([r["func"][m][mn] for r in sub]) for mn in ("hit@10", "mrr")} for m in METHODS}}
    # grid + leave-one-repository-out selection of (mu, m) by function MRR
    keys = sorted(ok[0]["grid"])
    out["grid_mrr"] = {k: rb.mean([r["grid"][k] for r in ok]) for k in keys}
    held = []
    for rp in sorted({r["repo"] for r in ok}):
        tr = [r for r in ok if r["repo"] != rp]
        best = max(keys, key=lambda k: (rb.mean([r["grid"][k] for r in tr]), k))
        held += [(r["grid"][best], r["grid"]["0_0"]) for r in ok if r["repo"] == rp]
        out.setdefault("loro_choice", {})[rp] = best
    out["loro_mrr"] = rb.mean([h[0] for h in held])
    out["loro_vs_ast_graph"] = rb.paired_diff([h[0] for h in held], [h[1] for h in held])
    with open(os.path.join(PAPER, "results", "graph_embed.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    with open(os.path.join(PAPER, "results", "graph_embed_per_task.jsonl"), "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    print(json.dumps(out["coverage"]), "missing", len(out["missing"]), "n", len(ok))
    for m in METHODS:
        print("  %-12s" % m + "".join("  %s %.3f [%.2f,%.2f]" % (mn, out["methods"][m][mn]["mean"], *out["methods"][m][mn]["ci95"])
                                      for mn in ("hit@1", "hit@10", "mrr")), " found", out["methods"][m]["found_any"])
    for k, v in out["paired"].items():
        print("  %-34s %+.3f [%+.3f,%+.3f]" % (k, v["mean"], v["ci95"][0], v["ci95"][1]))
    print("strata", json.dumps(out["strata"]))
    print("grid", {k: round(v, 3) for k, v in out["grid_mrr"].items()}, "loro", out["loro_choice"], round(out["loro_mrr"], 3), out["loro_vs_ast_graph"])


if __name__ == "__main__":
    main()
