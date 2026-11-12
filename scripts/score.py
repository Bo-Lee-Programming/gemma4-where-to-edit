"""Score a localizer against the Where-to-Edit labels.  Standard library only.

    python -I scripts/score.py PREDICTIONS.jsonl [--labels data/where_to_edit.jsonl] [--json]

PREDICTIONS.jsonl has one object per task (tasks you omit count as misses):
    {"instance_id": "rich_3521",
     "files":     ["rich/segment.py", "rich/console.py", ...],                 # ranked, best first
     "functions": ["rich/segment.py::Segment._split_cells", ...]}             # ranked "path::qualname"
Either list may be missing.  A qualname is Class.method or a top-level name; nested functions belong to
their enclosing function (see README).  Prints Hit@k and MRR with 95% bootstrap intervals (2,000 resamples,
seed 0), overall and split by whether the task text names an edited function.
For SWE-bench Verified pass --labels swebench/where_to_edit_swebench_verified.jsonl.
"""
import argparse
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FILE_K = (1, 5, 10)
FUNC_K = (1, 5, 10)


def first_rank(ranked, gold):
    for i, x in enumerate(ranked):
        if x in gold:
            return i + 1
    return None


def ci(vals, b=2000, seed=0):
    if not vals:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(vals)
    st = sorted(sum(vals[rng.randrange(n)] for _ in range(n)) / n for _ in range(b))
    return (st[int(0.025 * b)], st[int(0.975 * b) - 1])


def score(preds, labels):
    """Return {level: {"n":..., "rows": [per-task dict]}} for level in files/functions."""
    out = {"files": [], "functions": []}
    for lab in labels:
        p = preds.get(lab["instance_id"], {})
        named = bool(lab["features"].get("symbol_named"))
        if lab["gold_files"]:
            r = first_rank(p.get("files", []), set(lab["gold_files"]))
            out["files"].append({"id": lab["instance_id"], "rank": r, "named": named})
        if lab["gold_symbols"]:
            r = first_rank(p.get("functions", []), {g["sid"] for g in lab["gold_symbols"]})
            out["functions"].append({"id": lab["instance_id"], "rank": r, "named": named})
    return out


def summarize(rows, ks):
    res = {"n": len(rows)}
    for k in ks:
        v = [1.0 if r["rank"] is not None and r["rank"] <= k else 0.0 for r in rows]
        res["hit@%d" % k] = (sum(v) / len(v) if v else float("nan"), ci(v))
    v = [1.0 / r["rank"] if r["rank"] else 0.0 for r in rows]
    res["mrr"] = (sum(v) / len(v) if v else float("nan"), ci(v))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description="Score a localizer against Where-to-Edit labels.")
    ap.add_argument("predictions")
    ap.add_argument("--labels", default=os.path.join(ROOT, "data", "where_to_edit.jsonl"))
    ap.add_argument("--json", action="store_true", help="print machine-readable results")
    a = ap.parse_args(argv)
    with open(a.labels, encoding="utf-8") as f:
        labels = [json.loads(l) for l in f if l.strip()]
    preds = {}
    with open(a.predictions, encoding="utf-8") as f:
        for l in f:
            if l.strip():
                d = json.loads(l)
                preds[d["instance_id"]] = d
    known = {l["instance_id"] for l in labels}
    unknown = sorted(set(preds) - known)
    S = score(preds, labels)
    result = {}
    for level, ks in (("files", FILE_K), ("functions", FUNC_K)):
        rows = S[level]
        result[level] = {"all": summarize(rows, ks),
                         "named": summarize([r for r in rows if r["named"]], ks),
                         "not_named": summarize([r for r in rows if not r["named"]], ks)}
    if a.json:
        print(json.dumps(result, indent=1))
        return 0
    print("%s: %d of %d labelled tasks predicted%s" % (os.path.basename(a.predictions), len(set(preds) & known),
                                                       len(labels), (", %d unknown ids ignored" % len(unknown)) if unknown else ""))
    for level, ks in (("files", FILE_K), ("functions", FUNC_K)):
        cols = ["hit@%d" % k for k in ks] + ["mrr"]
        print("\n%-10s %-22s" % (level, "subset (n)") + "".join("%20s" % c for c in cols))
        for sub, name in (("all", "all"), ("named", "task names target"), ("not_named", "target not named")):
            s = result[level][sub]
            cells = "".join("%7.3f [%.2f,%.2f]" % (s[c][0], s[c][1][0], s[c][1][1]) for c in cols)
            print("%-10s %-22s%s" % ("", "%s (%d)" % (name, s["n"]), cells))
    return 0


if __name__ == "__main__":
    sys.exit(main())
