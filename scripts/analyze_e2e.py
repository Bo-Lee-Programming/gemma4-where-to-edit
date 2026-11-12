"""End-to-end comparison of agent runs on the 129 public tasks, scored with the Where-to-Edit labels.

Each run directory is the output of the evaluation notebook: eval_tasks.csv (resolved, tool_calls, ...),
results/patches/<instance_id>.patch and results/traces/trace_<instance_id>.json.  Besides the resolution rate we
label every agent patch with the code that labels the gold patches and report whether the agent edited a gold file /
a gold unit ("edit localization").  From the traces we read every run_skill_script call: how many files the skill
indexed, which units it listed, and whether an edited unit was among them.
Usage: python -I paper/scripts/analyze_e2e.py [--out=e2e] name=dir [name=dir ...]   ->  paper/results/<out>.json, <out>_per_task.jsonl
The first run is the reference of every paired comparison; all pairs are also reported.
"""
import csv
import json
import os
import re
import sys
from math import comb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_baselines as rb  # noqa: E402
from build_labels import label_one  # noqa: E402
from wte_common import PAPER, Repo, load_labels, load_tasks  # noqa: E402

LISTED = re.compile(r"\d+\. (\S+\.py):(\d+)-(\d+) ([\w.]+)")


def sign_p(a, b):
    m = a + b
    return min(1.0, 2 * sum(comb(m, j) for j in range(min(a, b) + 1)) / 2 ** m) if m else 1.0


def read_trace(d, iid):
    """Skill usage in one trace: calls, files indexed by the last answered call, units it listed (path, qualname)."""
    p = os.path.join(d, "results", "traces", "trace_%s.json" % iid)
    if not os.path.exists(p):
        return None
    tr = json.load(open(p, encoding="utf-8"))
    out = {"skill_calls": 0, "files_indexed": 0, "listed": [], "tools": {}}
    for s in tr["steps"]:
        for tc in s.get("tool_calls") or []:
            out["tools"][tc["function_name"]] = out["tools"].get(tc["function_name"], 0) + 1
            if tc["function_name"] != "run_skill_script":
                continue
            out["skill_calls"] += 1
            obs = json.dumps(s.get("observation"))
            for _ in range(4):  # the tool result is JSON nested in JSON
                obs = obs.replace("\\\\", "\\").replace("\\n", "\n").replace('\\"', '"')
            m = re.search(r"(\d+) files, (\d+) defs indexed", obs)
            if m:
                out["files_indexed"] = int(m.group(1))
                out["listed"] = [(a, q) for a, _, _, q in LISTED.findall(obs)]
    return out


def load_run(d, tasks, labels, repos):
    rows = {r["instance_id"]: r for r in csv.DictReader(open(os.path.join(d, "eval_tasks.csv"), encoding="utf-8"))}
    out = {}
    for iid, r in rows.items():
        t, lab = tasks[iid], labels[iid]
        pp = os.path.join(d, "results", "patches", iid + ".patch")
        patch = open(pp, encoding="utf-8", errors="replace").read() if os.path.exists(pp) else ""
        files, units = set(), set()
        if patch.strip():
            try:
                al = label_one(dict(t, patch=patch), repos.setdefault(t["repo"], Repo(t["repo"])))
                files, units = set(al["gold_files"]), {(g["path"], g["qualname"]) for g in al["gold_symbols"]}
            except Exception as e:  # malformed agent patch
                print("patch parse failed", iid, repr(e)[:80])
        gold_units = {(g["path"], g["qualname"]) for g in lab["gold_symbols"]}
        tr = read_trace(d, iid)
        listed = set(map(tuple, tr["listed"])) if tr else set()
        out[iid] = {"resolved": int(str(r["resolved"]).lower() == "true"), "tool_calls": float(r["tool_calls"] or 0),
                    "empty": int(not patch.strip()), "has_file_target": bool(lab["gold_files"]),
                    "has_unit_target": bool(lab["gold_symbols"]), "symbol_named": lab["features"]["symbol_named"],
                    "file_hit": int(bool(files & set(lab["gold_files"]))), "unit_hit": int(bool(units & gold_units)),
                    "trace": tr is not None, "skill_calls": tr["skill_calls"] if tr else 0,
                    "skill_indexed_repo": int(bool(tr and tr["files_indexed"] > 1)),
                    "skill_listed_gold": int(bool(listed & gold_units)),
                    "skill_listed_gold_top3": int(bool(set(map(tuple, tr["listed"][:3])) & gold_units)) if tr else 0,
                    "edited_listed_unit": int(bool(listed & units)),
                    "graph_tool_calls": sum(v for k, v in tr["tools"].items() if k in ("get_code_neighbors", "search_similar_code", "get_code_subgraph")) if tr else 0}
    return out


def summarize(R, ids):
    f = [i for i in ids if R[i]["has_file_target"]]
    u = [i for i in ids if R[i]["has_unit_target"]]
    un = [i for i in u if not R[i]["symbol_named"]]
    nm = [i for i in u if R[i]["symbol_named"]]
    lg = [i for i in u if R[i]["skill_listed_gold"]]
    nl = [i for i in u if R[i]["skill_indexed_repo"] and not R[i]["skill_listed_gold"]]
    c = lambda key, sub: sum(R[i][key] for i in sub)
    return {"resolved": c("resolved", ids), "empty_patches": c("empty", ids),
            "mean_tool_calls": rb.mean([R[i]["tool_calls"] for i in ids]),
            "edited_gold_file": c("file_hit", f), "n_file": len(f), "edited_gold_unit": c("unit_hit", u), "n_unit": len(u),
            "edited_gold_unit_unnamed": c("unit_hit", un), "n_unit_unnamed": len(un),
            "edited_gold_unit_named": c("unit_hit", nm), "n_unit_named": len(nm),
            "traces": c("trace", ids), "tasks_calling_skill": sum(1 for i in ids if R[i]["skill_calls"]),
            "tasks_skill_indexed_repo": c("skill_indexed_repo", ids),
            "skill_listed_gold_unit": len(lg), "skill_listed_gold_unit_top3": c("skill_listed_gold_top3", u),
            "edited_gold_unit_when_listed": c("unit_hit", lg), "resolved_when_listed": c("resolved", lg),
            "n_indexed_not_listed": len(nl), "edited_gold_unit_when_not_listed": c("unit_hit", nl),
            "resolved_when_not_listed": c("resolved", nl),
            "tasks_editing_a_listed_unit": c("edited_listed_unit", ids),
            "graph_tool_calls": c("graph_tool_calls", ids)}


def main():
    stem = ([a[6:] for a in sys.argv[1:] if a.startswith("--out=")] or ["e2e"])[0]
    specs = [a.split("=", 1) for a in sys.argv[1:] if not a.startswith("--out=")]
    tasks = {t["instance_id"]: t for t in load_tasks()}
    labels = {l["instance_id"]: l for l in load_labels()}
    repos = {}
    runs = {name: load_run(d, tasks, labels, repos) for name, d in specs}
    names = [n for n, _ in specs]
    ids = sorted(set.intersection(*(set(r) for r in runs.values())))
    res = {"n": len(ids), "order": names, "runs": {n: summarize(runs[n], ids) for n in names}, "paired": {}}
    for ai, a in enumerate(names):
        for b in names[ai + 1:]:
            A, B = runs[a], runs[b]
            pr = {}
            for key, sub in (("resolved", ids), ("file_hit", [i for i in ids if A[i]["has_file_target"]]),
                             ("unit_hit", [i for i in ids if A[i]["has_unit_target"]]),
                             ("unit_hit_unnamed", [i for i in ids if A[i]["has_unit_target"] and not A[i]["symbol_named"]])):
                k = key.replace("_unnamed", "")
                gain = [i for i in sub if B[i][k] and not A[i][k]]
                lost = [i for i in sub if A[i][k] and not B[i][k]]
                pr[key] = {"n": len(sub), "gained": len(gain), "lost": len(lost), "sign_p": sign_p(len(gain), len(lost)),
                           "diff": rb.paired_diff([B[i][k] for i in sub], [A[i][k] for i in sub]), "gained_ids": gain, "lost_ids": lost}
            pr["tool_calls"] = rb.paired_diff([B[i]["tool_calls"] for i in ids], [A[i]["tool_calls"] for i in ids])
            res["paired"]["%s-%s" % (b, a)] = pr
    with open(os.path.join(PAPER, "results", stem + ".json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=1, sort_keys=True)
    with open(os.path.join(PAPER, "results", stem + "_per_task.jsonl"), "w", encoding="utf-8", newline="\n") as f:
        for i in ids:
            f.write(json.dumps(dict({"instance_id": i}, **{n: runs[n][i] for n in names}), sort_keys=True) + "\n")
    for n in names:
        print(n, json.dumps(res["runs"][n]))
    for k, pr in res["paired"].items():
        print(k)
        for key, v in pr.items():
            if "gained" in v:
                print("  %-18s n=%d +%d -%d sign p=%.3f diff %+.3f [%+.3f,%+.3f]" % (key, v["n"], v["gained"], v["lost"], v["sign_p"],
                                                                                 v["diff"]["mean"], *v["diff"]["ci95"]))
            else:
                print("  %-18s %+.2f [%+.2f,%+.2f]" % (key, v["mean"], *v["ci95"]))


if __name__ == "__main__":
    main()
