"""Build Where-to-Edit labels from gold patches + base-commit ASTs.

Output: paper/data/where_to_edit.jsonl (one record per task).
Usage:  python -I paper/scripts/build_labels.py
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wte_common import LABELS, Repo, load_tasks, locate, merge_ranges, parse_patch, query_text  # noqa: E402


def indent(s):
    return len(s) - len(s.lstrip(" \t"))


def scope_for_insertion(scopes, lines, anchor, text):
    best = None
    for s in scopes:
        inside = s.start <= anchor and anchor + 1 <= s.end
        tail = (s.end == anchor and text.strip() and s.start - 1 < len(lines)
                and indent(text) > indent(lines[s.start - 1]))
        if (inside or tail) and (best is None or s.end - s.start < best.end - best.start):
            best = s
    return best


def word_in(word, text):
    return re.search(r"(?<![A-Za-z0-9_])" + re.escape(word) + r"(?![A-Za-z0-9_])", text) is not None


def file_mentions(path):
    stem = path[:-3]
    mod = stem[4:] if stem.startswith("src/") else stem
    return [path, os.path.basename(path), mod.replace("/", "."), mod]


def label_one(t, repo):
    tree = repo.ls_tree(t["base_commit"])
    pf = parse_patch(t["patch"])
    gold_paths = [p for p in pf if p in tree and locate.is_candidate(p)
                  and not pf[p]["old_dev_null"]]
    blobs = repo.read_blobs([tree[p] for p in gold_paths])
    cand = [p for p in tree if locate.is_candidate(p)]
    q = query_text(t)
    rec = {"instance_id": t["instance_id"], "repo": t["repo"],
           "base_commit": t["base_commit"], "gold_files": [], "gold_symbols": [],
           "edit_spans": {}, "new_source_files": [], "other_files": []}
    adds = dels = 0
    module_only = []
    for p, info in pf.items():
        if p not in gold_paths:
            if locate.is_candidate(p) and (p not in tree or info["old_dev_null"]):
                rec["new_source_files"].append(p)
            else:
                rec["other_files"].append(p)
            continue
        src = blobs[tree[p]]
        lines = src.splitlines()
        scopes = locate.extract_scopes(p, src)
        touched = {}
        for ln in info["deleted"]:
            s = locate.innermost_scope(scopes, ln)
            if s is not None:
                touched[s.sid] = s
        for anchor, text in info["anchors"]:
            s = scope_for_insertion(scopes, lines, anchor, text)
            if s is not None:
                touched[s.sid] = s
        rec["gold_files"].append(p)
        adds += info["adds"]
        dels += info["dels"]
        rec["edit_spans"][p] = merge_ranges(
            info["deleted"] + [max(a, 1) for a, _ in info["anchors"]])
        if not touched:
            module_only.append(p)
        for s in sorted(touched.values(), key=lambda x: x.start):
            rec["gold_symbols"].append({"sid": s.sid, "path": p, "qualname": s.qualname,
                                        "kind": s.kind, "start": s.start, "end": s.end})
    names = {g["qualname"].split(".")[-1] for g in rec["gold_symbols"]}
    comps = {c for g in rec["gold_symbols"] for c in g["qualname"].split(".")}
    ql = q.lower()
    rec["features"] = {
        "n_gold_files": len(rec["gold_files"]),
        "n_gold_symbols": len(rec["gold_symbols"]),
        "n_new_source_files": len(rec["new_source_files"]),
        "n_other_files": len(rec["other_files"]),
        "adds": adds, "dels": dels,
        "module_level_only_files": module_only,
        "file_named": any(m.lower() in ql for p in rec["gold_files"] for m in file_mentions(p)),
        "file_path_named": any(m.lower() in ql for p in rec["gold_files"]
                               for m in file_mentions(p)[:2]),
        "symbol_named": any(word_in(n, q) for n in names),
        "symbol_component_named": any(word_in(n, q) for n in comps),
        "n_explicit_ids": len(locate.explicit_identifiers(q)),
        "issue_tokens": len(locate.tokenize(q)),
        "has_code_block": "```" in q,
        "has_traceback": "Traceback (most recent call last)" in q,
        "has_hints": bool((t.get("hints_text") or "").strip()),
        "n_candidate_files": len(cand),
        "created_at": t.get("created_at"),
    }
    return rec


def main():
    tasks = load_tasks()
    repos = {}
    out = []
    for t in tasks:
        repo = repos.setdefault(t["repo"], Repo(t["repo"]))
        rec = label_one(t, repo)
        out.append(rec)
    os.makedirs(os.path.dirname(LABELS), exist_ok=True)
    with open(LABELS, "w", encoding="utf-8", newline="\n") as f:
        for r in out:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    # summary
    n = len(out)
    has_file = [r for r in out if r["gold_files"]]
    has_sym = [r for r in out if r["gold_symbols"]]
    print("tasks", n)
    print("with existing source-file target", len(has_file))
    print("with function/class target", len(has_sym))
    print("symbol named (of those with symbol)", sum(r["features"]["symbol_named"] for r in has_sym))
    print("symbol component named", sum(r["features"]["symbol_component_named"] for r in has_sym))
    print("file named (of those with file)", sum(r["features"]["file_named"] for r in has_file))
    print("file path/basename named", sum(r["features"]["file_path_named"] for r in has_file))
    from collections import Counter
    print("n_gold_files dist", sorted(Counter(r["features"]["n_gold_files"] for r in out).items()))
    print("tasks with new source files", sum(1 for r in out if r["new_source_files"]))


if __name__ == "__main__":
    main()
