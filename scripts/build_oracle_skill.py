"""Build the oracle-localization agent: the where-to-edit skill call returns the gold files and units.

The oracle keeps the prompt, the skill name, the call and the output format of the real skill.  Its locate.py
recognises the task from a fingerprint of the repository snapshot (candidate paths and their non-blank sizes) and
prints the gold files and up to 12 gold units from the Where-to-Edit labels.  This gives the upper end of a
three-level comparison through one channel: no information (control), BM25 + graph (skill), gold (oracle).

Usage: python -I paper/scripts/build_oracle_skill.py   ->  variants/G7_{e4b,31b}_oracle_locate[.zip], results/oracle_build.json
"""
import json
import os
import re
import shutil
import subprocess
import sys
import zlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wte_common import PAPER, ROOT, Repo, load_labels, load_tasks  # noqa: E402

TEMPLATE = r'''#!/usr/bin/env python3
"""Oracle variant of where-to-edit for the upper-bound experiment: prints the gold edit locations of the task whose
repository snapshot matches the workspace.  Same call and output format as the real skill."""
import argparse
import os
import re
import sys
import zlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from locate import find_repo, load_repo  # noqa: E402

TABLE = __TABLE__


def tokens(files):
    return {zlib.crc32(("%s:%d" % (p, len("".join(t.split())))).encode("utf-8")) & 0xffffffff for p, t in files.items()}


def words(text):
    return {w for w in re.findall(r"[a-z_][a-z0-9_]{3,}", text.lower())}


def pick(tk, query):
    """Task whose snapshot matches the workspace; tasks sharing a snapshot are separated by word overlap with the query."""
    best, score, qw = None, (0.0, 0.0), words(query)
    for e in TABLE:
        t = set(e["t"])
        s = (round(len(tk & t) / float(max(len(tk | t), 1)), 3), len(qw & set(e["q"])) / float(max(len(qw), 1)))
        if s > score:
            best, score = e, s
    return best, score[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="/workspace")
    ap.add_argument("--query", default="")
    a = ap.parse_args()
    files = load_repo(find_repo(a.repo))
    tk = tokens(files)
    best, score = pick(tk, a.query)
    if best is None or score < 0.8:
        print("no candidate .py files under the repository")
        return 1
    print("Where-to-edit (BM25 over functions + one call-graph hop). %d files, %d defs indexed." % (len(files), best["d"]))
    print("Top files:")
    for i, p in enumerate(best["f"], 1):
        print("  %d. %s (%d lines)" % (i, p, files[p].count("\n") + 1 if p in files else 0))
    print("Top functions/classes (path:start-end qualname):")
    for i, u in enumerate(best["u"][:12], 1):
        print("  %d. %s:%d-%d %s" % (i, u[0], u[1], u[2], u[3]))
    print("Next: open the top 1-3 functions with sed -n 'START,ENDp' FILE, then edit.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''


def tokens(files):
    return sorted({zlib.crc32(("%s:%d" % (p, len("".join(t.split())))).encode("utf-8")) & 0xffffffff for p, t in files.items()})


def main():
    import locate
    tasks = {t["instance_id"]: t for t in load_tasks()}
    repos, groups, table, ndefs, snaps = {}, {}, [], {}, {}
    for lab in load_labels():
        t = tasks[lab["instance_id"]]
        files, _ = repos.setdefault(t["repo"], Repo(t["repo"])).snapshot(t["base_commit"])
        tk = tuple(tokens(files))
        if tk not in ndefs:
            ndefs[tk] = len(locate.Index(files).sids)
        g = groups.setdefault(tk, {"ids": []})
        g["ids"].append(lab["instance_id"])
        table.append({"t": list(tk), "d": ndefs[tk], "f": list(lab["gold_files"]),
                      "q": sorted(set(re.findall(r"[a-z_][a-z0-9_]{3,}", (t.get("problem_statement") or "").lower())))[:400],
                      "u": [[s["path"], s["start"], s["end"], s["qualname"]]
                            for s in sorted(lab["gold_symbols"], key=lambda s: (s["path"], s["start"]))]})
        snaps[lab["instance_id"]] = files
    shared = [g["ids"] for g in groups.values() if len(g["ids"]) > 1]
    code = TEMPLATE.replace("__TABLE__", json.dumps(table, separators=(",", ":")))
    compile(code, "oracle", "exec")
    V = os.path.join(ROOT, "variants")
    for model, base in (("e4b", "G3c_e4b_where_to_edit"), ("31b", "G3b_31b_where_to_edit")):
        name = "G7_%s_oracle_locate" % model
        d = os.path.join(V, name)
        shutil.rmtree(d, ignore_errors=True)
        shutil.copytree(os.path.join(V, base), d)
        sd = os.path.join(d, "skills", "where-to-edit", "scripts")
        os.replace(os.path.join(sd, "locate.py"), os.path.join(sd, "locate_impl.py"))
        with open(os.path.join(sd, "locate.py"), "wb") as f:
            f.write(code.replace("from locate import", "from locate_impl import").encode("utf-8"))
        for s in ("pack_variant.py",):
            subprocess.run([sys.executable, os.path.join(ROOT, "scripts", s), d, d + ".zip"], check=True)
    capped = sum(1 for e in table if len(e["u"]) > 12)
    # self-test: with the full task text every task gets its own gold set; with a title-only query too
    ns = {"__file__": os.path.abspath(__file__)}
    exec(compile(code.replace("from locate import find_repo, load_repo  # noqa: E402", "").split("if __name__ ==")[0], "oracle", "exec"), ns)
    labs = {l["instance_id"]: l for l in load_labels()}
    ok_full = ok_title = 0
    for iid, files in snaps.items():
        want = [[s["path"], s["start"], s["end"], s["qualname"]] for s in sorted(labs[iid]["gold_symbols"], key=lambda s: (s["path"], s["start"]))]
        ps = tasks[iid].get("problem_statement") or ""
        e, sc = ns["pick"](ns["tokens"](files), ps)
        ok_full += int(sc == 1.0 and e["u"] == want and e["f"] == list(labs[iid]["gold_files"]))
        e, sc = ns["pick"](ns["tokens"](files), ps.strip().splitlines()[0])
        ok_title += int(e["u"] == want and e["f"] == list(labs[iid]["gold_files"]))
    out = {"tasks": len(tasks), "snapshots": len(groups), "snapshots_shared_by_tasks": shared,
           "tasks_with_more_than_12_gold_units": capped,
           "selftest_full_text_query_own_gold": ok_full, "selftest_title_only_query_own_gold": ok_title, "locate_bytes": len(code)}
    with open(os.path.join(PAPER, "results", "oracle_build.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out)[:600])


if __name__ == "__main__":
    main()
