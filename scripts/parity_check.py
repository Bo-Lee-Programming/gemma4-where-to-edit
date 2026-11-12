"""Check that the skill (reading a checked-out repo from disk) produces exactly
the ranking the benchmark computes from git objects.

Usage: python -I paper/scripts/parity_check.py [N_PER_REPO]
Writes paper/results/parity.json.  Checkouts go to paper/tmp/ (deleted after).
"""
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wte_common import PAPER, Repo, load_labels, load_tasks, locate, query_text  # noqa: E402


def main():
    n_per = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    tasks = {t["instance_id"]: t for t in load_tasks()}
    picked, per_repo = [], {}
    for lab in load_labels():
        if per_repo.get(lab["repo"], 0) < n_per:
            per_repo[lab["repo"]] = per_repo.get(lab["repo"], 0) + 1
            picked.append(lab["instance_id"])
    out = []
    tmp = os.path.join(PAPER, "tmp", "parity")
    for iid in picked:
        t = tasks[iid]
        repo = Repo(t["repo"])
        files, _ = repo.snapshot(t["base_commit"])
        ws = os.path.join(tmp, iid)
        shutil.rmtree(ws, ignore_errors=True)
        os.makedirs(ws)
        data = subprocess.run(["git", "-C", repo.git, "archive", t["base_commit"]],
                              capture_output=True, check=True).stdout
        with tarfile.open(fileobj=io.BytesIO(data)) as tf:
            tf.extractall(ws, filter="data")
        disk = locate.load_repo(ws)
        q = query_text(t)
        a, _ = locate.Index(files).run_all(q)
        b, _ = locate.Index(disk).run_all(q)
        same_files = sorted(files) == sorted(disk)
        same_rank = a["graph"][0] == b["graph"][0] and a["graph"][1] == b["graph"][1]
        out.append({"id": iid, "same_file_set": same_files, "same_graph_ranking": same_rank,
                    "n_files": len(files)})
        print(iid, same_files, same_rank, len(files), len(disk))
        shutil.rmtree(ws, ignore_errors=True)
    ok = all(r["same_file_set"] and r["same_graph_ranking"] for r in out)
    with open(os.path.join(PAPER, "results", "parity.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump({"all_equal": ok, "tasks": out}, f, indent=1)
    print("ALL EQUAL" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
