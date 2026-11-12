"""Shared helpers for the Where-to-Edit benchmark (paper/ only)."""
import json
import os
import re
import subprocess
import sys

PAPER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(PAPER)
REPOS = os.environ.get("WTE_REPOS") or os.path.join(PAPER, "repos")
# released folder: data/tasks.jsonl next to scripts/; working repo: ../data/tasks.jsonl
TASKS = os.path.join(PAPER, "data", "tasks.jsonl")
if not os.path.exists(TASKS):
    TASKS = os.path.join(ROOT, "data", "tasks.jsonl")
LABELS = os.path.join(PAPER, "data", "where_to_edit.jsonl")
sys.path.insert(0, os.path.join(PAPER, "skill", "where-to-edit", "scripts"))
import locate  # noqa: E402


def load_tasks():
    with open(TASKS, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def load_labels():
    with open(LABELS, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def query_text(task, hints=True):
    t = task.get("problem_statement") or ""
    if hints and task.get("hints_text"):
        t += "\n" + task["hints_text"]
    return t


class Repo(object):
    def __init__(self, full_name):
        self.git = os.path.join(REPOS, full_name.split("/")[1] + ".git")

    def ls_tree(self, commit):
        out = subprocess.run(["git", "-C", self.git, "ls-tree", "-r", commit],
                             capture_output=True, check=True).stdout.decode("utf-8")
        res = {}
        for line in out.splitlines():
            meta, path = line.split("\t", 1)
            mode, typ, sha = meta.split()
            if typ == "blob":
                res[path] = sha
        return res

    def read_blobs(self, shas):
        """Batch read blobs -> {sha: text}."""
        shas = list(dict.fromkeys(shas))
        if not shas:
            return {}
        p = subprocess.run(["git", "-C", self.git, "cat-file", "--batch"],
                           input=("\n".join(shas) + "\n").encode(), capture_output=True,
                           check=True)
        data, pos, out = p.stdout, 0, {}
        for sha in shas:
            nl = data.index(b"\n", pos)
            header = data[pos:nl].split()
            size = int(header[2])
            out[sha] = data[nl + 1:nl + 1 + size].decode("utf-8", errors="replace")
            pos = nl + 1 + size + 1
        return out

    def snapshot(self, commit, only_candidates=True):
        tree = self.ls_tree(commit)
        paths = [p for p in tree if (locate.is_candidate(p) or not only_candidates)
                 and p.endswith(".py")]
        blobs = self.read_blobs([tree[p] for p in paths])
        return {p: blobs[tree[p]] for p in paths}, tree


HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def parse_patch(patch):
    """Unified diff (--- a/ / +++ b/ form) -> {path: {...}}.

    For each file: deleted old-line numbers, insertion anchors (old line after
    which new lines were inserted; 0 = top of file), counts and new/deleted flags.
    """
    files, cur, old = {}, None, 0
    lines = patch.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("--- ") and i + 1 < len(lines) and lines[i + 1].startswith("+++ "):
            a = line[4:].split("\t")[0]
            b = lines[i + 1][4:].split("\t")[0]
            path = (b[2:] if b.startswith("b/") else a[2:] if a.startswith("a/") else b)
            if b == "/dev/null":
                path = a[2:] if a.startswith("a/") else a
            cur = files.setdefault(path, {"deleted": [], "anchors": [], "adds": 0, "dels": 0,
                                          "old_dev_null": a == "/dev/null",
                                          "new_dev_null": b == "/dev/null", "hunks": []})
            i += 2
            continue
        m = HUNK_RE.match(line)
        if m and cur is not None:
            old = int(m.group(1))
            cur["hunks"].append([int(m.group(1)), int(m.group(2) or 1)])
            if int(m.group(2) or 1) == 0:
                old += 1  # "-N,0": insertion after line N
            i += 1
            continue
        if cur is not None:
            if line.startswith("-"):
                cur["deleted"].append(old)
                cur["dels"] += 1
                old += 1
            elif line.startswith("+"):
                cur["anchors"].append((old - 1, line[1:]))
                cur["adds"] += 1
            elif line.startswith(" ") or line == "":
                old += 1
        i += 1
    return files


def merge_ranges(nums):
    nums = sorted(set(nums))
    out = []
    for n in nums:
        if out and n <= out[-1][1] + 1:
            out[-1][1] = n
        else:
            out.append([n, n])
    return out
