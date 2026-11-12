#!/usr/bin/env python3
"""where-to-edit: CPU-only, stdlib-only localization for Python repositories.

Given an issue text, rank the repository's source files and functions/classes
by how likely they are to need an edit.  The same module is used (1) by the
Where-to-Edit benchmark scripts and (2) as an ADK skill script inside the task
container, so the numbers in the paper are produced by exactly this code.

Methods (all deterministic, no network, no model):
  grep      literal identifiers from the issue, definition-first
  bm25      BM25 over files / over functions and classes
  graph     BM25 scores propagated one hop over a name-resolved call graph
  rrf       reciprocal-rank fusion of the above
"""
import argparse
import ast
import math
import os
import re
import sys
from collections import Counter, defaultdict

# ----------------------------------------------------------------------------
# Candidate files

EXCLUDE_DIRS = {
    "tests", "test", "testing", "docs", "doc", "docs_src", "examples",
    "example", "benchmarks", "benchmark", ".git", ".venv", "venv", "env",
    "node_modules", "build", "dist", "site-packages", "__pycache__", ".tox",
    ".nox", ".mypy_cache", ".pytest_cache",
}


def is_candidate(path):
    """Source-file rule shared by the benchmark labels and the skill."""
    if not path.endswith(".py"):
        return False
    parts = path.replace("\\", "/").split("/")
    if any(p in EXCLUDE_DIRS or p.startswith(".") for p in parts[:-1]):
        return False
    name = parts[-1]
    if name.startswith("test_") or name.endswith("_test.py") or name == "conftest.py":
        return False
    return True


# ----------------------------------------------------------------------------
# Tokenisation

ID_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
CAMEL_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+")

STOP = set("""
a an the and or not no of to in on at by for from with without as is are was
were be been being it its this that these those there here i we you he she they
me my our your their them his her if else elif then than so but do does did done
can could should would will shall may might must have has had having get got
use used using also just only very more most some any all each every other such
what which who whom whose when where why how into out up down over under again
further once about above below between through during before after while because
until against same own too both few nor off def class return none true false
self cls import pass raise try except finally lambda yield global nonlocal assert
del async await print str int float bool dict list set tuple type object
like want need work works working way make makes made new one two see seem
issue bug error problem fix expected actual please thanks thank example
""".split())


def split_ident(tok):
    out = []
    for piece in tok.split("_"):
        if piece:
            out.extend(CAMEL_RE.findall(piece))
    return out


def tokenize(text):
    out = []
    for m in ID_RE.findall(text):
        low = m.lower()
        if len(low) > 1 and low not in STOP:
            out.append(low)
        parts = split_ident(m)
        if len(parts) > 1:
            for p in parts:
                pl = p.lower()
                if len(pl) > 1 and pl not in STOP:
                    out.append(pl)
    return out


def path_tokens(path):
    stem = path[:-3] if path.endswith(".py") else path
    return tokenize(stem.replace("/", " ").replace(".", " "))


# ----------------------------------------------------------------------------
# Literal identifiers mentioned in the issue (what a "grep the exact name"
# prompt would search for)

CODE_BLOCK_RE = re.compile(r"```.*?```", re.S)
INLINE_RE = re.compile(r"`([^`\n]{1,200})`")
CALL_RE = re.compile(r"(?<![A-Za-z0-9_])([A-Za-z_][A-Za-z0-9_]*)\s*\(")
DOTTED_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+")


def explicit_identifiers(text):
    found = set()
    for block in CODE_BLOCK_RE.findall(text):
        found.update(ID_RE.findall(block))
    for span in INLINE_RE.findall(CODE_BLOCK_RE.sub(" ", text)):
        found.update(ID_RE.findall(span))
    found.update(CALL_RE.findall(text))
    for d in DOTTED_RE.findall(text):
        found.update(d.split("."))
    for t in ID_RE.findall(text):
        core = t.strip("_")
        if "_" in core or re.search(r"[a-z][A-Z]", t) or re.match(r"[A-Z][a-z]+[A-Z]", t):
            found.add(t)
    return {t for t in found if len(t) >= 3 and t.lower() not in STOP}


# ----------------------------------------------------------------------------
# AST scopes and call graph


class Scope(object):
    __slots__ = ("sid", "path", "qualname", "name", "kind", "start", "end",
                 "text", "calls")

    def __init__(self, path, qualname, name, kind, start, end):
        self.sid = path + "::" + qualname
        self.path, self.qualname, self.name, self.kind = path, qualname, name, kind
        self.start, self.end = start, end
        self.text = ""
        self.calls = set()


def _node_start(n):
    return min([n.lineno] + [d.lineno for d in getattr(n, "decorator_list", [])])


def extract_scopes(path, src):
    """Top-level functions, methods and classes with line ranges (1-based,
    inclusive).  Functions nested inside functions are part of their parent.

    Function text = its full source; class text = class lines that are not
    inside a nested def/class (header, attributes, docstring).
    """
    try:
        tree = ast.parse(src)
    except (SyntaxError, ValueError):
        return []
    lines = src.splitlines()
    scopes = []

    def visit(body, prefix):
        for n in body:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                q = prefix + n.name
                kind = "class" if isinstance(n, ast.ClassDef) else "function"
                s = Scope(path, q, n.name, kind, _node_start(n), n.end_lineno)
                if kind == "function":
                    s.text = "\n".join(lines[s.start - 1:s.end])
                    for c in ast.walk(n):
                        if isinstance(c, ast.Call):
                            f = c.func
                            if isinstance(f, ast.Name):
                                s.calls.add(f.id)
                            elif isinstance(f, ast.Attribute):
                                s.calls.add(f.attr)
                else:
                    nested = set()
                    for c in n.body:
                        if isinstance(c, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                            nested.update(range(_node_start(c), c.end_lineno + 1))
                    s.text = "\n".join(
                        lines[i - 1] for i in range(s.start, s.end + 1) if i not in nested)
                scopes.append(s)
                if kind == "class":  # nested defs inside functions belong to the function
                    visit(n.body, q + ".")
            elif isinstance(n, (ast.If, ast.Try, ast.With, ast.For, ast.While)):
                # defs under `if TYPE_CHECKING:` / try-import blocks
                for field in ("body", "orelse", "finalbody"):
                    visit(getattr(n, field, []) or [], prefix)
                for h in getattr(n, "handlers", []) or []:
                    visit(h.body, prefix)

    visit(tree.body, "")
    return scopes


def innermost_scope(scopes, line):
    best = None
    for s in scopes:
        if s.start <= line <= s.end and (best is None or s.end - s.start < best.end - best.start):
            best = s
    return best


# ----------------------------------------------------------------------------
# BM25


class BM25(object):
    def __init__(self, docs, k1=1.2, b=0.75):
        self.k1, self.b = k1, b
        self.n = len(docs)
        self.dl = [len(d) for d in docs]
        self.avgdl = (sum(self.dl) / float(self.n)) if self.n else 1.0
        self.post = defaultdict(list)
        for i, d in enumerate(docs):
            for t, c in Counter(d).items():
                self.post[t].append((i, c))

    def scores(self, query_tokens):
        sc = [0.0] * self.n
        for t in set(query_tokens):
            p = self.post.get(t)
            if not p:
                continue
            idf = math.log(1.0 + (self.n - len(p) + 0.5) / (len(p) + 0.5))
            for i, c in p:
                norm = c + self.k1 * (1 - self.b + self.b * self.dl[i] / max(self.avgdl, 1e-9))
                sc[i] += idf * c * (self.k1 + 1) / norm
        return sc


# ----------------------------------------------------------------------------
# Ranking


def _rank(keys, scores, keep_zero=False):
    """Sort keys by score desc, ties by key; drop zero scores unless asked."""
    items = [(s, k) for k, s in zip(keys, scores) if keep_zero or s > 0]
    items.sort(key=lambda x: (-x[0], x[1]))
    return [k for _, k in items]


def rrf(rankings, k=60):
    score = defaultdict(float)
    for r in rankings:
        for i, key in enumerate(r):
            score[key] += 1.0 / (k + i + 1)
    return sorted(score, key=lambda x: (-score[x], x))


def files_from_scopes(ranked_sids, sid2path):
    seen, out = set(), []
    for sid in ranked_sids:
        p = sid2path[sid]
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


class Index(object):
    """Index of one repository snapshot: {path: source text}."""

    def __init__(self, files, k1=1.2, b=0.75):
        self.paths = sorted(files)
        self.src = files
        self.scopes = []
        for p in self.paths:
            self.scopes.extend(extract_scopes(p, files[p]))
        self.sids = [s.sid for s in self.scopes]
        self.sid2scope = {s.sid: s for s in self.scopes}
        self.sid2path = {s.sid: s.path for s in self.scopes}
        # file docs: path tokens (x3) + content
        self.file_tok = [path_tokens(p) * 3 + tokenize(files[p]) for p in self.paths]
        self.bm25_file = BM25(self.file_tok, k1, b)
        self.scope_tok = [tokenize(s.qualname) * 2 + path_tokens(s.path) + tokenize(s.text)
                          for s in self.scopes]
        self.bm25_scope = BM25(self.scope_tok, k1, b)
        # raw identifier sets for grep
        self.file_ids = [set(ID_RE.findall(files[p])) for p in self.paths]
        self.file_defs = []
        for p in self.paths:
            self.file_defs.append({s.name for s in self.scopes if s.path == p} |
                                  set(re.findall(r"(?m)^([A-Za-z_][A-Za-z0-9_]*)\s*[:=]", files[p])))
        self.scope_ids = [set(ID_RE.findall(s.text)) for s in self.scopes]
        # call graph (undirected, name-resolved, functions and classes)
        by_name = defaultdict(list)
        for i, s in enumerate(self.scopes):
            by_name[s.name].append(i)
        self.nbr = [set() for _ in self.scopes]
        for i, s in enumerate(self.scopes):
            for c in s.calls:
                targets = by_name.get(c, [])
                if 0 < len(targets) <= 5:  # skip hopelessly ambiguous names
                    for j in targets:
                        if j != i:
                            self.nbr[i].add(j)
                            self.nbr[j].add(i)

    # --- individual methods -------------------------------------------------

    def grep(self, text):
        ids = explicit_identifiers(text)
        df_f = Counter(t for fi in self.file_ids for t in ids & fi)
        fs = []
        for fi, fd in zip(self.file_ids, self.file_defs):
            hit = ids & fi
            fs.append(1000.0 * len(ids & fd) + sum(1.0 / df_f[t] for t in hit))
        df_s = Counter(t for si in self.scope_ids for t in ids & si)
        ss = []
        for s, si in zip(self.scopes, self.scope_ids):
            hit = ids & si
            ss.append(1000.0 * (s.name in ids) + sum(1.0 / df_s[t] for t in hit))
        return _rank(self.paths, fs), _rank(self.sids, ss), ss, ids

    def bm25(self, text):
        q = tokenize(text)
        fs = self.bm25_file.scores(q)
        ss = self.bm25_scope.scores(q)
        return _rank(self.paths, fs), _rank(self.sids, ss), ss

    def graph(self, scope_scores, lam=0.5):
        m = max(scope_scores) if scope_scores else 0.0
        if m <= 0:
            return []
        base = [x / m for x in scope_scores]
        g = [base[i] + lam * max([base[j] for j in self.nbr[i]] or [0.0])
             for i in range(len(base))]
        return _rank(self.sids, g)

    def run_all(self, text, lam=0.5, rrf_k=60):
        """Return {method: (ranked_files, ranked_scope_ids)}."""
        gf, gs, gscore, ids = self.grep(text)
        bf, bs, bscore = self.bm25(text)
        gr = self.graph(bscore, lam)
        out = {
            "grep": (gf, gs),
            "bm25_file": (bf, []),
            "bm25_func": (files_from_scopes(bs, self.sid2path), bs),
            "graph": (files_from_scopes(gr, self.sid2path), gr),
        }
        # size prior: sanity baseline (largest first)
        out["size_prior"] = (
            sorted(self.paths, key=lambda p: (-self.src[p].count("\n"), p)),
            sorted(self.sids, key=lambda s: (-(self.sid2scope[s].end - self.sid2scope[s].start), s)))
        out["rrf_all"] = (
            rrf([gf, bf, out["bm25_func"][0], out["graph"][0]], rrf_k),
            rrf([gs, bs, gr], rrf_k))
        # cascade: definitions whose name the issue spells out, then BM25+graph
        named = [sid for sid, sc in sorted(zip(self.sids, gscore), key=lambda x: (-x[1], x[0]))
                 if sc >= 1000]
        seen = set(named)
        casc = named + [s for s in gr if s not in seen]
        out["cascade"] = (files_from_scopes(casc, self.sid2path), casc)
        return out, ids


# ----------------------------------------------------------------------------
# Skill entry point


def load_repo(root):
    files = {}
    for d, dirs, names in os.walk(root):
        dirs[:] = sorted(x for x in dirs if x not in EXCLUDE_DIRS and not x.startswith("."))
        for n in names:
            full = os.path.join(d, n)
            rel = os.path.relpath(full, root).replace("\\", "/")
            if is_candidate(rel) and not n.startswith("."):  # ADK's wrapper script is a dot-file in the workspace
                try:
                    with open(full, encoding="utf-8", errors="replace") as fh:
                        files[rel] = fh.read()
                except OSError:
                    pass
    return files


def find_repo(arg):
    """Repository root.  ADK's run_skill_script copies the skill into a temporary directory and changes into it,
    so the task workspace is the directory the wrapper started in: its `_orig_cwd` local, the folder of the wrapper
    script (sys.orig_argv), or $PWD.  An existing --repo wins; the current directory is the last resort."""
    cands = [arg]
    f = sys._getframe()
    while f is not None:
        if "_orig_cwd" in f.f_locals:
            cands.append(f.f_locals["_orig_cwd"])
        f = f.f_back
    for x in getattr(sys, "orig_argv", [])[1:]:
        if x.endswith(".py"):
            cands.append(os.path.dirname(os.path.abspath(x)))
    cands += [os.environ.get("PWD"), os.getcwd()]
    here = os.path.realpath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for c in cands:
        if c and os.path.isdir(c) and not (os.path.realpath(c) + os.sep).startswith(here + os.sep):
            return c
    return os.getcwd()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--repo", default="/workspace", help="repository root")
    ap.add_argument("--query", default="", help="issue text (or use --query_file)")
    ap.add_argument("--query_file", default="", help="file containing the issue text")
    ap.add_argument("--top_files", type=int, default=8)
    ap.add_argument("--top_funcs", type=int, default=12)
    a = ap.parse_args(argv)
    text = a.query
    if a.query_file:
        with open(a.query_file, encoding="utf-8", errors="replace") as fh:
            text += "\n" + fh.read()
    if not text.strip():
        print("error: pass the issue text with --query")
        return 2
    root = find_repo(a.repo)
    idx = Index(load_repo(root))
    if not idx.paths:
        print("no candidate .py files under %s" % root)
        return 1
    res, ids = idx.run_all(text)
    files, funcs = res["graph"]
    print("Where-to-edit (BM25 over functions + one call-graph hop). %d files, %d defs indexed."
          % (len(idx.paths), len(idx.sids)))
    defined = sorted({s.name for s in idx.scopes if s.name in ids})
    if defined:
        print("Defined names the issue mentions (often the API used, not the edit site): "
              + ", ".join(defined[:12]))
    print("Top files:")
    for i, p in enumerate(files[:a.top_files], 1):
        print("  %d. %s (%d lines)" % (i, p, idx.src[p].count("\n") + 1))
    print("Top functions/classes (path:start-end qualname):")
    for i, sid in enumerate(funcs[:a.top_funcs], 1):
        s = idx.sid2scope[sid]
        print("  %d. %s:%d-%d %s" % (i, s.path, s.start, s.end, s.qualname))
    print("Next: open the top 1-3 functions with sed -n 'START,ENDp' FILE, then edit.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
