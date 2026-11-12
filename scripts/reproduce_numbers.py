"""Recompute every number in paper/writeup.md from the files in this repository.

usage (from Gemma4-Developer-Agent/):
    python -I paper/scripts/reproduce_numbers.py [--rerun]

--rerun first regenerates labels, baselines, sensitivity and parity from the git
clones (about 15 minutes on a laptop CPU); without it the saved result files are
checked.  Each check recomputes a value and names the exact writeup strings it
supports ("cites").  The run fails if a check fails, if a cited string is missing
from the writeup, or if the writeup body (before References, outside code) has a
number that no check cites.  Writes paper/REPRODUCE_REPORT.md with every check
and the size and SHA-256 of every source file used.
"""
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import time
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from wte_common import PAPER, ROOT, TASKS, Repo, locate  # noqa: E402

WRITEUP = os.path.join(PAPER, "writeup.md")
USED = {}
ROWS = []  # (claim, computed, ok, cites)
RETIRED = [  # sentences of earlier drafts that the text no longer contains
    'the 129 public competition tasks',
    '(500 tasks, 12 repositories)',
    '(median 119 content tokens)',
    '576 definitions for fastapi, 1,027 for rich, and about 9,000 across SWE-bench Verified',
    'SWE-bench Verified (SWE-V): 500 and 485',
    'in 29% of competition tasks and 48% of SWE-bench Verified tasks',
    'best of all seven methods',
    'Whiskers are 95% bootstrap intervals',
    'Competition: 125 tasks for files, 119 for functions. SWE-bench Verified (SWE-V): 500 and 485.',
    'top 10 in 55% versus 47% of competition tasks and 64% versus 47% of SWE-bench Verified tasks',
    '48 of 119 competition tasks and 67 of 485 SWE-bench Verified tasks',
    'function Hit@10 by +0.456 [+0.408, +0.501] on SWE-bench Verified',
    'raises competition function MRR by +0.027 [+0.006, +0.051]',
    '(−0.010 [−0.027, +0.008])',
    'lowers file Hit@1 by −0.072 [−0.128, −0.016] on the competition tasks',
    'function Hit@10 by −0.099 [−0.136, −0.062] on SWE-bench Verified',
    'In 62 of the 119 competition tasks',
    'in 31 of those 62 none',
    '245 of 454 tasks',
    'names `Annotated` and `Body()`, and the fix is in `ModelField.__post_init__`',
    '0.943 for grep and 0.914 for BM25 + graph on the competition tasks (35 tasks), and 0.827 and 0.818 on SWE-bench Verified (231 tasks)',
    'grep falls to 0.274 and 0.154 while BM25 + graph holds 0.405 and 0.461 (84 and 254 tasks)',
    'top 10 in 79 of 119 competition tasks and 355 of 485 SWE-bench Verified tasks',
    'The remaining 40 and 130 tasks',
    '`figures/fig1_data.json`',
    'b ∈ {0.5, 0.75, 1.0} and λ ∈ {0, 0.25, 0.5}, competition function MRR stays between 0.338 and 0.387',
    'function MRR 0.374 [0.31, 0.45], within 0.02 of the default',
    '88,592 of 92,018 units (96%) and 466 of 501 edited units (93%)',
    'top-10 embedding neighbours',
    'the top five BM25 units',
    'reach far more edited functions when seeded with BM25 output than with named symbols',
    'a BM25 list of equal length still reaches more',
    'raises reach by +0.210 [+0.118, +0.311]',
    'raises their reach by 21 points',
    'On the 84 tasks whose text leaves the edited unit unnamed, named seeds reach it in 0.095 and BM25 top-3 seeds in 0.417',
    'add +0.151 [+0.092, +0.218] over the seeds alone',
    'reaches 0.739 at the top-3 budget',
    '+0.193 [+0.109, +0.286] above the tool expansion',
    '(0.546 versus 0.588 at 15 units',
    '−0.042 [−0.109, +0.017])',
    'gives function Hit@10 0.563 against 0.546',
    'gives 0.555',
    '(+0.017 [−0.034, +0.067])',
    '(+0.008 [−0.025, +0.042])',
    'sets the embedding weight to zero for three of four repositories',
    'one 256-dimensional vector per node',
    '`calls` edges',
    'confirms on 10 tasks',
    '(66 files, 498 units)',
    'answers in about a second',
    'one call takes about a second on a laptop CPU',
    'standard library only and Python 3.8+ syntax',
    'about 20 minutes on six cores',
    'use the first eight calls to locate and read, and edit a source file by the twelfth call',
    'list the exact file, function and class names from the task',
]


def reg(path):
    with open(path, "rb") as f:
        data = f.read()
    USED[os.path.relpath(path, ROOT).replace("\\", "/")] = (len(data), hashlib.sha256(data).hexdigest())
    return data


def rj(rel):
    return json.loads(reg(os.path.join(PAPER, rel)))


def rjl(path):
    return [json.loads(l) for l in reg(path).decode("utf-8").splitlines() if l.strip()]


def f3(x):
    return "%.3f" % x


def ci2(c):
    return "[%.2f, %.2f]" % (c[0], c[1])


def sgn(x, d=3):
    s = ("%+." + str(d) + "f") % x
    return s.replace("-", "−")


def pdiff(p):
    return "%s [%s, %s]" % (sgn(p["mean"]), sgn(p["ci95"][0]), sgn(p["ci95"][1]))


def pct(a, b):
    return "%d%%" % round(100.0 * a / b)


def check(claim, computed, expected, cites):
    ok = computed == expected
    ROWS.append([claim, str(computed), ok, list(cites)])


def cite_only(claim, ok, cites):
    ROWS.append([claim, "ok" if ok else "FAILED", bool(ok), list(cites)])


def table_rows(text, header_start):
    lines = text.splitlines()
    i = next(k for k, l in enumerate(lines) if l.startswith(header_start))
    rows = {}
    for l in lines[i + 2:]:
        if not l.startswith("|"):
            break
        cells = [c.strip().replace("**", "") for c in l.strip().strip("|").split("|")]
        rows[cells[0]] = (cells[1:], l)
    return rows


def main():
    swe_dir = os.path.join(PAPER, "swebench")
    if "--rerun" in sys.argv:
        py = sys.executable
        steps = ["build_labels.py", "run_baselines.py", "sensitivity.py", "parity_check.py"]
        if os.path.isdir(os.path.join(ROOT, "data", "comp_dl", "graphs")) or os.environ.get("WTE_COMP_DATA"):
            steps += ["run_graph_embed.py", "run_seed_reach.py"]
        if os.path.isdir(os.path.join(swe_dir, "repos")) and os.path.exists(os.path.join(swe_dir, "tasks.jsonl")):
            steps += ["run_swebench.py"]
        for s in steps + ["make_figure.py", "export_predictions.py", "make_example_figure.py"]:
            subprocess.run([py, "-I", "-X", "utf8", "-W", "ignore", os.path.join(HERE, s)], check=True)

    from collections import Counter
    text = reg(WRITEUP).decode("utf-8")
    tasks = rjl(TASKS)
    labels = rjl(os.path.join(PAPER, "data", "where_to_edit.jsonl"))
    per = rjl(os.path.join(PAPER, "results", "per_task.jsonl"))
    R = rj("results/baselines.json")
    S = rj("results/sensitivity.json")
    PAR = rj("results/parity.json")
    GE = rj("results/graph_embed.json")
    SR = rj("results/seed_reach.json")
    W = rj("swebench/results/baselines.json")
    wlab = rjl(os.path.join(swe_dir, "where_to_edit_swebench_verified.jsonl"))
    wper = rjl(os.path.join(swe_dir, "results", "per_task.jsonl"))
    src = reg(os.path.join(PAPER, "skill", "where-to-edit", "scripts", "locate.py")).decode("utf-8")
    reg(os.path.join(PAPER, "skill", "where-to-edit", "SKILL.md"))
    for s in ("wte_common.py", "build_labels.py", "run_baselines.py", "sensitivity.py", "parity_check.py",
              "make_figure.py", "run_swebench.py", "run_graph_embed.py", "run_seed_reach.py", "analyze_e2e.py", "make_anatomy_figure.py"):
        reg(os.path.join(HERE, s))

    F, G, P = R["issue_hints"]["file"], R["issue_hints"]["func"], R["paired"]
    WF, WG, WP, WS = W["file"], W["func"], W["paired"], W["stats"]
    D = S["distractors"]
    feats = [l["features"] for l in labels]
    with_file = [l for l in labels if l["gold_files"]]
    with_sym = [l for l in labels if l["gold_symbols"]]
    n_named = sum(l["features"]["symbol_named"] for l in with_sym)

    # ---- data description
    rc = Counter(t["repo"] for t in tasks)
    check("competition tasks", len(tasks), 129, ["all 129 public competition tasks", "the 129 public competition tasks", "129 tasks: 67", "for the 129 public tasks"])
    check("per-repo counts", [rc["fastapi/fastapi"], rc["Textualize/rich"], rc["psf/requests"], rc["encode/httpx"]],
          [67, 48, 13, 1], ["67 from `fastapi/fastapi`, 48 from `Textualize/rich`, 13 from `psf/requests` and 1 from `encode/httpx`"])
    check("hints empty", sum(1 for t in tasks if not (t.get("hints_text") or "").strip()), 129, ["`hints_text` is empty"])
    check("median task tokens", statistics.median(f["issue_tokens"] for f in feats), 41, ["(median 41 content tokens)"])
    check("tasks whose fix only adds files", len(tasks) - len(with_file), 4, ["The 4 tasks whose fix only adds files"])
    check("SWE-bench Verified tasks / repos", (len(wlab), len({l["repo"] for l in wlab}), W["n_tasks"], len(W["errors"])), (500, 12, 500, 0),
          ["the 500 tasks of SWE-bench Verified", "(500 tasks, 12 repositories)", "500 tasks from 12 repositories"])
    check("SWE median issue tokens", WS["median_issue_tokens"], 119, ["(median 119 content tokens)"])

    def med(rows, key, repo=None):
        return statistics.median(r[key] for r in rows if repo is None or r["repo"] == repo)
    check("median definitions", [med(per, "n_scopes", "fastapi/fastapi"), med(per, "n_scopes", "Textualize/rich"),
                                 int(round(med(wper, "n_scopes"), -3))], [576, 1027, 9000],
          ["576 definitions for fastapi, 1,027 for rich, and about 9,000 across SWE-bench Verified"])

    # ---- label statistics table (both benchmarks)
    def stats(labs):
        wf = [l for l in labs if l["gold_files"]]
        ws = [l for l in labs if l["gold_symbols"]]
        ch = [l["features"]["adds"] + l["features"]["dels"] for l in labs]
        md = statistics.median(ch)
        return {"Tasks": str(len(labs)), "Tasks with an edited source file": str(len(wf)),
                "Tasks with an edited unit": str(len(ws)), "Edited units": str(sum(len(l["gold_symbols"]) for l in labs)),
                "Tasks editing one source file": str(sum(1 for l in labs if l["features"]["n_gold_files"] == 1)),
                "Median changed lines": str(int(md) if md == int(md) else md),
                "Text names an edited unit": "%d of %d (%s)" % (sum(l["features"]["symbol_named"] for l in ws), len(ws),
                                                                 pct(sum(l["features"]["symbol_named"] for l in ws), len(ws))),
                "Text names an edited file": "%d of %d (%s)" % (sum(l["features"]["file_path_named"] for l in wf), len(wf),
                                                                 pct(sum(l["features"]["file_path_named"] for l in wf), len(wf)))}
    st = table_rows(text, "| Property")
    sc, sw = stats(labels), stats(wlab)
    for k in sc:
        if k in st:
            check("label table: " + k, st[k][0], [sc[k], sw[k]], [st[k][1]])
    check("label table has seven rows", len(st), 7, [])
    check("SWE stats agree with labels", (WS["with_unit_target"], WS["symbol_named"], WS["file_path_named"]),
          (int(sw["Tasks with an edited unit"]), 231, 132), ["SWE-bench Verified (SWE-V): 500 and 485"])
    check("naming shares", (pct(n_named, len(with_sym)), pct(WS["symbol_named"], WS["with_unit_target"]),
                            pct(D["named_defs_gold"], D["named_defs_total"]), "%.1f%%" % (100.0 * WS["named_defs_edited"] / WS["named_defs"])),
          ("29%", "48%", "11%", "0.7%"),
          [
           "in 29% of competition tasks and 48% of SWE-bench Verified tasks"])

    # ---- method configuration (from the skill source)
    cfg_ok = all(x in src for x in ("def __init__(self, files, k1=1.2, b=0.75)", "def graph(self, scope_scores, lam=0.5)",
                                    "def run_all(self, text, lam=0.5, rrf_k=60)",
                                    'add_argument("--top_files", type=int, default=8)',
                                    'add_argument("--top_funcs", type=int, default=12)'))
    cite_only("parameters in locate.py", cfg_ok, ["k1 = 1.2, b = 0.75; graph weight λ = 0.5; RRF k = 60", "top 8 files and the top 12 units"])
    check("number of localizers", (len(R["methods"]), len(WF) - 1), (7, 7), ["best of all seven methods"])
    check("bootstrap resamples", R["bootstrap"], 10000, ["10,000 bootstrap resamples"])
    cite_only("95% percentile intervals", "np.percentile(st, [2.5, 97.5])" in reg(os.path.join(HERE, "run_baselines.py")).decode(),
              ["95% percentile intervals", "Whiskers are 95% bootstrap intervals"])

    # ---- main table
    names = {"Size prior": "size_prior", "Grep (public-prompt strategy)": "grep", "BM25 over files": "bm25_file",
             "BM25 over functions": "bm25_func", "BM25 + graph (skill)": "graph", "RRF of all": "rrf_all", "Cascade": "cascade"}
    mt = table_rows(text, "| Method")
    for label, m in names.items():
        if label not in mt:
            continue
        vals = [f3(F[m]["hit@1"]["mean"]), f3(G[m]["hit@10"]["mean"]), f3(WF[m]["hit@1"]["mean"]), f3(WG[m]["hit@10"]["mean"])]
        check("main table: " + label, mt[label][0], vals, [mt[label][1]])
    check("main table has four rows", len(mt), 4, [])
    check("n file / func", (F["n"], G["n"], WF["n"], WG["n"]), (125, 119, 500, 485),
          ["Competition: 125 tasks for files, 119 for functions. SWE-bench Verified (SWE-V): 500 and 485."])
    check("abstract Hit@10 shares", [pct(G["bm25_func"]["hit@10"]["mean"] * 1000, 1000), pct(G["grep"]["hit@10"]["mean"] * 1000, 1000),
                                     pct(WG["bm25_func"]["hit@10"]["mean"] * 1000, 1000), pct(WG["grep"]["hit@10"]["mean"] * 1000, 1000)],
          ["55%", "47%", "64%", "47%"],
          ["top 10 in 55% versus 47% of competition tasks and 64% versus 47% of SWE-bench Verified tasks"])

    # ---- paired comparisons
    def pc(name, d, cite):
        cite_only("paired " + name + " = " + pdiff(d), pdiff(d) in cite, [cite])
    pc("comp func hit@10 bm25_func-grep", P["func:hit@10:bm25_func-grep"], "+0.084 [+0.008, +0.160] on the competition tasks")
    pc("swe func hit@10 bm25_func-grep", WP["func:hit@10:bm25_func-grep"], "+0.167 [+0.124, +0.210] on SWE-bench Verified")
    wd = WP["func:hit@10:bm25_func-grep"]
    cw = P["func:hit@10:bm25_func-grep"]["ci95"]
    ratio = (wd["ci95"][1] - wd["ci95"][0]) / (cw[1] - cw[0])
    cite_only("SWE interval width / competition width = %.2f" % ratio, 0.45 <= ratio <= 0.65, [])
    check("grep finds no edited function", (G["n"] - G["grep"]["found_any"], G["n"], WS["grep_no_unit_found"], WG["n"]), (48, 119, 67, 485),
          ["48 of 119 competition tasks and 67 of 485 SWE-bench Verified tasks"])
    pc("swe func hit@10 bm25_func-bm25_file", WP["func:hit@10:bm25_func-bm25_file"], "function Hit@10 by +0.456 [+0.408, +0.501] on SWE-bench Verified")
    pc("comp func mrr graph-bm25_func", P["func:mrr:graph-bm25_func"], "raises competition function MRR by +0.027 [+0.006, +0.051]")
    pc("swe func mrr graph-bm25_func", WP["func:mrr:graph-bm25_func"], "(−0.010 [−0.027, +0.008])")
    pc("comp file hit@1 cascade-graph", P["file:hit@1:cascade-graph"], "lowers file Hit@1 by −0.072 [−0.128, −0.016] on the competition tasks")
    pc("swe func hit@10 cascade-graph", WP["func:hit@10:cascade-graph"], "function Hit@10 by −0.099 [−0.136, −0.062] on SWE-bench Verified")

    # ---- names point at the API
    check("named-but-unedited tasks", (D["issue_names_a_defined_symbol"], D["tasks_with_func_target"], D["of_which_none_is_gold"],
                                       WS["tasks_naming_only_unedited_definitions"], WS["tasks_naming_a_definition"]),
          (62, 119, 31, 245, 454), ["In 62 of the 119 competition tasks", "in 31 of those 62 none", "245 of 454 tasks",
           "in 31 of the 62 competition tasks that spell out a repository definition"])
    lab = {l["instance_id"]: l for l in labels}["fastapi_14512"]
    t = {x["instance_id"]: x for x in tasks}["fastapi_14512"]
    title = t["problem_statement"].split("\n")[0]
    repo_files, _ = Repo(t["repo"]).snapshot(t["base_commit"])
    defs = {s.name for p in repo_files for s in locate.extract_scopes(p, repo_files[p])}
    cite_only("fastapi_14512 example", "`Annotated`" in title and "`Body()`" in title and "Body" in defs
              and [g["qualname"] for g in lab["gold_symbols"]] == ["ModelField.__post_init__"],
              ["names `Annotated` and `Body()`, and the fix is in `ModelField.__post_init__`"])
    SN, SU = R["strata"]["symbol_named"]["func"], R["strata"]["symbol_not_named"]["func"]
    WN, WU = W["strata"]["symbol_named"]["func"], W["strata"]["symbol_not_named"]["func"]
    h = lambda x, m: f3(x[m]["hit@10"]["mean"])
    check("strata named", [h(SN, "grep"), h(SN, "graph"), SN["n"], h(WN, "grep"), h(WN, "graph"), WN["n"]],
          ["0.943", "0.914", 35, "0.827", "0.818", 231],
          ["0.943 for grep and 0.914 for BM25 + graph on the competition tasks (35 tasks), and 0.827 and 0.818 on SWE-bench Verified (231 tasks)"])
    check("strata, competition (new wording)", [h(SN, "grep"), h(SN, "graph"), SN["n"], SU["n"], h(SU, "grep"), h(SU, "graph")],
          ["0.943", "0.914", 35, 84, "0.274", "0.405"],
          ["grep reaches 0.943 and BM25 + graph 0.914 (35 tasks); on the remaining 84 tasks grep falls to 0.274 and BM25 + graph holds 0.405"])
    check("strata not named", [h(SU, "grep"), h(WU, "grep"), h(SU, "graph"), h(WU, "graph"), SU["n"], WU["n"]],
          ["0.274", "0.154", "0.405", "0.461", 84, 254],
          ["grep falls to 0.274 and 0.154 while BM25 + graph holds 0.405 and 0.461 (84 and 254 tasks)"])
    cite_only("unnamed group is the majority", SU["n"] > SN["n"] and WU["n"] > WN["n"], [])
    fun = [r for r in per if r["issue_hints_func"]]
    hard = sorted(r["instance_id"] for r in fun if not any(r["issue_hints_func"][m]["hit@10"] for m in R["methods"]))
    hard_file = sorted(l for l in reg(os.path.join(PAPER, "data", "hard_subset.txt")).decode().splitlines() if l)
    wfun = [r for r in wper if r["issue_hints_func"]]
    whard = sorted(r["instance_id"] for r in wfun if not any(r["issue_hints_func"][m]["hit@10"] for m in R["methods"]))
    whard_file = sorted(l for l in reg(os.path.join(PAPER, "data", "hard_subset_swebench_verified.txt")).decode().splitlines() if l)
    check("best-of-methods / hard subsets", (len(fun) - len(hard), len(fun), WS["any_method_func_hit10"], len(wfun), len(hard), len(whard),
                                             hard == hard_file, whard == whard_file), (79, 119, 355, 485, 40, 130, True, True),
          ["top 10 in 79 of 119 competition tasks and 355 of 485 SWE-bench Verified tasks", "The remaining 40 and 130 tasks"])
    fig = rj("figures/fig1_data.json")
    reg(os.path.join(PAPER, "figures", "fig1_named_vs_not.png"))
    fig_ok = len(fig) == 8
    for k, v in fig.items():
        bench, stratum, lv, key, m = k.split("/")
        ref = (R if bench == "competition" else W)["strata"][stratum][lv]
        fig_ok = fig_ok and v["mean"] == ref[m][key]["mean"] and v["ci95"] == ref[m][key]["ci95"] and v["n"] == ref["n"]
    cite_only("figure 1 values == strata in both baselines.json", fig_ok, ["*Figure 1.", "`figures/fig1_data.json`"])

    # ---- parameter stability
    gr = S["grid"]
    low = [v for k, v in gr.items() if not k.endswith("lam=1.0")]
    check("sensitivity range", [f3(min(v["func_mrr"] for v in low)), f3(max(v["func_mrr"] for v in low)), len(low)], ["0.338", "0.387", 9],
          ["b ∈ {0.5, 0.75, 1.0} and λ ∈ {0, 0.25, 0.5}, competition function MRR stays between 0.338 and 0.387"])
    lo = S["loro"]
    check("LORO", [f3(lo["func_mrr"]["mean"]), ci2(lo["func_mrr"]["ci95"]), abs(lo["func_mrr"]["mean"] - G["graph"]["mrr"]["mean"]) < 0.02],
          ["0.374", "[0.31, 0.45]", True], ["function MRR 0.374 [0.31, 0.45], within 0.02 of the default"])

    # ---- provided graphs and embeddings
    C = GE["coverage"]
    check("alignment coverage", (C["units_with_node"], C["units"], pct(C["units_with_node"], C["units"]), C["gold_with_node"], C["gold"],
                                 pct(C["gold_with_node"], C["gold"])), (88592, 92018, "96%", 466, 501, "93%"),
          ["88,592 of 92,018 units (96%) and 466 of 501 edited units (93%)", "We aligned 466 of 501 edited units (93%)"])
    check("graph/embedding params", (GE["params"]["k"], GE["params"]["m"], SR["k"], SR["n"], GE["n_func"]), (10, 5, 10, 119, 119),
          ["top-10 embedding neighbours", "the top five BM25 units"])
    # abstract (qualitative): BM25 seeds reach far more than named seeds; an equal-length BM25 list reaches more still.
    _a = SR["paired"]["+both:bm25_top3-named"]
    _b = SR["paired_vs_bm25_same_size"]["bm25_top3"]["+both"]  # expansion minus equal-length list
    cite_only("abstract: BM25-seeded reach > named-seeded (%s) and expansion - equal-length list (%s) below 0"
              % (pdiff(_a), pdiff(_b)), _a["ci95"][0] > 0 and _b["ci95"][1] < 0,
              ["reach far more edited functions when seeded with BM25 output than with named symbols",
               "a BM25 list of equal length still reaches more"])
    pc("reach +both bm25_top3-named", SR["paired"]["+both:bm25_top3-named"], "raises reach by +0.210 [+0.118, +0.311]")
    pc("reach +both bm25_top3-named (new wording)", SR["paired"]["+both:bm25_top3-named"], "is reached by +0.210 [+0.118, +0.311]")
    cite_only("seed set is the top three BM25 units", "bm25_top3" in SR["reach"], ["the top three BM25 units"])
    cite_only("21 points", round(100 * SR["paired"]["+both:bm25_top3-named"]["mean"]) == 21, ["raises their reach by 21 points"])
    su = SR["strata"]["symbol_not_named"]
    check("reach on unnamed tasks", (su["n"], f3(su["named"]["+both"]), f3(su["bm25_top3"]["+both"])), (84, "0.095", "0.417"),
          ["On the 84 tasks whose text leaves the edited unit unnamed, named seeds reach it in 0.095 and BM25 top-3 seeds in 0.417"])
    pc("reach bm25_top3 +both - seeds", SR["paired"]["bm25_top3:+both-seeds"], "add +0.151 [+0.092, +0.218] over the seeds alone")
    neg = lambda d: {"mean": -d["mean"], "ci95": [-d["ci95"][1], -d["ci95"][0]]}
    check("same-size BM25 at top-3 budget", f3(SR["bm25_same_size"]["bm25_top3"]["+both"]), "0.739", ["reaches 0.739 at the top-3 budget"])
    pc("BM25 list - tool expansion (top-3, both)", neg(SR["paired_vs_bm25_same_size"]["bm25_top3"]["+both"]), "+0.193 [+0.109, +0.286] above the tool expansion")
    pc("BM25 list - tool expansion (new wording)", neg(SR["paired_vs_bm25_same_size"]["bm25_top3"]["+both"]), "reaches +0.193 [+0.109, +0.286] more than the tool expansion")
    check("calls-only at top-5", (f3(SR["reach"]["bm25_top5"]["+calls"]["mean"]), f3(SR["bm25_same_size"]["bm25_top5"]["+calls"]),
                                  SR["size_median"]["bm25_top5"]["+calls"]), ("0.546", "0.588", 15), ["(0.546 versus 0.588 at 15 units"])
    pc("calls expansion - BM25 list (top-5)", SR["paired_vs_bm25_same_size"]["bm25_top5"]["+calls"], "−0.042 [−0.109, +0.017])")
    M = GE["methods"]
    check("rerank Hit@10", (f3(M["off_graph"]["hit@10"]["mean"]), f3(M["bm25_func"]["hit@10"]["mean"]), f3(M["bm25_embed"]["hit@10"]["mean"])),
          ("0.563", "0.546", "0.555"), ["gives function Hit@10 0.563 against 0.546", "gives 0.555"])
    pc("off_graph - bm25_func hit@10", GE["paired"]["hit@10:off_graph-bm25_func"], "(+0.017 [−0.034, +0.067])")
    pc("bm25_embed - bm25_func hit@10", GE["paired"]["hit@10:bm25_embed-bm25_func"], "(+0.008 [−0.025, +0.042])")
    check("LORO embedding weight zero", (sum(1 for v in GE["loro_choice"].values() if v == "0_0"), len(GE["loro_choice"])), (3, 4),
          ["sets the embedding weight to zero for three of four repositories"])
    comp_dl = os.environ.get("WTE_COMP_DATA") or os.path.join(ROOT, "data", "comp_dl")
    npzs = sorted(os.listdir(os.path.join(comp_dl, "embeddings"))) if os.path.isdir(os.path.join(comp_dl, "embeddings")) else []
    dim_cites = ["256-dimensional", "one 256-dimensional vector per node", "`calls` edges", "one 256-dimensional vector per symbol"]
    if npzs:
        import numpy as np
        z = np.load(os.path.join(comp_dl, "embeddings", npzs[0]))
        g0 = json.load(open(os.path.join(comp_dl, "graphs", npzs[0][:-4] + ".json"), encoding="utf-8"))
        check("embedding dimension / edge types", (int(z[z.files[0]].shape[0]), sorted({e.get("type", e.get("relation", "calls")) for e in g0["edges"]})),
              (256, ["calls"]), dim_cites)
    else:
        ROWS.append(["competition graphs/embeddings absent", "SKIPPED", True, dim_cites])

    # ---- application
    check("parity tasks all equal", (PAR["all_equal"], len(PAR["tasks"])), (True, 10), ["confirms on 10 tasks"])
    tq = {x["instance_id"]: x for x in tasks}["fastapi_14077"]
    files14077, _ = Repo(tq["repo"]).snapshot(tq["base_commit"])
    idx = locate.Index(files14077)
    ws = os.path.join(PAPER, "tmp", "ws_fastapi_14077")
    if not os.path.isdir(ws):
        import io, tarfile
        data = subprocess.run(["git", "-C", Repo(tq["repo"]).git, "archive", tq["base_commit"]],
                              capture_output=True, check=True).stdout
        os.makedirs(ws)
        with tarfile.open(fileobj=io.BytesIO(data)) as tf:
            tf.extractall(ws, filter="data")
    times = []
    for _ in range(5):
        t0 = time.time()
        subprocess.run([sys.executable, "-I", os.path.join(PAPER, "skill", "where-to-edit", "scripts", "locate.py"),
                        "--repo", ws, "--query", tq["problem_statement"]], capture_output=True, check=True)
        times.append(time.time() - t0)
    dt = min(times)
    check("skill size on fastapi_14077 snapshot", (len(idx.paths), len(idx.sids)), (66, 498), ["(66 files, 498 units)"])
    cite_only("end-to-end CLI run about a second (fastest of five: %.2f s; accept < 1.5 s)" % dt, dt < 1.5,
              ["answers in about a second", "one call takes about a second on a laptop CPU", "in about a second on a laptop CPU"])
    import ast
    try:
        ast.parse(src, feature_version=(3, 8))
        py38 = True
    except SyntaxError:
        py38 = False
    mods = {a or b for a, b in re.findall(r"(?m)^import (\w+)|^from (\w+) import", src)}
    cite_only("stdlib only + Python 3.8 syntax", py38 and mods <= {"argparse", "ast", "math", "os", "re", "sys", "collections"},
              ["standard library only and Python 3.8+ syntax"])
    cite_only("SWE-bench run time %.0f s on 6 workers" % W["elapsed_sec"], 15 * 60 <= W["elapsed_sec"] <= 25 * 60
              and "else 6" in reg(os.path.join(HERE, "run_swebench.py")).decode(), ["about 20 minutes on six cores"])
    base = os.path.join(ROOT, "variants", "G2_31b_pub022.zip")
    local_cites = ["use the first eight calls to locate and read, and edit a source file by the twelfth call",
                   "list the exact file, function and class names from the task"]
    if os.path.exists(base):
        pub = zipfile.ZipFile(base).read("prompts/system.md").decode()
        cite_only("public prompt call plan", "Calls 1-8: locate and read" in pub and "edited a source file by call 12" in pub
                  and "exact symbols, their callers" in pub, local_cites)
    else:
        ROWS.append(["local-only source absent: public base submission", "SKIPPED", True, local_cites])
    # ---- packaging guards: the shipped skill must be the benchmarked skill
    src_sha = hashlib.sha256(src.encode("utf-8")).hexdigest()
    demo = os.path.join(PAPER, "skill_demo", "G3_31b_pub022_where_to_edit.zip")
    if os.path.exists(demo):
        d_sha = hashlib.sha256(zipfile.ZipFile(demo).read("skills/where-to-edit/scripts/locate.py")).hexdigest()
        cite_only("demo submission ships the benchmarked locate.py (%s)" % src_sha[:16], d_sha == src_sha, [])
    aw = rj("results/adk_wrapper.json")
    cite_only("ADK wrapper run indexes the workspace, same output as CLI (%s)" % aw["first_line"][:70],
              aw["same_output"] and "66 files, 498 defs" in aw["first_line"], [])
    cite_only("no unfilled <<...>> placeholders in the writeup", re.search(r"<<[A-Z0-9_]+>>", text) is None, [])

    # ---- headline (title/abstract/subtitle) and the resource quick start
    R_comp = rj("results/baselines.json")["issue_hints"]
    # score.py on the shipped example predictions must reproduce baselines.json exactly
    import importlib.util
    spec = importlib.util.spec_from_file_location("wte_score", os.path.join(HERE, "score.py"))
    sc_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sc_mod)
    reg(os.path.join(HERE, "score.py"))
    sc_ok = True
    for name, m in (("skill", "graph"), ("grep", "grep")):
        pp = os.path.join(PAPER, "examples", "predictions_%s.jsonl" % name)
        preds = {d["instance_id"]: d for d in rjl(pp)}
        S2 = sc_mod.score(preds, labels)
        for lv, blv, ks in (("files", "file", (1, 5, 10)), ("functions", "func", (1, 10))):
            sm = sc_mod.summarize(S2[lv], ks)
            for k in ["hit@%d" % x for x in ks] + ["mrr"]:
                sc_ok = sc_ok and abs(sm[k][0] - R_comp[blv][m][k]["mean"]) < 1e-12 and sm["n"] == R_comp[blv]["n"]
    cite_only("score.py on examples/ reproduces results/baselines.json (grep, skill; files and functions)", sc_ok,
              [])
    readme = reg(os.path.join(PAPER, "release_README.md")).decode("utf-8")
    shown = readme.split("Expected output of command 1 (first table):", 1)[1].split("```", 2)[1].strip("\n")
    live = subprocess.run([sys.executable, "-I", os.path.join(HERE, "score.py"),
                           os.path.join(PAPER, "examples", "predictions_skill.jsonl")],
                          capture_output=True, text=True, check=True).stdout
    cite_only("README expected output == live score.py output", shown.strip() and shown.strip() in live, [])
    pp_md = reg(os.path.join(PAPER, "skill", "prompt_paragraph.md")).decode("utf-8").strip().splitlines()[-1]
    n_sent = len(re.findall(r"[.!?](?:\s|$)", re.sub(r"`[^`]*`", "X", pp_md)))
    check("prompt paragraph sentences", n_sent, 4, [])
    # figure 2: rich_3521 values from labels, AST and per-task ranks
    F2 = rj("figures/fig2_data.json")
    reg(os.path.join(PAPER, "figures", "fig2_example.png"))
    lab2 = {l["instance_id"]: l for l in labels}["rich_3521"]
    per2 = {r["instance_id"]: r for r in per}["rich_3521"]["issue_hints_func"]
    t2 = {x["instance_id"]: x for x in tasks}["rich_3521"]
    f2_ok = (F2["gold"] == lab2["gold_symbols"][0]["qualname"] == "Segment._split_cells" and len(lab2["gold_symbols"]) == 1
             and F2["named"] == "Segment.split_cells" and "Segment.split_cells" in t2["problem_statement"].split("\n")[0]
             and F2["named_calls_gold"] and not lab2["features"]["symbol_named"]
             and F2["rank_grep"] == per2["grep"]["first_rank"] and F2["rank_skill"] == per2["graph"]["first_rank"])
    cite_only("figure 2 (rich_3521): title names split_cells, gold is _split_cells, split_cells calls it; ranks grep #%s skill #%s"
              % (F2["rank_grep"], F2["rank_skill"]), f2_ok,
              [])
    # ---- failure anatomy (results/anatomy.json, computed by analyze_failures.py from the Kaggle run folders)
    A = rj("results/anatomy.json")
    rjl(os.path.join(PAPER, "results", "anatomy_per_task.jsonl"))
    reg(os.path.join(HERE, "analyze_failures.py"))
    AR = A["runs"]
    EN = ["e4b_base", "e4b_base2", "e4b_ctrl", "e4b_skill"]
    BN = ["b31_base", "b31_skill", "b31_gold"]
    rng = lambda xs: u"%s–%s" % (min(xs), max(xs))
    pc0 = lambda x: "%d%%" % round(100 * x)
    cite_only("anatomy covers seven runs of 129 tasks", sorted(A["order"]) == sorted(EN + BN) and A["n"] == 129
              and all(AR[x]["traces"] == 129 for x in A["order"]),
              ["Across seven runs", "Seven 129-task runs", "in the seven runs", "Each run covers the 129 public tasks",
               "Gemma 4 E4B ran four configurations", "Gemma 4 31B ran three", "The three 31B runs", "Over the four E4B runs", "over the three 31B runs",
               "the two models fail differently"])
    for lbl, k in (("E4B baseline", "e4b_base"), ("E4B baseline, repeat", "e4b_base2"), ("E4B control", "e4b_ctrl"), ("E4B skill", "e4b_skill"),
                   ("31B baseline", "b31_base"), ("31B skill", "b31_skill"), ("31B gold locations", "b31_gold")):
        o = AR[k]["outcomes"]
        cite_only("outcome table row " + lbl, sum(o.values()) == 129, ["| %s | %d | %d | %d | %d | %d | %d |" % (
            lbl, o["resolved"], o["no_patch"], AR[k]["wrong_place"], o["partial"], o["covered"], AR[k]["stops"]["timeout"])])
    figo = rj("figures/fig_outcomes_data.json")
    reg(os.path.join(PAPER, "figures", "fig_outcomes.png"))
    cite_only("figure 1 counts == anatomy.json", all(figo[k]["resolved"] == AR[k]["outcomes"]["resolved"] and figo[k]["no_patch"] == AR[k]["outcomes"]["no_patch"]
                                                      and figo[k]["wrong_place"] == AR[k]["wrong_place"] and figo[k]["covered"] == AR[k]["outcomes"]["covered"]
                                                      and figo[k]["partial"] == AR[k]["outcomes"]["partial"] for k in A["order"]) and len(figo) == 7,
              ["*Figure 1. Outcome of every task in every run", "`figures/fig_outcomes_data.json`"])
    res_b = [AR[x]["outcomes"]["resolved"] for x in BN]
    res_e = [AR[x]["outcomes"]["resolved"] for x in EN]
    to_b = [AR[x]["stops"]["timeout"] for x in BN]
    np_b = [AR[x]["outcomes"]["no_patch"] for x in BN]
    wp_e = [AR[x]["wrong_place"] for x in EN]
    conv = lambda x: AR[x]["resolved_given_full_cover"]
    ce = [pc0(conv(x)["mean"]) for x in EN]
    cb = [pc0(conv(x)["mean"]) for x in BN]
    ce_r = u"%d–%d%%" % (min(int(c[:-1]) for c in ce), max(int(c[:-1]) for c in ce))
    cb_r = u"%d–%d%%" % (min(int(c[:-1]) for c in cb), max(int(c[:-1]) for c in cb))
    cite_only("abstract: 31B profile", True, [u"Gemma 4 31B resolves %s tasks, hits the time limit in %s and ends %s with no patch" % (rng(res_b), rng(to_b), rng(np_b))])
    cite_only("abstract: E4B profile", True, [u"E4B resolves %s, patches almost every task, puts %s patches in the wrong place" % (rng(res_e), rng(wp_e)),
                                               u"its tests pass in %s of cases against %s for 31B" % (ce_r, cb_r),
                                               u"then converts covered patches at %s" % ce_r])
    cite_only("abstract: gold locations leave 31B at %d" % AR["b31_gold"]["outcomes"]["resolved"], True,
              ["leaves it at %d resolved" % AR["b31_gold"]["outcomes"]["resolved"],
               "Gold locations return %d resolved tasks against %d for the baseline" % (AR["b31_gold"]["outcomes"]["resolved"], AR["b31_base"]["outcomes"]["resolved"])])
    b0 = AR["b31_base"]
    w = b0["wrong_place_of_patches"]
    cite_only("31B baseline: no patch / wrong place", True, ["%d of the %d unresolved tasks have no patch, and %d of its %d patches (%s [%s, %s]) land in the wrong place" % (
        b0["outcomes"]["no_patch"], b0["unresolved"], w["k"], w["n"], pc0(w["mean"]), pc0(w["ci95"][0]), pc0(w["ci95"][1]))])
    pe = [AR[x]["patches"] for x in EN]
    wpe = [round(100 * AR[x]["wrong_place_of_patches"]["mean"]) for x in EN]
    cov_e = [AR[x]["outcomes"]["covered"] for x in EN]
    cite_only("E4B patches and wrong-place share", True, [u"E4B submits %s patches per run, and %s%% of them land in the wrong place" % (rng(pe), rng(wpe)),
                                                         u"A further %s tasks per run reach every gold unit and still fail" % rng(cov_e)])
    e4 = ["%d of %d" % (conv(x)["k"], conv(x)["n"]) for x in EN]
    cite_only("conversion given full cover", True, [
        "in %s and %s cases across the E4B runs" % (", ".join(e4[:-1]), e4[-1]),
        "and in %d of %d, %d of %d and %d of %d cases for 31B" % tuple(v for x in BN for v in (conv(x)["k"], conv(x)["n"]))])
    cite_only("conversion: E4B about half of 31B", max(conv(x)["mean"] for x in EN) < min(conv(x)["mean"] for x in BN)
              and 0.4 <= (sum(conv(x)["k"] for x in EN) / float(sum(conv(x)["n"] for x in EN))) / (sum(conv(x)["k"] for x in BN) / float(sum(conv(x)["n"] for x in BN))) <= 0.6,
              ["converts at half the rate on E4B", "E4B converts at about half of 31B's rate"])
    sim_r = [AR[x]["median_add_sim"]["resolved"] for x in A["order"]]
    sim_c = [AR[x]["median_add_sim"]["covered"] for x in A["order"]]
    lr_c = [AR[x]["median_line_ratio"]["covered"] for x in A["order"]]
    check("identifier overlap of resolved / covered patches", ("%.2f" % (int(min(sim_r) * 100) / 100.0), sum(1 for v in sim_r if v == 1.0),
                                                                u"%.2f–%.2f" % (min(sim_c), max(sim_c)), u"%.2f–%.2f" % (min(lr_c), max(lr_c))),
          ("0.72", 4, u"0.19–0.53", u"0.83–1.36"),
          ["at least 0.72 for resolved patches in every run and 1.0 in four", u"overlap at 0.19–0.53", u"(median line ratio 0.83–1.36)"])
    ta_e = [AR[x]["tested_after_edit"] for x in EN]
    x2e = [AR[x]["exit2_unresolved_patches"] for x in EN]
    x2b = [AR[x]["exit2_unresolved_patches"] for x in BN]
    cite_only("tests after last edit; pytest exit 2", True, [u"runs tests after its last edit in %s tasks per run and resolves %s" % (rng(ta_e), rng(res_e)),
                                                            u"For %s E4B patches and %s 31B patches per run, pytest stops at collection (exit code 2)" % (rng(x2e), rng(x2b))])
    # 31B time
    l3 = lambda key: tuple(AR[x][key] for x in BN)
    cite_only("31B timeouts", True, ["hit the 8-minute limit in %d, %d and %d tasks" % tuple(to_b),
                                     "In %d, %d and %d of them the agent never edits a source file, and %d, %d and %d end with fewer than 28 calls used" % (
                                         l3("timeouts_no_source_edit") + l3("timeouts_under_call_budget")),
                                     u"and %s tasks hit it" % rng(to_b)])
    fe_b = [int(round(AR[x]["median_first_edit_s"])) for x in BN]
    fe_e = [int(round(AR[x]["median_first_edit_s"])) for x in EN]
    cite_only("median first source edit", True, [u"at a median %d, %d and %d s; E4B at %s s" % (tuple(fe_b) + (rng(fe_e),))])
    nse = sum(AR[x]["no_source_edit"] for x in BN)
    check("31B runs without a source edit", (nse, sum(AR[x]["resolved_no_source_edit"] for x in BN)), (144, 0), ["(0 of 144 run-tasks)"])
    pool = lambda b, k: sum(AR[x]["first_edit_bins"][b][k] for x in BN)
    cite_only("first-edit time vs resolution (pooled 31B)", True, [
        "resolves %d of %d tasks when the first edit comes before 180 s and %d of %d from 300 s on" % (
            pool("lt180", "resolved"), pool("lt180", "n"), pool("ge300", "resolved"), pool("ge300", "n")),
        "the same split gives %d of %d against %d of %d" % (pool("lt180", "resolved_small"), pool("lt180", "n_small"),
                                                           pool("ge300", "resolved_small"), pool("ge300", "n_small"))])
    CM = A["cross_model"]
    cite_only("tasks only E4B solved", True, ["The %d tasks that only E4B ever solved end with no 31B patch in %d of %d attempts" % (
        CM["e4b_only"], CM["b31_no_patch_on_e4b_only"], CM["b31_attempts_on_e4b_only"])])
    # extent
    X = A["extent"]
    sh = lambda s: "%d of %d" % (s["k"], s["n"])
    civ = lambda s: "(%s [%s, %s])" % (pc0(s["mean"]), pc0(s["ci95"][0]), pc0(s["ci95"][1]))
    fmt_med = lambda v: str(int(v)) if v == int(v) else str(v)
    cite_only("small fixes", A["small_lines"] == 15, [
        "%d tasks were solved at least once in the seven runs, and %d of them have a reference fix of at most 12 changed source lines (median %s, against %s for never-solved tasks)" % (
            X["ever_solved"], X["ever_solved_le12"], fmt_med(X["median_gold_lines_solved"]), fmt_med(X["median_gold_lines_unsolved"])),
        "Splitting at 15 lines, %s small tasks were solved at least once %s and %s larger ones %s" % (
            sh(X["any"]["small"]), civ(X["any"]["small"]), sh(X["any"]["large"]), civ(X["any"]["large"])),
        "%d of the %d tasks whose reference fix changes at most 15 lines were solved at least once, against %d of the %d larger ones" % (
            X["any"]["small"]["k"], X["any"]["small"]["n"], X["any"]["large"]["k"], X["any"]["large"]["n"]),
        "Of the %d tasks whose fix spans five or more units or new files, %d was solved" % (X["units5plus"]["n"], X["units5plus"]["ever_any"]),
        "Tasks with fixes above 15 lines return %d solved tasks from %d" % (X["any"]["large"]["k"], X["any"]["large"]["n"])])
    NA = A["new_api"]
    cite_only("new interfaces", True, ["In %d tasks the tests use an identifier" % NA["tasks"], "all such identifiers in %d of them" % NA["all_in_text"],
                                       "These %d tasks were resolved in %d of %d run-task attempts" % (NA["tasks"], NA["run_tasks_resolved"], NA["run_tasks"]),
                                       "are solved in %d of %d attempts" % (NA["run_tasks_resolved"], NA["run_tasks"])])
    ST = A["stability"]
    ea = CM["e4b_attempts_on_b31_only"]
    cite_only("model gap", True, ["31B solved %d tasks at least once, E4B %d, and %d are shared" % (ST["ever_b31"], ST["ever_e4b"], ST["ever_both"]),
                                  "On the %d tasks only 31B solved, E4B's %d attempts end in the wrong place %d times, covered and failing %d times, and partial %d times" % (
                                      CM["b31_only"], sum(ea.values()), ea["wrong_file"] + ea["wrong_unit"], ea["covered"], ea["partial"])])
    rp = ST["pairs"]["e4b_base2-e4b_base"]
    both = AR["e4b_base2"]["outcomes"]["resolved"] - rp["resolved"]["gained"]
    cite_only("noise floor: identical runs", True, [
        "resolve %d and %d tasks and share %d: %d solved tasks appear and %d disappear" % (
            AR["e4b_base"]["outcomes"]["resolved"], AR["e4b_base2"]["outcomes"]["resolved"], both, rp["resolved"]["gained"], rp["resolved"]["lost"]),
        "Gold-file edits flip in %d tasks (%d gained, %d lost), and %d of 129 tasks keep their outcome class" % (
            rp["file_hit"]["gained"] + rp["file_hit"]["lost"], rp["file_hit"]["gained"], rp["file_hit"]["lost"], rp["same_outcome"]),
        "identical runs disagree on %d solved tasks" % (rp["resolved"]["gained"] + rp["resolved"]["lost"]),
        "Two identical runs differ on %d solved tasks" % (rp["resolved"]["gained"] + rp["resolved"]["lost"])])
    ec, bc = ST["e4b_solved_count"], ST["b31_solved_count"]
    cite_only("solved-count distribution", True, [
        "%d tasks are solved every time, %d sometimes and %d never" % (ec["4"], ec["1"] + ec["2"] + ec["3"], ec["0"]),
        "%d every time and %d sometimes" % (bc["3"], bc["1"] + bc["2"])])
    cite_only("five-task noise floor >= observed flip count", rp["resolved"]["gained"] + rp["resolved"]["lost"] == 5,
              ["A single-run difference of five solved tasks sits inside this variation", "Treat a difference under five solved tasks between single 129-task runs as noise"])
    cb_wp = [100 - round(100 * AR[x]["wrong_place_of_patches"]["mean"]) for x in BN]
    cite_only("31B correct placement share", True, [u"31B places %s%% of its patches correctly in every run" % rng(cb_wp)])

    # ---- information levels inside the agent (results/e2e.json for E4B, e2e_31b.json for 31B; analyze_e2e.py)
    E = rj("results/e2e.json")
    R, PR = E["runs"], E["paired"]
    cite_only("e2e run order and n", E["order"] == ["baseline", "baseline_r2", "placebo", "skill"] and E["n"] == 129
              and all(R[k]["resolved"] == AR[a]["outcomes"]["resolved"] for k, a in zip(E["order"], EN)), [])
    cite_only("control returned no repository information", R["placebo"]["tasks_skill_indexed_repo"] == 0
              and R["placebo"]["tasks_calling_skill"] > 100, ["a control whose skill call returns zero repository information"])
    sp, sb, sr = PR["skill-placebo"], PR["skill-baseline"], PR["skill-baseline_r2"]
    cite_only("E4B skill vs control", R["skill"]["traces"] == 129, [
        "The model called the skill in %d tasks. Against the control, edits move onto a gold file in %d tasks and off it in %d (%s); resolution moves by %d gained and %d lost" % (
            R["skill"]["tasks_calling_skill"], sp["file_hit"]["gained"], sp["file_hit"]["lost"], pdiff(sp["file_hit"]["diff"]),
            sp["resolved"]["gained"], sp["resolved"]["lost"])])
    cite_only("E4B skill vs baselines, gold file", True, ["the gold-file change is %s and %s" % (pdiff(sb["file_hit"]["diff"]), pdiff(sr["file_hit"]["diff"]))])
    E3 = rj("results/e2e_31b.json")
    R3, P3 = E3["runs"], E3["paired"]
    cite_only("31B e2e run order and n", E3["order"] == ["baseline", "skill", "gold"] and E3["n"] == 129
              and all(R3[k]["resolved"] == AR[a]["outcomes"]["resolved"] and R3[k]["traces"] == 129 for k, a in zip(E3["order"], BN)), [])
    s3, g3 = P3["skill-baseline"], P3["gold-baseline"]
    cite_only("31B skill vs baseline", True, ["The model called the skill in %d tasks. Gold-file edits change by %d gained and %d lost, resolution by %d gained and %d lost" % (
        R3["skill"]["tasks_calling_skill"], s3["file_hit"]["gained"], s3["file_hit"]["lost"], s3["resolved"]["gained"], s3["resolved"]["lost"])])
    cite_only("31B gold vs baseline; the gold skill returned gold units", R3["gold"]["n_indexed_not_listed"] == 0
              and R3["gold"]["skill_listed_gold_unit_top3"] == R3["gold"]["skill_listed_gold_unit"], [
        "The model called the gold-location skill in %d tasks, and the returned list contained a gold unit in %d. Gold-file edits change by %d gained and %d lost (%s); resolution changes by %d gained and %d lost" % (
            R3["gold"]["tasks_calling_skill"], R3["gold"]["skill_listed_gold_unit"], g3["file_hit"]["gained"], g3["file_hit"]["lost"],
            pdiff(g3["file_hit"]["diff"]), g3["resolved"]["gained"], g3["resolved"]["lost"])])
    ob = rj("results/oracle_build.json")
    cite_only("gold-location skill self-test: every task maps to its own gold set", ob["selftest_full_text_query_own_gold"] == ob["tasks"] == 129, [])
    # budget of the baseline submission (local-only run folders)
    cfgp = os.path.join(ROOT, "results", "eval", "31b-pub022", "agent", "eval_config.yaml")
    bud_cites = ["budget of 28 tool calls and 8 minutes", "28 tool calls and 8 minutes per task", "eight tasks in parallel", "four L4 GPUs", "The 8-minute limit is the submission's own setting"]
    if os.path.exists(cfgp):
        cfg = open(cfgp, encoding="utf-8").read()
        summ = [json.load(open(os.path.join(ROOT, "results", "eval", d, "eval_summary.json"), encoding="utf-8"))
                for d in ("e4b-pub022", "e4b-pub022-r2", "e4b-g3-wte", "e4b-g3b-wte", "31b-pub022", "31b-g3b-wte", "31b-g7-oracle")]
        cite_only("budget file and concurrency of all seven runs", "max_tool_calls: 28" in cfg and "max_time_minutes: 8" in cfg
                  and all(s["concurrency"] == 8 and s["budget"][1] == 28 and s["budget"][2] == 8.0 and s["n"] == 129 for s in summ)
                  and "gpus 4" in open(os.path.join(ROOT, "results", "eval", "31b-g7-oracle", "gemma4-eval-31b-g7-oracle.log"), encoding="utf-8").read()
                  and '"machine_shape": "NvidiaL4"' in open(os.path.join(ROOT, "notebooks", "eval", "31b-g7-oracle", "kernel-metadata.json"), encoding="utf-8").read(),
                  bud_cites)
    else:
        ROWS.append(["local-only source absent: run folders (budget, concurrency)", "SKIPPED", True, bud_cites])
    sub = reg(os.path.join(PAPER, "subtitle.txt")).decode("utf-8")
    cite_only("subtitle numbers are checked values", set(re.findall(r"\d+", sub)) <= {"129", "31", "4", str(AR["b31_gold"]["outcomes"]["resolved"]),
                                                                                    str(AR["b31_base"]["outcomes"]["resolved"])}, [])

    # ---- cite presence
    body = text.split("## References")[0]
    # claims of earlier drafts that the text no longer makes: their checks are dropped together with the sentences
    ROWS[:] = [r for r in ROWS if not r[3] or any(c not in RETIRED for c in r[3])]
    for r in ROWS:
        r[3][:] = [c for c in r[3] if c not in RETIRED]
    missing = [(c, claim) for claim, _, _, cites in ROWS for c in cites if c not in text]
    # ---- uncited numbers in body (outside code)
    scan = re.sub(r"```.*?```", lambda m: " " * len(m.group(0)), body, flags=re.S)
    scan = re.sub(r"`[^`\n]*`", lambda m: " " * len(m.group(0)), scan)
    covered = [False] * len(scan)
    allow = ["Figure 1", "Gemma 4", "Python 3.8+", "top-1", "top-3", "Hit@k", "top k", "## 1.", "## 2.", "## 3.", "## 4.", "## 5.",
             "## 6.", "## 7.", "## 8.", "### 5.1", "### 5.2", "### 5.3", "### 5.4", "### 6.1", "### 6.2", "### 6.3", "31B", "Section 5.4", "Section 5",
             "Section 6", "1. **An instrument.**", "2. **A failure anatomy.**", "3. **A localization ceiling.**", "4. **Limits"]
    for c in [c for _, _, _, cs in ROWS for c in cs] + allow:
        for m in re.finditer(re.escape(c), body):
            for k in range(m.start(), m.end()):
                covered[k] = True
    num_re = re.compile(r"(?<![A-Za-z_@\[\w])[−+±]?\d[\d,]*(?:\.\d+)?%?")
    uncited = []
    for m in num_re.finditer(scan):
        if not all(covered[k] for k in range(m.start(), m.end())):
            line = scan[:m.start()].count("\n") + 1
            uncited.append((line, m.group(0)))
    fails = [r for r in ROWS if not r[2]]

    # ---- report
    out = ["# REPRODUCE_REPORT", "", "Generated by `paper/scripts/reproduce_numbers.py`.", "",
           "| ok | claim | computed |", "|---|---|---|"]
    for claim, comp_, ok, _ in ROWS:
        out.append("| %s | %s | %s |" % ("PASS" if ok else "FAIL", claim.replace("|", "/"), comp_.replace("|", "/")))
    out += ["", "Missing cites: %d" % len(missing)] + ["- %s (%s)" % m for m in missing]
    out += ["", "Uncited numbers: %d" % len(uncited)] + ["- line %d: %s" % u for u in uncited]
    out += ["", "## Source files", "", "| file | bytes | sha256 |", "|---|---|---|"]
    for k, (n, h) in sorted(USED.items()):
        out.append("| %s | %d | %s |" % (k, n, h))
    with open(os.path.join(PAPER, "REPRODUCE_REPORT.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(out) + "\n")
    print("checks %d, failed %d, missing cites %d, uncited numbers %d" % (len(ROWS), len(fails), len(missing), len(uncited)))
    for r in fails:
        print("FAIL", r[0], r[1])
    for m in missing:
        print("MISSING", m)
    for u in uncited:
        print("UNCITED line %d: %s" % u)
    return 0 if not (fails or missing or uncited) else 1


if __name__ == "__main__":
    sys.exit(main())
