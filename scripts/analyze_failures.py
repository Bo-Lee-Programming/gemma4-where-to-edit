"""Failure anatomy of agent runs on the 129 public tasks: one feature row per (run, task).

Inputs per run directory (output of the evaluation notebook): eval_tasks.csv, results/patches/<id>.patch,
results/traces/trace_<id>.json.  Task-side inputs: gold patch, test patch, Where-to-Edit labels, base snapshot.

Task features
  gold_units / gold_files / gold_lines   extent of the reference fix
  new_api          identifiers that the test patch uses, the reference fix introduces, and the base commit lacks
  new_api_in_text  how many of those the task text spells out
Run features
  outcome          resolved | no_patch | wrong_file | wrong_unit | partial | covered (see classify())
  unit_cov         share of gold units the agent patch touches
  new_api_made     share of new_api identifiers present in the agent patch
  add_sim          Jaccard overlap between identifiers on the agent's and the reference fix's added source lines
  stop             harness stop reason: '', timeout, tool_budget, turns, other
  first_edit_call  index of the first successful source edit; tested_after_edit; submitted
Usage: python -I paper/scripts/analyze_failures.py name=dir [name=dir ...]  ->  results/anatomy_per_task.jsonl, anatomy.json
       python -I paper/scripts/analyze_failures.py --summary               ->  anatomy.json from the saved per-task file
Run names starting with e4b / b31 group the runs by model.
"""
import csv
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_labels import label_one  # noqa: E402
from wte_common import PAPER, Repo, load_labels, load_tasks, parse_patch  # noqa: E402

sys.path.insert(0, os.path.join(PAPER, "skill", "where-to-edit", "scripts"))
import locate  # noqa: E402

IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
TESTY = re.compile(r"(^|/)(tests?|testing)/|(^|/)test_[^/]*\.py$|_test\.py$|(^|/)conftest\.py$")


def added_idents(patch, want):
    """Identifiers on added lines of the files selected by want(path)."""
    out = set()
    for p, info in parse_patch(patch).items():
        if want(p):
            for _, text in info["anchors"]:
                out.update(IDENT.findall(text))
    return out


def is_test(p):
    return bool(TESTY.search(p))


def task_features(t, lab, repo, cache):
    key = (t["repo"], t["base_commit"])
    if key not in cache:
        files, _ = repo.snapshot(t["base_commit"], only_candidates=False)
        vocab = set()
        for src in files.values():
            vocab.update(IDENT.findall(src))
        cache[key] = vocab
    vocab = cache[key]
    gold_new = added_idents(t["patch"], lambda p: p.endswith(".py") and not is_test(p)) - vocab
    test_new = added_idents(t.get("test_patch") or "", lambda p: True) - vocab
    api = sorted(gold_new & test_new)
    text = t.get("problem_statement") or ""
    pf = parse_patch(t["patch"])
    src = [p for p in pf if p.endswith(".py") and not is_test(p)]
    return {"gold_units": len(lab["gold_symbols"]), "gold_files": len(lab["gold_files"]),
            "gold_new_files": len(lab["new_source_files"]),
            "gold_lines": sum(pf[p]["adds"] + pf[p]["dels"] for p in src),
            "symbol_named": bool(lab["features"]["symbol_named"]),
            "new_api": api, "new_api_in_text": [a for a in api if re.search(r"(?<![A-Za-z0-9_])%s(?![A-Za-z0-9_])" % re.escape(a), text)]}


def trace_features(d, iid):
    p = os.path.join(d, "results", "traces", "trace_%s.json" % iid)
    if not os.path.exists(p):
        return None
    tr = json.load(open(p, encoding="utf-8"))
    calls = []  # (name, ok, is_source_edit, is_test_run, elapsed)
    for s in tr["steps"]:
        obs = json.dumps(s.get("observation"))
        for tc in s.get("tool_calls") or []:
            name, a = tc["function_name"], tc.get("arguments") or {}
            ok = '\\"status\\": \\"error\\"' not in obs[:400]
            fp = str(a.get("filepath") or "")
            cmd = str(a.get("command") or "")
            edit = name in ("edit_file", "write_file") or (name == "run_command" and re.search(r"\bsed -i\b|>\s*\S+\.py|\bpatch\b|git apply", cmd) is not None)
            calls.append({"name": name, "ok": ok, "src_edit": bool(edit and ok and name != "run_command" and not is_test(fp)),
                          "test_edit": bool(edit and name != "run_command" and is_test(fp)),
                          "edit_try": bool(edit), "edit_fail": bool(edit and not ok),
                          "test_run": name == "run_command" and re.search(r"\bpytest\b|python[0-9.]* -m (pytest|unittest)|\btox\b", cmd) is not None,
                          "t": float((tc.get("extra") or {}).get("elapsed_s") or 0)})
    src = [i for i, c in enumerate(calls) if c["src_edit"]]
    fm = tr.get("final_metrics") or {}
    last = calls[-1]["t"] if calls else 0.0
    return {"calls": len(calls), "first_edit_call": src[0] + 1 if src else None, "first_edit_s": calls[src[0]]["t"] if src else None,
            "src_edits": len(src), "edit_fails": sum(c["edit_fail"] for c in calls), "test_edits": sum(c["test_edit"] for c in calls),
            "test_runs": sum(c["test_run"] for c in calls),
            "tested_after_edit": bool(src and any(c["test_run"] for c in calls[src[-1] + 1:])),
            "submitted": any(c["name"] == "submit_patch" for c in calls), "last_call_s": last,
            "completion_tokens": fm.get("total_completion_tokens"), "prompt_tokens": fm.get("total_prompt_tokens"),
            "skill_calls": sum(c["name"] == "run_skill_script" for c in calls),
            "read_calls": sum(c["name"] == "read_file" for c in calls),
            "cmd_calls": sum(c["name"] == "run_command" for c in calls)}


def stop_reason(err):
    e = (err or "").lower()
    return "" if not e else "timeout" if "session timeout" in e else "tool_budget" if "tool call budget" in e else "turns" if "turns budget" in e else "other"


def classify(r, tf):
    """Mutually exclusive outcome, first match wins.  Tasks without an existing-file target (new files only) are judged on
    patch presence alone: no_patch / covered."""
    if r["resolved"]:
        return "resolved"
    if r["empty"]:
        return "no_patch"
    if not tf["gold_files"]:
        return "covered"
    if not r["file_hit"]:
        return "wrong_file"
    if tf["gold_units"] and not r["unit_hit"]:
        return "wrong_unit"
    if r["unit_cov"] is not None and r["unit_cov"] < 1 or r["file_cov"] < 1:
        return "partial"
    return "covered"


def load_run(d, tasks, labels, repos, TF):
    rows = {r["instance_id"]: r for r in csv.DictReader(open(os.path.join(d, "eval_tasks.csv"), encoding="utf-8"))}
    out = {}
    for iid, r in rows.items():
        t, lab, tf = tasks[iid], labels[iid], TF[iid]
        pp = os.path.join(d, "results", "patches", iid + ".patch")
        patch = open(pp, encoding="utf-8", errors="replace").read() if os.path.exists(pp) else ""
        files, units, lines, tests_touched, made, sim = set(), set(), 0, 0, None, None
        if patch.strip():
            try:
                al = label_one(dict(t, patch=patch), repos.setdefault(t["repo"], Repo(t["repo"])))
                files, units = set(al["gold_files"]), {(g["path"], g["qualname"]) for g in al["gold_symbols"]}
            except Exception as e:  # malformed agent patch
                print("patch parse failed", iid, repr(e)[:80])
            pf = parse_patch(patch)
            lines = sum(v["adds"] + v["dels"] for p, v in pf.items() if p.endswith(".py") and not is_test(p))
            tests_touched = sum(1 for p in pf if is_test(p))
            have = added_idents(patch, lambda p: not is_test(p))
            gold = added_idents(t["patch"], lambda p: not is_test(p))
            sim = len(have & gold) / float(len(have | gold)) if have | gold else None
            if tf["new_api"]:
                made = sum(1 for a in tf["new_api"] if a in have) / float(len(tf["new_api"]))
        gu = {(g["path"], g["qualname"]) for g in lab["gold_symbols"]}
        gf = set(lab["gold_files"])
        row = {"resolved": int(str(r["resolved"]).lower() == "true"), "exit": int(float(r["test_exit_code"] or -1)),
               "tool_calls": int(float(r["tool_calls"] or 0)), "llm_calls": int(float(r["total_llm_calls"] or 0)),
               "duration": float(r["duration_seconds"] or 0), "stop": stop_reason(r["error"]), "empty": int(not patch.strip()),
               "file_hit": int(bool(files & gf)), "unit_hit": int(bool(units & gu)),
               "file_cov": len(files & gf) / float(len(gf)) if gf else 1.0,
               "unit_cov": len(units & gu) / float(len(gu)) if gu else None,
               "extra_units": len(units - gu), "extra_files": len(files - gf), "agent_units": len(units),
               "agent_lines": lines, "tests_touched": tests_touched, "new_api_made": made, "add_sim": sim,
               "trace": trace_features(d, iid)}
        row["full_cover"] = int(bool(gf) and not row["empty"] and row["file_cov"] == 1 and row["unit_cov"] in (None, 1.0))
        row["outcome"] = classify(row, tf)
        out[iid] = row
    return out


OUTCOMES = ("resolved", "no_patch", "wrong_file", "wrong_unit", "partial", "covered")
SMALL = 15  # reference-fix size (changed source lines) that splits the tasks


def summarize():
    """results/anatomy_per_task.jsonl -> results/anatomy.json: every aggregate the writeup cites."""
    import statistics
    import run_baselines as rb
    R = [json.loads(l) for l in open(os.path.join(PAPER, "results", "anatomy_per_task.jsonl"), encoding="utf-8")]
    names = list(R[0]["runs"])
    E = [n for n in names if n.startswith("e4b")]
    B = [n for n in names if n.startswith("b31")]
    med = lambda xs: statistics.median(xs) if xs else None
    share = lambda xs: {"k": int(sum(xs)), "n": len(xs), "mean": rb.mean(xs), "ci95": rb.boot_ci(xs)}
    ever = lambda r, rs: int(any(r["runs"][x]["resolved"] for x in rs))
    out = {"n": len(R), "order": names, "small_lines": SMALL, "runs": {}}
    for x in names:
        Q = [(r, r["runs"][x]) for r in R]
        o = {k: sum(q["outcome"] == k for _, q in Q) for k in OUTCOMES}
        unres = [q for _, q in Q if not q["resolved"]]
        pat = [q for _, q in Q if not q["empty"]]
        fc = [q for _, q in Q if q["full_cover"]]
        uh = [q for _, q in Q if q["unit_hit"]]
        tr = [(q, q["trace"]) for _, q in Q if q["trace"]]
        to = [(q, t) for q, t in tr if q["stop"] == "timeout"]
        bins = {}
        for lab, lo, hi in (("lt180", 0, 180), ("180_300", 180, 300), ("ge300", 300, 1e9)):
            g = [q for q, t in tr if t["first_edit_s"] is not None and lo <= t["first_edit_s"] < hi]
            gs = [q for (r, q) in Q if q["trace"] and q["trace"]["first_edit_s"] is not None
                  and lo <= q["trace"]["first_edit_s"] < hi and r["task"]["gold_lines"] <= SMALL]
            bins[lab] = {"n": len(g), "resolved": sum(q["resolved"] for q in g), "n_small": len(gs), "resolved_small": sum(q["resolved"] for q in gs)}
        out["runs"][x] = {
            "outcomes": o, "unresolved": len(unres), "patches": len(pat),
            "wrong_place": o["wrong_file"] + o["wrong_unit"],
            "share_no_patch": share([int(q["outcome"] == "no_patch") for _, q in Q]),
            "share_wrong_place": share([int(q["outcome"] in ("wrong_file", "wrong_unit")) for _, q in Q]),
            "wrong_place_of_patches": share([int(q["outcome"] in ("wrong_file", "wrong_unit")) for q in pat]),
            "stops": {k: sum(q["stop"] == k for _, q in Q) for k in ("timeout", "tool_budget", "turns", "other")},
            "exit1_unresolved_patches": sum(q["exit"] == 1 for q in unres if not q["empty"]),
            "exit2_unresolved_patches": sum(q["exit"] == 2 for q in unres if not q["empty"]),
            "unit_hit": len(uh), "resolved_given_unit_hit": sum(q["resolved"] for q in uh),
            "full_cover": len(fc), "resolved_given_full_cover": share([q["resolved"] for q in fc]),
            "resolved_without_full_cover": sum(q["resolved"] and not q["full_cover"] for _, q in Q),
            "median_add_sim": {k: med([q["add_sim"] for _, q in Q if q["outcome"] == k and q["add_sim"] is not None]) for k in OUTCOMES},
            "median_line_ratio": {k: med([q["agent_lines"] / float(max(r["task"]["gold_lines"], 1)) for r, q in Q if q["outcome"] == k and not q["empty"]]) for k in OUTCOMES},
            "traces": len(tr), "timeouts": len(to), "timeouts_no_source_edit": sum(1 for q, t in to if not t["src_edits"]),
            "timeouts_under_call_budget": sum(1 for q, t in to if t["calls"] < 28), "median_calls_at_timeout": med([t["calls"] for q, t in to]),
            "no_source_edit": sum(1 for q, t in tr if not t["src_edits"]), "resolved_no_source_edit": sum(q["resolved"] for q, t in tr if not t["src_edits"]),
            "median_first_edit_s": med([t["first_edit_s"] for q, t in tr if t["first_edit_s"] is not None]),
            "median_first_edit_call": med([t["first_edit_call"] for q, t in tr if t["first_edit_call"]]),
            "first_edit_bins": bins,
            "tested_after_edit": sum(1 for q, t in tr if t["tested_after_edit"]), "with_source_edit": sum(1 for q, t in tr if t["src_edits"]),
            "skill_calls_tasks": sum(1 for q, t in tr if t["skill_calls"]),
            "median_duration": med([q["duration"] for _, q in Q]), "median_tool_calls": med([q["tool_calls"] for _, q in Q]),
        }
    # extent of the reference fix
    sm = [r for r in R if r["task"]["gold_lines"] <= SMALL]
    lg = [r for r in R if r["task"]["gold_lines"] > SMALL]
    ext = {}
    for lab, rs in (("e4b", E), ("b31", B), ("any", E + B)):
        ext[lab] = {"small": share([ever(r, rs) for r in sm]), "large": share([ever(r, rs) for r in lg])}
    ext["per_run_small"] = {x: sum(r["runs"][x]["resolved"] for r in sm) for x in names}
    ext["per_run_large"] = {x: sum(r["runs"][x]["resolved"] for r in lg) for x in names}
    sl = sorted(r["task"]["gold_lines"] for r in R if ever(r, E + B))
    ext["ever_solved"] = len(sl)
    ext["ever_solved_le12"] = sum(1 for v in sl if v <= 12)
    ext["median_gold_lines_solved"] = med(sl)
    ext["median_gold_lines_unsolved"] = med([r["task"]["gold_lines"] for r in R if not ever(r, E + B)])
    nu = lambda r: r["task"]["gold_units"] + r["task"]["gold_new_files"]
    ext["units5plus"] = {"n": sum(1 for r in R if nu(r) >= 5), "ever_any": sum(ever(r, E + B) for r in R if nu(r) >= 5)}
    ext["units1"] = {"n": sum(1 for r in R if nu(r) <= 1), "ever_any": sum(ever(r, E + B) for r in R if nu(r) <= 1)}
    out["extent"] = ext
    na = [r for r in R if r["task"]["new_api"]]
    out["new_api"] = {"tasks": len(na), "all_in_text": sum(1 for r in na if len(r["task"]["new_api_in_text"]) == len(r["task"]["new_api"])),
                      "none_in_text": sum(1 for r in na if not r["task"]["new_api_in_text"]),
                      "ever_any": sum(ever(r, E + B) for r in na), "run_tasks": len(na) * len(names),
                      "run_tasks_resolved": sum(r["runs"][x]["resolved"] for r in na for x in names),
                      "other_tasks": len(R) - len(na), "other_ever_any": sum(ever(r, E + B) for r in R if not r["task"]["new_api"])}
    # stability
    cnt = lambda r, rs: sum(r["runs"][x]["resolved"] for x in rs)
    st = {"e4b_solved_count": {str(k): sum(1 for r in R if cnt(r, E) == k) for k in range(len(E) + 1)},
          "b31_solved_count": {str(k): sum(1 for r in R if cnt(r, B) == k) for k in range(len(B) + 1)},
          "ever_e4b": sum(ever(r, E) for r in R), "ever_b31": sum(ever(r, B) for r in R),
          "ever_both": sum(ever(r, E) and ever(r, B) for r in R), "ever_any": sum(ever(r, E + B) for r in R), "pairs": {}}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if a[:3] != b[:3]:
                continue
            pr = {}
            for k in ("resolved", "file_hit", "unit_hit", "empty"):
                pr[k] = {"gained": sum(1 for r in R if r["runs"][b][k] and not r["runs"][a][k]),
                         "lost": sum(1 for r in R if r["runs"][a][k] and not r["runs"][b][k]),
                         "diff": rb.paired_diff([r["runs"][b][k] for r in R], [r["runs"][a][k] for r in R])}
            pr["same_outcome"] = sum(1 for r in R if r["runs"][a]["outcome"] == r["runs"][b]["outcome"])
            st["pairs"]["%s-%s" % (b, a)] = pr
    out["stability"] = st
    # what E4B did on the tasks only 31B solved, and the reverse
    o31 = [r for r in R if ever(r, B) and not ever(r, E)]
    oe = [r for r in R if ever(r, E) and not ever(r, B)]
    out["cross_model"] = {"b31_only": len(o31), "e4b_only": len(oe),
                          "e4b_attempts_on_b31_only": {k: sum(r["runs"][x]["outcome"] == k for r in o31 for x in E) for k in OUTCOMES},
                          "b31_no_patch_on_e4b_only": sum(r["runs"][x]["outcome"] == "no_patch" for r in oe for x in B),
                          "b31_attempts_on_e4b_only": len(oe) * len(B)}
    with open(os.path.join(PAPER, "results", "anatomy.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    for x in names:
        print(x, out["runs"][x]["outcomes"], "full_cover", out["runs"][x]["full_cover"], out["runs"][x]["resolved_given_full_cover"]["k"])


def main():
    if sys.argv[1:] == ["--summary"]:
        return summarize()
    specs = [a.split("=", 1) for a in sys.argv[1:]]
    tasks = {t["instance_id"]: t for t in load_tasks()}
    labels = {l["instance_id"]: l for l in load_labels()}
    repos, cache, TF = {}, {}, {}
    for iid, t in tasks.items():
        TF[iid] = task_features(t, labels[iid], repos.setdefault(t["repo"], Repo(t["repo"])), cache)
    runs = {n: load_run(d, tasks, labels, repos, TF) for n, d in specs}
    with open(os.path.join(PAPER, "results", "anatomy_per_task.jsonl"), "w", encoding="utf-8", newline="\n") as f:
        for iid in sorted(tasks):
            f.write(json.dumps({"instance_id": iid, "repo": tasks[iid]["repo"], "task": TF[iid],
                                "runs": {n: runs[n][iid] for n, _ in specs if iid in runs[n]}}, sort_keys=True) + "\n")
    summarize()


if __name__ == "__main__":
    main()
