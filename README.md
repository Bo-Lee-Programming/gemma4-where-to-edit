# Where-to-Edit v1: edit-location labels and a failure anatomy for Gemma 4 Developer Agent tasks

This release labels the edited files and functions (with line ranges) for the 129
public tasks of the Kaggle competition "Google - The Gemma 4 Developer Agent" and
the 500 tasks of SWE-bench Verified, and uses the labels as an instrument: every
patch from seven 129-task agent runs (Gemma 4 E4B and 31B) is assigned an outcome
class (resolved, no patch, wrong place, partial, covered and failing). It also
scores any localizer in one command and ships the best CPU localizer as a one-call
ADK skill.

Per-task outcomes of all seven runs: `results/anatomy_per_task.jsonl`. Aggregates
cited in the paper: `results/anatomy.json`.

## Five-minute start (Python 3.8+, standard library only, no download)

```
python -I scripts/score.py examples/predictions_skill.jsonl     # 1. score the shipped skill ranking
python -I scripts/score.py examples/predictions_grep.jsonl      # 2. score literal-name grep
python -I skill/where-to-edit/scripts/locate.py --repo scripts --query "bootstrap interval for hit at k"
```

Expected output of command 1 (first table):

```
files      subset (n)                           hit@1               hit@5              hit@10                 mrr
           all (125)               0.472 [0.38,0.56]  0.728 [0.64,0.81]  0.848 [0.78,0.91]  0.583 [0.51,0.66]
           task names target (35)  0.829 [0.69,0.94]  0.943 [0.86,1.00]  1.000 [1.00,1.00]  0.885 [0.79,0.96]
           target not named (90)   0.333 [0.23,0.43]  0.644 [0.54,0.73]  0.789 [0.71,0.87]  0.466 [0.39,0.55]
```

Command 3 runs the skill on any folder of Python code (here, this release's own
scripts) and prints ranked files and `path:start-end qualname` lines.

**Score your own localizer.** Write one JSON line per task and run command 1 on it:

```
{"instance_id": "rich_3521", "files": ["rich/segment.py", ...], "functions": ["rich/segment.py::Segment._split_cells", ...]}
```

Lists are ranked, best first; omitted tasks count as misses. Function ids are
`path::qualname` (`Class.method` or a top-level name; a nested function belongs to
its enclosing function). Add `--labels swebench/where_to_edit_swebench_verified.jsonl`
for SWE-bench Verified, `--json` for machine-readable output.

**Put the skill in a submission.** Copy `skill/where-to-edit/` to
`skills/where-to-edit/` in your submission folder, then:

```
# agent.yaml
skills:
  - skills/where-to-edit
```

and append `skill/prompt_paragraph.md` (four sentences) to your system prompt. The
agent then makes one call, `run_skill_script(skill_name="where-to-edit",
file_path="scripts/locate.py", args={"repo": "/workspace", "query": "<task text>"})`.

## Files

| Path | Content |
|---|---|
| `data/where_to_edit.jsonl` | Competition labels, one record per task: `instance_id`, `repo`, `base_commit`, `gold_files`, `gold_symbols` (path, qualname, kind, start, end), `edit_spans`, `new_source_files`, `other_files`, `features` |
| `swebench/where_to_edit_swebench_verified.jsonl` | The same record for each SWE-bench Verified task |
| `data/hard_subset.txt`, `data/hard_subset_swebench_verified.txt` | 40 and 130 task ids where no CPU localizer has an edited function in its top 10 |
| `results/baselines.json`, `swebench/results/baselines.json` | Metrics with 95% bootstrap intervals, strata and paired differences |
| `results/per_task.jsonl`, `swebench/results/per_task.jsonl` | Per-task ranks for every method |
| `results/sensitivity.json` | Parameter grid, leave-one-repository-out selection, named-definition counts |
| `results/graph_embed.json`, `results/graph_embed_per_task.jsonl` | The organisers' call graph and embeddings as ranking features |
| `results/seed_reach.json`, `results/seed_reach_per_task.jsonl` | Reach of `get_code_neighbors` / `search_similar_code` from named seeds and from BM25 seeds |
| `results/anatomy.json`, `results/anatomy_per_task.jsonl` | Failure anatomy of seven agent runs: per task and run the outcome class, gold-unit coverage, stop reason, first-edit time, test exit code; aggregates with bootstrap intervals |
| `results/e2e.json`, `results/e2e_per_task.jsonl` | Gemma 4 E4B runs (baseline, repeat, control, skill): resolution, edited-unit hits, skill usage read from traces |
| `results/e2e_31b.json`, `results/e2e_31b_per_task.jsonl` | Gemma 4 31B runs (baseline, skill, gold locations): the same record |
| `results/oracle_build.json` | Self-test of the gold-location skill variant: every task maps to its own gold set |
| `results/parity.json`, `results/adk_wrapper.json` | Skill (disk) vs benchmark (git) ranking equality; skill output under ADK's `run_skill_script` wrapper |
| `skill/where-to-edit/` | ADK skill: `SKILL.md` + `scripts/locate.py` (standard library only) |
| `scripts/score.py`, `examples/predictions_{skill,grep}.jsonl` | Scorer for any localizer (standard library) and two example prediction files |
| `figures/` | Outcome figure of the paper (`fig_outcomes.png`), named vs not named (`fig1_named_vs_not.png`), the `rich_3521` example, with the plotted values |
| `skill/prompt_paragraph.md` | The paragraph added to the agent prompt in the skill runs |
| `scripts/` | Labeling, baselines, graph/embedding experiments, end-to-end analysis, `reproduce_numbers.py` |
| `writeup.md`, `REPRODUCE_REPORT.md` | The paper and the check report (every number, every source file hash) |

Task texts and patches are read from their public sources: the competition's
task file, the SWE-bench Verified dataset, and the GitHub repositories at each
task's base commit.

## Field notes

- **Units** are top-level functions, classes, and methods of classes. A function
  nested in a function belongs to its parent. A class is a gold unit only when an
  edit falls outside its methods.
- **Line numbers** are 1-based and refer to the file at `base_commit`.
- **Candidate files**: `.py` files outside test/doc/example/benchmark/build/hidden
  directories, excluding `test_*.py`, `*_test.py`, `conftest.py` (`locate.is_candidate`).
- `features.symbol_named`: the task text contains an edited unit's name as a whole word.
  `features.file_path_named`: it contains an edited file's path or file name.

## Reproduce

From inside this folder (Python 3.12+, numpy, matplotlib, git; CPU only):

```
git clone --bare https://github.com/fastapi/fastapi.git repos/fastapi.git
git clone --bare https://github.com/Textualize/rich.git repos/rich.git
git clone --bare https://github.com/psf/requests.git repos/requests.git
git clone --bare https://github.com/encode/httpx.git repos/httpx.git
# competition files: tasks.jsonl -> data/tasks.jsonl ; graphs/ and embeddings/ -> a folder named in $WTE_COMP_DATA
python -I scripts/reproduce_numbers.py            # check the saved result files against the writeup
python -I scripts/reproduce_numbers.py --rerun    # regenerate the result files first
```

SWE-bench Verified: export the dataset's test split to `swebench/tasks.jsonl`
(fields `instance_id`, `repo`, `base_commit`, `patch`, `problem_statement`) and
clone the 12 repositories as bare clones into `swebench/repos/<name>.git`;
`--rerun` then includes `run_swebench.py` (about 20 minutes on six cores).
The agent runs are GPU runs on Kaggle. From the run folders (per-task csv, patches,
traces), `scripts/analyze_failures.py` recomputes `results/anatomy*.json*` and
`scripts/analyze_e2e.py` recomputes `results/e2e*.json`; `scripts/build_oracle_skill.py`
builds the gold-location skill variant.

## License (proposed)

Labels and results: CC BY 4.0. Code and skill: Apache-2.0. The underlying
repositories and datasets keep their own licenses.
