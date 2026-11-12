---
name: where-to-edit
description: Rank the repository's Python files and functions by how likely they need an edit for the current issue. CPU-only, a few seconds, one tool call. Use it first, before grepping, when the issue does not name the function to change.
---

# where-to-edit

Run once at the start of a task:

    run_skill_script(skill_name="where-to-edit", file_path="scripts/locate.py",
                     args={"repo": "/workspace", "query": "<the full issue text>"})

It prints the top files and the top functions/classes as `path:start-end qualname`.

How to use the output:
1. Open the top 1-3 functions with `sed -n 'START,ENDp' PATH` and decide which one to change.
2. The "defined names the issue mentions" line lists definitions the issue spells out.
   They are often the public API the user calls, not the place to edit; check callees.
3. If none of the top functions fits, fall back to `grep -rn` on words from the issue.

On the 129 public tasks this ranking puts an edited function in the top 10 for
55% of tasks and an edited file first for 47%, versus 47% and 36% for grepping
the identifiers written in the issue.
