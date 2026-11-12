## Localization skill
Before your first search, call `run_skill_script(skill_name="where-to-edit", file_path="scripts/locate.py", args={"repo": "/workspace", "query": "<the full issue text>"})`. It returns the most likely files and functions to edit as `path:start-end qualname`. Read the top 1-3 functions first. Names the issue mentions are often the API being called, not the place to edit.
