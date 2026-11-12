"""Run the skill the way ADK's run_skill_script does and confirm it indexes the task workspace.

ADK writes a wrapper script into the sandbox workspace; the wrapper copies the skill files into a temporary
directory, changes into it and executes scripts/locate.py with runpy.  This check reproduces that sequence with the
wrapper lines of google-adk's skill_toolset._build_wrapper_code and compares the output with a direct CLI run.
Usage: python -I paper/scripts/adk_wrapper_check.py <workspace_dir>  ->  paper/results/adk_wrapper.json
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)
SKILL = os.path.join(PAPER, "skill", "where-to-edit")
QUERY = "Fix validation error when `File` is declared after `Form` parameter"

WRAPPER = '''import os
import tempfile
import sys
import runpy
_files = %r
def _materialize_and_run():
  _orig_cwd = os.getcwd()
  with tempfile.TemporaryDirectory() as td:
    for rel_path, content in _files.items():
      full_path = os.path.join(td, rel_path)
      os.makedirs(os.path.dirname(full_path), exist_ok=True)
      with open(full_path, 'w', encoding='utf-8') as f:
        f.write(content)
    os.chdir(td)
    try:
      sys.argv = ['scripts/locate.py', '--query', %r]
      try:
        runpy.run_path('scripts/locate.py', run_name='__main__')
      except SystemExit as e:
        if e.code is not None and e.code != 0:
          raise e
    finally:
      os.chdir(_orig_cwd)
_materialize_and_run()
'''


def main():
    ws = os.path.abspath(sys.argv[1])
    files = {}
    for rel in ("SKILL.md", "scripts/locate.py"):
        with open(os.path.join(SKILL, rel), encoding="utf-8") as f:
            files[rel] = f.read()
    wrapper = os.path.join(ws, ".adk_exec_check.py")
    with open(wrapper, "w", encoding="utf-8") as f:
        f.write(WRAPPER % (files, QUERY))
    try:
        via_adk = subprocess.run([sys.executable, "-I", wrapper], cwd=ws, capture_output=True, text=True, encoding="utf-8")
    finally:
        os.remove(wrapper)
    direct = subprocess.run([sys.executable, "-I", os.path.join(SKILL, "scripts", "locate.py"), "--repo", ws, "--query", QUERY],
                            capture_output=True, text=True, encoding="utf-8")
    out = {"same_output": via_adk.stdout == direct.stdout and via_adk.returncode == 0, "first_line": via_adk.stdout.split("\n")[0],
           "stderr": via_adk.stderr[-300:]}
    with open(os.path.join(PAPER, "results", "adk_wrapper.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
