"""Tiny nonnumerical probe of the same venv interpreter and identity verifier."""
import json
import os
from pathlib import Path
import subprocess
import sys
from launch_identity import probe_process


def main():
    code = """import json,os,sys
from launch_identity import probe_process,verify_claim
request=json.loads(sys.stdin.readline())
worker=dict(pid=os.getpid(),created=probe_process(os.getpid())[1])
mode=verify_claim(request['launcher'],worker,request['owner'],os.getppid())
print(json.dumps(dict(worker=worker,parent=os.getppid(),mode=mode,numerical_imports=any(m in sys.modules for m in ('numpy','torch')))),flush=True)
sys.stdin.readline()
"""
    child = subprocess.Popen([sys.executable, "-B", "-c", code], cwd=Path(__file__).parent,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    try:
        request = dict(owner=dict(pid=os.getpid(), created=probe_process(os.getpid())[1]),
                       launcher=dict(pid=child.pid, created=probe_process(child.pid)[1]))
        out, err = child.communicate(json.dumps(request)+"\nexit\n", timeout=15)
        if child.returncode:
            raise RuntimeError(err)
        result = dict(**request, result=json.loads(out), exit=child.returncode,
                      executable=sys.executable, base_executable=sys._base_executable,
                      parent_numerical_imports=any(m in sys.modules for m in ("numpy", "torch")))
        if result["parent_numerical_imports"] or result["result"]["numerical_imports"]:
            raise RuntimeError("Unexpected numerical imports")
        print(json.dumps(result))
    finally:
        if child.poll() is None:
            child.kill()
            child.communicate(timeout=15)


if __name__ == "__main__":
    main()
