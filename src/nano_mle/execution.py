"""Persist requests and logs, and stop the entire child process group on timeout."""

import json
import os
import signal
import subprocess
import sys
from pathlib import Path


def run_plan(directory: Path, source: str, request: dict, timeout: int):
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "plan.py").write_text(source)
    (directory / "request.json").write_text(json.dumps(request, indent=2))
    env = os.environ.copy()
    env.update({"MPLCONFIGDIR": str(directory / "mpl-cache"), "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    with (directory / "execution.log").open("w") as log:
        child = subprocess.Popen([sys.executable, "-m", "nano_mle.worker", str(directory)],
                                 cwd=directory, env=env, stdout=log, stderr=log, start_new_session=True)
        try:
            child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
            usage = directory / "usage.json"
            result = {"status": "failed", "error": f"Execution timed out after {timeout}s",
                      "evaluation_count": json.loads(usage.read_text())["evaluation_count"] if usage.exists() else 0}
            if usage.exists():
                result["variants"] = [{**v, "error": result["error"]}
                                      for v in json.loads(usage.read_text())["variants"]]
            (directory / "response.json").write_text(json.dumps(result))
            return result
        except BaseException:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGKILL)
            child.wait()
            raise
    response = directory / "response.json"
    if not response.exists():
        usage = directory / "usage.json"
        reserved = json.loads(usage.read_text()) if usage.exists() else {"evaluation_count": 0}
        result = {"status": "failed", "error": f"Worker exited {child.returncode} without a response",
                  "evaluation_count": reserved["evaluation_count"]}
        if "variants" in reserved:
            result["variants"] = [{**v, "error": result["error"]} for v in reserved["variants"]]
        response.write_text(json.dumps(result))
        return result
    return json.loads(response.read_text())
