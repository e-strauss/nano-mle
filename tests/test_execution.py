import json

from nano_mle.execution import run_plan


def test_worker_timeout_is_recorded(tmp_path):
    directory = tmp_path / "timeout"
    result = run_plan(directory, "def build():\n    return {}\n", {}, timeout=0.01)
    assert result["status"] == "failed"
    assert "timed out" in result["error"]
    assert json.loads((directory / "response.json").read_text()) == result
