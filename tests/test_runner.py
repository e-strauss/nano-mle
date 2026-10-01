import fcntl
import json
import pandas as pd
import pytest

from nano_mle.cli import make_demo
from nano_mle.demo import DemoBackend
from nano_mle.models import Budget, EvaluationSpec, Task
from nano_mle.plans import PlanContext
from nano_mle.runner import Runner, initialize
from nano_mle.store import Store


@pytest.fixture
def workspace(tmp_path):
    train = tmp_path / "train.csv"
    pd.DataFrame({"a": range(18), "b": [1, 2, 3] * 6,
                  "target": [2 * i + 0.1 for i in range(18)]}).to_csv(train, index=False)
    task = Task(description="Independent regression data", sources={"train": str(train)}, target="target")
    root = tmp_path / "workspace"
    initialize(root, task, Budget(max_expansions=2, max_evaluations=4, max_repairs=0))
    return root


def test_offline_workflow_and_graph_artifacts(tmp_path):
    root = make_demo(tmp_path / "demo")
    store = Store(root)
    candidates = store.records("candidate")
    explorations = store.records("exploration")
    contract = store.meta("contract")
    assert len(candidates) == 3
    assert all(c["status"] == "ok" for c in candidates)
    assert len(explorations) == 2
    assert explorations[0]["expansion_id"] is None
    assert explorations[1]["expansion_id"] is not None
    assert all(c["contract_id"] == contract["id"] for c in candidates)
    assert all(len(c["fold_scores"]) == 3 for c in candidates)
    siblings = candidates[1:]
    assert siblings[0]["parent_id"] == siblings[1]["parent_id"] == candidates[0]["id"]
    assert siblings[0]["batch_id"] == siblings[1]["batch_id"]
    assert siblings[0]["configuration"] != siblings[1]["configuration"]
    assert len(store.records("attempt")) == 4
    assert store.meta("search_stats")["root"]["visits"] == 3
    assert all(e["status"] != "running" for e in explorations)
    for exploration in explorations:
        artifact = root / exploration["artifact_path"]
        graph = json.loads((artifact / "missing_counts.graph.json").read_text())
        operations = [n["operation"] for n in graph["nodes"]]
        assert "Call 'read_csv'" in operations
        assert "CallMethod 'isna'" in operations
        assert "CallMethod 'sum'" in operations
        assert (artifact / "missing_counts.csv").exists()
    runner = Runner(root, DemoBackend())
    selection = type("Selection", (), {"parent_id": siblings[0]["id"], "reference_ids": []})()
    context = runner.context(selection)
    assert "choose_from" not in context["parent"]["resolved_source"]
    runner.store.close()
    # A completed run never calls the backend again.
    before_calls = len(store.records("model_call"))
    Runner(root, DemoBackend()).run()
    assert len(store.records("model_call")) == before_calls
    store.close()


def test_contract_requires_exploration_and_refuses_relocking(workspace):
    runner = Runner(workspace, DemoBackend())
    spec = EvaluationSpec(scoring="r2", cv="kfold", rationale="test")
    with pytest.raises(ValueError, match="Successful exploration"):
        runner.establish(spec)
    runner.store.put("exploration", {"id": "initial", "status": "ok"})
    runner.establish(spec)
    with pytest.raises(ValueError, match="already locked"):
        runner.establish(spec)
    runner.store.close()


def test_training_alias_cannot_bypass_marking(workspace):
    runner = Runner(workspace, DemoBackend())
    sources = runner.store.meta("sources")
    sources["alias"] = sources["train"]
    context = PlanContext(runner.task, sources, {"locked": True})
    with pytest.raises(ValueError, match="aliases"):
        context.read("alias")
    runner.store.close()


def test_call_budget_stops_without_another_backend_call(workspace):
    runner = Runner(workspace, DemoBackend())
    runner.store.set_meta("budget", Budget(max_model_calls=1).model_dump())
    runner.store.close()
    Runner(workspace, DemoBackend()).run()
    store = Store(workspace)
    assert len(store.records("model_call")) == 1
    assert store.meta("state") == "complete"
    assert all(e["status"] == "failed" for e in store.records("exploration"))
    store.close()


def test_competing_runner_does_not_change_owner_state(workspace):
    store = Store(workspace)
    store.set_meta("state", "running")
    with (workspace / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with pytest.raises(ValueError, match="Another runner"):
            Runner(workspace, DemoBackend()).run()
    assert store.meta("state") == "running"
    store.close()


def test_interrupted_execution_is_charged_without_replay(workspace):
    runner = Runner(workspace, DemoBackend())
    directory = workspace / "artifacts" / "interrupted"
    directory.mkdir()
    (directory / "usage.json").write_text(json.dumps({"evaluation_count": 3}))
    runner.store.put("attempt", {"id": "unfinished", "status": "running", "evaluation_count": 0,
                                 "path": "artifacts/interrupted"})
    runner.store.put("expansion", {"id": "expansion", "status": "running"})
    runner.recover()
    assert runner.counts()["evaluations"] == 3
    assert runner.store.get("unfinished")["status"] == "interrupted"
    assert runner.store.get("expansion")["status"] == "interrupted"
    runner.store.close()


class OversizedGrid(DemoBackend):
    def implement(self, kind, context, intent):
        if kind == "exploration":
            return super().implement(kind, context, intent)
        return """import skrub
from sklearn.linear_model import Ridge
def build(ctx):
    X, y = ctx.load_xy()
    model = Ridge(alpha=skrub.choose_from([1.0, 2.0, 3.0], name='alpha'))
    return X.skb.apply(model, y=y)
"""


def test_oversized_grid_is_rejected_before_scoring(workspace):
    runner = Runner(workspace, OversizedGrid())
    runner.store.set_meta("budget", Budget(max_expansions=1, max_evaluations=2, max_repairs=0,
                                          max_actions=4).model_dump())
    runner.store.close()
    Runner(workspace, OversizedGrid()).run()
    store = Store(workspace)
    attempts = store.records("attempt")
    assert sum(a["evaluation_count"] for a in attempts) == 0
    candidates = store.records("candidate")
    assert len(candidates) == 1 and candidates[0]["status"] == "failed"
    error = candidates[0]["error"]
    assert "exceeding remaining evaluation budget" in error
    store.close()


class FailingGrid(OversizedGrid):
    def implement(self, kind, context, intent):
        source = super().implement(kind, context, intent)
        return source.replace("[1.0, 2.0, 3.0]", "[-1.0, -2.0]")


def test_failed_grid_preserves_and_charges_every_variant(workspace):
    runner = Runner(workspace, FailingGrid())
    runner.store.set_meta("budget", Budget(max_expansions=1, max_evaluations=2,
                                          max_repairs=0, max_actions=4).model_dump())
    runner.store.close()
    Runner(workspace, FailingGrid()).run()
    store = Store(workspace)
    candidates = store.records("candidate")
    assert len(candidates) == 2
    assert all(c["status"] == "failed" for c in candidates)
    assert candidates[0]["parent_id"] == candidates[1]["parent_id"] == "root"
    assert sum(a["evaluation_count"] for a in store.records("attempt")) == 2
    assert store.meta("search_stats")["root"] == {"visits": 2, "reward_sum": -2}
    store.close()
