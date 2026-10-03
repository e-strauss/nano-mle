import pandas as pd
import pytest

from nano_mle.cli import make_demo
from nano_mle.demo import DemoBackend
from nano_mle.models import Task
from nano_mle.store import Store
from nano_mle.submit import submit

FINAL = '''import pandas as pd
import skrub
from sklearn.linear_model import Ridge

FIT = {{"rows": {train!r}}}
PREDICT = {{"rows": {test!r}}}
FORMAT = {sample!r}


def build():
    rows = skrub.var("rows").skb.apply_func(pd.read_csv)
    y = rows["target"].skb.mark_as_y()
    X = rows.drop(columns=["id", "target"], errors="ignore").skb.mark_as_X()
    pred = X.skb.apply(Ridge(), y=y)
    return {{"submission": rows[["id"]].assign(target=pred)}}
'''


class FinalBackend(DemoBackend):
    def __init__(self, source):
        self.source = source
        self.contexts = []

    def implement(self, kind, context, intent):
        assert kind == "final"
        self.contexts.append(context)
        return self.source


@pytest.fixture
def finished(tmp_path):
    workspace = make_demo(tmp_path / "demo")
    train = pd.read_csv(tmp_path / "demo" / "train.csv")
    train.insert(0, "id", range(len(train)))
    train.to_csv(tmp_path / "demo" / "train.csv", index=False)
    train.drop(columns="target").head(20).assign(id=range(1000, 1020)).to_csv(tmp_path / "demo" / "test.csv", index=False)
    pd.DataFrame({"id": range(1000, 1020), "target": 0.0}).to_csv(tmp_path / "demo" / "sample.csv", index=False)
    paths = {name: str(tmp_path / "demo" / f"{name}.csv") for name in ("train", "test", "sample")}
    task = Task(description="Synthetic regression; submit target for test rows", sources=paths)
    return workspace, task, paths


def test_submit_refits_best_candidate_and_checks_format(finished):
    workspace, task, paths = finished
    backend = FinalBackend(FINAL.format(**paths))
    record = submit(workspace, backend, task=task, notify=lambda *_: None)
    assert record["status"] == "ok", record
    info = record["result"]["submission"]
    table = pd.read_csv(info["path"])
    assert list(table.columns) == ["id", "target"] and len(table) == 20
    assert table["target"].notna().all() and not info["warnings"]
    assert "columns match the format" in info["checks"]
    context = backend.contexts[0]
    assert "def build" in context["resolved_source"] and "sample" in context["source_heads"]
    assert Store(workspace).records("submission")[0]["candidate_id"] == context["candidate"]["id"]


def test_submit_rejects_prediction_rows_outside_sources(finished, tmp_path):
    workspace, task, paths = finished
    stray = tmp_path / "elsewhere.csv"
    pd.read_csv(paths["test"]).to_csv(stray, index=False)
    backend = FinalBackend(FINAL.format(train=paths["train"], test=str(stray), sample=paths["sample"]))
    record = submit(workspace, backend, task=task, max_repairs=0, notify=lambda *_: None)
    assert record["status"] == "failed" and "not a task source" in record["result"]["error"]
