import pandas as pd
import pytest

from nano_mle.contracts import create_contract, source_manifest, verify_contract
from nano_mle.models import EvaluationSpec, Task


def task_at(tmp_path, frame):
    path = tmp_path / "train.csv"
    frame.to_csv(path, index=False)
    return Task(description="test", sources={"train": str(path)}, target="target")


def test_source_and_contract_changes_are_rejected(tmp_path):
    task = task_at(tmp_path, pd.DataFrame({"a": range(12), "target": range(12)}))
    spec = EvaluationSpec(scoring="neg_mean_absolute_error", cv="kfold", rationale="test")
    contract = create_contract(task, source_manifest(task), spec)
    verify_contract(contract)
    changed = {**contract, "rows": 100}
    with pytest.raises(ValueError, match="modified"):
        verify_contract(changed)
    (tmp_path / "train.csv").write_text("a,target\n1,2\n")
    with pytest.raises(ValueError, match="changed"):
        verify_contract(contract)


def test_group_and_time_membership(tmp_path):
    frame = pd.DataFrame({"group": [0, 1, 2] * 4, "target": range(12),
                          "time": pd.date_range("2025-01-01", periods=12)[::-1]})
    task = task_at(tmp_path, frame)
    manifest = source_manifest(task)
    grouped = create_contract(task, manifest, EvaluationSpec(scoring="r2", cv="group",
                              group_column="group", rationale="Repeated entities"))
    for split in grouped["splits"]:
        assert set(frame.iloc[split["train"]]["group"]).isdisjoint(frame.iloc[split["test"]]["group"])
    timed = create_contract(task, manifest, EvaluationSpec(scoring="r2", cv="time",
                            time_column="time", rationale="Future observations"))
    for split in timed["splits"]:
        assert frame.iloc[split["train"]]["time"].max() < frame.iloc[split["test"]]["time"].min()
