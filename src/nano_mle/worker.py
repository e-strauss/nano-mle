"""Execute one generated plan in a time-bounded child process."""

import json
import math
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import skrub
from sklearn.base import BaseEstimator
from sklearn.model_selection import ParameterGrid

from .contracts import ContractDrift, verify_contract
from .evaluation import audit_boundary
from .graphs import validate_graph
from .plans import validate_source


def graph_artifact(plan, path):
    # Isolate the sole private Skrub API here; public describe_steps is also saved.
    from skrub._data_ops._evaluation import graph

    validate_graph(plan)
    structure = graph(plan)
    node_ids = {id(value): key for key, value in structure["nodes"].items()}

    def encode(value):
        if isinstance(value, skrub.DataOp):
            return {"node": node_ids.get(id(value)), "type": type(value._skrub_impl).__name__}
        if value is None or isinstance(value, (str, bool, int)):
            return value
        if isinstance(value, (float, np.floating)):
            return float(value) if math.isfinite(value) else repr(value)
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, dict):
            return {str(k): encode(v) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return [encode(v) for v in value]
        if isinstance(value, BaseEstimator):
            return {"estimator": f"{type(value).__module__}.{type(value).__qualname__}",
                    "parameters": encode(value.get_params(deep=False))}
        if callable(value):
            return {"callable": f"{getattr(value, '__module__', '')}.{getattr(value, '__qualname__', repr(value))}"}
        return {"type": type(value).__name__, "repr": repr(value)}

    serial = {"skrub_version": skrub.__version__, "edge_direction": "operation -> dependencies",
              "nodes": [{"id": key, "type": type(value._skrub_impl).__name__,
                         "operation": value.skb.describe_steps().splitlines()[-1],
                         "attributes": {field: encode(getattr(value._skrub_impl, field))
                                        for field in value._skrub_impl._fields}}
                        for key, value in structure["nodes"].items()],
              "dependencies": structure["children"]}
    path.with_suffix(".graph.json").write_text(json.dumps(serial, indent=2))
    path.with_suffix(".steps.txt").write_text(plan.skb.describe_steps())


def output_artifact(name, value, directory):
    """Full tabular output on disk, bounded preview for model context."""
    if isinstance(value, (pd.DataFrame, pd.Series)):
        value.to_csv(directory / f"{name}.csv")
        return {"artifact": f"{name}.csv", "shape": list(value.shape),
                "preview": value.head(20).to_string()[:6000]}
    if isinstance(value, np.ndarray):
        np.save(directory / f"{name}.npy", value)
        return {"artifact": f"{name}.npy", "shape": list(value.shape),
                "preview": repr(value)[:6000]}
    return {"preview": str(value)[:6000]}


def execute(request, directory):
    contract = request.get("contract")
    if contract:
        verify_contract(contract)
    source = (directory / "plan.py").read_text()
    validate_source(source)
    namespace = {"__name__": "generated_plan"}
    started = time.monotonic()
    with skrub.config_context(eager_data_ops=False):
        exec(compile(source, str(directory / "plan.py"), "exec"), namespace)
        result = namespace["build"]()
    if request["kind"] == "exploration":
        if not isinstance(result, dict) or not result:
            raise ValueError("Exploration must return a nonempty dict of named DataOp outputs")
        outputs = {}
        for name, plan in result.items():
            if not isinstance(name, str) or not name.isidentifier() or name.startswith("_"):
                raise ValueError("Output names must be plain identifiers")
            if not isinstance(plan, skrub.DataOp):
                raise ValueError(f"Output {name!r} is not a DataOp")
            graph_artifact(plan, directory / name)
            outputs[name] = output_artifact(name, plan.skb.eval(), directory)
        return {"status": "ok", "outputs": outputs, "duration_s": time.monotonic() - started}
    if request.get("expected_scoring") is not None and result.get("scoring") != request["expected_scoring"]:
        raise ValueError("Setup scoring must match the proposed scorer")
    snapshot = audit_boundary(result, contract)
    (directory / "evaluation.graph.json").write_text(json.dumps(snapshot["boundary_graph"], indent=2))
    if request["kind"] == "evaluation":
        graph_artifact(result["X"], directory / "X")
        graph_artifact(result["y"], directory / "y")
        outputs = {}
        for name, plan in result.get("audit", {}).items():
            if not isinstance(name, str) or not name.isidentifier() or name.startswith("_") or not isinstance(plan, skrub.DataOp):
                raise ValueError("Audit outputs must be named DataOps")
            graph_artifact(plan, directory / f"audit_{name}")
            outputs[name] = output_artifact(f"audit_{name}", plan.skb.eval(), directory)
        return {"status": "ok", "snapshot": snapshot, "outputs": outputs,
                "duration_s": time.monotonic() - started}
    if contract is None:
        raise ValueError("Scoring requires an audited, locked evaluation setup")
    pred = result["pred"]
    graph_artifact(pred, directory / "pipeline")
    learner = pred.skb.make_learner()
    grid = ParameterGrid(learner.get_param_grid())
    if len(grid) > request["remaining_evaluations"]:
        raise ValueError(f"Grid has {len(grid)} variants, exceeding remaining evaluation budget")
    # Require named choices so configurations can be resolved across rebuilt graphs.
    learner.get_named_params()
    pending = []
    for index, params in enumerate(grid):
        resolved = pred.skb.make_learner().set_params(**params)
        pending.append({"index": index, "status": "failed", "score": None,
                        "configuration": resolved.get_named_params(),
                        "configuration_description": resolved.describe_params()})
    # Reserve before scoring: a crash/timeout is charged conservatively for the grid.
    (directory / "usage.json").write_text(json.dumps({"evaluation_count": len(grid), "variants": pending}))
    frozen = [(np.asarray(s["train"]), np.asarray(s["test"])) for s in contract["splits"]]
    search = pred.skb.make_grid_search(fitted=True, refit=False, n_jobs=1, cv=frozen,
                                     scoring=contract["scoring"], error_score=np.nan)
    raw = search.cv_results_
    variants = []
    for index, params in enumerate(raw["params"]):
        resolved = pred.skb.make_learner().set_params(**params)
        folds = [float(raw[f"split{k}_test_score"][index]) for k in range(len(contract["splits"]))]
        valid = all(math.isfinite(score) for score in folds)
        variants.append({"index": index, "status": "ok" if valid else "failed",
                         "configuration": resolved.get_named_params(),
                         "configuration_description": resolved.describe_params(),
                         "score": float(raw["mean_test_score"][index]) if valid else None,
                         "std": float(raw["std_test_score"][index]) if valid else None,
                         "fold_scores": folds if valid else None,
                         "mean_fit_time": float(raw["mean_fit_time"][index]),
                         "mean_score_time": float(raw["mean_score_time"][index])})
    verify_contract(contract)
    return {"status": "ok", "variants": variants, "duration_s": time.monotonic() - started,
            "contract_id": contract["id"], "fold_fingerprint": contract["fold_fingerprint"]}


def main():
    directory = Path(sys.argv[1]).resolve()
    try:
        response = execute(json.loads((directory / "request.json").read_text()), directory)
    except Exception as error:
        response = {"status": "failed", "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc()[-12000:]}
        if isinstance(error, ContractDrift):
            response.update(warning="contract_drift", changed_components=error.changes)
    usage = directory / "usage.json"
    reserved = json.loads(usage.read_text()) if usage.exists() else {"evaluation_count": 0}
    response["evaluation_count"] = reserved["evaluation_count"]
    if response["status"] == "failed" and "variants" in reserved:
        response["variants"] = [{**v, "error": response["error"]} for v in reserved["variants"]]
    (directory / "response.json").write_text(json.dumps(response, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
