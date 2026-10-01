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
from sklearn.model_selection import ParameterGrid

from .contracts import verify_contract, verify_sources
from .models import Task
from .plans import PlanContext, validate_source


def graph_artifact(plan, path):
    # Isolate the sole private Skrub API here; public describe_steps is also saved.
    from skrub._data_ops._evaluation import graph

    structure = graph(plan)
    for node in structure["nodes"].values():
        impl = node._skrub_impl
        if type(impl).__name__ == "Call" and impl.func is not pd.read_csv:
            raise ValueError("Opaque function node found: only harness CSV reads are allowed")
    serial = {"skrub_version": skrub.__version__, "edge_direction": "operation -> dependencies",
              "nodes": [{"id": key, "type": type(value._skrub_impl).__name__,
                         "operation": value.skb.describe_steps().splitlines()[-1]}
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
    task = Task.model_validate(request["task"])
    manifest = request["sources"]
    contract = request.get("contract")
    verify_sources(manifest)
    if contract:
        verify_contract(contract)
    source = (directory / "plan.py").read_text()
    validate_source(source)
    context = PlanContext(task, manifest, contract)
    namespace = {"__name__": "generated_plan"}
    started = time.monotonic()
    with skrub.config_context(eager_data_ops=False):
        exec(compile(source, str(directory / "plan.py"), "exec"), namespace)
        result = namespace["build"](context)
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
        verify_sources(manifest)
        return {"status": "ok", "outputs": outputs, "duration_s": time.monotonic() - started}
    if not isinstance(result, skrub.DataOp) or context.marked is None:
        raise ValueError("Pipeline must return a DataOp using ctx.load_xy()")
    found = result.skb.find_X_y()
    if any(found.get(key) is None or found[key].skb.id != expected.skb.id
           for key, expected in zip(("X", "y"), context.marked)):
        raise ValueError("Prediction must use the harness-owned marked X and y")
    graph_artifact(result, directory / "pipeline")
    learner = result.skb.make_learner()
    grid = ParameterGrid(learner.get_param_grid())
    if len(grid) > request["remaining_evaluations"]:
        raise ValueError(f"Grid has {len(grid)} variants, exceeding remaining evaluation budget")
    # Require named choices so configurations can be resolved across rebuilt graphs.
    learner.get_named_params()
    (directory / "usage.json").write_text(json.dumps({"evaluation_count": len(grid)}))
    search = result.skb.make_grid_search(fitted=True, refit=False, n_jobs=1,
                                       scoring=contract["spec"]["scoring"], error_score=np.nan)
    raw = search.cv_results_
    variants = []
    for index, params in enumerate(raw["params"]):
        resolved = result.skb.make_learner().set_params(**params)
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
    usage = directory / "usage.json"
    response["evaluation_count"] = json.loads(usage.read_text())["evaluation_count"] if usage.exists() else 0
    (directory / "response.json").write_text(json.dumps(response, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
