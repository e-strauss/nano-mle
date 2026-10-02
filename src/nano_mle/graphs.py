"""Logical graph fingerprints, independent of Skrub UUIDs and Python variable names.

Private Skrub fields are isolated here. Comparison is structural, not a proof of
semantic equivalence, and performs no execution optimization.
"""

import ast
import inspect
import math
import textwrap
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import skrub
from sklearn.base import BaseEstimator

from .contracts import canonical_hash


PRIMITIVES = {pd.read_csv, pd.read_parquet, pd.to_datetime, pd.to_numeric, pd.concat,
              np.where, np.select, np.isfinite, np.isinf, np.isnan, len}


def is_dtype(value):
    """Types passed as dtypes (astype(str), np.float64) are values, not callbacks."""
    return value in (str, int, float, bool) or (isinstance(value, type) and issubclass(value, np.generic))


def validate_graph(plan):
    from skrub._data_ops._evaluation import graph

    def check_callbacks(value, method):
        if isinstance(value, skrub.DataOp) or is_dtype(value):
            return
        if callable(value) and value not in PRIMITIVES:
            name = getattr(value, "__name__", None) or type(value).__name__
            raise ValueError(f"Opaque callable argument {name!r} to .{method}(): use explicit recorded operations")
        if isinstance(value, Mapping):
            for item in value.values():
                check_callbacks(item, method)
        elif isinstance(value, (tuple, list)):
            for item in value:
                check_callbacks(item, method)

    for node in graph(plan)["nodes"].values():
        impl = node._skrub_impl
        if type(impl).__name__ == "Call" and impl.func not in PRIMITIVES:
            raise ValueError("Opaque function node: use fine-grained DataOps and supported library primitives")
        if type(impl).__name__ == "Value" and isinstance(impl.value, (pd.DataFrame, pd.Series, np.ndarray)):
            raise ValueError("Materialized data leaf: record reads and transformations inside the graph")
        if type(impl).__name__ == "CallMethod":
            check_callbacks(impl.args, impl.method_name)
            check_callbacks(impl.kwargs, impl.method_name)
            if impl.method_name in {"apply", "transform", "pipe"}:
                raise ValueError("Opaque dataframe callback: use explicit recorded operations")
            if impl.method_name == "map" and impl.args and callable(impl.args[0]):
                raise ValueError("Callable map is opaque; dictionary/Series maps are supported")


def canonical_graph(roots):
    nodes = {}
    seen = {}

    def encode(value):
        if isinstance(value, skrub.DataOp):
            identity = id(value)
            if identity not in seen:
                impl = value._skrub_impl
                node = {"op": type(impl).__name__,
                        "fields": {field: encode(getattr(impl, field)) for field in impl._fields},
                        "is_X": bool(impl.is_X), "is_y": bool(impl.is_y)}
                key = canonical_hash(node)
                nodes[key] = node
                seen[identity] = key
            return {"node": seen[identity]}
        if value is None or isinstance(value, (str, bool, int)):
            return value
        if isinstance(value, (float, np.floating)):
            return float(value) if math.isfinite(value) else {"float": repr(float(value))}
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.bool_):
            return bool(value)
        if isinstance(value, np.ndarray):
            return {"array": encode(value.tolist()), "dtype": str(value.dtype)}
        if isinstance(value, (pd.Timestamp, pd.Timedelta)):
            return {type(value).__name__: value.isoformat()}
        if isinstance(value, pd.DateOffset):
            return {"DateOffset": {"n": value.n, "normalize": value.normalize, "kwds": encode(value.kwds)}}
        if isinstance(value, Path):
            return {"path": str(value)}
        if isinstance(value, slice):
            return {"slice": [encode(value.start), encode(value.stop), encode(value.step)]}
        if isinstance(value, Mapping):
            return {"mapping": [[encode(k), encode(v)] for k, v in value.items()]}
        if isinstance(value, (tuple, list)):
            return {type(value).__name__: [encode(v) for v in value]}
        if isinstance(value, BaseEstimator):
            return {"estimator": f"{type(value).__module__}.{type(value).__qualname__}",
                    "parameters": encode(value.get_params(deep=False))}
        if callable(value):
            if getattr(value, "__module__", "") == "generated_plan":
                raise ValueError("Custom functions cannot be embedded as opaque graph operations")
            return {"callable": f"{getattr(value, '__module__', '')}.{getattr(value, '__qualname__', getattr(value, '__name__', ''))}"}
        if callable(getattr(value, "split", None)):
            cls = type(value)
            description = {"splitter": f"{cls.__module__}.{cls.__qualname__}", "state": encode(vars(value))}
            if cls.__module__ == "generated_plan":
                description["methods"] = {name: ast.dump(ast.parse(textwrap.dedent(inspect.getsource(method))))
                                           for name, method in vars(cls).items() if inspect.isfunction(method)}
            return description
        if value is pd.NA:
            return {"missing": "pd.NA"}
        raise ValueError(f"Cannot fingerprint {type(value).__name__}; use explicit deterministic values")

    encoded = {name: encode(root) for name, root in roots.items()}
    return {"roots": encoded, "nodes": nodes}


def fingerprint(value):
    return canonical_hash(canonical_graph({"root": value}))


def is_custom_scorer(value):
    return inspect.isfunction(value) and value.__module__ == "generated_plan"


def scorer_fingerprint(scorer):
    """Locks a setup-defined scorer by its source and the plan helpers/constants it uses.

    Pipelines paste the locked setup source, so an unchanged scorer reproduces the
    same fingerprint; editing it or any helper it calls is reported as drift.
    """
    parts, seen, pending = {}, set(), [scorer]
    while pending:
        fn = pending.pop()
        if fn.__name__ in seen:
            continue
        seen.add(fn.__name__)
        parts[fn.__name__] = ast.dump(ast.parse(textwrap.dedent(inspect.getsource(fn))))
        for name in fn.__code__.co_names:
            value = fn.__globals__.get(name)
            if is_custom_scorer(value):
                pending.append(value)
            elif isinstance(value, (int, float, str, bool, tuple)) and name.isupper():
                parts[f"const:{name}"] = repr(value)
    return canonical_hash({"scorer": scorer.__name__, "parts": parts})
