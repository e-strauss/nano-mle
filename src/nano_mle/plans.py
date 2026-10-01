"""Generated code policy and harness-owned graph roots.

This is a graph-discipline lint, not a security sandbox or a proof of no leakage.
"""

import ast

import numpy as np
import pandas as pd
import skrub


GUIDE = """
Write Python source containing imports and exactly one function: build(ctx).
Exploration returns a dict of named DataOp outputs. Pipeline returns a prediction DataOp.
ctx.read('source_name') returns a recorded CSV read as a DataOp. For pipelines,
start with X, y = ctx.load_xy(); the harness marks RAW X/y and attaches frozen CV.
Use explicit DataOp indexing, dataframe methods (drop, assign, merge, groupby,
agg, isna, sum, describe, value_counts, etc.), operators and .skb.concat.
Keep each meaningful operation visible in the computation graph.
For estimators use .skb.apply(skrub.TableVectorizer()) and .skb.apply(model, y=y).
Only skrub.choose_from with named, explicit discrete choices is supported.
No UDFs, lambdas, custom transformers, deferred, apply_func, pandas imports,
plain dataframe reads, eval/preview/get_data, feature caches, or manual scoring.
No extra functions, classes, loops, filesystem access, or helper imports.
Do not change CV/scoring or row population/order. Do not read the training source
as a side table in pipelines: labels must come from marked y. Exploration may
inspect labels, but findings are evidence, never precomputed pipeline features.
Estimator hyperparameters are literal values or choose_from values. Seed models.
The parent is a resolved candidate: preserve its configuration unless the proposal
explicitly changes it. Return source only, without markdown fences.
"""

ALLOWED_IMPORTS = {"skrub", "sklearn.ensemble", "sklearn.linear_model",
                   "sklearn.preprocessing", "sklearn.impute", "sklearn.dummy",
                   "sklearn.tree", "sklearn.neighbors", "sklearn.svm"}
FORBIDDEN = {"apply_func", "deferred", "apply", "map", "transform", "pipe", "eval", "exec",
             "preview", "get_data", "set_data", "get_vars", "mark_as_X", "mark_as_y",
             "with_scoring", "make_grid_search", "make_randomized_search", "make_learner",
             "cross_validate", "iter_cv_splits", "train_test_split", "subsample",
             "choose_float", "choose_int", "choose_bool", "optional", "var", "X", "y",
             "as_data_op", "fit", "fit_transform", "predict", "to_csv", "to_parquet",
             "to_pickle", "to_json", "to_sql", "save", "dump"}


def validate_source(source: str):
    tree = ast.parse(source)
    choice_names = set()
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if len(functions) != 1 or functions[0].name != "build":
        raise ValueError("Plan must define exactly one build(ctx) function")
    build = functions[0]
    if ([a.arg for a in build.args.args] != ["ctx"] or build.args.defaults
            or build.args.vararg or build.args.kwarg or build.args.kwonlyargs
            or build.args.posonlyargs or build.decorator_list or build.returns):
        raise ValueError("Use undecorated build(ctx) without annotations or defaults")
    for node in tree.body:
        if not isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef)):
            if not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
                    and isinstance(node.value.value, str)):
                raise ValueError("Only imports and build(ctx) are allowed at module level")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(a.name not in ALLOWED_IMPORTS for a in node.names):
                raise ValueError("Unsupported import")
        if isinstance(node, ast.ImportFrom):
            if node.level or node.module not in ALLOWED_IMPORTS or any(a.name == "*" for a in node.names):
                raise ValueError("Unsupported import")
        if isinstance(node, (ast.ClassDef, ast.Lambda, ast.AsyncFunctionDef, ast.For, ast.While,
                             ast.With, ast.Try, ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp)):
            raise ValueError("Opaque functions, classes, comprehensions, and control loops are not supported")
        if isinstance(node, ast.FunctionDef) and node is not build:
            raise ValueError("Custom functions are not allowed")
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("_"):
                raise ValueError("Private attributes are not allowed")
            if node.attr in FORBIDDEN:
                # skb.apply is the explicit estimator node, distinct from pandas apply.
                if not (node.attr == "apply" and isinstance(node.value, ast.Attribute)
                        and node.value.attr == "skb"):
                    raise ValueError(f"Unsupported opaque or harness-owned operation: {node.attr}")
            if isinstance(node.value, ast.Name) and node.value.id == "ctx":
                if node.attr not in {"read", "load_xy"}:
                    raise ValueError("ctx exposes only read and load_xy")
        if isinstance(node, ast.Name) and (node.id.startswith("__") or node.id in
                {"open", "eval", "exec", "compile", "getattr", "setattr", "globals", "locals", "vars", "type", "print", "FunctionTransformer"}):
            raise ValueError(f"Unsupported name: {node.id}")
        if isinstance(node, ast.Call):
            name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
            if name == "choose_from":
                names = [k.value for k in node.keywords if k.arg == "name"]
                if (len(names) != 1 or not isinstance(names[0], ast.Constant)
                        or not isinstance(names[0].value, str) or not names[0].value):
                    raise ValueError("Every choose_from needs a literal nonempty name")
                if names[0].value in choice_names:
                    raise ValueError("Choice names must be unique")
                choice_names.add(names[0].value)
            if name == "FunctionTransformer":
                raise ValueError("Custom function transformers are not supported")
        if isinstance(node, ast.Assign):
            if any(isinstance(n, (ast.Attribute, ast.Subscript)) for t in node.targets for n in ast.walk(t)):
                raise ValueError("Use recorded assign(), not mutation")
    return tree


def resolve_source(source, configuration):
    """Lock a selected parent's named choices in the code supplied to the coder."""
    tree = validate_source(source)

    class Resolve(ast.NodeTransformer):
        def visit_Call(self, node):
            name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
            if name == "choose_from":
                choice = next(k.value.value for k in node.keywords if k.arg == "name")
                if choice in configuration:
                    values = node.args[0]
                    outcomes = values.values if isinstance(values, ast.Dict) else values.elts
                    return self.visit(outcomes[int(configuration[choice])])
            return self.generic_visit(node)

    return ast.unparse(ast.fix_missing_locations(Resolve().visit(tree))) + "\n"


class PlanContext:
    def __init__(self, task, manifest, contract=None):
        self.task = task
        self.manifest = manifest
        self.contract = contract
        self.marked = None

    def read(self, name):
        if self.contract and name == self.task.train_source:
            raise ValueError("Pipeline training data must come through load_xy()")
        return self._read(name)

    def _read(self, name):
        return skrub.as_data_op(self.manifest[name]["path"]).skb.apply_func(pd.read_csv)

    def load_xy(self):
        if self.contract is None:
            raise ValueError("load_xy is only available to scored pipelines")
        if self.marked is None:
            table = self._read(self.task.train_source)
            splits = [(np.asarray(s["train"]), np.asarray(s["test"])) for s in self.contract["splits"]]
            y = table[self.task.target].skb.mark_as_y()
            drops = list(dict.fromkeys([self.task.target, *self.task.drop_columns]))
            X = table.drop(columns=drops).skb.mark_as_X(cv=splits)
            self.marked = X, y
        return self.marked
