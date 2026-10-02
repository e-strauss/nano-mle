"""Generated plans own their reads and X/y construction; graph checks enforce style.

Source lint is not a security sandbox or a proof of no leakage.
"""

import ast


GUIDE = """
Plan contract (checked by the harness):
- Standalone Python with imports, constants, graph-building helpers and one
  zero-argument build(). The harness builds the graph lazily and evaluates it.
- Record reads: skrub.as_data_op(path).skb.apply_func(pd.read_csv | pd.read_parquet, ...)
  with columns/filters/sep as needed. Paths come from the task sources.
- Compute with fine-grained DataOps: indexing, filters, joins, groupby/agg, assign,
  string/date operations, .skb.concat. Build-time helpers and loops are fine when they
  emit explicit graph nodes. apply_func is limited to known library primitives
  (pandas readers, to_datetime, to_numeric, concat; numpy where/select/isfinite/
  isinf/isnan; len). No deferred, UDFs, custom transformers, callable
  apply/map/transform, eager reads, materialised data or files. Dictionary and Series
  maps are allowed. Estimators are applied with .skb.apply(...).
- Exploration returns a dict of named DataOps; all are evaluated in one pass.
- Evaluation setup returns {'X', 'y', 'scoring', optional 'row_keys', optional
  'audit'}: X marked with mark_as_X(cv=..., split_kwargs=...) and the raw y marked
  with mark_as_y(). scoring is a sklearn scorer string or a plain function
  scorer(estimator, X, y) -> float defined in the setup (higher is better), called on
  each test fold with that fold's marked X. CV is explicit and deterministic: sklearn
  splitters, custom splitter classes or explicit folds. For KFold use split_kwargs={}
  (n_splits/shuffle/random_state go to the constructor); split_kwargs are arguments
  to split(), e.g. groups for GroupKFold.
- Pipelines start from locked_evaluation_source: paste it, call build_evaluation(),
  build features and the model downstream, and return {'pred', 'scoring':
  setup scoring, 'row_keys': the same DataOp if declared}. The harness compares the
  X/y/CV/scoring/row-key graphs and the folds with the lock before fitting; drift is
  rejected. Keep the row count and order of X.
- Named skrub.choose_from grids become one candidate per variant (within the
  evaluation budget). Children receive the parent's resolved configuration.
- Return code without markdown fences. No credentials, no manual fitting or scoring,
  no caching.

Skrub and library notes (exact signatures; do not guess other keywords):
- a.skb.concat([b, c], axis=0) takes only a list and axis; no ignore_index.
  Chain .reset_index(drop=True) afterwards if a fresh index is needed.
- X.skb.apply(estimator, y=y) for the final model; there is no method= argument.
  Scorers call predict_proba/decision_function on the learner themselves.
- skrub.TableVectorizer(cardinality_threshold=40, low_cardinality=..., high_cardinality=...,
  numeric=..., datetime=..., specific_transformers=..., drop_null_fraction=...) takes only
  these keyword arguments; there is no categorical= argument.
- Casts use dtypes: .astype("string"), .astype("float64") or .astype(str) are fine.
- .skb.apply keeps DataFrame output, so sklearn encoders must produce dense output:
  OneHotEncoder(sparse_output=False, handle_unknown="ignore").
- Any installed library can be imported; nothing is installed on demand. Installed
  ML libraries include scikit-learn, lightgbm, xgboost, catboost, torch (CUDA),
  skorch, sentence-transformers, polars, faiss and rank_bm25. Modules for processes,
  files or network access (os, subprocess, pathlib, requests, ...) are not allowed.
- Custom CV splitters are plain classes with no base class defining
  split(self, X, y=None, groups=None) and get_n_splits(self, X=None, y=None, groups=None).
- Boolean masks over nullable columns must not contain NA: use .fillna(False) or
  str methods with na=False before combining masks with & or |.
- Output names are dict keys that must be valid Python identifiers.
"""

# Any installed library may be imported, except modules that reach outside the plan
# (processes, files, network, interpreter internals). Plans that import a library
# which is not installed fail with MissingLibrary; the runner records it.
DENIED_MODULES = {"os", "sys", "subprocess", "shutil", "pathlib", "socket", "ctypes", "importlib",
                  "builtins", "multiprocessing", "threading", "pickle", "dill", "joblib", "marshal",
                  "requests", "urllib", "http", "ftplib", "smtplib", "tempfile", "glob", "io", "gc"}


class MissingLibrary(ValueError):
    def __init__(self, modules):
        self.modules = sorted(modules)
        super().__init__(f"Library not installed: {', '.join(self.modules)}. The harness does not install "
                         "packages; use an installed library instead (see the plan guide).")


def installed(module):
    import importlib.util

    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


FORBIDDEN = {"deferred", "eval", "exec", "preview", "get_data", "set_data", "get_vars",
             "make_grid_search", "make_randomized_search", "make_learner", "with_scoring",
             "cross_validate", "iter_cv_splits", "train_test_split", "subsample",
             "choose_float", "choose_int", "choose_bool", "optional", "fit", "fit_transform",
             "to_csv", "to_parquet", "to_pickle", "to_json", "to_sql", "save", "dump"}


def validate_source(source):
    tree = ast.parse(source)
    builders = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "build"]
    if len(builders) != 1:
        raise ValueError("Plan must define exactly one zero-argument build() function")
    build = builders[0]
    if (build.args.args or build.args.posonlyargs or build.args.kwonlyargs or build.args.vararg
            or build.args.kwarg or build.decorator_list):
        raise ValueError("Use undecorated build() without arguments")
    # Eager reads are calls that resolve to pandas readers: pd.read_*(...) through a
    # pandas alias, or names imported from pandas. A plan's own helper that happens
    # to be called read_parquet (and records the read) is not one.
    reader_names = {"read_csv", "read_parquet"}
    pandas_aliases = {"pandas"}
    imported_readers = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            pandas_aliases.update(a.asname or a.name for a in node.names if a.name == "pandas")
        if isinstance(node, ast.ImportFrom) and node.module == "pandas":
            imported_readers.update(a.asname or a.name for a in node.names if a.name in reader_names)
    local_functions = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    readers = imported_readers | (reader_names - local_functions)

    def is_eager_read(call):
        if isinstance(call.func, ast.Attribute):
            return (call.func.attr in reader_names and isinstance(call.func.value, ast.Name)
                    and call.func.value.id in pandas_aliases)
        return getattr(call.func, "id", "") in readers
    choice_names = set()
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        if isinstance(node, ast.ImportFrom):
            if node.level or not node.module:
                raise ValueError("Relative imports are not supported")
            if any(a.name == "*" for a in node.names):
                raise ValueError(f"Wildcard import from {node.module} is not supported")
            imported.add(node.module)
    denied = sorted(m for m in imported if m.split(".")[0] in DENIED_MODULES)
    if denied:
        raise ValueError(f"Import of {', '.join(denied)} is not allowed: plans may not access processes, "
                         "files, the network or interpreter internals")
    missing = {m.split(".")[0] for m in imported if not installed(m.split(".")[0])}
    if missing:
        raise MissingLibrary(missing)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if any(a.name == "FunctionTransformer" for a in node.names):
                raise ValueError("Custom function transformers are not supported")
        if isinstance(node, ast.ClassDef):
            methods = {m.name for m in node.body if isinstance(m, ast.FunctionDef)}
            if "split" not in methods or "get_n_splits" not in methods or node.bases or node.decorator_list:
                raise ValueError(f"Class {node.name}: custom classes are limited to plain CV splitters defining "
                                 "split and get_n_splits, without base classes (e.g. not BaseCrossValidator) "
                                 "or decorators")
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("_") or node.attr in FORBIDDEN:
                raise ValueError(f"Unsupported opaque or harness-owned operation: {node.attr}")
        if isinstance(node, ast.Name) and (node.id.startswith("__") or node.id in
                {"open", "eval", "exec", "compile", "getattr", "setattr", "globals", "locals", "vars", "print", "FunctionTransformer"}):
            raise ValueError(f"Unsupported name: {node.id}")
        if isinstance(node, ast.Call):
            name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
            if is_eager_read(node):
                raise ValueError("Record readers with as_data_op(path).skb.apply_func(reader), never eager reads")
            if name == "choose_from":
                names = [k.value for k in node.keywords if k.arg == "name"]
                if (len(names) != 1 or not isinstance(names[0], ast.Constant)
                        or not isinstance(names[0].value, str) or not names[0].value):
                    raise ValueError("Every choose_from needs a literal nonempty name")
                if names[0].value in choice_names:
                    raise ValueError("Choice names must be unique")
                choice_names.add(names[0].value)
        if isinstance(node, ast.Assign):
            # Dictionary construction is allowed in graph-building helpers.
            # DataOp item mutation fails at construction; use recorded assign().
            for target in node.targets:
                if isinstance(target, ast.Attribute) and not (isinstance(target.value, ast.Name) and target.value.id == "self"):
                    raise ValueError("Use recorded assign(), not dataframe mutation")
    return tree


def evaluation_source(source):
    tree = validate_source(source)
    # Preserve existing graph helpers, including a helper called build_evaluation.
    # Rename the entry point to a fresh name, then expose a stable wrapper.
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    names.update(n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.ClassDef)))
    entry = "locked_setup_entry"
    while entry in names:
        entry += "_"
    helper = "locked_setup_helper"
    while helper in names or helper == entry:
        helper += "_"

    class Rename(ast.NodeTransformer):
        def visit_Name(self, node):
            if node.id == "build":
                node.id = entry
            elif node.id == "build_evaluation":
                node.id = helper
            return node

        def visit_FunctionDef(self, node):
            if node.name == "build":
                node.name = entry
            elif node.name == "build_evaluation":
                node.name = helper
            return self.generic_visit(node)

    tree = Rename().visit(tree)
    tree.body.extend(ast.parse(f"def build_evaluation():\n    return {entry}()\n").body)
    return ast.unparse(tree) + "\n"


def resolve_source(source, configuration):
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
