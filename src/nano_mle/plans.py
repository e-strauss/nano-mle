"""Generated plans own their reads and X/y construction; graph checks enforce style.

Source lint is not a security sandbox or a proof of no leakage.
"""

import ast


GUIDE = """
Write standalone Python containing imports, constants, graph-building helpers and
one zero-argument build() function.
Record reads directly: skrub.as_data_op(path).skb.apply_func(pd.read_csv, ...),
or pd.read_parquet with columns/filters/storage_options. Paths come from task context.
Input files are assumed frozen: do not hash, snapshot, or inspect them for changes.
Exploration returns a dict of named DataOp outputs. Evaluation setup returns a dict
with marked X, marked raw y, a sklearn scoring string, optional row_keys DataOp and
optional audit dict of DataOps. Pipeline returns {'pred': prediction_DataOp,
'scoring': scoring_string, 'row_keys': same_optional_DataOp}.
Build the modelling population and labels with explicit recorded operations. Mark
X/y as soon as population and raw labels exist, before feature engineering. Attach
an explicit deterministic cv and split_kwargs to mark_as_X. Built-in sklearn CV,
custom splitter classes, and explicit fold sequences are supported.
For KFold use split_kwargs={}; shuffle/random_state/n_splits belong to the splitter
constructor. split_kwargs are arguments to splitter.split, e.g. groups for GroupKFold.
Use fine-grained indexing, string/date operations, filters, joins, groupby/agg,
assign and .skb.concat. Build-time helpers and loops are fine when they generate
explicit graph nodes; never wrap a custom data-processing function as one node.
apply_func is limited to known library primitives: pandas readers, to_datetime,
to_numeric, concat; numpy where/select/isfinite/isinf/isnan; len.
No deferred, UDFs, custom transformers, callable dataframe apply/map/transform,
eager reads, materialized feature blocks, feature files, manual fit/scoring or eval.
Dictionary and Series maps ARE allowed. Standard estimators use .skb.apply(...).
Use named explicit skrub.choose_from grids only, downstream of the marked inputs.
Once evaluation is locked, reuse locked_evaluation_source from context. It defines
build_evaluation(); paste it into the standalone pipeline and call it, then build
features/model downstream. Equivalent inline construction is also allowed, but the
harness compares X/y/CV/scoring/row_keys graphs and effective folds before fitting.
Drift must be repaired or evaluated in a fresh workspace, never silently relocked.
Return code without markdown fences. Never embed credentials or add caching.

Skrub API notes (exact signatures; do not guess other keywords):
- a.skb.concat([b, c], axis=0) takes only a list and axis; no ignore_index.
  Chain .reset_index(drop=True) afterwards if a fresh index is needed.
- X.skb.apply(estimator, y=y) for the final classifier; there is no method= argument.
  Scorers such as average_precision/roc_auc call predict_proba/decision_function
  on the learner themselves. Optional *_kwargs args only pass extra arguments.
- Chain preprocessing as successive .skb.apply(...) steps; sklearn.pipeline is not
  importable. Allowed imports: skrub, pandas, numpy and sklearn submodules
  model_selection, ensemble, linear_model, preprocessing, impute, dummy, tree,
  neighbors, svm.
- Custom CV splitters are plain classes with no base class (not BaseCrossValidator)
  defining split(self, X, y=None, groups=None) and get_n_splits(self, X=None,
  y=None, groups=None).
- Boolean masks over nullable columns must not contain NA: use .fillna(False) or
  str methods with na=False before combining masks with & or |.
- Exploration and audit output names are dict keys that must be valid Python
  identifiers (letters, digits, underscores; e.g. coverage_2020, not "coverage-2020").
  All outputs are evaluated together in one graph evaluation, so shared reads run
  once; prefer a few focused summary tables over many large outputs.
"""

ALLOWED_IMPORTS = {"skrub", "pandas", "numpy", "sklearn.model_selection", "sklearn.ensemble",
                   "sklearn.linear_model", "sklearn.preprocessing", "sklearn.impute",
                   "sklearn.dummy", "sklearn.tree", "sklearn.neighbors", "sklearn.svm"}
FORBIDDEN = {"deferred", "eval", "exec", "preview", "get_data", "set_data", "get_vars",
             "make_grid_search", "make_randomized_search", "make_learner", "with_scoring",
             "cross_validate", "iter_cv_splits", "train_test_split", "subsample",
             "choose_float", "choose_int", "choose_bool", "optional", "fit", "fit_transform",
             "predict", "to_csv", "to_parquet", "to_pickle", "to_json", "to_sql", "save", "dump"}


def validate_source(source):
    tree = ast.parse(source)
    builders = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "build"]
    if len(builders) != 1:
        raise ValueError("Plan must define exactly one zero-argument build() function")
    build = builders[0]
    if (build.args.args or build.args.posonlyargs or build.args.kwonlyargs or build.args.vararg
            or build.args.kwarg or build.decorator_list):
        raise ValueError("Use undecorated build() without arguments")
    readers = {"read_csv", "read_parquet"}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "pandas":
            readers.update(a.asname or a.name for a in node.names if a.name in {"read_csv", "read_parquet"})
    choice_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            rejected = [a.name for a in node.names if a.name not in ALLOWED_IMPORTS]
            if rejected:
                raise ValueError(f"Unsupported import {', '.join(rejected)}; allowed: {', '.join(sorted(ALLOWED_IMPORTS))}")
        if isinstance(node, ast.ImportFrom):
            if node.level or node.module not in ALLOWED_IMPORTS:
                raise ValueError(f"Unsupported import from {'.' * node.level}{node.module or ''}; "
                                 f"allowed: {', '.join(sorted(ALLOWED_IMPORTS))}")
            if any(a.name == "*" for a in node.names):
                raise ValueError(f"Wildcard import from {node.module} is not supported")
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
            if name in readers:
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
