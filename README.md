# nano-mle

A small, sequential harness for LLM-driven ML experimentation on tabular data.
A language model proposes data explorations, an evaluation setup and pipeline
experiments; the harness executes them, enforces budgets, locks the evaluation,
records every attempt and runs an interchangeable search policy (greedy, MCTS or
MCGS) over the resulting candidates.

All model-written code is expressed as [Skrub DataOps](https://skrub-data.org)
computation graphs: data reads, joins, aggregations, features and estimators are
recorded graph nodes rather than opaque Python functions. This makes every
exploration and pipeline inspectable and comparable, and keeps the evaluation
boundary checkable.

## Quick start

```bash
uv sync
uv run pytest                                  # offline test suite
uv run nano-mle demo /tmp/nano-mle-demo        # deterministic run, no API calls
uv run nano-mle show /tmp/nano-mle-demo/workspace
```

The demo uses a scripted backend on a synthetic regression table. It walks the
full loop: exploration, evaluation setup, a baseline, a planner-requested
exploration and a two-variant grid.

## Running a task

1. **Describe the task** in a JSON file. Paths are relative to the file, or remote
   URIs such as `gs://bucket/table/`:

   ```json
   {
     "description": "Predict … Explain the entities, prediction setting, metric and data here.",
     "sources": {"train": "train.csv", "metadata": "metadata.csv"},
     "train_source": "train",
     "target": "target",
     "drop_columns": ["id"]
   }
   ```

   Only `description` and `sources` are required. `train_source`, `target` and
   `drop_columns` are hints; the model constructs the actual modelling population
   and labels.

2. **Configure credentials** in the repository `.env` (gitignored), for example
   `OPENAI_API_KEY=…`, `GEMINI_API_KEY=…`, and
   `GOOGLE_APPLICATION_CREDENTIALS=/path/key.json` for `gs://` sources. The model
   backend loads it when a run starts; workers inherit the environment.

3. **Initialise a workspace** (no model calls) and **run** it:

   ```bash
   uv run nano-mle init workspaces/my-run --task task.json --policy greedy \
     --max-model-calls 30 --max-expansions 3 --max-explorations 4 \
     --max-evaluations 8 --max-repairs 2 --execution-timeout 1800
   uv run nano-mle run workspaces/my-run --model openai/gpt-6.1-sol
   uv run nano-mle show workspaces/my-run
   uv run nano-mle draw workspaces/my-run/artifacts/<step>/<attempt> > graph.svg
   ```

   `--model` is any DSPy/LiteLLM model id. `run` also takes `--max-tokens`
   (completion cap per call including reasoning, default 16,000) and
   `--request-timeout` (seconds, default 180). Provider retries are disabled.

A run resumes where it stopped if `run` is invoked again on an interrupted
workspace; interrupted work stays recorded and is not replayed. A completed
workspace is immutable; use a new one for new budgets or evaluation rules.

## Configuration

Harness settings live in `nano-mle.toml` at the repository root. It is gitignored;
`nano-mle.example.toml` is the tracked template and holds the defaults, which apply
when no local file exists. `NANO_MLE_CONFIG` points to another file. Each run records
the configuration it used in its workspace metadata.

| Setting | Default | Effect |
|---|---|---|
| `[execution] cpu_threads` | 32 | Threads one attempt may use (OpenMP/BLAS for numpy, LightGBM, XGBoost, torch); 0 means all cores. |
| `[execution] grid_n_jobs` | 1 | Fits run in parallel processes by the grid search; each gets `cpu_threads // grid_n_jobs` threads and its own copy of X and y, so values above 1 only pay off for small data. Probes fit sequentially with all threads. |
| `[plans] restrict_primitives` | false | Limit `apply_func` to the curated primitives in `graphs.PRIMITIVES`. Disabled for now, so plans may call any library function; plan-defined functions and lambdas are rejected either way. |
| `[prompts] data_volume_study` | true | Adds the data-volume-study convention for the planner and controller: for large sources, explore which rows and table parts are needed before the lock, and measure a learning curve over training-set size after it. |

## Submission

`nano-mle submit WORKSPACE --model MODEL [--candidate ID] [--task TASK.json]` runs after
the search, like mle-claude's ml-submit skill. The harness takes the best candidate
(or the named one) and resolves its winning grid variant. One writer call (with
repairs) re-roots the plan so it can be applied to new rows: rows that differ between
fitting and predicting are read through `skrub.var` holding a path, bound by the
plan's `FIT` and `PREDICT` dicts. The harness fits once on `FIT`, predicts on
`PREDICT`, checks columns, keys and row count against `FORMAT` (the sample
submission) and writes `submission/<id>/attempt_*/submission.csv`. The writer sees
the first lines of each text source. `--task` passes an updated task file that adds
the prediction rows and the sample submission as sources. Submissions are never
scored and never become candidates.

## How a run works

A single controller loop chooses one action at a time:

| Action | What happens | Scored? |
|---|---|---|
| `explore` | The writer produces a graph answering a concrete question; outputs are evaluated and an interpreter turns them into scoped findings. | No |
| `establish_evaluation` | The writer constructs the modelling population, raw labels, CV and scorer. The harness audits and locks them. | No |
| `expand` | The search policy selects a parent; the planner proposes a bounded change (or requests an exploration first); the writer implements it; every grid variant is scored. | Yes |
| `probe` | Fits one configuration (an existing candidate, or a pipeline the writer builds) on the locked folds and reports its fold scores and a harness-computed error summary of its out-of-fold predictions. | No |
| `stop` | Ends the run. | |

Who decides what, compared with a single-agent harness such as mle-claude:

| Decision | mle-claude | nano-mle |
|---|---|---|
| Next action (explore, lock evaluation, expand, probe, stop) | LLM | LLM controller |
| Which candidate to build on | LLM | search policy (greedy / MCTS / MCGS) |
| What to change in the experiment | LLM | LLM planner, given the selected node |
| Writing and fixing code | LLM | LLM writer / repairer |
| Turning outputs into findings | LLM | LLM interpreter |
| Scoring and the evaluation lock | harness | harness |

Who knows what. mle-claude is one agent with one continuous context; nano-mle's roles
are stateless calls that see what the harness passes them:

| Knowledge | mle-claude | nano-mle |
|---|---|---|
| Task description and sources | the agent | every role (task and source manifest) |
| Raw data | read directly, any slice | only through a plan's graph, evaluated by the harness |
| Exploration results | everything it printed | output previews of the last four explorations, plus the interpreter's scoped findings |
| Earlier code | any file in the workspace | writer sees the parent's and references' resolved source |
| Model errors | loads out-of-fold predictions and slices freely | probe fold scores, preview and a harness-computed error summary |
| Intermediate artifacts | reads its own files (convention: no cached features) | never: plans read task sources only, artifact paths are hidden |
| History and reasoning | its own context window | journal summaries: findings, leaderboard, recent failures, trajectory |
| Run time | sees wall time | attempt wall time, phase timings and per-variant fit times in the records it is shown |

Every implementation runs in a time-bounded subprocess. If it fails, the repairer
gets the source and traceback and may fix it, up to `--max-repairs` times. A
repair must keep the planned experiment; a different hypothesis needs a new proposal.

Budgets cover controller actions, model calls (counted before dispatch, including
failures), explorations, setup attempts, expansions, scored variants, probes, repairs
and execution time. They bound call volume, not dollar cost.

## Plans

Every plan is standalone Python with imports, optional graph-building helpers and
one zero-argument `build()`. Graphs are built lazily; the harness evaluates them.

**Exploration** returns named DataOps. They are evaluated together in a single
graph evaluation, so shared reads and transformations run once:

```python
import pandas as pd
import skrub

def build():
    data = skrub.as_data_op("train.csv").skb.apply_func(pd.read_csv)
    return {"missing_counts": data.isna().sum(), "summary": data.describe()}
```

**Evaluation setup** marks X and the raw y as soon as population and labels exist,
attaches an explicit deterministic CV, and names a scikit-learn scorer. It may also
return `row_keys` (unique, aligned) and `audit` (named DataOps saved as evidence).
Joins, filters, derived labels, custom splitter classes and split kwargs are allowed:

```python
from sklearn.model_selection import KFold

def build():
    data = skrub.as_data_op("train.csv").skb.apply_func(pd.read_csv)
    X = data.drop(columns=["target"]).skb.mark_as_X(
        cv=KFold(3, shuffle=True, random_state=42), split_kwargs={})
    y = data["target"].skb.mark_as_y()
    return {"X": X, "y": y, "scoring": "neg_root_mean_squared_error"}
```

**Pipelines** receive the locked setup as source code defining `build_evaluation()`,
call it, and build features and models downstream. A named `skrub.choose_from` grid
becomes one candidate per variant:

```python
from sklearn.linear_model import Ridge

def build():
    setup = build_evaluation()
    encoded = setup["X"].skb.apply(skrub.TableVectorizer())
    model = Ridge(alpha=skrub.choose_from([0.1, 10.0], name="alpha"))
    pred = encoded.skb.apply(model, y=setup["y"])
    return {"pred": pred, "scoring": setup["scoring"]}
```

**Scoring.** The setup's `scoring` is either a scikit-learn scorer string or a plain
function `scorer(estimator, X, y) -> float` defined in the setup (higher is better).
It runs on each test fold with that fold's marked X, so ids and group keys kept in X
can drive grouped metrics such as per-entity recall@k. A custom scorer is locked like
a custom splitter: by its source plus the plan helpers and upper-case constants it uses.

**Probes.** A probe is written like a pipeline with a single configuration. The
harness wraps the locked scorer to capture each test fold's predictions during the
normal fold loop, so nothing is fitted twice, and writes `oof_predictions.parquet`
with `row` (position in the locked X), `fold`, `y`, `prediction` (positive-class
probability, or the predicted label plus one `proba_<class>` column per class) and
`row_key`. The file is for people and the dashboard. The model sees the fold scores,
a preview and an error summary computed by the harness: per-class recall, precision
and confusion for classification, calibration by prediction decile for binary
probabilities, error quantiles for regression.

**Plan rules.** These are enforced by a source lint (`plans.py`) and a runtime graph
check (`graphs.py`):
- Readers are recorded with `skrub.as_data_op(path).skb.apply_func(pd.read_csv | pd.read_parquet, ...)`.
- Plans read only the task sources. Every path or URL in a plan's graph must be a
  source or lie inside a source directory; files written by earlier steps (outputs,
  predictions, manifests) are rejected, so anything worth reusing is recomputed. The
  model context lists task sources but no artifact paths.
- Computation is expressed as fine-grained DataOps. No UDFs, `deferred`, callable
  `apply`/`map`, eager reads, materialised data, files, or manual fitting in the graph.
- Custom classes are allowed when they are estimators or transformers (`fit`), torch
  modules (`forward`) or CV splitters, applied with `.skb.apply`. Inside them, fitting,
  private state and `super().__init__()` are allowed. The guide asks for them only
  when a step needs fitted state or is a new model. A plan-defined class inside the
  locked X/y graph is fingerprinted by its source.
- Any installed library may be imported, including lightgbm, xgboost, catboost, torch
  (CUDA), skorch, sentence-transformers, polars, faiss and rank_bm25. Modules that
  reach processes, files, the network or interpreter internals are denied.
- The harness never installs packages. An import of a missing library fails the
  attempt and is recorded in `missing_libraries.json` at the repository root (also
  shown in the dashboard), so you can decide what to add.

These checks enforce the plan style. They are not a security sandbox or a proof of
no leakage.

## Evaluation lock

When a setup passes its audit, the harness locks a contract with these parts:
- the structural fingerprints of the X, y, CV/split-kwargs, scorer and row-key graphs;
- the row count;
- the exact positional fold memberships.

**Fingerprints.** They ignore Skrub UUIDs and variable names, so an equivalent graph
constructed independently is accepted. The comparison is structural, not a proof of
semantic equivalence.

**Checks on every pipeline.** Before fitting, the pipeline's boundary is
re-fingerprinted and re-evaluated, and the folds are re-derived and compared with
the lock.

**Drift.** If any part drifts, the attempt is sent back for repair. If the drift is
not repaired, no candidate or search reward is created. A run has exactly one
evaluation setup; a different one needs a new workspace.

**Inputs are assumed frozen.** Source contents and remote object versions are not
hashed. Hashes protect the generated artifacts and the contract itself.

## Search

Candidates are search nodes; explorations and findings are evidence and carry no reward.

**Policies.**
- `greedy` expands the best valid candidate.
- `mcts` uses UCT with progressive widening.
- `mcgs` adds an elite-selection schedule and passes cross-branch candidates to the
  planner as references.

`mcgs` is inspired by [MLEvolve](https://arxiv.org/html/2606.06473); it does not
reproduce its full operator set.

**Rewards.** Every scored variant is one observation:
- −1 for a failure;
- +1 for a valid non-improvement;
- +2 for an improvement over the best score before the batch.

The reward is backpropagated along the parent chain. Variants of one grid are
siblings under the same parent.

## Workspace

```
state.db         authoritative journal: metadata, records and events (SQLite)
workspace.json   task, sources, budget, policy, model, state and evaluation contract
graph.json       candidates with parent, reference and evidence edges, search statistics
report.md        leaderboard, evaluation audit, drift warnings and findings
artifacts/
  calls/                         every model call's inputs and outputs
  <exploration|setup|expansion>_<id>/<attempt>/
    plan.py, request.json, response.json, execution.log
    *.graph.json, *.steps.txt    DataOps graphs (nodes, arguments, dependencies)
    *.csv / *.npy                full exploration and audit outputs
    timings.json                 per-phase time breakdown, updated while running
```

**`timings.json` phases.** It records imports, request loading, contract
verification, lazy graph building, boundary fingerprinting, X/y evaluation and
checks, graph export, learner/grid construction and grid search. The grid search is
split into fold fitting, fold scoring and search overhead. A worker that times out
still shows which phase was running.

## Code map

| File | Role |
|---|---|
| `cli.py` | Entry point: `init`, `run`, `submit`, `show`, `draw`, `demo`. |
| `runner.py` | Controller loop, budgets, repair loop, exploration/setup/expansion, report export, resume. |
| `agents.py` | DSPy backend: controller, planner, writer, repairer and interpreter signatures. |
| `prompts.py` | Instructions for the controller, planner and writer. |
| `plans.py` | Plan guide shown to the writer, source lint, locked-setup export, grid-variant resolution. |
| `models.py` | Pydantic types: task, decisions, proposals, findings, budget. |
| `execution.py` | Runs one plan in a killable subprocess with a timeout. |
| `worker.py` | Subprocess body: builds the graph, evaluates outputs, audits the boundary, runs the grid search, records timings. |
| `evaluation.py` | Evaluation-boundary audit: X/y/CV/row-key checks, folds, drift detection. |
| `graphs.py` | Runtime graph rules and structural fingerprints (the only private-Skrub-API use besides the graph export). |
| `contracts.py` | Contract creation, hashing and drift comparison. |
| `search.py` | Greedy, MCTS and MCGS policies and reward updates. |
| `store.py` | SQLite journal. |
| `config.py` | Loads `nano-mle.toml` (threads, grid parallelism, plan checks). |
| `timing.py` | Phase timer for workers. |
| `libraries.py` | Repository-level record of imported-but-missing libraries. |
| `submit.py` | `nano-mle submit`: final refit of a candidate, prediction on new rows, format checks. |
| `draw.py` | `nano-mle draw ATTEMPT`: rebuilds a recorded plan lazily and prints its Skrub `draw_graph` SVG. |
| `demo.py` | Scripted offline backend. |

`dashboard/` is an optional, harness-agnostic web UI for browsing runs, search trees
and launching new runs (see its README). `FUTURE_IDEAS.md` collects design notes that
are not implemented, such as multiple evaluation branches.

## Limitations

- **Execution:** sequential throughout, with one evaluation setup per workspace.
  There is no intermediate caching: shared reads are recomputed across attempts,
  and inside the grid search for every fold.
- **Data access:** only CSV and Parquet readers are supported; local and `gs://`
  paths work with the installed `gcsfs`/`pyarrow`. Shell access, plots, and final
  refitting or submission files are out of scope.
- **Trust:** generated code is trusted. Workers are timeout-bounded subprocesses,
  not a sandbox.
