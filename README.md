# nano-mle

A small sequential DSPy harness for adaptive data exploration and scored Skrub
DataOps pipeline search. Both exploration and pipelines are computation graphs;
generated plans cannot hide feature construction inside UDFs or custom transformers.

The controller can initiate exploration, and the planner can request it while
expanding a selected candidate. The harness owns evaluation, execution, budgets,
history and search statistics. Exploration produces scoped findings, not rewards.

## Offline prototype

```bash
uv sync
uv run nano-mle demo /tmp/nano-mle-demo
uv run nano-mle show /tmp/nano-mle-demo/workspace
uv run pytest
```

The deterministic demo makes **no API calls**. It explores a synthetic dataset,
locks CV/scoring, evaluates a baseline, runs a planner-requested investigation,
and evaluates a two-variant grid. Each grid variant becomes a sibling candidate.
Use a fresh demo directory for each run.

Live runs are explicit: `nano-mle run WORKSPACE --model MODEL`. Model credentials
are loaded from `.env` only by that backend. Never commit credentials.

## Task and workspace

Task JSON describes local CSV sources, target and columns excluded before marking:

```json
{
  "description": "Predict target; evaluate RMSE. Explain the row population and prediction setting here.",
  "sources": {"train": "train.csv", "metadata": "metadata.csv"},
  "train_source": "train",
  "target": "target",
  "drop_columns": ["id"]
}
```

Paths are relative to the task JSON. Initialize without making any model calls:

```bash
uv run nano-mle init workspaces/example --task task.json --policy greedy \
  --max-expansions 3 --max-explorations 5 --max-evaluations 10 --max-repairs 1 \
  --max-model-calls 20
```

Review [the prompts](src/nano_mle/prompts.py), task, and budgets before starting a
live run. For example, the DSPy/LiteLLM model identifier for Gemini is supplied
as `--model gemini/gemini-3.8-flash`. The prototype does not automatically test
credentials or start a live model.

Each backend call has a 6,000 output-token limit and a 60-second request timeout.
Provider retries and DSPy's adapter fallback calls are disabled. The workspace's
model-call limit counts calls before dispatch, including failures. This bounds
call volume, not total dollar cost; input size and provider pricing still matter.

The workspace contains `state.db` (authoritative journal), `workspace.json`
(exported evaluation contract), `graph.json`, `report.md`, and `artifacts/`.
Artifacts preserve model inputs/outputs, generated source, repair attempts,
execution logs, DataOps graph JSON (operations, arguments and dependencies), step
descriptions and exploration results.
Completed workspaces are immutable runs; use a new workspace for new budgets or
evaluation rules. Interrupted runs can resume; interrupted work remains recorded
and is not silently replayed.

## Plan interface

Plans contain imports and one `build(ctx)` function. The harness disables eager
previews during construction and evaluates the returned graph.

Exploration:

```python
def build(ctx):
    data = ctx.read("train")
    missing = data.isna()
    counts = missing.sum()
    summary = data.describe()
    return {"missing_counts": counts, "summary": summary}
```

Pipeline:

```python
import skrub
from sklearn.linear_model import Ridge

def build(ctx):
    X, y = ctx.load_xy()
    encoded = X.skb.apply(skrub.TableVectorizer())
    model = Ridge(alpha=skrub.choose_from([0.1, 10.0], name="alpha"))
    pred = encoded.skb.apply(model, y=y)
    return pred
```

`ctx.read` creates a recorded CSV-read node; `ctx.load_xy` creates recorded reads,
marks raw X/y early and attaches the harness's frozen folds. Pipeline features
must be built downstream of these marks. Pipelines cannot reread the training
source as a constant side table. The harness evaluates all explicit variants
with `make_grid_search`, saves their complete fold scores/configurations, and
resolves a selected parent's choices before supplying its code to the coder.

Inputs are SHA256-pinned. Scoring and exact positional fold membership are locked
after successful initial exploration. Supported CV: shuffled KFold, stratified,
group and time. Time CV currently requires unique timestamps. Changes to input
contents or the evaluation contract are rejected.

## Search experiments

`greedy` selects the best valid candidate. `mcts` uses UCT and progressive widening
on primary edges. `mcgs` adds an elite-selection schedule and explicit candidate
references. The latter is MLEvolve-inspired, not a reproduction of all its
operators and stagnation logic.

Selection, reference construction, execution and reward updates have separate
interfaces. Every evaluated grid variant contributes one observation. Reward is
-1 for failure, 1 for a valid non-improvement, and 2 for improvement against the
global best **before the batch**. References and findings receive no propagated
reward. Sibling variants share their triggering expansion's selected parent.

Debugging is bounded inside construction of an experiment. Attempts preserve their
source and errors; changing the experimental hypothesis requires a new proposal.

## Prototype boundaries

- Local CSV sources and standard sklearn/Skrub estimators; GCS discovery, remote
  source pinning, arbitrary raw-file population construction, custom/weighted
  scorers, plots and final submission/refitting are not implemented yet.
- No intermediate reuse, execution rewrites or caching layer. Graphs remain the
  deliverable for a future execution engine.
- Source lint and runtime graph checks enforce the supported plan style. They
  are not a security sandbox or proof of no leakage; use trusted model-generated
  code and inspect findings and plans. Side tables still require provenance audits.
- Workers are subprocesses with timeouts; the runner and grid execution are sequential.
- One private Skrub graph-inspection API is isolated in `worker.py`, alongside
  public textual graph exports. Dependencies are pinned in `uv.lock`.
