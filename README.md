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

Task JSON supplies source-location hints and a description of the prediction setting.
`train_source`, `target`, and `drop_columns` are optional planning hints; the writer
constructs the actual modelling population and labels. Sources may be local paths
or remote URIs such as `gs://bucket/table.parquet`:

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

Each backend call has a 6,000 completion-token limit and a 60-second request timeout.
Provider retries and DSPy's adapter fallback calls are disabled. The workspace's
model-call limit counts calls before dispatch, including failures. This bounds
call volume, not total dollar cost; input size and provider pricing still matter.
For OpenAI GPT-6 models, the backend uses low reasoning effort, omits a custom
temperature, and includes reasoning tokens in the completion cap. See the
[GPT-6.1 Sol documentation](https://developers.openai.com/api/docs/models/gpt-6.1-sol).

A live `openai/gpt-6.1-sol` smoke test on 120 synthetic regression rows completed
with 10 model calls, one exploration, two expansions, four scored candidates and
no repairs. Three-fold CV RMSE improved from 0.21441 to 0.20773. This verifies the
live execution path; it is not a benchmark of model quality. Local artifacts are
under `workspaces/gpt61-sol-smoke-20261002/` and are excluded from Git.

The workspace contains `state.db` (authoritative journal), `workspace.json`
(exported evaluation contract), `graph.json`, `report.md`, and `artifacts/`.
Artifacts preserve model inputs/outputs, generated source, repair attempts,
execution logs, DataOps graph JSON (operations, arguments and dependencies), step
descriptions and exploration results.
Completed workspaces are immutable runs; use a new workspace for new budgets or
evaluation rules. Interrupted runs can resume; interrupted work remains recorded
and is not silently replayed.

## Plan interface

Plans contain imports, optional graph-building helpers and one zero-argument
`build()` function. No `common.py`, fixed reader adapter or `load_xy()` is required.
The harness disables eager previews and evaluates returned DataOps.

Exploration uses recorded readers and fine-grained operations:

```python
import pandas as pd
import skrub

def build():
    data = skrub.as_data_op("train.csv").skb.apply_func(pd.read_csv)
    return {"missing_counts": data.isna().sum(), "summary": data.describe()}
```

After initial exploration, a **special unscored setup phase** constructs the
population, raw labels and evaluation. Joins, filters, derived labels, reader
options, custom CV splitters and graph-defined split kwargs are supported:

```python
from sklearn.model_selection import KFold

def build():
    data = skrub.as_data_op("train.csv").skb.apply_func(pd.read_csv)
    X = data.drop(columns=["target"]).skb.mark_as_X(
        cv=KFold(3, shuffle=True, random_state=42))
    y = data["target"].skb.mark_as_y()
    return {"X": X, "y": y, "scoring": "neg_root_mean_squared_error"}
```

Optional `row_keys` must be unique, nonmissing and aligned with X/y. Optional
`audit` is a dict of named DataOps whose outputs are saved. The harness checks
population/label alignment, missing labels and valid nonoverlapping fold positions.
It locks the logical X/y graphs, CV/split kwargs, scorer, optional row-key graph,
row count and exact positional fold memberships. CV planning hints do not define
splits; the recorded setup does.

The writer receives this standalone setup as `locked_evaluation_source`, with
`build()` renamed to `build_evaluation()`. A subsequent pipeline includes that code:

```python
from sklearn.linear_model import Ridge

def build():
    setup = build_evaluation()
    encoded = setup["X"].skb.apply(skrub.TableVectorizer())
    model = Ridge(alpha=skrub.choose_from([0.1, 10.0], name="alpha"))
    pred = encoded.skb.apply(model, y=setup["y"])
    return {"pred": pred, "scoring": setup["scoring"]}
```

If setup declares row keys, return the same row-key DataOp with the prediction.
Equivalent independently constructed graphs are accepted. Graph fingerprints
ignore Skrub UUIDs and local variable names; comparison is structural, not a proof
of semantic equivalence. Downstream features and models may change freely.
Evaluation drift is warned about before fitting and passed into bounded repair.
An unresolved drift creates no candidate or search reward. Restore the lock or
start a fresh workspace; this prototype has one evaluation branch.

**Inputs are assumed static/frozen.** Input contents and remote object versions
are not checked or hashed. Hashes protect generated artifacts and contract metadata.
All scoring uses the locked folds. Early marking does not prove freedom from
leakage: side tables and downstream features still need investigation.

Graph-building helpers and loops are allowed; opaque runtime UDFs, custom
transformers and callable dataframe callbacks are rejected. Native library
primitives are documented in [the plan guide](src/nano_mle/plans.py). The harness
saves graphs and scores for every explicit grid variant and supplies resolved
parent code to the next writer.

Legacy `load_xy` workspaces remain readable, but cannot resume under this interface;
create a new workspace. The historical live smoke run above used the old interface.

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

- Recorded CSV/Parquet readers support local/remote paths when the relevant pandas
  engines/filesystem dependencies and credentials are installed. Remote GCS access
  has not been tested here. Arbitrary shell/GCS discovery commands, custom scorers,
  plots and final submission/refitting are outside this prototype.
- No intermediate reuse, execution rewrites or caching layer. Graphs remain the
  deliverable for a future execution engine.
- Source lint and runtime graph checks enforce the supported plan style. They
  are not a security sandbox or proof of no leakage; use trusted model-generated
  code and inspect findings and plans. Side tables still require provenance audits.
- Workers are subprocesses with timeouts; the runner and grid execution are sequential.
- Private Skrub graph inspection is isolated in `graphs.py` and `worker.py`, alongside
  public textual graph exports. Dependencies are pinned in `uv.lock`.
