"""Versioned generation instructions, adapted from mle-claude's Skrub guide.

The former exploration exception for plain pandas is intentionally removed.
"""

from .plans import GUIDE


PLAN_INSTRUCTIONS = GUIDE + """

Pipeline skeleton:
    import skrub
    from sklearn.ensemble import RandomForestRegressor

    def build(ctx):
        X, y = ctx.load_xy()
        features = X.drop(columns=['id'])
        encoded = features.skb.apply(skrub.TableVectorizer())
        model = RandomForestRegressor(n_estimators=40, random_state=42, n_jobs=1)
        pred = encoded.skb.apply(model, y=y)
        return pred

Exploration skeleton (all report computations are DataOps, not eager pandas):
    def build(ctx):
        data = ctx.read('train')
        missing = data.isna()
        missing_counts = missing.sum()
        summary = data.describe()
        target_counts = data['target'].value_counts()
        return {'missing_counts': missing_counts, 'summary': summary,
                'target_counts': target_counts}

Feature construction: use separate selections, groupby/agg, renames, merges and
assigns; avoid packing an entire feature builder into one node. No custom helpers.
The ctx read is the single harness-controlled primitive for a CSV read.
The harness evaluates returned exploration outputs and saves their graphs and values.
Do not print, evaluate, or write results yourself. Never persist features for reuse.

Grid example:
    model = skrub.choose_from({
        'small': RandomForestRegressor(n_estimators=20, random_state=42, n_jobs=1),
        'large': RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=1),
    }, name='model')
    pred = encoded.skb.apply(model, y=y)
Every resolved grid variant becomes a sibling candidate. Respect the remaining
evaluation budget. Use explicit choices only. Named choice configurations store
outcome INDICES, alongside human-readable descriptions; preserve a selected parent's
resolved outcome when building from its code. Do not accidentally rerun its full grid.

Leakage discipline:
- X and RAW y are marked by load_xy before any feature engineering.
- Labels used by learned transforms must originate from marked y, not a side file.
- Other sources can also contain labels: early marking does not make those safe.
- Investigate suspicious gains, near-perfect feature signals and coverage differences
  between training and prediction populations. Small fold std alone is not proof.
- Maintain row count/order for predictions. Drop identifier/group/time columns from
  model features when appropriate; the contract's CV still uses its frozen rows.
- Target transforms require inversion at prediction; prefer raw targets in this MVP.
- Use no caching or execution optimization: preserve the logical computation graph.
- A repair fixes the planned experiment. Changing its hypothesis is a new proposal.

Only local CSV task sources and standard sklearn/Skrub estimators are supported in
this prototype. Do not invent ctx methods or assume arbitrary shell access.
"""

CONTROL_INSTRUCTIONS = """
Choose one next action for a sequential ML experiment. Explore first and adaptively
when uncertainties, suspicious gains, join feasibility, or coverage need evidence.
An exploration needs a concrete question and an observable stopping condition.
Use establish_evaluation only after successful exploration supplies enough evidence
to justify a single sklearn scorer string and kfold/stratified/group/time CV.
The harness freezes all folds, sources and scoring. Once locked, expand or explore;
do not change the contract. Honor remaining budgets. Stop if no useful work remains.
Explorations produce evidence, not search rewards. Pipelines use fine-grained Skrub.
"""

PLANNING_INSTRUCTIONS = """
Plan a bounded change to the selected resolved pipeline, or request exploration
before proposing it. Use the parent's resolved configuration, findings, branch
trajectory and explicit candidate references. Explain the hypothesis and changes.
For exploration state its question and stopping condition. After the requested
rounds are exhausted, propose a feasible experiment using known evidence.
Do not add candidate references that the search policy did not supply.
"""
