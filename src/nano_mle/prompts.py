"""Instructions for the controller, planner and writer.

The plan contract and library notes live in plans.GUIDE. The working conventions
below are task-independent and adapted from mle-claude.
"""
from .config import load_config
from .plans import GUIDE

GOAL = """
Goal: a model that scores well on the task's metric for the population it must
predict. Cross-validation under the locked evaluation is how progress is measured;
it is only as good as the evaluation setup is representative.
"""

CONVENTIONS = """
Working conventions:
- Evaluation. Choose the population, labels, CV and scorer once, so scores stay
  comparable. Make the rows and folds mirror the prediction setting (entities,
  time, groups). Use the task's own metric when you can express it as a scorer.
- Leakage 1, mark early. Mark X and the raw y as soon as rows and labels exist and
  build features after the marks, so anything fit on the target sees training folds
  only.
- Leakage 2, where labels come from. Marking protects only the route through y. A
  feature that depends on labels must take them from the marked y or from rows
  provably disjoint from the modelled rows; reading the modelled rows' labels back
  from a source table leaks identically in every fold. Prefer statistics estimated
  on disjoint rows over leave-one-out corrections. In graph or neighbour features,
  a walk that leaves a row and returns (d -> n -> d) hands it its own label.
- Leakage 3, how to catch it. Distrust a large gain with an unusually small fold
  std, and a feature whose standalone ranking is near-perfect on the rows it covers.
- Audit new features. Check a feature's standalone signal on the rows it covers and
  that its coverage and distribution match between modelled rows and the rows to be
  predicted; a feature rich in training and empty at prediction time hurts.
- Experiments. Change one thing per experiment so its effect is attributable. When a
  change has natural variants (hyperparameters, estimators, feature blocks for an
  ablation), express them as a named choose_from grid in one experiment instead of
  stepping through them one at a time. Differences within the fold noise are not
  evidence.
- Data volume. Read only the columns you need, filter large tables to the relevant
  rows before joining, and subsample deterministically in the plan when a full
  table is unnecessary. Otherwise write the clearest plan: do not precompute,
  memoise or restructure for speed.
"""

# Optional convention, switched by [prompts] data_volume_study in nano-mle.toml.
DATA_VOLUME_STUDY = """- Data volume study. When sources are large (millions of rows), decide the modelled
  population with evidence rather than defaulting to all rows or to a small sample.
  Before the evaluation setup, explore which rows and which parts of large tables
  the task needs (e.g. only edges or events that touch the modelled and predicted
  entities) and whether a deterministic sample matches the prediction population.
  After the lock, measure a learning curve: keep the test folds fixed and train on
  growing deterministic fractions of the training rows, e.g. a choose_from over the
  fraction with an estimator that subsamples in fit. Build the added rows exactly
  like the others, including their label-derived features; a curve that falls as
  rows are added signals a construction error, not saturation. The curve only
  reaches the locked population; if it is still rising there, record that a larger
  population is likely to help.
"""
if load_config()["prompts"]["data_volume_study"]:
    CONVENTIONS += DATA_VOLUME_STUDY

PLAN_INSTRUCTIONS = GOAL + GUIDE + CONVENTIONS + """
Evaluation setup is unscored: it builds and audits the rows, labels, CV and scorer
that every later pipeline is scored with. Use exploration evidence for it.
A repair fixes the planned experiment; a different hypothesis needs a new proposal.
"""

CONTROL_INSTRUCTIONS = GOAL + ("Working convention:\n" + DATA_VOLUME_STUDY
                               if load_config()["prompts"]["data_volume_study"] else "") + """
Choose the next action. Explore with a concrete question and an observable stopping
condition whenever evidence is missing: before the evaluation setup, and later
whenever results are unclear or progress stalls. Propose establish_evaluation once
the population, labels, CV and metric are understood; it is locked afterwards.
Once candidates exist, a probe saves out-of-fold predictions of a candidate
(candidate_id) or of a new single-configuration pipeline for the question; follow it
with an exploration that analyses the predictions (errors by segment, entity or
feature). Exploration and probes produce evidence, not search reward. Stay within
the budgets.
"""

PLANNING_INSTRUCTIONS = GOAL + CONVENTIONS + """
Propose the next experiment on the selected pipeline, or request exploration first.
State the hypothesis and the changes, grounded in findings, references and history.
"""
