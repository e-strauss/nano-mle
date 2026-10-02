"""Instructions for the controller, planner and writer.

The plan contract and library notes live in plans.GUIDE. The working conventions
below are task-independent and adapted from mle-claude.
"""
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

PLAN_INSTRUCTIONS = GOAL + GUIDE + CONVENTIONS + """
Evaluation setup is unscored: it builds and audits the rows, labels, CV and scorer
that every later pipeline is scored with. Use exploration evidence for it.
A repair fixes the planned experiment; a different hypothesis needs a new proposal.
"""

CONTROL_INSTRUCTIONS = GOAL + """
Choose the next action. Explore with a concrete question and an observable stopping
condition whenever evidence is missing: before the evaluation setup, and later
whenever results are unclear or progress stalls. Propose establish_evaluation once
the population, labels, CV and metric are understood; it is locked afterwards.
Exploration produces evidence, not search reward. Stay within the budgets.
"""

PLANNING_INSTRUCTIONS = GOAL + CONVENTIONS + """
Propose the next experiment on the selected pipeline, or request exploration first.
State the hypothesis and the changes, grounded in findings, references and history.
"""
