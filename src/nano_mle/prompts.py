"""Instructions for auditable agent-authored evaluation and pipeline graphs."""
from .plans import GUIDE

PLAN_INSTRUCTIONS = GUIDE + """
Evaluation setup is a special unscored phase. Use exploration evidence to construct
representative modelling rows, raw labels and a defensible CV. Return marked X/y,
scoring (a sklearn scorer string, or a scorer function when the task metric needs it),
optional aligned unique row_keys and optional audit outputs.
Investigate temporal label windows, join multiplicity, population coverage and leakage.
Freeze all randomness in splitters. Custom splitter classes are permitted.
Record readers with skrub.as_data_op(path).skb.apply_func(pd.read_csv, ...) or
pd.read_parquet. Source paths are hints; construction is your responsibility.

For pipelines, include context.locked_evaluation_source and call build_evaluation().
Take its X/y, build downstream features and estimators, and return a dict containing
pred, the same scoring and the same row_keys if supplied. Equivalent inline graphs
are allowed. Do not change population, labels or CV. Drift requires restoring the
lock or a new workspace. Maintain row count and ordering downstream.
Fine-grained graph-building helpers and loops are allowed; opaque runtime UDFs are not.
Exploration returns named DataOps; the harness saves their graphs and outputs.
Use separate selection, groupby, merge, assign and aggregation nodes.
Never evaluate graphs, print results, cache features or score manually.
A named choose_from grid becomes sibling candidates; respect evaluation budgets.
Use the selected parent's resolved configuration rather than rerunning its grid.
A repair fixes the planned experiment; a new hypothesis needs a new proposal.
"""
CONTROL_INSTRUCTIONS = """
Choose the next sequential action. Explore adaptively with concrete questions and
observable stopping conditions. After sufficient evidence, propose establish_evaluation
with a scorer, CV planning hints and rationale. The writer builds and audits this setup
before any scored search. Once locked, explore or expand without changing evaluation.
Inputs are assumed static/frozen; content verification is out of scope.
Exploration produces evidence, not search reward. Honor all remaining budgets.
"""
PLANNING_INSTRUCTIONS = """
Propose a bounded change to the selected resolved pipeline or request exploration.
Use findings, supplied references and trajectory. Explain hypothesis and changes.
Keep locked evaluation intact; use fine-grained Skrub graphs for all computations.
"""
