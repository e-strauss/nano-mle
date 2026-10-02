# Future ideas

These are design notes, not implemented behavior or requirements for the current
prototype.

## Multiple evaluation branches

Allow one task to contain multiple evaluation branches. Each branch owns one locked
X/y construction, CV setup and scoring contract, together with its own candidate
search graph. These evaluation branches are distinct from the pipeline lineage
branches inside an individual search graph.

The motivation is validation quality: a search policy can optimize its internal CV
score successfully while selecting pipelines that perform poorly on the prediction
population. Exploration may reveal that another population definition or CV setup
better represents the actual task.

MLE-claude demonstrated a useful behavior here: the model could reason about a
different evaluation setup using evidence from additional data exploration. Preserve
that ability without silently changing the contract of existing scored candidates.

### Proposing a branch

The controller or planner can propose another X/y/CV setup and explain why it should
better represent the prediction setting. Further exploration can support or challenge
the proposal with statistical evidence: population and coverage differences,
temporal drift, repeated entities, or differences between validation and prediction
conditions. Such evidence supports a hypothesis; it does not prove future performance.

Audit and lock the proposed setup independently. Preserve existing branches and
their results. Candidate code and relevant findings can inform the new branch, but
transferred candidates must be evaluated under its contract.

### Search policy

Selection becomes two decisions:

1. Which evaluation branch should receive the next allocation of work?
2. Which candidate inside that branch should be expanded?

Each branch can retain a greedy, MCTS or MCGS candidate policy. A separate allocation
policy must balance investigating evaluation quality with improving pipelines under
an established setup. Keep the first implementation sequential; multiple branches
do not require concurrent execution.

Scores, candidate rankings and reward backpropagation remain local to each contract.
Raw CV scores from different setups must not form a shared leaderboard: an easier
split can produce a larger score without producing a better model. Choosing between
branches needs evidence about evaluation representativeness as well as progress and
cost within each branch. Exploration findings can be shared with their scope and
provenance preserved.

### Open questions

- What evidence warrants opening another evaluation branch?
- How should the outer policy allocate budget without favoring optimistic CV setups?
- How should the final pipeline be selected when branch scores are incomparable?
- What context and candidate references should transfer between branches?

Multiple evaluation branches and their allocation policy are deferred. The current
nano-mle design should first support one agent-authored, locked evaluation setup.
