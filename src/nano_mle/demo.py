"""Deterministic backend exercises the harness without any model/API calls."""

from .models import Decision, EvaluationSpec, Finding, Proposal


class DemoBackend:
    def control(self, context):
        counts = context["counts"]
        if counts["explorations"] == 0:
            return Decision(action="explore", reason="Understand the raw table",
                            question="What are the distributions and missing-value counts?",
                            stopping_condition="Summary and missing counts computed")
        if context["contract"] is None:
            return Decision(action="establish_evaluation", reason="Independent synthetic regression rows",
                            evaluation=EvaluationSpec(scoring="neg_root_mean_squared_error", cv="kfold",
                                                      rationale="Synthetic independent regression observations"))
        if counts["expansions"] >= 2:
            return Decision(action="stop", reason="Offline demonstration complete")
        return Decision(action="expand", reason="Evaluate baseline then a small grid")

    def plan(self, context):
        if context["selection"]["parent_id"] != "root" and not context["requested_explorations"]:
            return Proposal(action="explore", description="Check feature coverage before expanding",
                            question="Are feature values missing?",
                            stopping_condition="Per-column missing counts computed")
        return Proposal(action="experiment", description="Ridge baseline" if not context["parent"] else
                        "Compare two Ridge regularization strengths", changes=["Regularization strength"])

    def implement(self, kind, context, intent):
        if kind == "exploration":
            return """def build(ctx):
    data = ctx.read('train')
    missing = data.isna()
    counts = missing.sum()
    summary = data.describe()
    return {'missing_counts': counts, 'summary': summary}
"""
        alpha = "skrub.choose_from([0.1, 10.0], name='alpha')" if context["parent"] else "1.0"
        return f"""import skrub
from sklearn.linear_model import Ridge

def build(ctx):
    X, y = ctx.load_xy()
    encoded = X.skb.apply(skrub.TableVectorizer())
    model = Ridge(alpha={alpha})
    pred = encoded.skb.apply(model, y=y)
    return pred
"""

    def repair(self, kind, context, intent, source, error):
        return source

    def interpret(self, context, question, result):
        outputs = result["outputs"]
        return [Finding(statement="Computed per-column missing counts and numeric distributions",
                        scope="Full pinned training CSV", evidence=str(outputs)[:6000])]
