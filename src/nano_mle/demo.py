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
        path = repr(context["sources"]["train"]["path"])
        reader = f"import pandas as pd\nimport skrub\nfrom sklearn.model_selection import KFold\n\ndef build():\n    data = skrub.as_data_op({path}).skb.apply_func(pd.read_csv)\n"
        if kind == "exploration":
            return reader + "    return {'missing_counts': data.isna().sum(), 'summary': data.describe()}\n"
        if kind == "evaluation":
            target = repr(context["task"]["target"])
            return reader + f"    X = data.drop(columns=[{target}]).skb.mark_as_X(cv=KFold(3, shuffle=True, random_state=42))\n    y = data[{target}].skb.mark_as_y()\n    return {{'X': X, 'y': y, 'scoring': 'neg_root_mean_squared_error'}}\n"
        alpha = "skrub.choose_from([0.1, 10.0], name='alpha')" if context["parent"] else "1.0"
        return context["locked_evaluation_source"] + f"""
from sklearn.linear_model import Ridge

def build():
    setup = build_evaluation()
    encoded = setup['X'].skb.apply(skrub.TableVectorizer())
    pred = encoded.skb.apply(Ridge(alpha={alpha}), y=setup['y'])
    return {{'pred': pred, 'scoring': setup['scoring']}}
"""

    def repair(self, kind, context, intent, source, error):
        return source

    def interpret(self, context, question, result):
        outputs = result["outputs"]
        return [Finding(statement="Computed per-column missing counts and numeric distributions",
                        scope="Full frozen training table", evidence=str(outputs)[:6000])]
