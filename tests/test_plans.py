import pytest

from nano_mle.plans import resolve_source, validate_source


@pytest.mark.parametrize("body", [
    "return ctx.read('train').skb.eval()",
    "return ctx.read('train').apply(lambda row: row)",
    "return ctx.read('train').skb.apply_func(sum)",
    "return ctx.load_xy()[0].skb.mark_as_X(cv=2)",
    "return open('features.csv')",
    "return ctx.read('train').to_csv('features.csv')",
])
def test_opaque_computation_and_harness_overrides_rejected(body):
    with pytest.raises(ValueError):
        validate_source("def build(ctx):\n    " + body + "\n")


def test_parent_grid_is_resolved_before_implementation():
    source = """import skrub
from sklearn.linear_model import Ridge
def build(ctx):
    X, y = ctx.load_xy()
    model = Ridge(alpha=skrub.choose_from([0.1, 10.0], name='alpha'))
    return X.skb.apply(model, y=y)
"""
    resolved = resolve_source(source, {"alpha": 1})
    assert "choose_from" not in resolved
    assert "alpha=10.0" in resolved


def test_unnamed_choices_rejected():
    with pytest.raises(ValueError, match="name"):
        validate_source("import skrub\ndef build(ctx):\n    return skrub.choose_from([1,2])\n")
