import pandas as pd
import pytest
import skrub
from nano_mle.graphs import validate_graph
from nano_mle.plans import resolve_source, validate_source

@pytest.mark.parametrize('body', ["return open('features.csv')", "return data.to_csv('features.csv')", "return data.skb.eval()"])
def test_forbidden_execution(body):
    with pytest.raises(ValueError):
        validate_source('def build():\n    '+body+'\n')


def test_opaque_callbacks_and_materialized_data_rejected(tmp_path):
    with skrub.config_context(eager_data_ops=False):
        table = skrub.as_data_op(str(tmp_path / 'rows.csv')).skb.apply_func(pd.read_csv)
        for plan in [table.apply(lambda x:x), table.skb.apply_func(lambda x:x), table['a'].map(lambda x:x), skrub.as_data_op(pd.DataFrame({'a':[1]}))]:
            with pytest.raises(ValueError):
                validate_graph(plan)
        validate_graph(table['a'].map({1:2}))


def test_parent_grid_is_resolved_before_implementation():
    source = """import skrub
from sklearn.linear_model import Ridge
def build():
    return Ridge(alpha=skrub.choose_from([0.1, 10.0], name='alpha'))
"""
    resolved = resolve_source(source, {'alpha':1})
    assert 'choose_from' not in resolved
    assert 'alpha=10.0' in resolved


def test_unnamed_choices_rejected():
    with pytest.raises(ValueError, match='name'):
        validate_source('import skrub\ndef build():\n    return skrub.choose_from([1,2])\n')


def test_transformer_alias_cannot_hide_function_nodes():
    with pytest.raises(ValueError, match='function transformers'):
        validate_source('from sklearn.preprocessing import FunctionTransformer as F\ndef build():\n    return F()\n')


def test_graph_building_dictionary_helpers_allowed():
    validate_source('''def helper(data):
    outputs = {}
    for col in ['a', 'b']:
        outputs[col] = data[col].isna().sum()
    return outputs

def build():
    return helper(data)
''')


def test_eager_reader_alias_rejected():
    with pytest.raises(ValueError, match='Record readers'):
        validate_source('from pandas import read_csv as reader\ndef build():\n    return reader("data.csv")\n')


def test_locked_setup_export_preserves_existing_helper():
    from nano_mle.plans import evaluation_source
    source = '''def build_evaluation():
    return {'scoring': 'r2'}

def locked_setup_entry():
    return None

def build():
    return build_evaluation()
'''
    exported = evaluation_source(source)
    namespace = {}
    exec(exported, namespace)
    assert namespace['build_evaluation']() == {'scoring': 'r2'}
