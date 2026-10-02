import numpy as np
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


def test_lint_errors_name_the_rejected_construct():
    with pytest.raises(ValueError, match="Import of os is not allowed"):
        validate_source("import os\ndef build():\n    return {}\n")
    helper = "class Helper:\n    def run(self, x):\n        return x\ndef build():\n    return {}\n"
    with pytest.raises(ValueError, match="Class Helper.*fine-grained DataOps"):
        validate_source(helper)


def test_custom_estimators_and_torch_modules_are_allowed():
    validate_source('''
import numpy as np
import torch
from torch import nn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression

class Net(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.layer = nn.Linear(width, 1)
    def forward(self, x):
        return self.layer(x)

class Wrapped(ClassifierMixin, BaseEstimator):
    def __init__(self, C=1.0):
        self.C = C
    def fit(self, X, y):
        self._model = LogisticRegression(C=self.C).fit(X, y)
        self.classes_ = self._model.classes_
        net = Net(X.shape[1])
        net.eval()
        return self
    def predict_proba(self, X):
        return self._model.predict_proba(X)

def build():
    return {}
''')
    for body in ["        self.__dict__['a'] = 1", "        X.to_csv('x.csv')", "        torch.save(self, 'm.pt')"]:
        with pytest.raises(ValueError):
            validate_source("class E:\n    def fit(self, X, y=None):\n" + body + "\n        return self\n"
                            "def build():\n    return {}\n")
    with pytest.raises(ValueError, match="harness-owned"):
        validate_source("def build():\n    return model.fit(X)\n")

def test_installed_libraries_import_and_missing_ones_are_named():
    from nano_mle.plans import MissingLibrary

    validate_source("import lightgbm\nfrom xgboost import XGBClassifier\nimport polars as pl\n"
                    "def build():\n    return {}\n")
    with pytest.raises(MissingLibrary) as error:
        validate_source("import definitely_not_installed_pkg\ndef build():\n    return {}\n")
    assert error.value.modules == ["definitely_not_installed_pkg"]


def test_dtype_arguments_are_not_opaque_callables():
    data = skrub.var("data")  # no value: no eager preview
    validate_graph(data.assign(b=data["a"].astype(str), c=data["a"].astype(float)))
    validate_graph(data["a"].astype(np.float64))
    with pytest.raises(ValueError, match="'helper'.*clip"):
        def helper(x):
            return x
        validate_graph(data["a"].clip(upper=helper))


def test_local_reader_helpers_are_not_eager_reads():
    helper = ("import pandas as pd\nimport skrub\n"
              "def read_parquet(name):\n"
              "    return skrub.as_data_op(name).skb.apply_func(pd.read_parquet)\n"
              "def build():\n    return {'t': read_parquet('t.parquet')}\n")
    validate_source(helper)
    for eager in ("import pandas as pd\ndef build():\n    return pd.read_parquet('t')\n",
                  "import pandas\ndef build():\n    return pandas.read_csv('t')\n",
                  "from pandas import read_csv as rc\ndef build():\n    return rc('t')\n"):
        with pytest.raises(ValueError, match="Record readers"):
            validate_source(eager)


def test_getattr_only_with_literal_public_name():
    validate_source('def build():\n    return getattr(est, "classes_", None)\n')
    for call in ['getattr(est, name)', 'getattr(est, "__class__")', 'getattr(est, "_private")', 'getattr(est, "fit")']:
        with pytest.raises(ValueError):
            validate_source('def build():\n    return ' + call + '\n')
