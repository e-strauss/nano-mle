"""Realistic multi-table, derived-label setup with an agent-defined splitter."""
import pandas as pd
from nano_mle.execution import run_plan
from nano_mle.contracts import create_contract
from nano_mle.models import EvaluationSpec
from nano_mle.plans import evaluation_source


def test_custom_population_labels_and_cutoff_cv(tmp_path):
    lots = tmp_path / 'lots.csv'
    events = tmp_path / 'events.csv'
    pd.DataFrame({'bbl':range(1,13),'units':[2,4,5]*4,'cutoff':[0]*4+[1]*4+[2]*4}).to_csv(lots,index=False)
    pd.DataFrame({'bbl':range(1,13),'violations':range(12)}).to_csv(events,index=False)
    source = f'''import pandas as pd
import numpy as np
import skrub
class CutoffSplit:
    def split(self, X, y=None):
        cutoffs = X['cutoff'].to_numpy()
        for cutoff in [1, 2]:
            yield np.flatnonzero(cutoffs < cutoff), np.flatnonzero(cutoffs == cutoff)
    def get_n_splits(self, X=None, y=None):
        return 2

def read_table(path):
    return skrub.as_data_op(path).skb.apply_func(pd.read_csv)

def build():
    lots = read_table({str(lots)!r})
    events = read_table({str(events)!r})
    eligible = lots[lots['units'] >= 3]
    population = eligible.merge(events, on='bbl', how='left')
    X = population.drop(columns=['violations']).skb.mark_as_X(cv=CutoffSplit())
    y = population['violations'].fillna(0).skb.mark_as_y()
    return {{'X':X, 'y':y, 'row_keys':population[['bbl','cutoff']], 'scoring':'neg_mean_absolute_error'}}
'''
    setup = run_plan(tmp_path/'setup',source,{'kind':'evaluation'},60)
    assert setup['status'] == 'ok', setup
    assert setup['snapshot']['rows'] == 8
    assert len(setup['snapshot']['splits']) == 2
    contract = create_contract(setup['snapshot'], evaluation_source(source), EvaluationSpec(scoring='neg_mean_absolute_error',rationale='Future cutoffs'))
    candidate = contract['setup_source'] + '''
from sklearn.linear_model import Ridge
def build():
    setup = build_evaluation()
    pred = setup['X'].drop(columns=['bbl']).skb.apply(Ridge(), y=setup['y'])
    return {'pred':pred, 'scoring':setup['scoring'], 'row_keys':setup['row_keys']}
'''
    scored = run_plan(tmp_path/'candidate',candidate,{'kind':'pipeline','contract':contract,'remaining_evaluations':1},60)
    assert scored['status'] == 'ok', scored
    assert scored['evaluation_count'] == 1
