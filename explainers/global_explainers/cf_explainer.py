import pandas as pd

from explanation import ExplanationSet


class BaseExplainer:
    def _explain(self, test_item, n_cf=1) -> dict:
        pass

    def explain(self, test_item, n_cf=1):
        expl_dict = self._explain(test_item, n_cf)
        expl_set = ExplanationSet(**expl_dict)
        return expl_set

def _empty_explanation_dict(test_item):
    record, label, pred, proba, target = test_item
    features = record.index.to_list()
    return {
        'record': pd.Series(record, index=features), 'label': label, 'pred': pred,
        'proba': proba, 'target': target, 'list_new_probs': pd.Series(dtype=float),
        'list_expl_full': pd.DataFrame(columns=features),
        'list_expl_changes': pd.DataFrame(columns=features)
    }
