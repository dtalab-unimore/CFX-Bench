import pandas as pd

from explanation import ExplanationSet


class BaseExplainer:

    def __init__(
            self, model, X_train, y_train, features, cat_features, num_features, act_features, target,
            monotonic_features: dict = None,
    ):
        """
        :param monotonic_features: Dictionary with monotonic feature names as keys.
        For categorical features, the value is a list of categories in monotonic order.
        For numerical features, the value is either 1 (monotonic increasing) or -1 (monotonic decreasing).
        """
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.features = features
        self.cat_features = cat_features
        self.num_features = num_features
        self.act_features = act_features
        self.target = target
        self.monotonic_features = monotonic_features if monotonic_features is not None else {}

        if self.X_train is not None:
            self.X_train = self.X_train.copy()
            if self.features is not None:
                self.X_train = self.X_train[self.features]
        self._init()

    def _init(self):
        pass

    def _explain(self, test_item, n_cf=1) -> dict:
        pass

    def explain(self, test_item, n_cf=1):
        expl_dict = self._explain(test_item, n_cf)
        expl_set = ExplanationSet(**expl_dict)
        return expl_set


def prepare_output(record: pd.Series, label, pred, proba, target, cfs, cfs_proba):
    list_expl_full = cfs.reset_index(drop=True)
    list_expl_changes = []
    for _, cf in list_expl_full.iterrows():
        changes = []
        for col in list_expl_full.columns:
            if cf[col] != record[col]:
                changes.append(cf[col])
            else:
                changes.append('-')
        list_expl_changes.append(changes)
    list_expl_changes = pd.DataFrame(list_expl_changes, columns=list_expl_full.columns)

    expl_dict = {
        'record': record, 'label': label, 'pred': pred, 'proba': proba, 'target': target,
        'list_expl_full': list_expl_full, 'list_expl_changes': list_expl_changes,
        'list_new_probs': pd.Series(cfs_proba)
    }
    return expl_dict


def _empty_explanation_dict(test_item):
    record, label, pred, proba, target = test_item
    features = record.index.to_list()
    return {
        'record': pd.Series(record, index=features), 'label': label, 'pred': pred,
        'proba': proba, 'target': target, 'list_new_probs': pd.Series(dtype=float),
        'list_expl_full': pd.DataFrame(columns=features),
        'list_expl_changes': pd.DataFrame(columns=features)
    }
