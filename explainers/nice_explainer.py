import numpy as np
import pandas as pd
from optbinning import Scorecard

from explainers.base import BaseExplainer
from explainers.nice import NICE


class NiceExplainer(BaseExplainer):
    def __init__(self, model: Scorecard, X_train, y_train, features, cat_features, num_features, **kwargs):
        super().__init__(
            model, X_train, y_train, features, cat_features, num_features, [], None, None
        )

    def _init(self):
        X_train, y_train = self.X_train, self.y_train
        num_features, cat_features = self.num_features, self.cat_features
        predict_fn = lambda x: self.model.predict_proba(pd.DataFrame(x, columns=self.features))
        ix_cat_features = [self.features.index(x) for x in cat_features]
        ix_num_features = [self.features.index(x) for x in num_features]
        if not isinstance(X_train, np.ndarray): X_train = X_train.to_numpy()
        if not isinstance(y_train, np.ndarray): y_train = y_train.to_numpy()
        self.cf_model = NICE(
            X_train=X_train,
            predict_fn=predict_fn,
            y_train=y_train,
            cat_feat=ix_cat_features,
            num_feat=ix_num_features,
            distance_metric='HEOM',
            num_normalization='minmax',
            optimization='proximity',
            justified_cf=True
        )
        self.predict_fn = predict_fn

    def _explain(self, test_item, n_cf=1):
        record, label, pred, proba, target = test_item
        cfs = self.cf_model.explain(record.to_numpy().reshape(1, -1))
        list_new_probs = self.predict_fn(cfs)[:, 1]
        list_expl_full = pd.DataFrame(cfs, columns=self.features)
        list_expl_changes = []
        for cf in cfs:
            changes = []
            for i in range(len(cf)):
                if cf[i] != record[i]:
                    changes.append(cf[i])
                else:
                    changes.append('-')
            list_expl_changes.append(changes)
        list_expl_changes = pd.DataFrame(list_expl_changes, columns=self.features)

        expl_dict = {
            'record': record, 'label': label, 'pred': pred,
            'proba': proba, 'target': target, 'list_expl_full': list_expl_full,
            'list_expl_changes': list_expl_changes, 'list_new_probs': pd.Series(list_new_probs)
        }

        return expl_dict
