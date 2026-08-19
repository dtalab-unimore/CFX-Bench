from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from classifiers import OneHotDataClassifierAdapter
from explainers.AR import ActionableRecourse
from explainers.AR.recourse.action_set import ActionSet
from explainers.base import BaseExplainer, _empty_explanation_dict, prepare_output
# from recourse import ActionSet
from utils import get_binning_maps


class ARExplainer(BaseExplainer):

    def __init__(self, model, X_train: pd.DataFrame, y_train: pd.Series, features, cat_features, num_features,
                 act_features, target, monotonic_features: dict = None, **kwargs):
        # TODO: adapt for datasets with numerical features (currently assumes all features are categorical)
        super().__init__(
            model, X_train, y_train, features, features, [], act_features, target, monotonic_features
        )

    def _init(self):
        ohe_sep = '§'  # use a separator that is not likely to appear in feature names or category names
        self.OHE = OneHotEncoder(sparse_output=False, drop=None, feature_name_combiner=lambda a, b: f"{a}{ohe_sep}{b}")
        X_bins_ohe = self.OHE.fit_transform(self.X_train)
        X_bins_ohe = pd.DataFrame(X_bins_ohe, columns=self.OHE.get_feature_names_out())

        self.model_ohe = OneHotDataClassifierAdapter(self.model, self.OHE.get_feature_names_out(), ohe_separator=ohe_sep)
        self.model_ohe.fit(X_bins_ohe, self.y_train)
        self._ohe_sep = ohe_sep

        self.X_train = X_bins_ohe
        ohe_features = self.OHE.get_feature_names_out()
        immutables_ohe = {f: False for f in ohe_features}
        for f in self.features:
            if f not in self.act_features:
                # mark all ohe features corresponding to f as immutable
                ohe_feature_groups = self.model_ohe.ohe_feature_groups_
                ids = ohe_feature_groups[f]
                for id_ in ids:
                    immutables_ohe[ohe_features[id_]] = True
        immutables_ohe = [k for k, v in immutables_ohe.items() if v]
        self.features = self.OHE.get_feature_names_out().tolist()
        self.cat_features = self.features
        self.immutable_features = immutables_ohe

        # TODO: pass coeffs and intercept as arguments to the constructor, to generalize to non-linear models
        lr_coef_map = dict(zip(self.model.estimator_.feature_names_in_, self.model.estimator_.coef_.flatten()))
        woe_map, _ = get_binning_maps(self.model.binning_process_)
        coeffs = []
        for feature_category in self.OHE.get_feature_names_out():
            feature, category = feature_category.split(ohe_sep, 1)
            woe = woe_map[feature][category]
            coef = lr_coef_map[feature] * woe
            coeffs.append(coef)
        self.coeffs, self.intercept = np.array(coeffs), self.model.estimator_.intercept_

    def _setup_ar_explainer(self, record, target=1):
        # Implement monotonicity constraints by marking features as immutable
        # IMPORTANT: immutable features must be indicated in the constructor and excluded from 'subset limit' constraints
        immutables_ohe = deepcopy(self.immutable_features)
        for f, categories in self.monotonic_features.items():
            curr_val = record[f]
            idx = categories.index(curr_val)
            if idx == len(categories) - 1:
                # mark the feature as immutable
                for bin_to_imm in categories:
                    ohe_name = f"{f}{self._ohe_sep}{bin_to_imm}"
                    immutables_ohe.append(ohe_name)
                continue
            for bin_to_imm in categories[:idx]:
                ohe_name = f"{f}{self._ohe_sep}{bin_to_imm}"
                immutables_ohe.append(ohe_name)

        self.cf_model = ActionableRecourse(
            self.model_ohe.predict_proba, self.X_train, self.features,
            continuous_features=[], immutable_features=immutables_ohe, coeffs=self.coeffs, intercepts=self.intercept,
            y_desired=target,
        )

        A: ActionSet = self.cf_model.action_set
        ohe_features = np.asarray(self.features)
        ohe_feature_groups = self.model_ohe.ohe_feature_groups_
        for f in self.act_features:
            ids = ohe_feature_groups[f]
            names = ohe_features[ids].tolist()
            names = [n for n in names if n not in immutables_ohe]
            if len(names) == 0:
                continue
            A.add_constraint(constraint_type='subset_limit', names=names, lb=1, ub=1, id=f)

    def _explain(self, test_item, n_cf=1):
        record, label, pred, proba, target = test_item
        record_bins = record.copy().to_frame().T
        record_ohe = self.OHE.transform(record_bins)
        record_ohe = pd.DataFrame(record_ohe, columns=self.OHE.get_feature_names_out())

        self._setup_ar_explainer(record, target=target)

        cfs_ohe = self.cf_model.get_counterfactuals(record_ohe)

        if pd.isna(cfs_ohe).all().all():
            return _empty_explanation_dict(test_item)

        list_new_probs = self.model_ohe.predict_proba(cfs_ohe)[:, 1]
        cfs_bins = self.OHE.inverse_transform(cfs_ohe)
        cfs_bins = pd.DataFrame(cfs_bins, columns=self.OHE.feature_names_in_)

        expl_dict = prepare_output(*test_item, cfs_bins, list_new_probs)
        return expl_dict
