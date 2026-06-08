from copy import deepcopy

import numpy as np
import pandas as pd
from optbinning import BinningProcess
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

from utils import get_binning_maps


def get_classifier(name, params=None):
    if name == 'lr':
        from sklearn.linear_model import LogisticRegression
        if params is None:
            params = dict(solver="newton-cholesky", penalty=None)
        return LogisticRegression(**params)
    raise NotImplementedError("Supported models: ['lr']")


class ClassifierForBinnedData(ClassifierMixin, BaseEstimator):

    def __init__(self, estimator, binning_process_: BinningProcess, transform_type=None, estimator_fit_kwargs=None):
        self.estimator = deepcopy(estimator)  # estimator to fit
        self.binning_process = deepcopy(binning_process_)  # already fitted binning process
        if transform_type in [None, 'woe', 'id']:
            self.transform_type = transform_type
        else:
            raise ValueError(f"transform_ must be in [None, 'woe', 'id'], found '{transform_type}'")
        self.estimator_fit_kwargs_ = estimator_fit_kwargs or {}
        self.is_fitted_ = False

    @staticmethod
    def _replace_via_map(X, mapping):
        out = {}
        for col in X.columns:
            s = X[col]
            d = mapping.get(col)
            if d is None:
                out[col] = s.copy()
                continue
            mapped = s.map(d)
            # .map sends unmapped values to NaN; .replace leaves them unchanged
            miss = mapped.isna() & s.notna()
            if miss.any():
                mapped = mapped.astype(object)
                mapped[miss] = s[miss]
            out[col] = mapped
        return pd.DataFrame(out, index=X.index)

    def transform(self, X, transform_type=None):
        X = X.copy()
        if transform_type is None: transform_type = self.transform_type
        if isinstance(X, np.ndarray):  # assume two-dimensional array
            X = pd.DataFrame(X, columns=self.feature_names_in_)
        if transform_type == 'woe':
            return X.replace(self.woe_map_)
            # return self._replace_via_map(X, self.woe_map_)
        elif transform_type == 'id':
            return X.replace(self.ids_map_)
            # return self._replace_via_map(X, self.ids_map_)
        return X

    def fit(self, X: pd.DataFrame, y):
        self.feature_names_in_ = np.array(X.columns)

        self.woe_map_, self.ids_map_ = get_binning_maps(self.binning_process)
        if self.transform_type in ['woe', 'id']:
            X = self.transform(X)

        self.estimator.fit(X, y, **self.estimator_fit_kwargs_)
        self.binning_process_ = self.binning_process
        self.estimator_ = self.estimator
        self.is_fitted_ = True
        return self

    def predict_proba(self, X):
        check_is_fitted(self)
        if isinstance(X, np.ndarray):  # assume two-dimensional array
            X = pd.DataFrame(X, columns=self.feature_names_in_)
        X = self.transform(X)
        return self.estimator_.predict_proba(X)

    def predict(self, X):
        check_is_fitted(self)
        if isinstance(X, np.ndarray):  # assume two-dimensional array
            X = pd.DataFrame(X, columns=self.feature_names_in_)
        X = self.transform(X)
        return self.estimator_.predict(X)


class OneHotDataClassifierAdapter(ClassifierMixin, BaseEstimator):

    def __init__(self, model, ohe_feature_names, ohe_separator="_"):
        self.model = model
        self.ohe_feature_names = ohe_feature_names
        self.ohe_separator = ohe_separator
        self.is_fitted_ = False

    def fit(self, X=None, y=None):
        # self.model.fit(X, y)

        categories = []
        ohe_feature_groups = {}
        for i, feature_category in enumerate(self.ohe_feature_names):
            feature, category = feature_category.split(self.ohe_separator)
            categories.append(category)
            if feature not in ohe_feature_groups:
                ohe_feature_groups[feature] = []
            ohe_feature_groups[feature].append(i)

        self.categories_ = np.array(categories)
        self.feature_names_in_ = np.array(self.ohe_feature_names)
        self.ohe_feature_groups_ = ohe_feature_groups
        self.is_fitted_ = True
        return self

    def _validate_ohe_structure(self, X):
        """Ensures exactly one active bin per original feature group."""
        X_arr = X.values if hasattr(X, "values") else X

        for feat, indices in self.ohe_feature_groups_.items():
            subset = X_arr[:, indices]
            row_sums = subset.sum(axis=1)

            if not np.allclose(row_sums, 1):
                bad_idx = np.where(~np.isclose(row_sums, 1))[0][0]
                raise ValueError(
                    f"Scorecard Structural Error: Feature '{feat}' must have exactly 1 active bin. "
                    f"Row {bad_idx} sum was {row_sums[bad_idx]}."
                )

    def predict_proba(self, X):
        check_is_fitted(self)
        X = X.values if hasattr(X, "values") else np.asarray(X)

        if X.shape[1] != len(self.categories_):
            raise ValueError(f"Expected {len(self.categories_)} features, got {X.shape[1]}.")

        self._validate_ohe_structure(X)
        categories = np.broadcast_to(self.categories_, X.shape)
        X_bins = categories[X > 0.5].reshape(X.shape[0], -1)
        return self.model.predict_proba(X_bins)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)
