from copy import deepcopy

import numpy as np
import pandas as pd
from optbinning import BinningProcess
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import RobustScaler, MinMaxScaler, QuantileTransformer, OneHotEncoder, OrdinalEncoder
from sklearn.utils.validation import check_is_fitted

from utils import get_binning_maps


class _ClassifierForBinnedData(ClassifierMixin, BaseEstimator):

    def __init__(self, estimator, binning_process: BinningProcess):
        self.estimator = deepcopy(estimator)  # already fitted scorecard model
        self.binning_process = deepcopy(binning_process)  # already fitted binning process
        self.is_fitted_ = False

    def fit(self, X=None, y=None):
        check_is_fitted(self.estimator)
        self.woe_map_, _ = get_binning_maps(self.binning_process)
        self.feature_names_in_ = self.estimator.feature_names_in_
        self.binning_process_ = self.binning_process
        self.estimator_ = self.estimator
        self.is_fitted_ = True
        return self

    def predict_proba(self, X):
        check_is_fitted(self)
        if isinstance(X, np.ndarray):  # assume two-dimensional array
            X = pd.DataFrame(X, columns=self.feature_names_in_)
        X_woe = X.replace(self.woe_map_)
        return self.estimator_.predict_proba(X_woe)

    def predict(self, X):
        check_is_fitted(self)
        if isinstance(X, np.ndarray):  # assume two-dimensional array
            X = pd.DataFrame(X, columns=self.feature_names_in_)
        X_woe = X.replace(self.woe_map_)
        return self.estimator_.predict(X_woe)


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

    def transform(self, X, transform_type=None):
        X = X.copy()
        if transform_type is None: transform_type = self.transform_type
        if isinstance(X, np.ndarray):  # assume two-dimensional array
            X = pd.DataFrame(X, columns=self.feature_names_in_)
        if transform_type == 'woe':
            return X.replace(self.woe_map_)
        elif transform_type == 'id':
            return X.replace(self.ids_map_)
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


class ClassifierForMixedData(ClassifierMixin, BaseEstimator):

    def __init__(self, estimator, binning_process_: BinningProcess, cat_columns, num_columns):
        self.estimator = deepcopy(estimator)
        self.binning_process = deepcopy(binning_process_)
        # self.scaler = MinMaxScaler()
        self.scaler = QuantileTransformer(n_quantiles=100)
        self.cat_columns = cat_columns
        self.num_columns = num_columns
        self.is_fitted_ = False

    def transform(self, X):
        X = X.copy()
        X.loc[:, self.cat_columns] = X.loc[:, self.cat_columns].replace(self.woe_map_)
        X.loc[:, self.num_columns] = self.scaler_.transform(X.loc[:, self.num_columns])
        return X.astype(float)

    def fit(self, X: pd.DataFrame, y):
        self.woe_map_, _ = get_binning_maps(self.binning_process)
        X_num = X[self.num_columns]
        self.scaler_ = self.scaler.fit(X_num)
        X = self.transform(X)
        self.estimator_ = self.estimator.fit(X, y)
        self.binning_process_ = self.binning_process
        self.feature_names_in_ = self.estimator.feature_names_in_
        self.is_fitted_ = True
        return self

    def predict_proba(self, X):
        check_is_fitted(self)
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.feature_names_in_)
        X = self.transform(X)
        return self.estimator_.predict_proba(X)

    def predict(self, X):
        check_is_fitted(self)
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.feature_names_in_)
        X = self.transform(X)
        return self.estimator_.predict(X)


class ClassifierForMixedDataV2(ClassifierForMixedData):

    def __init__(self, estimator, binning_process_: BinningProcess, cat_columns, num_columns, ord_columns):
        super().__init__(estimator, binning_process_, ord_columns, num_columns)
        # for compatibility with superclass:
        # - super.cat_columns are ordinal features
        # - self.ohe_columns are categorical features
        self.ohe_columns = cat_columns

    def transform(self, X: pd.DataFrame):
        X = X.copy()
        X_num_cat = super().transform(X[self.num_columns + self.cat_columns])
        X_oh = self.OHE_.transform(X[self.ohe_columns])
        X_oh = pd.DataFrame(X_oh, columns=self.OHE_.get_feature_names_out(), index=X.index)
        X_ = [X_num_cat, X_oh]
        X = pd.concat(X_, axis=1)
        return X.astype(float)

    def fit(self, X: pd.DataFrame, y):
        self.OHE_ = OneHotEncoder(sparse_output=False, drop='if_binary').fit(X[self.ohe_columns])
        super().fit(X, y)
        return self


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


class UniversalProbabilityRescaler(BaseEstimator, ClassifierMixin):
    """
    Un wrapper universale che applica lo shift logistico alle probabilità
    di QUALSIASI classificatore pre-addestrato per spostare la soglia operativa.
    """

    def __init__(self, estimator, threshold=0.5):
        self.estimator = estimator
        self.threshold = threshold

    def fit(self, X, y=None):
        # Il modello interno DEVE essere già addestrato
        check_is_fitted(self.estimator)
        self.threshold_ = self.threshold

        if hasattr(self.estimator, 'classes_'):
            self.classes_ = self.estimator.classes_

        return self

    def _warp_logistic(self, prob, thr):
        """Trasformazione logistica valida per qualsiasi array di probabilità"""
        prob = np.asarray(prob)
        # Formula veloce ed efficiente che evita np.log e np.exp
        numerator = prob * (1.0 - thr)
        denominator = numerator + (1.0 - prob) * thr

        warped_prob = np.divide(numerator, denominator,
                                out=np.zeros_like(numerator),
                                where=(denominator != 0))

        warped_prob[(prob == 1.0) & (thr == 1.0)] = 1.0
        return warped_prob

    def predict_proba(self, X):
        check_is_fitted(self)

        # 1. Ottiene le probabilità "grezze" dal modello complesso (es. CatBoost)
        original_probs = self.estimator.predict_proba(X)
        pos_probs = original_probs[:, 1]

        # 2. Applica la "deformazione" matematica
        warped_pos_probs = self._warp_logistic(pos_probs, self.threshold_)

        warped_pos_probs = np.clip(warped_pos_probs, 0.0, 1.0)
        return np.vstack([1.0 - warped_pos_probs, warped_pos_probs]).T

    def predict(self, X):
        check_is_fitted(self)
        warped_probs = self.predict_proba(X)
        return self.classes_[np.argmax(warped_probs, axis=1)]

    def __getattr__(self, name):
        """Delega gli attributi al modello interno (es. feature_importances_)"""
        return getattr(self.estimator, name)
