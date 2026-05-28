import numpy as np
import pandas as pd
from optbinning import Scorecard
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
import re
from sklearn.preprocessing import OneHotEncoder
from copy import deepcopy

from utils import get_binning_maps


def get_woe_map(scorecard: Scorecard):
    woe_map = {}
    binning_table = scorecard.table("detailed")
    binning_table = binning_table.query("Bin not in ['Missing', 'Special']").copy()
    for v, table in binning_table.groupby("Variable"):
        table = table.copy()
        table["Bin"] = table["Bin"].apply(
            lambda bin_: str([str(b_) for b_ in bin_])
            if isinstance(bin_, (list, np.ndarray))
            else bin_
        )
        woe_map[v.item()] = table[['Bin', 'WoE']].set_index("Bin")['WoE'].to_dict()
    return woe_map

def normalize_ohe_bin_label(bin_label: str) -> str:
    if "', '" in bin_label:
        return bin_label
    if bin_label.startswith("[") and bin_label.endswith("]"):
        cleaned = re.sub(r"\s+", " ", bin_label.strip())
        items = re.findall(r"'([^']*)'", cleaned)
        return str(items)
    return bin_label

def prepare_data_ohe(df_train, features, target, model):
    """
    prepare data using binning and one-hot encoding.
    """

    X, y = df_train[features], df_train[target]
    # model: Scorecard = model
    # model.fit(X, y)
    binning_process = model.binning_process_
    # X_bins = binning_process.transform(X, metric="bins")
    # X_bins = X_bins.replace(r"np\.str_\((['\"].*?['\"])\)", r"\1", regex=True)
    X_bins = X
    # X_bins = X.replace('\n', '', regex=True)
    OHE = OneHotEncoder(sparse_output=False, drop=None)
    X_oh = OHE.fit_transform(X_bins)
    X_oh = pd.DataFrame(X_oh, columns=OHE.get_feature_names_out(), index=X_bins.index)
    LR = deepcopy(model.estimator_)
    # woe_map = get_woe_map(model)
    woe_map, _ = get_binning_maps(binning_process)
    model_ohe = ScorecardForOneHotData(base_lr=LR, woe_map=woe_map)
    model_ohe.fit(X_oh)

    return OHE, model_ohe, X_oh, binning_process, y

def prepare_data_bin(df_train, features, target, model):
    """
    prepare data using binning.
    """

    X, y = df_train[features], df_train[target]
    model: Scorecard = model
    model.fit(X, y)
    binning_process = model.binning_process_
    X_bins = binning_process.transform(X, metric="bins")
    X_bins = X_bins.replace(r"np\.str_\((['\"].*?['\"])\)", r"\1", regex=True)
    model_bin = ScorecardForBinnedData(model)
    model_bin.fit(X_bins)

    return model_bin, X_bins, binning_process, y

class ScorecardForOneHotData(BaseEstimator, ClassifierMixin):
    """
    Wraps a fitted Logistic Regression (WoE based) and applies it to One-Hot Encoded data.
    Enforces that exactly one bin is active per feature.
    """
    def __init__(self, base_lr, woe_map, ohe_separator="_"):
        self.base_lr = base_lr
        self.woe_map = woe_map
        self.ohe_separator = ohe_separator

    def fit(self, X, y=None):
        """
        Calculates effective coefficients based on the OHE column names of X.
        X must be a DataFrame or have 'columns' attribute to map features.
        """
        # Validate input has column names (Crucial for mapping OHE -> WoE)
        if not hasattr(X, "columns"):
            raise ValueError("X must be a pandas DataFrame for fitting to parse OHE names.")

        self.feature_names_in_ = np.array(X.columns)
        self.n_features_in_ = len(self.feature_names_in_)
        self.intercept_ = self.base_lr.intercept_

        # 1. Map Base LR Coefficients: FeatureName -> Beta
        # relying on the base_lr having feature_names_in_ (standard in sklearn > 1.0)
        lr_coef_map = dict(zip(self.base_lr.feature_names_in_, self.base_lr.coef_[0]))

        # 2. Calculate Effective Coefficients for OHE columns
        effective_coefs = []
        self.ohe_feature_groups_ = {}  # For validation structure

        for col_idx, col_name in enumerate(self.feature_names_in_):
            # Parse column name: e.g., "ohe__age__(25, 40]"
            # Using the split logic from your snippet
            original_feature, bin_label = col_name.rsplit(self.ohe_separator, 1)

            # Lookup
            if original_feature not in lr_coef_map:
                raise ValueError(f"Feature '{original_feature}' found in OHE columns but not in Base LR.")

            base_beta = lr_coef_map[original_feature]

            # Handle potential string/type mismatches in dictionary lookup
            # woe_map keys are strings, bin_label is string.
            # bin_label = normalize_ohe_bin_label(bin_label)
            try:
                woe_val = self.woe_map[original_feature][bin_label]
            except KeyError:
                raise KeyError(f"Bin '{bin_label}' for feature '{original_feature}' not found in woe_map.")

            # Calculate and Store
            weight = base_beta * woe_val
            effective_coefs.append(weight)

            # Track group for validation
            if original_feature not in self.ohe_feature_groups_:
                self.ohe_feature_groups_[original_feature] = []
            self.ohe_feature_groups_[original_feature].append(col_idx)

        # Store final vector
        self.coef_ = np.array(effective_coefs).reshape(1, -1)
        self.is_fitted_ = True
        return self

    def _validate_ohe_structure(self, X):
        """Ensures exactly one active bin per original feature group."""
        # Convert to numpy if dataframe
        X_arr = X.values if hasattr(X, "values") else X

        for feat, indices in self.ohe_feature_groups_.items():
            # Slice the columns for this feature
            subset = X_arr[:, indices]
            row_sums = subset.sum(axis=1)

            if not np.allclose(row_sums, 1):
                bad_idx = np.where(~np.isclose(row_sums, 1))[0][0]
                raise ValueError(
                    f"Scorecard Structural Error: Feature '{feat}' must have exactly 1 active bin. "
                    f"Row {bad_idx} sum was {row_sums[bad_idx]}."
                )

    def predict_proba(self, X):
        check_is_fitted(self, 'is_fitted_')
        # We allow dataframe or array, but structure check needs array
        self._validate_ohe_structure(X)

        X_arr = X.values if hasattr(X, "values") else X
        linear_pred = X_arr.dot(self.coef_.T) + self.intercept_
        probs = 1 / (1 + np.exp(-linear_pred))
        return np.hstack([1 - probs, probs])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)


class ScorecardForBinnedData(BaseEstimator, ClassifierMixin):

    def __init__(self, scorecard):
        self.scorecard = deepcopy(scorecard)  # already fitted scorecard model

    def _set_woe_map(self):
        self.woe_map_ = {
            col: {
                (k.replace("', '", "' '") if isinstance(k, str) else k): v
                for k, v in mapping.items()
            }
            for col, mapping in get_woe_map(self.scorecard).items()
        }

    def _bin_to_woe(self, s):
        s_ = s.to_dict()
        for k, v in s_.items():
            s_[k] = self.woe_map_[k][v]
        return pd.Series(s_)

    def fit(self, X, y=None):
        check_is_fitted(self.scorecard)
        self._set_woe_map()
        self.feature_names_in_ = self.scorecard.estimator_.feature_names_in_
        return self

    def predict_proba(self, X):
        check_is_fitted(self)
        if isinstance(X, np.ndarray):  # assume two-dimensional array
            X = pd.DataFrame(X, columns=self.feature_names_in_)
        X = X.replace('\n', '', regex=True)
        X_woe = X.replace(self.woe_map_)
        return self.scorecard.estimator_.predict_proba(X_woe)

    def predict(self, X):
        check_is_fitted(self)
        if isinstance(X, np.ndarray):  # assume two-dimensional array
            X = pd.DataFrame(X, columns=self.feature_names_in_)
        X = X.replace('\n', '', regex=True)
        X_woe = X.replace(self.woe_map_)
        return self.scorecard.estimator_.predict(X_woe)

    def table(self, style="summary"):
        return self.scorecard.table(style)

    def __getattr__(self, item):
        return getattr(self.scorecard, item)