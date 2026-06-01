import numpy as np
import pandas as pd
from optbinning import Scorecard
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
import re
from sklearn.preprocessing import OneHotEncoder
from copy import deepcopy

from aux_models import OneHotDataClassifierAdapter
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
    # ohe_sep = '§'
    ohe_sep = '_'
    X, y = df_train[features], df_train[target]
    OHE = OneHotEncoder(sparse_output=False, drop=None, feature_name_combiner=lambda a, b: f"{a}{ohe_sep}{b}")  # feature_name_combiner?
    X_oh = OHE.fit_transform(X, y)
    X_oh = pd.DataFrame(X_oh, columns=OHE.get_feature_names_out(), index=X.index)
    model_oh = OneHotDataClassifierAdapter(model, OHE.get_feature_names_out(), ohe_sep)
    model_oh.fit(X_oh)
    bp = model.binning_process_  # for backward compatibility

    return OHE, model_oh, X_oh, bp, y


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