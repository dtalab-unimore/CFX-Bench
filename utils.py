import re

import numpy as np
import pandas as pd
from optbinning import BinningProcess
from optbinning.binning.binning_statistics import bin_str_format
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder, FunctionTransformer, QuantileTransformer, OneHotEncoder
from sklearn.utils.validation import check_is_fitted


from typing import TypeVar
T = TypeVar('T', str, list, pd.Series, pd.DataFrame)
def clean_numpy2_strings(X: T) -> T:
    to_replace_regex = r"np\.str_\((['\"].*?['\"])\)"
    if isinstance(X, str):
        return re.sub(to_replace_regex, r"\1", X)
    if isinstance(X, list):
        return [re.sub(to_replace_regex, r"\1", x) for x in X]
    if isinstance(X, pd.Series) or isinstance(X, pd.DataFrame):
        return X.replace(to_replace_regex, r"\1", regex=True)
    raise TypeError(f"Unsupported type: {type(X)}")


class MidpointECDFTransformer(BaseEstimator, TransformerMixin):
    """
    Transforms ordinal features into continuous values based on the midpoint
    of their Empirical Cumulative Distribution Function (ECDF).
    Assumes ALL columns passed to fit() and transform() are ordinal.
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        # 1. Capture feature names if a pandas DataFrame is provided
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.array(X.columns)

        X_df = pd.DataFrame(X)
        self.mappings_ = {}

        # 2. Compute the ECDF mappings
        for col in X_df.columns:
            proportions = X_df[col].value_counts(normalize=True).sort_index()
            cdf_prev = proportions.cumsum().shift(1).fillna(0.0)
            ecdf_midpoints = cdf_prev + (proportions / 2.0)
            self.mappings_[col] = ecdf_midpoints.to_dict()

        self.is_fitted_ = True
        return self

    def transform(self, X):
        check_is_fitted(self, 'is_fitted_')
        X_df = pd.DataFrame(X)

        if len(X_df.columns) != len(self.mappings_):
            raise ValueError(
                f"X has {len(X_df.columns)} features, but ECDFEncoder "
                f"is expecting {len(self.mappings_)} features."
            )

        X_transformed = X_df.copy()

        # 3. Apply mappings by column position
        for col_idx, col in enumerate(X_df.columns):
            fitted_col_key = list(self.mappings_.keys())[col_idx]
            mapping = self.mappings_[fitted_col_key]

            X_transformed[col] = X_transformed[col].map(mapping)

        if isinstance(X, pd.DataFrame):
            return X_transformed
        return X_transformed.to_numpy()


def _transform_bin(bin_):
    if isinstance(bin_, list):
        return str([str(b_) for b_ in bin_])
    if isinstance(bin_, np.ndarray):
        # return transform_bin(bin_.tolist())
        return str(bin_)
    return bin_


# scorecard.binning_process_.get_binned_variable("property").binning_table.build(add_totals=False)
def get_binning_maps(binning_process: BinningProcess):
    bin_woe_map, bin_id_map = {}, {}
    for v in binning_process.variable_names:
        table = binning_process.get_binned_variable(v).binning_table.build(add_totals=False)
        table = table.query("Bin not in ['Missing', 'Special']").copy()
        table["Bin"] = table["Bin"].apply(_transform_bin)
        bin_woe_map[v] = table[['Bin', 'WoE']].set_index("Bin")['WoE'].to_dict()
        bin_id_map[v] = table[['Bin']].reset_index().set_index("Bin")['index'].to_dict()
    return bin_woe_map, bin_id_map


def OrdinalBinsEncoder(features, monotonic_features, binning_process: BinningProcess):
    categories = []
    for feature in features:
        if feature in monotonic_features:
            # for monotonic features, use the order specified in monotonic_features
            categories.append(monotonic_features[feature])
        else:
            # for non-monotonic features, use the order provided by optimal binning
            # it doesn't matter if ascending or descending, distance is symmetric
            binned_variable = binning_process.get_binned_variable(feature)
            dtype, splits = binned_variable.dtype, binned_variable.splits
            if dtype == 'numerical':
                splits = bin_str_format(np.concatenate([[-np.inf], splits, [np.inf]]), 2)
            else:  # categorical
                # splits = [str(split_) for split_ in splits]
                splits = [clean_numpy2_strings(str(split_)) for split_ in splits]
                # splits = [str([x if type(x)==str else x.item() for x in split_]) for split_ in splits]  # Numpy 2
            categories.append(splits)
    ordenc = OrdinalEncoder(categories=categories)
    return ordenc


def OrdinalBinsMinMaxEncoder(X_fit, cat_features, num_features, monotonic_features, binning_process):
    """Ordinal bins for categorical features, percentiles for numeric features"""
    cat_tf = OrdinalBinsEncoder(cat_features, monotonic_features, binning_process)
    num_tf = QuantileTransformer(n_quantiles=100)

    cat_tf.fit(X_fit[cat_features])
    num_tf.fit(X_fit[num_features])

    def _transform(X: pd.DataFrame):
        X = X.copy()
        X.loc[:, cat_features] = cat_tf.transform(X[cat_features])
        X.loc[:, num_features] = num_tf.transform(X[num_features])
        return X

    def _inverse_transform(X: pd.DataFrame):
        X = X.copy()
        X.loc[:, cat_features] = cat_tf.inverse_transform(X[cat_features])
        X.loc[:, num_features] = num_tf.inverse_transform(X[num_features])
        return X

    ft = FunctionTransformer(_transform, _inverse_transform, feature_names_out=(lambda a, b: X_fit.columns))
    return ft


def OHOrdinalBinsMixMaxEncoder(X_fit, cat_features, ord_features, num_features, monotonic_features, binning_process):
    X_fit = X_fit.copy()

    cat_tf = OneHotEncoder(sparse_output=False)
    ord_tf = OrdinalBinsEncoder(ord_features, monotonic_features, binning_process)
    num_tf = QuantileTransformer(n_quantiles=100)

    cat_tf.fit(X_fit[cat_features])
    ord_tf.fit(X_fit[ord_features])
    num_tf.fit(X_fit[num_features])

    feature_names_in_ = X_fit.columns.tolist()
    feature_names_out = num_features + ord_features + cat_tf.get_feature_names_out().tolist()

    def _transform(X: pd.DataFrame):
        # global cat_features, ord_features, num_features, cat_tf, ord_tf, num_tf
        X = X.copy()
        X_num = num_tf.transform(X[num_features])
        X_ord = ord_tf.transform(X[ord_features])
        X_cat = cat_tf.transform(X[cat_features])
        X_ = np.concatenate([X_num, X_ord, X_cat], axis=1)
        X_ = pd.DataFrame(X_, columns=feature_names_out)
        return X_.astype(float)

    def _inverse_transform(X: pd.DataFrame):
        # global cat_features, ord_features, num_features, cat_tf, ord_tf, num_tf
        X = X.copy()
        X_num = num_tf.inverse_transform(X[num_features])
        X_ord = ord_tf.inverse_transform(X[ord_features])
        X_cat = cat_tf.inverse_transform(X[cat_tf.get_feature_names_out()])
        X_ = np.concatenate([X_num, X_ord, X_cat], axis=1)
        X_ = pd.DataFrame(X_, columns=(num_features + ord_features + cat_features))
        return X_[feature_names_in_]

    ft = FunctionTransformer(_transform, _inverse_transform, feature_names_out=(lambda a, b: feature_names_out))
    return ft


def conf_to_str(conf):
    dataset = conf['dataset']
    model = conf['model_name']
    explainer = conf['explainer_name']
    test_case = conf['test_case_sel_method']
    seed = conf['seed']

    if explainer == 'dice':
        dice_solver = conf['dice_solver']
        explainer = f'{explainer}/{dice_solver}'

    conf_key = f'{dataset}/{explainer}/{model}__{test_case}__s{seed}'
    return conf_key
