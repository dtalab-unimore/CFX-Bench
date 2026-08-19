import re
from typing import TypeVar

import numpy as np
import pandas as pd
from optbinning import BinningProcess
from optbinning.binning.binning_statistics import bin_str_format
from sklearn.preprocessing import OrdinalEncoder

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


def _transform_bin(bin_):
    if isinstance(bin_, list):
        return str([str(b_) for b_ in bin_])
    if isinstance(bin_, np.ndarray):
        # return transform_bin(bin_.tolist())
        return str(bin_)
    return bin_


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


def conf_to_str(conf):
    dataset = conf['dataset']
    model = conf['model_name']
    explainer = conf['explainer_name']
    test_case = conf['test_case']
    seed = conf['seed']

    if explainer == 'dice':
        dice_solver = conf['dice_solver']
        explainer = f'{explainer}/{dice_solver}'

    conf_key = f'{dataset}/{explainer}/{model}__{test_case}__s{seed}'
    return conf_key
