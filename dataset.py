import json
from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class Dataset:
    """
        Data loader class for classification tasks, handling data ingestion,
        stratified splitting, and subsampling for both single-file and pre-split datasets.

        Parameters
        ----------
        features : list
            List of column names to be used as predictive features.
        target : str
            The name of the target variable column.
        act_features : list, optional
            List of activity or behavioral feature column names.
        demo_features : list, optional
            List of demographic feature column names.
        monotonic_features : dict or list, optional
            Features expected to have a monotonic relationship with the target
            (often used for monotonic binning constraints).
        feature_costs : dict, optional
            Mapping of feature names to their associated acquisition or computational costs.
        id : str, optional
            The name of the identifier column. If provided, this column will be set as the DataFrame index.
        threshold_pos : float, optional
            Threshold value for defining the positive class in continuous targets or probabilities.
        binning_fit_params : dict, optional
            Parameters to be passed directly to the binning algorithm during preprocessing.
        data_path : str, optional
            Path to a single dataset file. If provided, `train_path` and `test_path` are ignored.
        data_sample : int, optional
            Absolute number of samples to draw from the single dataset (via `data_path`)
            before any train/test splitting occurs. Stratified by the target.
        test_split : float, optional
            The proportion of the dataset to include in the test split when using `data_path`.
            If `data_path` is provided but `test_split` is None, the entire dataset is
            copied and used for BOTH train and test sets.
        train_path : str, optional
            Path to the pre-split training dataset file. Used if `data_path` is None.
        test_path : str, optional
            Path to the pre-split testing dataset file. Used if `data_path` is None.
        train_sample : int, optional
            Absolute number of samples to draw from the resulting training set.
            Stratified by the target.
        test_sample : int, optional
            Absolute number of samples to draw from the resulting testing set.
            Stratified by the target.
        random_state : int, optional
            Controls the shuffling applied to the data before applying the splits or subsamples.
            Pass an int for reproducible output across multiple function calls.
        kwargs : dict
            Additional keyword arguments to catch unexpected configuration parameters safely.

        Notes
        -----
        Use Cases:

        1. Single File with Train/Test Split: Provide `data_path` and `test_split`. The data
           will be loaded, optionally subsampled (if `data_sample` is set), and split into
           train and test sets.
        2. Single File for Both Train/Test: Provide `data_path` but leave `test_split` as None.
           The exact same, full dataset will be loaded into both `train_data` and `test_data`.
        3. Pre-split Files: Leave `data_path` as None and provide both `train_path` and
           `test_path`. The files are loaded directly into their respective splits.
    """
    def __init__(self, *, features: list, target: str, act_features: list, demo_features: list=None,
                 monotonic_features: dict=None, ordinal_features: list=None, feature_costs: dict=None, id: str=None,
                 threshold_pos: float=0.5, binning_fit_params: dict=None, data_path: str=None, test_split: float=None,
                 train_path: str=None, test_path: str=None, test_sample: int=None, random_state: int=None, **kwargs):
        self.features = features
        self.target = target
        self.act_features = act_features
        self.demo_features = demo_features if demo_features is not None else []
        self.monotonic_features = monotonic_features if monotonic_features is not None else {}
        self.ordinal_features = ordinal_features or []
        self.feature_costs = feature_costs if feature_costs is not None else {}
        self.id = id
        self.threshold_pos = threshold_pos
        self.binning_fit_params = binning_fit_params  # None is the default argument for BinningProcess
        self.data_path = data_path
        # self.data_sample = data_sample
        self.test_split = test_split
        self.train_path = train_path
        self.test_path = test_path
        # self.train_sample = train_sample
        self.test_sample = test_sample
        self.random_state = random_state
        if kwargs:
            for k, v in kwargs.items():
                setattr(self, k, v)

        data, train_data, test_data = self._read_from_file()
        self._prepare_data(data, train_data, test_data)

        self.X_train = self.train_data[self.features]
        self.X_test = self.test_data[self.features]
        self.num_features = self.X_train.select_dtypes(include=["number"]).columns.tolist()
        self.cat_features = self.X_train.select_dtypes(exclude=["number"]).columns.tolist()
        self.y_train = self.train_data[self.target]
        self.y_test = self.test_data[self.target]

        if not set(self.ordinal_features).issubset(set(self.cat_features)):
            raise ValueError(f"Ordinal features {self.ordinal_features} must be a subset of categorical features {self.cat_features}")

    def _read_from_file(self):
        data, train_data, test_data = None, None, None
        if self.data_path is None:
            # consider train_path and test_path if data_path is not provided
            train_data, test_data = pd.read_csv(self.train_path), pd.read_csv(self.test_path)
        else:
            # consider data_path if provided, ignore train_path and test_path
            data = pd.read_csv(self.data_path)
        return data, train_data, test_data

    def _prepare_data(self, data=None, train_data=None, test_data=None):
        columns = self.features + [self.target]

        if data is None:
            if self.id is not None and self.id in train_data.columns:
                train_data = train_data.set_index(self.id)
                test_data = test_data.set_index(self.id)
            train_data, test_data = train_data[columns].copy(), test_data[columns].copy()

        else:
            if self.id is not None and self.id in data.columns:
                data = data.set_index(self.id)
            # if self.data_sample is not None:
            #     data, _ = train_test_split(data, train_size=self.data_sample,
            #                                random_state=self.random_state, stratify=data[self.target])
            if self.test_split is not None:
                train_data, test_data = train_test_split(data[columns].copy(), test_size=self.test_split,
                                                         random_state=self.random_state, stratify=data[self.target])
            else:
                train_data, test_data = data[columns].copy(), data[columns].copy()

        # if data is None or self.data_sample is None:
        #     if self.train_sample is not None:
        #         train_data, _ = train_test_split(train_data, train_size=self.train_sample,
        #                                          random_state=self.random_state, stratify=train_data[self.target])
        #     if self.test_sample is not None:
        #         test_data, _ = train_test_split(test_data, train_size=self.test_sample,
        #                                         random_state=self.random_state, stratify=test_data[self.target])

        self.train_data, self.test_data = train_data, test_data

    def get_X(self):
        return self.X_train, self.X_test

    def get_y(self):
        return self.y_train, self.y_test

    def get_features(self):
        return self.features

    def get_target(self):
        return self.target

    def get_num_features(self):
        return self.num_features

    def get_cat_features(self, include_ord=True):
        if include_ord:
            return self.cat_features
        return [f for f in self.cat_features if f not in self.ordinal_features]

    def get_ord_features(self):
        return self.ordinal_features

    def get_act_features(self):
        return self.act_features

    def get_demo_features(self):
        return self.demo_features

    def get_monotonic_features(self):
        return self.monotonic_features

    def get_threshold_pos(self):
        return self.threshold_pos

    def get_binning_fit_params(self):
        return self.binning_fit_params

    def get_feature_costs(self):
        return self.feature_costs

    def get_config(self):
        return {
            "features": self.features,
            "target": self.target,
            "act_features": self.act_features,
            "demo_features": self.demo_features,
            "monotonic_features": self.monotonic_features,
            "feature_costs": self.feature_costs,
            "id": self.id,
            "threshold_pos": self.threshold_pos,
            "binning_fit_params": self.binning_fit_params,
            "data_path": self.data_path,
            "test_split": self.test_split,
            "train_path": self.train_path,
            "test_path": self.test_path,
            "test_sample": self.test_sample,
            "random_state": self.random_state,
        }


class GermanCreditDataset(Dataset):
    def __init__(self, random_state=None):
        with open("data/german_credit_config.json", 'r') as f:
            config = json.load(f)
        config = _binning_fit_params_wildcard(config)
        config = _load_user_splits(config)
        super().__init__(random_state=random_state, **config)

    def _prepare_data(self, data=None, train_data=None, test_data=None):
        super()._prepare_data(data, train_data, test_data)
        # invert target
        self.train_data = self.train_data.rename(columns={self.target: "creditability"})
        self.test_data = self.test_data.rename(columns={self.target: "creditability"})
        self.target = "creditability"
        self.train_data[self.target] = 1 - self.train_data[self.target]
        self.test_data[self.target] = 1 - self.test_data[self.target]


class GermanCreditCRIFFullDataset(Dataset):

    _conf_path = "data/german_credit_crif_full_config.json"

    def __init__(self, random_state=None):
        with open(self._conf_path, 'r') as f:
            config = json.load(f)
        config = _binning_fit_params_wildcard(config)
        config = _load_user_splits(config)
        super().__init__(random_state=random_state, **config)

    def _read_from_file(self):
        data = pd.read_excel(self.data_path)
        return data, None, None

    def _prepare_data(self, data=None, train_data=None, test_data=None):
        # rename column with typo
        # self.train_data = self.train_data.rename(columns={"installment_rate": "installment.rate"})
        # self.test_data = self.test_data.rename(columns={"installment_rate": "installment.rate"})
        if data is not None:
            data = data.rename(columns={"installment_rate": "installment.rate"})
        else:
            train_data = train_data.rename(columns={"installment_rate": "installment.rate"})
            test_data = test_data.rename(columns={"installment_rate": "installment.rate"})
        super()._prepare_data(data, train_data, test_data)
        # invert target
        self.train_data[self.target] = 1 - self.train_data[self.target]
        self.test_data[self.target] = 1 - self.test_data[self.target]
        return


def _binning_fit_params_wildcard(config):
    """
    If the dictionary for "binning_fit_params" contains the "*" wildcard, such as::

        {
            "feature1": {
                "param1": "A",
                "param2": 1,
                ...
            },
            "feature2": {
                "param2": 2,
                ...
            },
            ...
            "*": {
                "param1": "B",
                "param3": "XYZ",
                ...
            }
        }

    apply the specified binning fit parameters to all features that don't have
    their own specific values for them. In the example above, "feature2" would have "param1" set to "B" and "param3"
    set to "XYZ" from the wildcard, while "feature1" would keep its own "param1" value of "A" and receive "param3"
    from the wildcard.
    """
    config = deepcopy(config)
    if "binning_fit_params" in config and config["binning_fit_params"] is not None:
        binning_fit_params = config.pop("binning_fit_params")
        if "*" in binning_fit_params:
            wildcard_params = binning_fit_params.pop("*")
            for feature in config["features"]:
                if feature not in binning_fit_params:
                    binning_fit_params[feature] = {}
                for param, value in wildcard_params.items():
                    if param not in binning_fit_params[feature]:
                        binning_fit_params[feature][param] = value
        config["binning_fit_params"] = binning_fit_params
    return config


def _load_user_splits(config):
    config = deepcopy(config)
    if "binning_user_splits_path" in config:
        user_splits_path = config.pop("binning_user_splits_path")
        binning_fit_params_user_splits = load_user_splits(user_splits_path)
        if "binning_fit_params" in config and config["binning_fit_params"] is not None:
            binning_fit_params = config.pop("binning_fit_params")
            # update with user splits, user splits take precedence over default binning params
            for feature, user_splits_dict in binning_fit_params_user_splits.items():
                if feature in binning_fit_params:
                    binning_fit_params[feature].update(user_splits_dict)
                else:
                    binning_fit_params[feature] = user_splits_dict
        else:
            binning_fit_params = binning_fit_params_user_splits
        config["binning_fit_params"] = binning_fit_params
    return config


class GermanCreditCRIFDataset(Dataset):

    _conf_path = "data/german_credit_crif_config.json"

    def __init__(self, conf_path=None, random_state=None):
        if conf_path is not None:
            self._conf_path = conf_path
        with open(self._conf_path, 'r') as f:
            config = json.load(f)
        config = _binning_fit_params_wildcard(config)
        config = _load_user_splits(config)
        super().__init__(random_state=random_state, **config,
                         ordinal_features=['savings.account.and.bonds', 'property', 'present.employment.since',
                                           'housing', 'other.installment.plans']
                         )

    def _read_from_file(self):
        data = pd.read_excel(self.data_path)
        return data, None, None

    def _prepare_data(self, data=None, train_data=None, test_data=None):
        super()._prepare_data(data, train_data, test_data)
        # invert target
        self.train_data[self.target] = 1 - self.train_data[self.target]
        self.test_data[self.target] = 1 - self.test_data[self.target]
        return


class GermanCreditCRIFMonotonicDataset(GermanCreditCRIFDataset):
    _conf_path = "data/german_credit_crif_mt_config.json"


class AdultIncomeDataset(Dataset):

    _all_features = [
        "age", "workclass", "fnlwgt", "education", "education-num",
        "marital-status", "occupation", "relationship", "race", "sex",
        "capital-gain", "capital-loss", "hours-per-week","native-country",
        "income"
    ]

    def __init__(self, random_state=None):
        with open("data/adult_config.json", 'r') as f:
            config = json.load(f)
        config = _binning_fit_params_wildcard(config)
        config = _load_user_splits(config)
        super().__init__(random_state=random_state, **config)

    def _read_from_file(self):
        train_data = pd.read_csv(self.train_path, sep=",", header=None)
        test_data = pd.read_csv(self.test_path, sep=",", header=None, skiprows=1)  # skip first row which is not data
        # drop rows that contain missing values (marked as ' ?' in the dataset)
        train_data = train_data[~train_data.isin([' ?']).any(axis=1)]
        test_data = test_data[~test_data.isin([' ?']).any(axis=1)]
        train_data.columns = self._all_features
        test_data.columns = self._all_features
        train_data[self.target] = (train_data[self.target].values == ' >50K').astype(int)
        test_data[self.target] = (test_data[self.target].values == ' >50K.').astype(int)  # note the '.' at the end of '>50K' in the test set
        return None, train_data, test_data


class LendingClubDataset(Dataset):

    _conf_file = "data/lending_club_config.json"

    def __init__(self, random_state=None):
        with open(self._conf_file, 'r') as f:
            config = json.load(f)
        config = _binning_fit_params_wildcard(config)
        config = _load_user_splits(config)
        super().__init__(random_state=random_state, **config)


class LendingClub2Dataset(LendingClubDataset):
    _conf_file = "data/lending_club_2_config.json"


class LendingClub2MonotonicDataset(LendingClubDataset):
    _conf_file = "data/lending_club_2_mt_config.json"


class LendingClub3Dataset(LendingClubDataset):
    _conf_file = "data/lending_club_3_config.json"


class HELOCDataset(Dataset):
    def __init__(self, random_state=None):
        with open("data/heloc_config.json", 'r') as f:
            config = json.load(f)
        config = _binning_fit_params_wildcard(config)
        config = _load_user_splits(config)
        super().__init__(random_state=random_state, **config)

    def _read_from_file(self):
        data = pd.read_csv(self.data_path)
        data[self.target] = (data[self.target].values == 'Good').astype(int)  # convert target to binary
        # drop special codes
        data = data[~data.isin([-7, -8, -9]).any(axis=1)]
        return data, None, None


class CompasDataset(Dataset):
    def __init__(self, random_state=None):
        with open("data/compas_config.json", 'r') as f:
            config = json.load(f)
        config = _binning_fit_params_wildcard(config)
        config = _load_user_splits(config)
        super().__init__(random_state=random_state, **config)

    def _prepare_data(self, data=None, train_data=None, test_data=None):
        super()._prepare_data(data, None, None)

        def f(data):
            data['days_b_screening_arrest'] = np.abs(data['days_b_screening_arrest'])
            data['c_jail_out'] = pd.to_datetime(data['c_jail_out'])
            data['c_jail_in'] = pd.to_datetime(data['c_jail_in'])
            data['length_of_stay'] = (data['c_jail_out'] - data['c_jail_in']).dt.days
            data['length_of_stay'] = np.abs(data['length_of_stay'])
            data = data.dropna()

            data['days_b_screening_arrest'] = data['days_b_screening_arrest'].astype(int)
            data['length_of_stay'] = data['length_of_stay'].astype(int)
            data['is_recid'] = data['is_recid'].astype(bool)
            data['is_violent_recid'] = data['is_violent_recid'].astype(bool)
            data = data.drop(columns=['c_jail_in', 'c_jail_out'])
            data = data.drop(columns=['is_recid'])  # is_recid raises an error with binning and woe
            return data

        self.train_data = f(self.train_data)
        self.test_data = f(self.test_data)

        self.features.remove('c_jail_in'), self.act_features.remove('c_jail_in')
        self.features.remove('c_jail_out'), self.act_features.remove('c_jail_out')
        self.features.append('length_of_stay'), self.act_features.append('length_of_stay')
        self.features.remove('is_recid'), self.act_features.remove('is_recid')

        return


def get_dataset(dataset_name: str, **kwargs):
    datasets = {
        'german-credit': GermanCreditDataset,
        'german-credit-crif-full': GermanCreditCRIFFullDataset,
        'german-credit-crif': GermanCreditCRIFDataset,
        'german-credit-crif-mt': GermanCreditCRIFMonotonicDataset,
        'adult': AdultIncomeDataset,
        'lending-club': LendingClubDataset,
        'lending-club-2': LendingClub2Dataset,
        'lending-club-2-mt': LendingClub2MonotonicDataset,
        'lending-club-3': LendingClub3Dataset,
        'heloc': HELOCDataset,
        'compas': CompasDataset
    }
    if dataset_name in datasets:
        return datasets[dataset_name](**kwargs)
    else:
        raise ValueError(f'Unknown dataset {dataset_name}')


def load_user_splits(file_path):
    binning_dict = {}
    with open(file_path, 'r') as f:
        binning_dict = json.load(f)
    for feature, user_splits in binning_dict.items():
        binning_dict[feature] = {
            "user_splits": user_splits,
            "user_splits_fixed": [True] * (len(user_splits))
        }
    return binning_dict
