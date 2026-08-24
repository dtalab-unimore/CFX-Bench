import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from optbinning import Scorecard

from .GLOBE_CE_main.ares import AReS
from .GLOBE_CE_main.globe_ce import GLOBE_CE
from .adapters import GlobeCeAdapter
from explainers.base import BaseExplainer, _empty_explanation_dict
from .utils_explainers import prepare_data_ares_globece, prepare_output, DatasetLoader

NAME = "globe-ce"
SCHEME = "random"
# SCHEME = "features"


class GlobeCeExplainer(BaseExplainer):
    def __init__(self, model: Scorecard, X_train, y_train, features, cat_features, num_features, act_features, target,
                 dataset_name, **kwargs):
        # self.model = model
        # self.df_train = df_train
        # self.features = features
        # self.cat_features = cat_features
        # self.num_features = num_features
        # self.act_features = act_features
        # self.target = target
        self.dataset_name = dataset_name
        super().__init__(
            model, X_train, y_train, features, cat_features, num_features, act_features, target
        )

    def _init(self):
        # prepare data
        self.cat_features, _, self.immutables, self.ohe, self.model_ohe, X_oh, self.binning_process, _, n_bins = (
            prepare_data_ares_globece(self.X_train, self.y_train, self.features, self.act_features, self.model, self.dataset_name))
        dataset = DatasetLoader(X_oh, self.y_train, self.target, self.ohe.feature_names_in_, self.ohe.categories_,
                                name=self.dataset_name)

        # initialise AReS to determine bin widths for costs
        ares = AReS(model=self.model_ohe, dataset=dataset, X=dataset.data_oh, n_bins=n_bins,
                    dropped_features=self.immutables, normalise=False)
        bin_widths = ares.bin_widths

        # start timer to measure training efficiency
        start = time.perf_counter()

        # initialise Globe-CE
        self.globe_ce = GLOBE_CE(model=self.model_ohe, dataset=dataset, X=dataset.data_oh, affected_subgroup=None,
                            dropped_features=self.immutables, ordinal_features=[], delta_init='zeros',
                            normalise=None, bin_widths=bin_widths, monotonicity=None, p=1)
        self.immutables_oh = np.array(self.globe_ce.feature_values)[self.globe_ce.active_idx == False].tolist()
        self.globe_ce.sample(n_sample=10000, magnitude=30, sparsity_power=30, idxs=None, n_features=3, disable_tqdm=False,
                             plot=True, seed=0, scheme=SCHEME, dropped_features=self.immutables_oh)
        delta = self.globe_ce.best_delta
        self.globe_ce.select_n_deltas(n_div=3)

        # scale best delta and compute minimum costs/scalars per input
        n_div = self.globe_ce.deltas_div.shape[0]
        min_costs = np.zeros((n_div, self.globe_ce.x_aff.shape[0]))
        self.min_costs_idxs = np.zeros((n_div, self.globe_ce.x_aff.shape[0]))
        self.k_s = np.zeros((n_div, self.X_train.shape[0]))
        for i in range(n_div):
            cor_s, cos_s, self.k_s[i] = self.globe_ce.scale(self.globe_ce.deltas_div[i], disable_tqdm=False, vector=True)
            min_costs[i], self.min_costs_idxs[i] = self.globe_ce.min_scalar_costs(cos_s, return_idxs=True, inf=False)
            max_scalar_idxs = self.globe_ce.cluster_by_costs(cos_s, n_bins=5)
        min_costs = min_costs.min(axis=0)

        # compute best average costs given each input uses its minimum scalar
        costs_bound, corrects_bound = self.globe_ce.accuracy_cost_bounds(min_costs)

        self.scalars_div = []
        for i in range(n_div):
            max_scalar = max(self.globe_ce.bisection(self.globe_ce.deltas_div[i]), 1)
            self.scalars_div.append(np.linspace(0, max_scalar, 1000))

        # end timer to measure training efficiency
        end = time.perf_counter()

        # plot coverage-cost profile and minimum costs histogram
        fig, ax = plt.subplots(nrows=1, ncols=2, dpi=200)
        fig.set_figwidth(15)
        ax[0].step(costs_bound, corrects_bound, where='post')
        ax[0].set_ylim([-5, 105])
        ax[0].set_title('Accuracy-Cost Profile')
        ax[0].set_xlabel('Average Cost')
        ax[0].set_ylabel('Accuracy (%)')
        ax[1].set_title('Minimum Costs Histogram')
        ax[1].hist(min_costs, bins=100)
        ax[1].set_ylabel('Frequency')
        ax[1].set_xlabel('Cost')
        plt.show()

        self.training_efficiency = end - start
        self.adapter = GlobeCeAdapter(self.globe_ce, None, self.k_s,
                                 self.min_costs_idxs, self.binning_process, self.ohe)

    def _explain(self, test_item, n_cf=1):
        # unpack test item
        record, label, pred, proba, target = test_item

        record_oh = self.ohe.transform(record.to_frame().T)

        delta_used, lambda_coef = None, None
        for i in range(self.globe_ce.deltas_div.shape[0]):
            delta_i = self.globe_ce.deltas_div[i]
            scalars_i = self.scalars_div[i]
            _, cos_s, _ = self.globe_ce.scale(delta_i, scalars=scalars_i, x_aff=record_oh, vector=True,
                                              disable_tqdm=True)
            valid = cos_s[:, 0] != 0
            if valid.any():
                idx = int(np.argmax(valid))
                delta_used, lambda_coef = delta_i, scalars_i[idx]
                break

        if lambda_coef is None:
            return _empty_explanation_dict(test_item)
        else:
            cfs_oh = (record_oh + lambda_coef * delta_used)
            cfs = self.ohe.inverse_transform(cfs_oh)
            cfs = pd.DataFrame(cfs, columns=self.features)

        # prepare the output in the required format
        return prepare_output(self.model, cfs[self.features], test_item)