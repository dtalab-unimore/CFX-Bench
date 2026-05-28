from explainers.global_explainers.cf_explainer import BaseExplainer
from optbinning import Scorecard
from model_wrapper import ModelWrapper
from GLOBE_CE_main.globe_ce import GLOBE_CE
from GLOBE_CE_main.ares import AReS
import matplotlib.pyplot as plt
import numpy as np
from utils_explainers import prepare_data_ares_globece, prepare_output
import pandas as pd
from ares_globece_loader import DatasetLoader
from metrics_gcfes import (compute_metrics_global, compute_metrics_auc_global,
                           LOWER_LIMIT_RANGE_FOR_D, UPPER_LIMIT_RANGE_FOR_D, UPPER_LIMIT_FOR_K)
import time
from adapters import GlobeCeAdapter

NAME = "globe-ce"
SCHEME = "random"
# SCHEME = "features"


class GlobeCeExplainer(BaseExplainer):
    def __init__(self, model: Scorecard, df_train, features, cat_features, num_features, act_features, target, dataset_name, **kwargs):
        self.model = model
        self.df_train = df_train
        self.features = features
        self.cat_features = cat_features
        self.num_features = num_features
        self.act_features = act_features
        self.target = target
        self.dataset_name = dataset_name

        # prepare data
        self.cat_features, _, self.immutables, self.ohe, self.model_ohe, X_oh, self.binning_process, _, n_bins = (
            prepare_data_ares_globece(self.features, self.act_features, self.df_train, self.target, self.model, self.dataset_name))
        dataset = DatasetLoader(self.features, self.cat_features, [], self.df_train,
                                pd.concat([X_oh, self.df_train[self.target]], axis=1))
        self.wrapper_model = ModelWrapper(self.model_ohe)

        # initialise AReS to determine bin widths for costs
        ares = AReS(model=self.wrapper_model, dataset=dataset, X=dataset.data_oh, n_bins=n_bins,
                    dropped_features=self.immutables, normalise=False)
        bin_widths = ares.bin_widths

        # start timer to measure training efficiency
        start = time.perf_counter()

        # initialise Globe-CE
        self.globe_ce = GLOBE_CE(model=self.wrapper_model, dataset=dataset, X=dataset.data_oh, affected_subgroup=None,
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
        self.k_s = np.zeros((n_div, self.df_train.shape[0]))
        for i in range(n_div):
            cor_s, cos_s, self.k_s[i] = self.globe_ce.scale(self.globe_ce.deltas_div[i], disable_tqdm=False, vector=True)
            min_costs[i], self.min_costs_idxs[i] = self.globe_ce.min_scalar_costs(cos_s, return_idxs=True, inf=False)
            max_scalar_idxs = self.globe_ce.cluster_by_costs(cos_s, n_bins=5)
        min_costs = min_costs.min(axis=0)

        # compute best average costs given each input uses its minimum scalar
        costs_bound, corrects_bound = self.globe_ce.accuracy_cost_bounds(min_costs)

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

        # compute metrics
        name = self.dataset_name + "_" + NAME
        self.factuals = self.df_train[self.model.predict(self.df_train[self.features]) == 0]
        training_efficiency = end - start
        _, _ = compute_metrics_global(self.df_train, self.factuals, self.features, cat_features, num_features, self.target,
                                      self.model, self._explain, self.binning_process, training_efficiency, name, write=True)

        # compute metrics auc
        adapter = GlobeCeAdapter(self.globe_ce, self.factuals[self.features].reset_index(drop=True), self.k_s,
                                 self.min_costs_idxs, self.binning_process, self.ohe)
        _, _, _, _, _, _, _, _, _ = compute_metrics_auc_global(adapter, LOWER_LIMIT_RANGE_FOR_D, UPPER_LIMIT_FOR_K,
                                        UPPER_LIMIT_RANGE_FOR_D, name, None, plot=True, write=True)

    def _explain(self, test_item, n_cf=1):
        # unpack test item
        record, label, pred, proba, target = test_item

        # find factual index
        factuals = self.factuals[self.features].reset_index(drop=True)
        mask = (factuals[self.features] == record.values).all(axis=1)
        factual_idx = factuals.index[mask][0]
        # fetch correct scalar
        k_s, delta, scalar_idx = None, None, None
        for i in range(self.globe_ce.deltas_div.shape[0]):
            # pick the best global direction
            delta = self.globe_ce.deltas_div[i]
            # scale direction
            k_s = self.k_s[i]
            # minimal scalar indices for each affected point
            min_idxs = self.min_costs_idxs[i]
            scalar_idx = min_idxs[factual_idx]
            if not np.isnan(scalar_idx):
                break
        if np.isnan(scalar_idx):
            cfs = record.to_frame().T
            # raise RuntimeError("Globe-CE could not find a valid counterfactual.")
        else:
            lambda_coef = k_s[int(scalar_idx)]
            # construct CF
            cfs = (self.globe_ce.x_aff[factual_idx] + (lambda_coef * delta)).reshape(1, -1)
            cfs = pd.DataFrame(cfs, columns=self.ohe.get_feature_names_out())
            cfs = self.wrapper_model.decode(cfs, self.features, self.ohe.get_feature_names_out(),
                                            self.num_features, self.binning_process)

        # prepare the output in the required format
        return prepare_output(self.model, cfs[self.features], test_item)