from explainers.global_explainers.cf_explainer import BaseExplainer
from optbinning import Scorecard
from GLANCE_main.src.glance.glance.glance import GLANCE
from model_wrapper import ModelWrapper
from utils_explainers import prepare_output
from scorecard_one_hot import prepare_data_bin
import pandas as pd
from metrics_gcfes import (compute_metrics_global, compute_metrics_auc_global,
                           LOWER_LIMIT_RANGE_FOR_D, UPPER_LIMIT_RANGE_FOR_D, UPPER_LIMIT_FOR_K)
import time
from adapters import GlanceAdapter
from utils_explainers import compute_bounds
from sklearn.preprocessing import OneHotEncoder

NAME = "glance"
CLUSTERING_METHOD = "KMeans"
# CLUSTERING_METHOD = "Agglomerative"
# CLUSTERING_METHOD = "GMM"
# CF_GENERATOR = "NearestNeighbors"
# CF_GENERATOR = "Dice"
# CF_GENERATOR = "NearestNeighborsScaled"
CF_GENERATOR = "RandomSampling"
CLUSTER_ACTION_CHOICE_ALGO = "max-eff"
# CLUSTER_ACTION_CHOICE_ALGO = "mean-act"
# CLUSTER_ACTION_CHOICE_ALGO = "low-cost"
# CLUSTER_ACTION_CHOICE_ALGO = "min-cost-eff-thres"
# CLUSTER_ACTION_CHOICE_ALGO = "eff-thres-hybrid"


class GlanceExplainer(BaseExplainer):
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
        self.model_bin, X_bins, self.binning_process, y = prepare_data_bin(self.df_train, self.features, self.target, self.model)
        self.cat_features = self.features
        self.cont_features = []
        # define model wrapper
        self.wrapper_model = ModelWrapper(self.model_bin)

        # start timer to measure training efficiency
        start = time.perf_counter()

        # define GLANCE
        self.glance = GLANCE(self.wrapper_model, initial_clusters=10, final_clusters=3, num_local_counterfactuals=5)
        self.glance.fit(X_bins, self.df_train[self.target], pd.concat([X_bins, self.df_train[self.target]], axis=1),
                        self.act_features, self.cont_features, self.cat_features, clustering_method=CLUSTERING_METHOD,
                        cf_generator=CF_GENERATOR, cluster_action_choice_algo=CLUSTER_ACTION_CHOICE_ALGO)
        # self.glance.fit(X_bins, self.df_train[self.target], pd.concat([X_bins, self.df_train[self.target]], axis=1),
        #                 self.act_features, self.cont_features, self.cat_features, clustering_method=CLUSTERING_METHOD,
        #                 cf_generator=CF_GENERATOR, cluster_action_choice_algo=CLUSTER_ACTION_CHOICE_ALGO, nns__n_scalars=50)

        # generate counterfactuals for all factuals
        self.factuals_bin = X_bins[self.model_bin.predict(X_bins) == 0]
        eff, cost, clusters, self.clusters_res, self.chosen_actions, final_costs = self.glance.explain_group(self.factuals_bin)

        # end timer to measure training efficiency
        end = time.perf_counter()

        # compute metrics
        name = self.dataset_name + "_" + NAME
        self.factuals = self.df_train[self.model.predict(self.df_train[self.features]) == 0]
        training_efficiency = end - start
        _, _ = compute_metrics_global(self.df_train, self.factuals, self.features, cat_features, num_features, self.target,
                                      self.model, self._explain, self.binning_process, training_efficiency, name, write=True)

        # compute metrics auc
        OHE = OneHotEncoder(sparse_output=False, drop=None).fit(X_bins)
        adapter = GlanceAdapter(self.clusters_res, self.chosen_actions, self.factuals[self.features].reset_index(drop=True),
                                cat_features, self.num_features, OHE, self.binning_process)
        _, _, _, _, _, _, _, _, _ = compute_metrics_auc_global(adapter, LOWER_LIMIT_RANGE_FOR_D, UPPER_LIMIT_FOR_K,
                                        UPPER_LIMIT_RANGE_FOR_D, name, None, plot=True, write=True)

    def _explain(self, test_item, n_cf=1):
        # unpack test item
        record, label, pred, proba, target = test_item

        # find factual index
        factuals = self.factuals[self.features].reset_index(drop=True)
        mask = (factuals[self.features] == record.values).all(axis=1)
        factual_idx = factuals.index[mask][0]
        # pick correct actions
        chosen_action = self.chosen_actions[factual_idx]
        cfs = record.to_frame().T
        if chosen_action != -1:
            cluster_res = self.clusters_res[list(self.clusters_res)[chosen_action]]
            # build counterfactual
            for col, value in cluster_res['action'].items():
                if value != '-':
                    if col in self.num_features:
                        lb, ub = compute_bounds(value)
                        if lb != '-inf' and ub != 'inf':
                            value = (lb + ub) / 2
                        elif lb == '-inf' and ub != 'inf':
                            value = ub
                        elif ub == 'inf' and lb != '-inf':
                            value = lb
                    else:
                        if value.count("'") > 2:
                            value = value.strip("[]").replace("\n", "").replace(",", "").split("' '")[0].strip("'")
                        else:
                            value = value.strip("[]'")
                    cfs[col] = value

        # prepare the output in the required format
        return prepare_output(self.model, cfs[self.features], test_item)