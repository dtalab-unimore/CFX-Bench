import time

import pandas as pd
from optbinning import Scorecard
from sklearn.preprocessing import OneHotEncoder

from explainers.base import BaseExplainer
from .adapters import GlanceAdapter
from .glance.glance.glance import GLANCE, cumulative
from .utils_explainers import prepare_output

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
        self.target = 'target'
        self.y_train = pd.Series(self.y_train)
        self.y_train.name = self.target

        # prepare data
        self.model_bin, X_bins, self.binning_process, y = self.model, self.X_train.copy(), self.model.binning_process_, self.y_train.copy()
        self.cat_features = self.features
        self.cont_features = []

        # start timer to measure training efficiency
        start = time.perf_counter()

        # define GLANCE
        self.glance = GLANCE(self.model_bin, initial_clusters=10, final_clusters=3, num_local_counterfactuals=5)
        self.glance.fit(X_bins, self.y_train, pd.concat([X_bins, self.y_train], axis=1),
                        self.act_features, self.cont_features, self.cat_features, clustering_method=CLUSTERING_METHOD,
                        cf_generator=CF_GENERATOR, cluster_action_choice_algo=CLUSTER_ACTION_CHOICE_ALGO)
        # generate counterfactuals for all factuals
        self.factuals_bin = X_bins[self.model_bin.predict(X_bins) == 0]
        eff, cost, clusters, self.clusters_res, self.chosen_actions, final_costs = self.glance.explain_group(self.factuals_bin)

        self.global_actions = [self.clusters_res[k]['action'] for k in self.clusters_res]

        # end timer to measure training efficiency
        end = time.perf_counter()

        self.training_efficiency = end - start
        OHE = OneHotEncoder(sparse_output=False, drop=None).fit(X_bins)
        self.adapter = GlanceAdapter(self.clusters_res, self.chosen_actions, None, self.cat_features,
                                     self.num_features, OHE, self.binning_process)

    def _select_action(self, factual):
        if len(self.global_actions) == 0:
            return -1

        # align columns to the action index (apply_action_pandas asserts this)
        factual = factual[self.features].reindex(columns=self.global_actions[0].index)

        _, _, chosen, _ = cumulative(
            self.model_bin,
            factual,
            self.global_actions,
            self.glance.dist_func_dataframe,
            self.glance.numerical_features_names,
            self.glance.categorical_features_names,
            "-",
        )
        return int(chosen[0])

    def _explain(self, test_item, n_cf=1):
        # unpack test item
        record, label, pred, proba, target = test_item

        # the factual as a single-row frame, ordered like the action index
        factual = record.to_frame().T[self.features].reindex(columns=self.global_actions[0].index) \
            if self.global_actions else record.to_frame().T[self.features]

        # select the best frozen action FOR THIS record (works for unseen test rows)
        chosen_action = self._select_action(factual)

        cfs = record.to_frame().T.copy()
        if chosen_action != -1:  # todo: cosa significa? nessun controfattuale?
            cluster_res = self.clusters_res[list(self.clusters_res)[chosen_action]]['action']
            # build counterfactual
            for col, value in cluster_res.items():
                if value != '-':
                    cfs[col] = value

        # prepare the output in the required format
        return prepare_output(self.model, cfs[self.features], test_item)