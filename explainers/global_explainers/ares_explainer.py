import time
from copy import deepcopy

import pandas as pd
from optbinning import Scorecard

from explainers.global_explainers.GLOBE_CE_main.ares import AReS
from explainers.global_explainers.adapters import AReSAdapter
from explainers.global_explainers.ares_globece_loader import DatasetLoader
from explainers.global_explainers.cf_explainer import BaseExplainer, _empty_explanation_dict
from explainers.global_explainers.utils_explainers import prepare_data_ares_globece, prepare_output, rule_applies, \
    apply_then

NAME = "ares"


class AresExplainer(BaseExplainer):
    def __init__(self, model: Scorecard, df_train, features, cat_features, num_features, act_features, target,
                 dataset_name, monotonic_features=None, **kwargs):
        self.model = model
        self.df_train = df_train
        self.features = features
        self.cat_features = cat_features
        self.num_features = num_features
        self.act_features = act_features
        self.target = target
        self.dataset_name = dataset_name
        self.monotonic_features = monotonic_features or {}  # ordinal_features implementation is bugged

        # prepare data
        self.cat_features, self.cont_features, self.immutables, self.ohe, self.model_ohe, X_oh, self.binning_process, _, n_bins = (
            prepare_data_ares_globece(self.features, self.act_features, self.df_train, self.target, self.model, self.dataset_name))
        dataset = DatasetLoader(self.features, self.cat_features, self.cont_features, self.df_train,
                                pd.concat([X_oh, self.df_train[self.target]], axis=1))

        # start timer to measure training efficiency
        start = time.perf_counter()

        # initialize AReS
        apriori_threshold = 0.1
        constraints = [50, 3, 10]
        if self.dataset_name == "german-credit":
            apriori_threshold = 0.05
            constraints = [50, 3, 10]
        elif self.dataset_name == "lending":
            apriori_threshold = 0.05
            constraints = [50, 3, 10]
        elif self.dataset_name == "compas":
            apriori_threshold = 0.04
            constraints = [15, 3, 6]
        elif self.dataset_name == "adult":
            apriori_threshold = 0.05
            constraints = [50, 4, 5]
        self.ares = AReS(
            model=self.model_ohe, dataset=dataset, X=dataset.data_oh, dropped_features=self.immutables,
            n_bins=n_bins, ordinal_features=[], normalise=False, constraints=constraints,
        )
        self.ares.generate_itemsets(apriori_threshold=apriori_threshold, max_width=None, affected_subgroup=None, save_copy=True)
        self.ares.generate_groundset(max_width=None, RL_reduction=True, then_generation=None, save_copy=False)
        # set lams = [_, _, _, _] for original AReS
        # lams = [1, 1000]
        lams = [1, 0]
        r = min(5000, self.ares.V.length)
        self.ares.evaluate_groundset(lams=lams, r=r, save_mode=1, disable_tqdm=False, plot_accuracy=False)
        self.ares.select_groundset(s=r)
        self.ares.optimise_groundset(lams=lams, factor=1, print_updates=False, print_terms=False)

        # end timer to measure training efficiency
        end = time.perf_counter()

        # print("Accuracy:" + " {}%".format(round(self.ares.R.accuracy, 2)))
        # print("Average Cost:" + " {}".format(round(self.ares.R.average_cost, 2)))
        # print("\nAccuracy Upper Bound (Evaluated/Sorted Ground Set):" + " {}%".format(round(self.ares.V.accuracy, 2)))
        print("Final Triples Post-Optimisation\n")
        for triple in self.ares.R.triples:
            outer_ifs, inner_ifs, thens = [sorted([j for j in triple[i]]) for i in range(3)]
            print("If" + " {}".format(', '.join(outer_ifs)))
            print("\t  If" + " {}".format(', '.join(inner_ifs)))
            print("\tThen" + " {}\n".format(', '.join(thens)))

        self.training_efficiency = end - start
        self.adapter = AReSAdapter(self.ares, None, self.num_features, self.binning_process, self.ohe)

    def _explain(self, test_item, n_cf=1):
        # unpack test item
        record, label, pred, proba, target = test_item

        cfs, cumulative_cf_final = [], []
        cf, cumulative_cf = record.copy(), record.copy()
        flag = False
        for triple in self.ares.R.triples:
            outer_ifs, inner_ifs, thens = [sorted([j for j in triple[i]]) for i in range(3)]
            # chek if rule applies
            if rule_applies(record, outer_ifs, inner_ifs, self.num_features):
                # apply the rule
                cf = apply_then(record, thens, self.num_features)
                if self.model.predict(cf.to_frame().T)[0] == 1:
                    cfs.append(cf)
                    if len(cfs) >= n_cf:
                        break
                cumulative_cf = apply_then(cumulative_cf, thens, self.num_features)
                if self.model.predict(cumulative_cf.to_frame().T)[0] == 1 and not flag:
                    cumulative_cf_final.append(cumulative_cf)
                    if len(cumulative_cf_final) >= n_cf:
                        flag = True
        if len(cfs) == 0:
            if len(cumulative_cf_final) == 0:
                return _empty_explanation_dict(test_item)
            else:
                cfs = pd.DataFrame(cumulative_cf_final)
        else:
            cfs = pd.DataFrame(cfs)

        # prepare the output in the required format
        return prepare_output(self.model, cfs[self.features], test_item)
