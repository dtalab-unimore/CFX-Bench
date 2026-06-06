from explainers.global_explainers.utils_explainers import rule_applies, apply_then, bin_and_transform
from scipy.spatial import distance
import numpy as np
from tqdm import tqdm
import pandas as pd
from explainers.global_explainers.utils_explainers import compute_bounds

MAXIMUM_COST = 5000


class GlobalCFMethod:
    def run_kd(self, max_d, k):
        pass

    def run_c(self, k, cov):
        pass


class FaceGroupAdapter(GlobalCFMethod):
    def __init__(self, facegroup, subgroups, candidate_counterfactuals, factuals_oh, cost_function,
                 distances, k_selection_method, cfe_selection_method, factuals_by_group, alg):
        self.facegroup = facegroup
        self.subgroups = subgroups
        self.candidate_counterfactuals = candidate_counterfactuals
        self.factuals_oh = factuals_oh
        self.cost_function = cost_function
        self.distances = distances
        self.k_selection_method = k_selection_method
        self.cfe_selection_method = cfe_selection_method
        self.factuals_by_group = factuals_by_group
        self.alg = alg

    def run_kd(self, max_d, k):
        gcfes, not_possible_to_cover_fns_group, time_gcfes = self.facegroup.compute_gcfes(self.subgroups,
            self.candidate_counterfactuals, self.factuals_oh, max_d, self.cost_function, k, self.distances,
            self.k_selection_method, verbose=False, cfe_selection_method=self.cfe_selection_method)
        return self.facegroup.apply_cfes(gcfes, self.factuals_by_group, self.distances, not_possible_to_cover_fns_group,
                                        self.k_selection_method, self.cost_function, verbose=False)

    def run_c(self, k, cov):
        _, results, max_cost = self.facegroup.get_gcfes_approach_coverage_constrained_global(self.subgroups,
            self.distances, self.candidate_counterfactuals, self.factuals_oh, k, cov, self.alg)
        return results, max_cost


class AReSAdapter(GlobalCFMethod):
    def __init__(self, ares, factuals, num_features, binning_process, ohe):
        self.ares = ares
        self.factuals = factuals
        self.num_features = num_features
        self.binning_process = binning_process
        self.ohe = ohe

    def run_kd(self, max_d, k):
        selected_rules = set(sorted(self.ares.R.triples, key=lambda r: sorted(r))[:k])
        results = {}
        covered = 0
        total = len(self.factuals)
        for _, x in tqdm(self.factuals.iterrows(), total=total):
            recourse_found = False
            for triple in selected_rules:
                outer_ifs, inner_ifs, thens = [sorted([j for j in triple[i]]) for i in range(3)]
                if rule_applies(x, outer_ifs, inner_ifs, self.num_features):
                    x_cf = apply_then(x, thens, self.num_features)
                    x_oh = bin_and_transform(x.to_frame().T, self.binning_process, self.ohe)
                    x_cf_oh = bin_and_transform(x_cf.to_frame().T, self.binning_process, self.ohe)
                    d = distance.cdist(x_oh.to_numpy(), x_cf_oh.to_numpy(), 'euclidean')[0][0]
                    if d <= max_d:
                        recourse_found = True
                        break

            if recourse_found:
                covered += 1

        coverage = covered / total
        results[0.0] = {
            "Coverage": coverage * 100,
        }
        return results

    def run_c(self, k, cov):
        selected_rules = set(sorted(self.ares.R.triples, key=lambda r: sorted(r))[:k])
        results = {}
        max_cost_global, covered, total_cost = 0, 0, 0
        total = len(self.factuals)
        for _, x in tqdm(self.factuals.iterrows(), total=total):
            if covered / total >= cov:
                break
            for triple in selected_rules:
                outer_ifs, inner_ifs, thens = [sorted([j for j in triple[i]]) for i in range(3)]
                if rule_applies(x, outer_ifs, inner_ifs, self.num_features):
                    x_cf = apply_then(x, thens, self.num_features)
                    x_oh = bin_and_transform(x.to_frame().T, self.binning_process, self.ohe)
                    x_cf_oh = bin_and_transform(x_cf.to_frame().T, self.binning_process, self.ohe)
                    cost = distance.cdist(x_oh.to_numpy(), x_cf_oh.to_numpy(), 'euclidean')[0][0]
                    total_cost += cost
                    max_cost_global = np.maximum(max_cost_global, cost)
                    covered += 1
                    break

        coverage = covered / total
        total_cost = total_cost if total_cost > 0 else MAXIMUM_COST
        results[0.0] = {
            "Coverage": coverage * 100,
            "Total Cost": total_cost
        }
        return results, max_cost_global if max_cost_global > 0 else MAXIMUM_COST


class GlobeCeAdapter(GlobalCFMethod):
    def __init__(self, globe_ce, factuals, k_s, min_costs_idxs, binning_process, ohe):
        self.globe_ce = globe_ce
        self.factuals = factuals
        self.k_s = k_s
        self.min_costs_idxs = min_costs_idxs
        self.binning_process = binning_process
        self.ohe = ohe

    def run_kd(self, max_d, k):
        selected_deltas = self.globe_ce.deltas_div[:k]
        results = {}
        covered = 0
        total = len(self.factuals)
        for _, x in tqdm(self.factuals.iterrows(), total=total):
            recourse_found = False
            mask = (self.factuals == x.values).all(axis=1)
            factual_idx = self.factuals.index[mask][0]
            for i, delta in enumerate(selected_deltas):
                k_s = self.k_s[i]
                min_idxs = self.min_costs_idxs[i]
                scalar_idx = min_idxs[factual_idx]
                if not np.isnan(scalar_idx):
                    lambda_coef = k_s[int(scalar_idx)]
                    x_cf_oh = (self.globe_ce.x_aff[factual_idx] + (lambda_coef * delta)).reshape(1, -1)
                    x_cf_oh = pd.DataFrame(x_cf_oh, columns=self.ohe.get_feature_names_out())
                    x_oh = bin_and_transform(x.to_frame().T, self.binning_process, self.ohe)
                    d = distance.cdist(x_oh.to_numpy(), x_cf_oh.to_numpy(), 'euclidean')[0][0]
                    if d <= max_d:
                        recourse_found = True
                        break

            if recourse_found:
                covered += 1

        coverage = covered / total
        results[0.0] = {
            "Coverage": coverage * 100,
        }
        return results

    def run_c(self, k, cov):
        selected_deltas = self.globe_ce.deltas_div[:k]
        results = {}
        max_cost_global, covered, total_cost = 0, 0, 0
        total = len(self.factuals)
        for _, x in tqdm(self.factuals.iterrows(), total=total):
            if covered / total >= cov:
                break
            mask = (self.factuals == x.values).all(axis=1)
            factual_idx = self.factuals.index[mask][0]
            for i, delta in enumerate(selected_deltas):
                k_s = self.k_s[i]
                min_idxs = self.min_costs_idxs[i]
                scalar_idx = min_idxs[factual_idx]
                if not np.isnan(scalar_idx):
                    lambda_coef = k_s[int(scalar_idx)]
                    x_cf_oh = (self.globe_ce.x_aff[factual_idx] + (lambda_coef * delta)).reshape(1, -1)
                    x_cf_oh = pd.DataFrame(x_cf_oh, columns=self.ohe.get_feature_names_out())
                    x_oh = bin_and_transform(x.to_frame().T, self.binning_process, self.ohe)
                    cost = distance.cdist(x_oh.to_numpy(), x_cf_oh.to_numpy(), 'euclidean')[0][0]
                    total_cost += cost
                    max_cost_global = np.maximum(max_cost_global, cost)
                    covered += 1
                    break

        coverage = covered / total
        total_cost = total_cost if total_cost > 0 else MAXIMUM_COST
        results[0.0] = {
            "Coverage": coverage * 100,
            "Total Cost": total_cost
        }
        return results, max_cost_global if max_cost_global > 0 else MAXIMUM_COST


class GlanceAdapter(GlobalCFMethod):
    def __init__(self, clusters_res, chosen_actions, factuals, cat_features, num_features, ohe, binning_process):
        self.clusters_res = clusters_res
        self.chosen_actions = chosen_actions
        self.factuals = factuals
        self.cat_features = cat_features
        self.num_features = num_features
        self.ohe = ohe
        self.binning_process = binning_process

    def run_kd(self, max_d, k):
        results = {}
        covered = 0
        total = len(self.factuals)
        for idx, x in tqdm(self.factuals.iterrows(), total=total):
            chosen_action = self.chosen_actions[idx]
            if (chosen_action != -1) and (chosen_action < k):
                selected_actions = self.clusters_res[list(self.clusters_res)[chosen_action]]
                x, x_cf = x.to_frame().T, x.to_frame().T
                for col, value in selected_actions['action'].items():
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
                        x_cf[col] = value
                x_oh = bin_and_transform(x, self.binning_process, self.ohe)
                x_cf_oh = bin_and_transform(x_cf, self.binning_process, self.ohe)
                d = distance.cdist(x_oh.to_numpy(), x_cf_oh.to_numpy(), 'euclidean')[0][0]
                if d <= max_d:
                    covered +=1

        coverage = covered / total
        results[0.0] = {
            "Coverage": coverage * 100,
        }
        return results

    def run_c(self, k, cov):
        results = {}
        max_cost_global, covered, total_cost = 0, 0, 0
        total = len(self.factuals)
        for idx, x in tqdm(self.factuals.iterrows(), total=total):
            if covered / total >= cov:
                break
            chosen_action = self.chosen_actions[idx]
            if (chosen_action != -1) and (chosen_action < k):
                selected_actions = self.clusters_res[list(self.clusters_res)[chosen_action]]
                x, x_cf = x.to_frame().T, x.to_frame().T
                for col, value in selected_actions['action'].items():
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
                                value = value.strip("[]").replace("\n", "").split("' '")[0].strip("'")
                            else:
                                value = value.strip("[]'")
                        x_cf[col] = value
                x_oh = bin_and_transform(x, self.binning_process, self.ohe)
                x_cf_oh = bin_and_transform(x_cf, self.binning_process, self.ohe)
                cost = distance.cdist(x_oh.to_numpy(), x_cf_oh.to_numpy(), 'euclidean')[0][0]
                total_cost += cost
                max_cost_global = np.maximum(max_cost_global, cost)
                covered += 1

        coverage = covered / total
        total_cost = total_cost if total_cost > 0 else MAXIMUM_COST
        results[0.0] = {
            "Coverage": coverage * 100,
            "Total Cost": total_cost
        }
        return results, max_cost_global if max_cost_global > 0 else MAXIMUM_COST


class LlmAdapter(GlobalCFMethod):
    def __init__(self, factuals, rules, binning_process, ohe):
        self.factuals = factuals
        self.rules = rules
        self.binning_process = binning_process
        self.ohe = ohe

    def run_kd(self, max_d, k):
        selected_rules = self.rules[:k]
        results = {}
        covered = 0
        total = len(self.factuals)
        for _, x in tqdm(self.factuals.iterrows(), total=total):
            recourse_found = False
            for rule in selected_rules:
                x_cf = x.copy()
                for col, value in rule.items():
                    if value != x[col]:
                        x_cf[col] = value
                x_oh = bin_and_transform(x.to_frame().T, self.binning_process, self.ohe)
                x_cf_oh = bin_and_transform(x_cf.to_frame().T, self.binning_process, self.ohe)
                d = distance.cdist(x_oh.to_numpy(), x_cf_oh.to_numpy(), 'euclidean')[0][0]
                if d <= max_d:
                    recourse_found = True
                    break

            if recourse_found:
                covered += 1

        coverage = covered / total
        results[0.0] = {
            "Coverage": coverage * 100,
        }
        return results

    def run_c(self, k, cov):
        selected_rules = self.rules[:k]
        results = {}
        max_cost_global, covered, total_cost = 0, 0, 0
        total = len(self.factuals)
        for _, x in tqdm(self.factuals.iterrows(), total=total):
            if covered / total >= cov:
                break
            for rule in selected_rules:
                x_cf = x.copy()
                for col, value in rule.items():
                    if value != x[col]:
                        x_cf[col] = value
                x_oh = bin_and_transform(x.to_frame().T, self.binning_process, self.ohe)
                x_cf_oh = bin_and_transform(x_cf.to_frame().T, self.binning_process, self.ohe)
                cost = distance.cdist(x_oh.to_numpy(), x_cf_oh.to_numpy(), 'euclidean')[0][0]
                total_cost += cost
                max_cost_global = np.maximum(max_cost_global, cost)
                covered += 1
                break

        coverage = covered / total
        total_cost = total_cost if total_cost > 0 else MAXIMUM_COST
        results[0.0] = {
            "Coverage": coverage * 100,
            "Total Cost": total_cost
        }
        return results, max_cost_global if max_cost_global > 0 else MAXIMUM_COST
