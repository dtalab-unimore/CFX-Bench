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


class _AReSAdapter(GlobalCFMethod):
    def __init__(self, ares, factuals, num_features, binning_process, ohe):
        self.ares = ares
        self.factuals = factuals
        self.num_features = num_features
        self.binning_process = binning_process
        self.ohe = ohe

        # Sort the rule set once. run_kd / run_c are called ~200 times by the
        # AUC sweeps with different (k, d, cov) values, but the underlying rule
        # applicability and counterfactual distances do not depend on any of
        # those arguments — so we materialise them once here.
        self.sorted_rules = sorted(self.ares.R.triples, key=lambda r: sorted(r))
        self._precompute()

    def _precompute(self):
        """Build two (n_factuals, n_rules) matrices used by every later call:

        - self.applies[i, j]   : True iff rule j's outer/inner-if matches factual i
        - self.distances[i, j] : euclidean distance in one-hot space between
                                 factual i and the counterfactual obtained by
                                 applying rule j's "then" to it. Set to +inf
                                 where the rule does not apply (so any
                                 `distances <= d` comparison is False there).

        Rules are processed in `self.sorted_rules` order, so the top-k slice
        `[:, :k]` matches the top-k selection logic of the original code.
        """
        n_factuals = len(self.factuals)
        n_rules = len(self.sorted_rules)

        self.applies = np.zeros((n_factuals, n_rules), dtype=bool)
        self.distances = np.full((n_factuals, n_rules), np.inf, dtype=float)

        if n_rules == 0 or n_factuals == 0:
            return

        # Encode all factuals once instead of once per (factual, rule) pair.
        factuals_oh_np = bin_and_transform(
            self.factuals, self.binning_process, self.ohe).to_numpy()

        # Materialise rows once so the per-rule loop doesn't repeatedly pay
        # iterrows() overhead.
        factual_series = [x for _, x in self.factuals.iterrows()]
        factual_columns = self.factuals.columns

        for j, triple in enumerate(tqdm(self.sorted_rules,
                                        desc="Precomputing AReS rule distances")):
            outer_ifs = sorted(list(triple[0]))
            inner_ifs = sorted(list(triple[1]))
            thens = sorted(list(triple[2]))

            cf_rows = []
            cf_indices = []
            for i, x in enumerate(factual_series):
                if rule_applies(x, outer_ifs, inner_ifs, self.num_features):
                    self.applies[i, j] = True
                    cf_rows.append(apply_then(x, thens, self.num_features))
                    cf_indices.append(i)

            if not cf_indices:
                continue

            # Batch-encode the counterfactuals for this rule and compute the
            # paired euclidean distances row-by-row (NOT a full cdist matrix).
            cfs_df = pd.DataFrame(cf_rows).reset_index(drop=True)
            cfs_df = cfs_df[factual_columns]  # guard against column reordering
            cfs_oh_np = bin_and_transform(
                cfs_df, self.binning_process, self.ohe).to_numpy()
            factuals_subset = factuals_oh_np[cf_indices]
            dists = np.linalg.norm(factuals_subset - cfs_oh_np, axis=1)
            self.distances[cf_indices, j] = dists

    def run_kd(self, max_d, k):
        # Top-k is just a column slice of the precomputed matrices.
        k_eff = min(k, self.applies.shape[1])
        valid = self.applies[:, :k_eff] & (self.distances[:, :k_eff] <= max_d)
        total = len(self.factuals)
        covered = int(valid.any(axis=1).sum()) if total else 0
        coverage = covered / total if total else 0.0
        return {0.0: {"Coverage": coverage * 100}}

    def run_c(self, k, cov):
        k_eff = min(k, self.applies.shape[1])
        applies_k = self.applies[:, :k_eff]
        distances_k = self.distances[:, :k_eff]

        # For each factual, locate the first applicable rule among the top-k.
        # argmax on a bool array returns the index of the first True; we mask
        # rows where nothing applies via `any_applies`.
        any_applies = applies_k.any(axis=1)
        first_idx = applies_k.argmax(axis=1)

        total = len(self.factuals)
        covered = 0
        total_cost = 0.0
        max_cost_global = 0.0

        # Walk factuals in row order to preserve the original early-stop
        # behaviour of breaking once the coverage target is met.
        for i in range(total):
            if total > 0 and covered / total >= cov:
                break
            if any_applies[i]:
                cost = float(distances_k[i, first_idx[i]])
                total_cost += cost
                if cost > max_cost_global:
                    max_cost_global = cost
                covered += 1

        coverage = covered / total if total else 0.0
        total_cost = total_cost if total_cost > 0 else MAXIMUM_COST
        return (
            {0.0: {"Coverage": coverage * 100, "Total Cost": total_cost}},
            max_cost_global if max_cost_global > 0 else MAXIMUM_COST,
        )


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


class LlmAgentsAdapter(GlobalCFMethod):
    def __init__(self, factuals, rules, clusters, binning_process, ohe):
        self.factuals = factuals
        self.rules = rules
        self.clusters = clusters
        self.binning_process = binning_process
        self.ohe = ohe

    def run_kd(self, max_d, k):
        results = {}
        covered = 0
        total = len(self.factuals)
        for idx, x in tqdm(self.factuals.iterrows(), total=total):
            recourse_found = False
            selected_cluster = self.clusters[idx]
            if selected_cluster < k:
                selected_rules = self.rules[selected_cluster]
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
        results = {}
        max_cost_global, covered, total_cost = 0, 0, 0
        total = len(self.factuals)
        for idx, x in tqdm(self.factuals.iterrows(), total=total):
            if covered / total >= cov:
                break
            selected_cluster = self.clusters[idx]
            if selected_cluster < k:
                selected_rules = self.rules[selected_cluster]
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