from explainers.global_explainers.utils_explainers import rule_applies, apply_then, ohe_hot_transform
import numpy as np
from tqdm import tqdm
import pandas as pd
from explainers.global_explainers.utils_explainers import compute_bounds

MAXIMUM_COST = 5000


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _batch_bin_and_transform(df, binning_process, ohe):
    """
    Transform a whole DataFrame in a single bin_and_transform call.
    Falls back to the original row-by-row behaviour if the batched call
    fails or returns an unexpected shape.
    """
    try:
        out = ohe_hot_transform(df, ohe)
        arr = np.asarray(out, dtype=float)
        if arr.ndim == 2 and arr.shape[0] == len(df):
            return arr
    except Exception:
        pass
    rows = [
        np.asarray(ohe_hot_transform(row.to_frame().T, ohe),
                   dtype=float).ravel()
        for _, row in df.iterrows()
    ]
    return np.vstack(rows)


def _row_euclidean(a, b):
    """Row-wise euclidean distance between two equally shaped 2D arrays."""
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return np.sqrt(np.einsum('ij,ij->i', diff, diff))


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
    """
    The factual -> counterfactual distance induced by a rule depends only on
    (factual, rule), never on max_d, k or the coverage constraint. All those
    distances are therefore computed once (with batched transforms) and
    cached; run_kd / run_c become cheap numpy lookups over the cache.
    """

    def __init__(self, ares, factuals, num_features, binning_process, ohe):
        self.ares = ares
        self.factuals = factuals
        self.num_features = num_features
        self.binning_process = binning_process
        self.ohe = ohe
        self._cache_token = None

    # -- caching ------------------------------------------------------------

    def _ensure_cache(self):
        token = (id(self.factuals), len(self.factuals))
        if getattr(self, "_cache_token", None) == token:
            return
        self._cache_token = token
        self._rule_dists = {}
        # one-hot encode every factual once (batched)
        self._factuals_oh = _batch_bin_and_transform(
            self.factuals, self.binning_process, self.ohe)

    def _distances_for_rule(self, triple):
        """
        Vector of euclidean distances (in one-hot space) between each factual
        and the counterfactual obtained by applying `triple`. NaN where the
        rule does not apply.
        """
        self._ensure_cache()
        cached = self._rule_dists.get(triple)
        if cached is not None:
            return cached

        outer_ifs, inner_ifs, thens = [sorted([j for j in triple[i]]) for i in range(3)]
        n = len(self.factuals)
        dists = np.full(n, np.nan)
        applies = np.zeros(n, dtype=bool)
        cf_rows = []
        for pos, (_, x) in enumerate(tqdm(self.factuals.iterrows(), total=n,
                                          desc="Caching rule distances", leave=False)):
            if rule_applies(x, outer_ifs, inner_ifs, self.num_features):
                applies[pos] = True
                cf_rows.append(apply_then(x, thens, self.num_features))

        if cf_rows:
            cf_df = pd.DataFrame(cf_rows).reset_index(drop=True)
            cf_oh = _batch_bin_and_transform(cf_df, self.binning_process, self.ohe)
            dists[applies] = _row_euclidean(self._factuals_oh[applies], cf_oh)

        self._rule_dists[triple] = dists
        return dists

    # -- public API ----------------------------------------------------------

    def run_kd(self, max_d, k):
        selected_rules = set(sorted(self.ares.R.triples, key=lambda r: sorted(r))[:k])
        total = len(self.factuals)
        covered_mask = np.zeros(total, dtype=bool)
        for triple in selected_rules:
            d = self._distances_for_rule(triple)
            covered_mask |= (d <= max_d)  # NaN compares False

        coverage = covered_mask.sum() / total
        return {0.0: {"Coverage": coverage * 100}}

    def run_c(self, k, cov):
        selected_rules = set(sorted(self.ares.R.triples, key=lambda r: sorted(r))[:k])
        total = len(self.factuals)

        # Same iteration order as `for triple in selected_rules` in the
        # original implementation: the first applicable rule (in set
        # iteration order) determines the cost for each factual.
        rule_dists = [self._distances_for_rule(t) for t in selected_rules]
        if rule_dists:
            D = np.vstack(rule_dists)                       # (n_rules, n_factuals)
            applicable = ~np.isnan(D)
            has_rule = applicable.any(axis=0)
            first = applicable.argmax(axis=0)
            first_cost = D[first, np.arange(total)]
        else:
            has_rule = np.zeros(total, dtype=bool)
            first_cost = np.zeros(total)

        max_cost_global, covered, total_cost = 0, 0, 0
        for pos in range(total):
            if covered / total >= cov:
                break
            if has_rule[pos]:
                cost = first_cost[pos]
                total_cost += cost
                max_cost_global = np.maximum(max_cost_global, cost)
                covered += 1

        coverage = covered / total
        total_cost = total_cost if total_cost > 0 else MAXIMUM_COST
        results = {0.0: {
            "Coverage": coverage * 100,
            "Total Cost": total_cost
        }}
        return results, max_cost_global if max_cost_global > 0 else MAXIMUM_COST


class GlobeCeAdapter(GlobalCFMethod):
    def __init__(self, globe_ce, factuals, k_s, min_costs_idxs, binning_process, ohe):
        self.globe_ce = globe_ce
        self.factuals = factuals
        self.k_s = k_s
        self.min_costs_idxs = min_costs_idxs
        self.binning_process = binning_process
        self.ohe = ohe
        self._cache_token = None

    # -- caching ------------------------------------------------------------

    def _ensure_cache(self):
        token = (id(self.factuals), len(self.factuals))
        if getattr(self, "_cache_token", None) == token:
            return
        self._cache_token = token
        self._delta_dists = {}
        self._factuals_oh = _batch_bin_and_transform(
            self.factuals, self.binning_process, self.ohe)
        # Replicates the original per-row lookup
        # mask = (self.factuals == x.values).all(axis=1); factuals.index[mask][0]
        # i.e. each row maps to the index label of its first identical row.
        seen = {}
        factual_idx = []
        for idx, row in self.factuals.iterrows():
            key = tuple(row.values.tolist())
            if key not in seen:
                seen[key] = idx
            factual_idx.append(seen[key])
        self._factual_idx = factual_idx

    def _dists_for_delta(self, i):
        """Distance per factual for translation i; NaN where no valid scalar."""
        self._ensure_cache()
        cached = self._delta_dists.get(i)
        if cached is not None:
            return cached

        delta = np.asarray(self.globe_ce.deltas_div[i], dtype=float).ravel()
        k_s = self.k_s[i]
        min_idxs = self.min_costs_idxs[i]
        n = len(self.factuals)
        dists = np.full(n, np.nan)
        for pos, fidx in enumerate(self._factual_idx):
            # scalar_idx = min_idxs[fidx]
            scalar_idx = pos
            if not np.isnan(scalar_idx):
                lambda_coef = k_s[int(scalar_idx)]
                # x_cf_oh = np.asarray(self.globe_ce.x_aff[fidx], dtype=float).ravel() + (lambda_coef * delta)
                x_cf_oh = np.asarray(self.globe_ce.x_aff[pos], dtype=float).ravel() + (lambda_coef * delta)
                diff = self._factuals_oh[pos] - x_cf_oh
                dists[pos] = np.sqrt(np.dot(diff, diff))

        self._delta_dists[i] = dists
        return dists

    # -- public API ----------------------------------------------------------

    def run_kd(self, max_d, k):
        total = len(self.factuals)
        n_deltas = min(k, len(self.globe_ce.deltas_div))
        covered_mask = np.zeros(total, dtype=bool)
        for i in range(n_deltas):
            d = self._dists_for_delta(i)
            covered_mask |= (d <= max_d)  # NaN compares False

        coverage = covered_mask.sum() / total
        return {0.0: {"Coverage": coverage * 100}}

    def run_c(self, k, cov):
        total = len(self.factuals)
        n_deltas = min(k, len(self.globe_ce.deltas_div))

        delta_dists = [self._dists_for_delta(i) for i in range(n_deltas)]
        if delta_dists:
            D = np.vstack(delta_dists)                     # (n_deltas, n_factuals)
            applicable = ~np.isnan(D)
            has_delta = applicable.any(axis=0)
            first = applicable.argmax(axis=0)
            first_cost = D[first, np.arange(total)]
        else:
            has_delta = np.zeros(total, dtype=bool)
            first_cost = np.zeros(total)

        max_cost_global, covered, total_cost = 0, 0, 0
        for pos in range(total):
            if covered / total >= cov:
                break
            if has_delta[pos]:
                cost = first_cost[pos]
                total_cost += cost
                max_cost_global = np.maximum(max_cost_global, cost)
                covered += 1

        coverage = covered / total
        total_cost = total_cost if total_cost > 0 else MAXIMUM_COST
        results = {0.0: {
            "Coverage": coverage * 100,
            "Total Cost": total_cost
        }}
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
        self._cache_token = None

    # -- caching ------------------------------------------------------------

    def _apply_action(self, x, selected_actions, drop_commas):
        """
        Apply a GLANCE action to a factual. The original run_kd additionally
        stripped commas from multi-category strings while run_c did not, so
        both parsing variants are kept (drop_commas=True replicates run_kd,
        False replicates run_c).
        """
        x_cf = x.copy()
        for col, value in selected_actions['action'].items():
            if value != '-':
                x_cf[col] = value
        return x_cf

    def _ensure_cache(self):
        token = (id(self.factuals), len(self.factuals))
        if getattr(self, "_cache_token", None) == token:
            return
        self._cache_token = token

        cluster_keys = list(self.clusters_res)
        n = len(self.factuals)
        chosen = np.full(n, -1, dtype=int)
        cf_kd_rows, cf_c_rows, act_pos = [], [], []

        for pos, (idx, x) in enumerate(tqdm(self.factuals.iterrows(), total=n,
                                            desc="Caching action distances", leave=False)):
            # chosen_action = self.chosen_actions[idx]
            chosen_action = self.chosen_actions[pos]
            chosen[pos] = chosen_action
            if chosen_action != -1:
                selected_actions = self.clusters_res[cluster_keys[chosen_action]]
                cf_kd_rows.append(self._apply_action(x, selected_actions, drop_commas=True))
                cf_c_rows.append(self._apply_action(x, selected_actions, drop_commas=False))
                act_pos.append(pos)

        self._chosen = chosen
        self._dist_kd = np.full(n, np.nan)
        self._dist_c = np.full(n, np.nan)
        if act_pos:
            x_oh = _batch_bin_and_transform(
                self.factuals.iloc[act_pos], self.binning_process, self.ohe)
            cf_kd_oh = _batch_bin_and_transform(
                pd.DataFrame(cf_kd_rows).reset_index(drop=True), self.binning_process, self.ohe)
            cf_c_oh = _batch_bin_and_transform(
                pd.DataFrame(cf_c_rows).reset_index(drop=True), self.binning_process, self.ohe)
            self._dist_kd[act_pos] = _row_euclidean(x_oh, cf_kd_oh)
            self._dist_c[act_pos] = _row_euclidean(x_oh, cf_c_oh)

    # -- public API ----------------------------------------------------------

    def run_kd(self, max_d, k):
        self._ensure_cache()
        total = len(self.factuals)
        valid = (self._chosen != -1) & (self._chosen < k)
        covered = int(np.sum(valid & (self._dist_kd <= max_d)))

        coverage = covered / total
        return {0.0: {"Coverage": coverage * 100}}

    def run_c(self, k, cov):
        self._ensure_cache()
        total = len(self.factuals)

        max_cost_global, covered, total_cost = 0, 0, 0
        for pos in range(total):
            if covered / total >= cov:
                break
            chosen_action = self._chosen[pos]
            if (chosen_action != -1) and (chosen_action < k):
                cost = self._dist_c[pos]
                total_cost += cost
                max_cost_global = np.maximum(max_cost_global, cost)
                covered += 1

        coverage = covered / total
        total_cost = total_cost if total_cost > 0 else MAXIMUM_COST
        results = {0.0: {
            "Coverage": coverage * 100,
            "Total Cost": total_cost
        }}
        return results, max_cost_global if max_cost_global > 0 else MAXIMUM_COST


class LlmAdapter(GlobalCFMethod):
    def __init__(self, factuals, rules, binning_process, ohe):
        self.factuals = factuals
        self.rules = rules
        self.binning_process = binning_process
        self.ohe = ohe
        self._cache_token = None

    # -- caching ------------------------------------------------------------

    def _ensure_cache(self):
        token = (id(self.factuals), len(self.factuals))
        if getattr(self, "_cache_token", None) == token:
            return
        self._cache_token = token
        self._rule_dists = {}
        self._factuals_oh = _batch_bin_and_transform(
            self.factuals, self.binning_process, self.ohe)

    def _dists_for_rule(self, j):
        """Distance per factual for rule j (rules always apply)."""
        self._ensure_cache()
        cached = self._rule_dists.get(j)
        if cached is not None:
            return cached

        rule = self.rules[j]
        n = len(self.factuals)
        cf_rows = []
        for _, x in tqdm(self.factuals.iterrows(), total=n,
                         desc="Caching rule distances", leave=False):
            x_cf = x.copy()
            for col, value in rule.items():
                if value != x[col]:
                    x_cf[col] = value
            cf_rows.append(x_cf)

        cf_oh = _batch_bin_and_transform(
            pd.DataFrame(cf_rows).reset_index(drop=True), self.binning_process, self.ohe)
        dists = _row_euclidean(self._factuals_oh, cf_oh)

        self._rule_dists[j] = dists
        return dists

    # -- public API ----------------------------------------------------------

    def run_kd(self, max_d, k):
        total = len(self.factuals)
        n_rules = min(k, len(self.rules))
        covered_mask = np.zeros(total, dtype=bool)
        for j in range(n_rules):
            covered_mask |= (self._dists_for_rule(j) <= max_d)

        coverage = covered_mask.sum() / total
        return {0.0: {"Coverage": coverage * 100}}

    def run_c(self, k, cov):
        total = len(self.factuals)
        n_rules = min(k, len(self.rules))

        # The original always took the first selected rule for every factual
        # (the inner loop breaks unconditionally after the first rule).
        first_cost = self._dists_for_rule(0) if n_rules > 0 else None

        max_cost_global, covered, total_cost = 0, 0, 0
        for pos in range(total):
            if covered / total >= cov:
                break
            if first_cost is not None:
                cost = first_cost[pos]
                total_cost += cost
                max_cost_global = np.maximum(max_cost_global, cost)
                covered += 1

        coverage = covered / total
        total_cost = total_cost if total_cost > 0 else MAXIMUM_COST
        results = {0.0: {
            "Coverage": coverage * 100,
            "Total Cost": total_cost
        }}
        return results, max_cost_global if max_cost_global > 0 else MAXIMUM_COST
