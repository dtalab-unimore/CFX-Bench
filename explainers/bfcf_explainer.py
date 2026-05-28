import itertools
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from optbinning import BinningProcess, OptimalBinning
from optbinning.binning.binning_statistics import bin_str_format

from explainers.base import BaseExplainer, prepare_output


# ======================================================================
# Explainer
# ======================================================================

class BruteForceCounterfactualExplainer(BaseExplainer):
    # Convenience: string name -> factory that builds the strategy from kwargs.
    # Add a method by writing a CounterfactualMethod subclass and one entry here.
    _BY_NAME = {
        "optimal": lambda kw: OptimalMethod(),
        "expert": lambda kw: ExpertMethod(kw.get("efforts")),
    }

    def __init__(self, model, X_train, features, act_features,
                 monotonic_features=None, method="optimal", **kwargs):
        if isinstance(method, str):
            if method not in self._BY_NAME:
                raise ValueError(
                    f"Unknown method '{method}'. Available: {sorted(self._BY_NAME)}."
                )
            method = self._BY_NAME[method](kwargs)
        elif not isinstance(method, CounterfactualMethod):
            raise TypeError("method must be a name or a CounterfactualMethod instance.")
        self.method = method
        super().__init__(
            model, X_train, None, features, features, None, act_features, None, monotonic_features
        )
        # bind after init so act_features / model / X_train are available to the strategy
        self.method.bind(self)

    def _init(self):
        categories = {}
        for feature in self.act_features:
            if feature in self.monotonic_features:
                categories[feature] = self.monotonic_features[feature]
            else:
                # for non-monotonic features, use the order provided by optimal binning
                # it doesn't matter if ascending or descending, distance is symmetric
                binning_process: BinningProcess = self.model.binning_process_
                binned_variable: OptimalBinning = binning_process.get_binned_variable(feature)
                dtype, splits = binned_variable.dtype, binned_variable.splits
                if dtype == 'numerical':
                    splits = bin_str_format(np.concatenate([[-np.inf], splits, [np.inf]]), 2)
                else:  # categorical
                    splits = [str(split_) for split_ in splits]
                categories[feature] = splits
        self.categories = categories

    def _explain(self, test_item, n_cf=1):
        record, label, pred, proba, target = test_item
        cfs, list_new_probs = self._find_counterfactuals(record, n_cf=n_cf, target=target)
        expl_dict = prepare_output(record, label, pred, proba, target, cfs, list_new_probs)
        return expl_dict

    def _find_counterfactuals(self, record, n_cf=1, target=1):
        cfs_batch = self._generate_candidates(record)
        if cfs_batch.empty:
            return pd.DataFrame(), pd.Series()

        record_ids = self.model.transform(record.to_frame().T[self.act_features], 'id')
        cfs_ids = self.model.transform(cfs_batch[self.act_features], 'id')

        # Universal "zero-effort" drop: keep only candidates whose bins actually
        # changed. Derived from id differences, independent of any cost model.
        changed = (cfs_ids[self.act_features].to_numpy() != record_ids[self.act_features].to_numpy()).any(axis=1)
        if not changed.any():
            return pd.DataFrame(), pd.Series()
        cfs_batch, cfs_ids = cfs_batch[changed], cfs_ids[changed]

        # HARD CONSTRAINT: candidates must flip the prediction before any ranking.
        proba = self.model.predict_proba(cfs_batch)[:, 1]
        if target == 1:
            flip, proba_asc = proba >= 0.5, True
        else:
            flip, proba_asc = proba < 0.5, False
        cfs_batch, cfs_ids = cfs_batch[flip], cfs_ids[flip]

        candidates = cfs_batch.assign(proba=proba[flip])
        ranked = self.method.rank(candidates, cfs_ids, record_ids, proba_asc).head(n_cf)
        return ranked[self.features], ranked['proba']

    def _generate_candidates(self, record):
        options_map = {}
        for var, options in self.categories.items():
            valid_options = options
            if var in self.monotonic_features:
                idx = self.monotonic_features[var].index(record[var])
                valid_options = self.monotonic_features[var][idx:]
            options_map[var] = list(valid_options)

        keys = list(options_map.keys())
        changes_df = pd.DataFrame(itertools.product(*options_map.values()), columns=keys)

        candidates_df = pd.DataFrame(
            np.tile(record.to_numpy(), (len(changes_df), 1)), columns=record.index
        )
        candidates_df[keys] = changes_df[keys].values
        return candidates_df


# ======================================================================
# Counterfactual methods (Strategy pattern)
# ======================================================================

class CounterfactualMethod(ABC):
    """A counterfactual-selection strategy.

    A strategy owns any cost model it needs and decides how flipping candidates
    are ranked. It never sees non-flipping candidates: the explainer enforces the
    "must flip the prediction" constraint *before* calling rank(). Cost-based
    methods compute effort inside rank() — effort is not part of the explainer's
    vocabulary.

    bind(explainer) is called once after the explainer is initialized, so a
    strategy can derive/validate costs from act_features, the fitted model, or
    training data.

    Alignment contract for cost-based methods: `candidate_ids` is row-aligned with
    `candidates` ON ENTRY. Derive everything you need from `candidate_ids`, attach
    it to `candidates` as columns, THEN sort. Do not read `candidate_ids` after a
    reordering — sorting reorders `candidates` but not `candidate_ids`.
    """

    def bind(self, explainer):
        self.explainer = explainer
        return self

    @abstractmethod
    def rank(self, candidates, candidate_ids, record_ids, proba_asc):
        """Order the flipping candidates best-first.

        :param candidates:    flipping rows; has a 'proba' column and all features.
        :param candidate_ids: bin ids of `candidates` for the actionable features
                              (row-aligned with `candidates`).
        :param record_ids:    bin ids of the original record (1-row frame).
        :param proba_asc:     sort direction for proba (True when target == 1).
        :return: `candidates` reordered best-first (extra columns allowed; the
                 explainer reads only the feature columns and 'proba').
        """
        ...


class OptimalMethod(CounterfactualMethod):
    """No cost model. Rank purely by how decisively the prediction flips."""

    def rank(self, candidates, candidate_ids, record_ids, proba_asc):
        return candidates.sort_values(by='proba', ascending=proba_asc)


class ExpertMethod(CounterfactualMethod):
    """Rank by user-supplied per-feature effort costs, then by proba."""

    def __init__(self, efforts):
        if efforts is None:
            raise ValueError("Efforts must be provided for the expert method.")
        self._given = efforts

    def bind(self, explainer):
        super().bind(explainer)
        for var in explainer.act_features:
            if var not in self._given:
                raise ValueError(f"Effort for actionable feature '{var}' must be provided.")
        self.costs = self._given
        return self

    def rank(self, candidates, candidate_ids, record_ids, proba_asc):
        effort = np.zeros(len(candidates))
        for var, cost in self.costs.items():
            orig = record_ids[var].item()
            effort += np.abs(candidate_ids[var].to_numpy() - orig) * cost
        candidates = candidates.assign(effort=effort)
        return candidates.sort_values(by=['effort', 'proba'], ascending=[True, proba_asc])
