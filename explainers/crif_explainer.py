import itertools

import numpy as np
import pandas as pd
from optbinning import BinningProcess, OptimalBinning
from optbinning.binning.binning_statistics import bin_str_format

from explainers.base import BaseExplainer, prepare_output


class CRIFCounterfactualExplainer(BaseExplainer):

    def __init__(self, model, X_train, features, act_features,
                 monotonic_features=None, method="optimal", efforts=None, **kwargs):
        self.method = method
        if method == "optimal":
            efforts = {var: 1 for var in act_features}
        else: # enforce that keys in efforts correspond to features in act_features
            if efforts is None:
                raise ValueError("Efforts must be provided for 'expert' method.")
            for var in act_features:
                if var not in efforts:
                    raise ValueError(f"Effort for actionable feature '{var}' must be provided.")
        self.efforts = efforts
        super().__init__(
            model, X_train, None, features, features, None, act_features, None, monotonic_features
        )

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

        """list_expl_full = cfs.reset_index(drop=True)
        list_expl_changes = []
        for _, cf in cfs.iterrows():
            changes = []
            for col in cfs.columns:
                # if cf[col] != record_bin[col]:
                if cf[col] != record[col]:
                    changes.append(cf[col])
                else:
                    changes.append('-')
            list_expl_changes.append(changes)
        list_expl_changes = pd.DataFrame(list_expl_changes, columns=cfs.columns)

        expl_dict = {
            'record': record, 'label': label, 'pred': pred,
            'proba': proba, 'target': target, 'list_expl_full': list_expl_full,
            'list_expl_changes': list_expl_changes, 'list_new_probs': pd.Series(list_new_probs)
        }"""
        expl_dict = prepare_output(record, label, pred, proba, target, cfs, list_new_probs)
        return expl_dict

    def _find_counterfactuals(self, record, n_cf=1, target=1):
        # 2. Generate Candidates (Extracted Method)
        cfs_batch = self._generate_candidates(record)

        if cfs_batch.empty:
            return pd.DataFrame(), pd.Series()

        # 3. Vectorized Effort Calculation
        # Calculate effort outside the generator as requested
        effort_scores = np.zeros(len(cfs_batch))

        record_ids, cfs_batch_ids = self.model.transform(record.to_frame().T, 'id'), self.model.transform(cfs_batch, 'id')
        for var, cost in self.efforts.items():
            """original_val = record[var]

            # Boolean mask: where is the candidate value different from original?
            mask_diff = cfs_batch[var] != original_val
            effort_scores[mask_diff] += cost"""
            original_val = record_ids[var].item()
            effort_scores += np.abs(cfs_batch_ids[var] - original_val) * cost

        cfs_batch['effort'] = effort_scores

        # 4. Filter "Zero Effort" (The Original Customer)
        # We only want actual changes
        cfs_batch = cfs_batch[cfs_batch['effort'] > 0].copy()

        if cfs_batch.empty:
            return pd.DataFrame(), pd.Series()

        # 5. Prediction
        proba = self.model.predict_proba(cfs_batch.drop('effort', axis=1))
        cfs_batch['proba'] = proba[:, 1]

        # 6. Filtering and Sorting
        if target == 1:
            _query = "proba >= 0.5"
            _proba_asc = True
        else:
            _query = "proba < 0.5"
            _proba_asc = False

        cfs_batch = cfs_batch.query(_query)  # Only consider counterfactuals that flip the prediction

        if self.method == 'optimal':
            cfs_batch = cfs_batch.sort_values(by='proba', ascending=_proba_asc)
        elif self.method == 'expert':
            cfs_batch = cfs_batch.sort_values(by=['effort', 'proba'], ascending=[True, _proba_asc])

        cfs_batch = cfs_batch.head(n_cf)

        return cfs_batch.drop(columns=['effort', 'proba']), cfs_batch['proba']

    def _generate_candidates(self, record):
        """
        Generates the search space of possible counterfactuals based on
        available categories and monotonicity constraints.

        Returns:
            pd.DataFrame: A DataFrame where every row is a candidate counterfactual
                          (including the original customer state).
        """
        # 1. Determine Valid Options per Variable (Monotonicity Check)
        options_map = {}
        for var, options in self.categories.items():
            # Default options (all bins)
            valid_options = options

            # Apply monotonicity filter if applicable
            if var in self.monotonic_features:
                curr_val = record[var]
                # Find index of current bin to slice the allowed options
                # Assumes monotonic_features[var] is sorted by 'difficulty' or 'progression'
                idx = self.monotonic_features[var].index(curr_val)
                valid_options = self.monotonic_features[var][idx:]

            options_map[var] = list(valid_options)

        # 2. Generate Cartesian Product of Changing Variables
        keys = list(options_map.keys())
        values = list(options_map.values())

        # itertools.product generates tuples of all possible combinations
        combinations = list(itertools.product(*values))

        # Create a DataFrame of just the changing parts
        changes_df = pd.DataFrame(combinations, columns=keys)

        # 3. Reconstruct Full Context (Broadcasting)
        # Create a DataFrame where every row is the 'current_customer'
        # This repeats the static (non-changing) variables N times
        candidates_df = pd.DataFrame(
            [record.values] * len(changes_df),
            columns=record.index
        )

        # Overwrite the columns that are changing with the generated combinations
        candidates_df[keys] = changes_df[keys].values

        return candidates_df
