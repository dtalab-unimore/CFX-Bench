import dice_ml
import pandas as pd
from optbinning import Scorecard
from raiutils.exceptions import UserConfigValidationException

from explainers.base import BaseExplainer, prepare_output, _empty_explanation_dict


class DiceExplainer(BaseExplainer):

    def __init__(self, model: Scorecard, X_train, y_train, features, cat_features, num_features, act_features, target,
                 monotonic_features=None, method='random', **kwargs):
        self.method = method
        super().__init__(
            model, X_train, y_train, features, cat_features, num_features, act_features, target, monotonic_features
        )

    def _init(self):
        df_train = pd.concat([self.X_train, self.y_train.rename(self.target)], axis=1)
        self._dice_data = dice_ml.Data(
            dataframe=df_train,
            continuous_features=self.num_features,
            outcome_name=self.target,
        )
        self._dice_model = dice_ml.Model(self.model, backend="sklearn")
        self._dice = dice_ml.Dice(self._dice_data, self._dice_model, method=self.method)

    def _explain(self, test_item, n_cf=1):
        record, label, pred, proba, target = test_item

        # --- monotonic bins
        if self.monotonic_features is None:
            permitted_range = None
        else:
            permitted_range = {}

            for feature in self.monotonic_features:
                if feature in self.num_features:
                    permitted_range[feature] = [record[feature], self._dice.data_interface.permitted_range[feature][1]]
                elif feature in self.cat_features:
                    categories = self.monotonic_features[feature]
                    cat_index = categories.index(record[feature])
                    permitted_range[feature] = categories[cat_index:]
        # ---

        try:
            cfs = self._dice.generate_counterfactuals(
                record.to_frame().T,
                total_CFs=n_cf,
                desired_class=target,
                features_to_vary=self.act_features,
                permitted_range=permitted_range,
            )
        except UserConfigValidationException as e:
            # print exception message, then return default dictionary with no counterfactual
            print(e)
            return _empty_explanation_dict(test_item)

        cfs = cfs.cf_examples_list
        cfs = pd.concat([cf.final_cfs_df for cf in cfs], ignore_index=True)[self.features]
        list_new_probs = self.model.predict_proba(cfs)[:, 1]
        """list_expl_full = cfs.reset_index(drop=True)
        list_expl_changes = []
        for _, cf in cfs.iterrows():
            changes = []
            for col in cfs.columns:
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
        expl_dict = prepare_output(*test_item, cfs, list_new_probs)
        return expl_dict
