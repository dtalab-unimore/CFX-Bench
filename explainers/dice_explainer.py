from copy import deepcopy

import dice_ml
import pandas as pd
from optbinning import Scorecard
from raiutils.exceptions import UserConfigValidationException

from explainers.base import BaseExplainer, prepare_output, _empty_explanation_dict
from utils import OrdinalBinsEncoder


class _ModelWrapperClass:

    def __init__(self, model, ordenc):
        self.model = model
        self.ordenc = ordenc

    def predict_proba(self, X):
        return self.model.predict_proba(self.ordenc.inverse_transform(X))

    def predict(self, X):
        return self.model.predict(self.ordenc.inverse_transform(X))


class DiceExplainer(BaseExplainer):

    def __init__(self, model: Scorecard, X_train, y_train, features, cat_features, num_features, act_features, target,
                 monotonic_features=None, method='random', **kwargs):
        self.method = method
        super().__init__(
            model, X_train, y_train, features, cat_features, num_features, act_features, target, monotonic_features
        )

    def _init(self):
        df_train = pd.concat([self.X_train, self.y_train.rename(self.target)], axis=1)

        # # === ordinal bins encoding
        # self.ordenc = OrdinalBinsEncoder(self.features, self.monotonic_features, self.model.binning_process_)
        # X_train = self.ordenc.fit_transform(self.X_train)
        # X_train = pd.DataFrame(X_train, columns=self.features)
        # df_train = pd.concat([X_train, self.y_train.rename(self.target)], axis=1)
        # self.num_features, self.cat_features = self.features, []
        # self._model = deepcopy(self.model)
        # self.model = _ModelWrapperClass(self._model, self.ordenc)
        # # ===

        self._dice_data = dice_ml.Data(
            dataframe=df_train,
            continuous_features=self.num_features,
            outcome_name=self.target,
        )
        self._dice_model = dice_ml.Model(self.model, backend="sklearn")
        self._dice = dice_ml.Dice(self._dice_data, self._dice_model, method=self.method)

    def _explain(self, test_item, n_cf=1):
        record, label, pred, proba, target = test_item

        # # === ordinal bins encoding
        # record = self.ordenc.transform(record.to_frame().T)
        # record = pd.DataFrame(record, columns=self.features)
        # record = record.iloc[0]
        # # ===

        features_to_vary = deepcopy(self.act_features)
        # --- monotonic bins
        if self.monotonic_features is None:
            permitted_range = None
        else:
            permitted_range = {}

            for feature in self.monotonic_features:
                if feature in self.num_features:
                    range_ = [record[feature], self._dice.data_interface.permitted_range[feature][1]]
                    if range_[0] == range_[1]:
                        features_to_vary.remove(feature)
                    else:
                        permitted_range[feature] = range_  # todo: adjust for monotonic descending
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
                features_to_vary=features_to_vary,
                permitted_range=permitted_range,
                # proximity_weight=0 if self.method == 'genetic' else 0.2  # 0.2 is the default value
            )
        except UserConfigValidationException as e:
            # print exception message, then return default dictionary with no counterfactual
            print(e)
            return _empty_explanation_dict(test_item)

        cfs = cfs.cf_examples_list
        cfs = pd.concat([cf.final_cfs_df for cf in cfs], ignore_index=True)[self.features]
        # cfs = cfs.astype(int)  # ordinal bins encoding
        list_new_probs = self.model.predict_proba(cfs)[:, 1]

        # cfs = pd.DataFrame(self.ordenc.inverse_transform(cfs), columns=self.features)  # ordinal bins encoding

        expl_dict = prepare_output(*test_item, cfs, list_new_probs)
        return expl_dict
