import pandas as pd
from optbinning import Scorecard
from optbinning.exceptions import CounterfactualsFoundWarning
from optbinning.scorecard import Counterfactual

from explainers.base import BaseExplainer, _empty_explanation_dict, prepare_output


class OptBinExplainer(BaseExplainer):
    def __init__(self, binning_process, model, X_train, y_train, features, act_features, **kwargs):
        model = Scorecard(
            binning_process=binning_process, estimator=model,
            scaling_method="min_max", scaling_method_params={"min": 300, "max": 850}
        )
        super().__init__(
            model, X_train, y_train, features, None, None, act_features, None, None
        )

    def _init(self):
        self.model.fit(self.X_train, self.y_train)
        self.cf_model = Counterfactual(scorecard=self.model)
        self.cf_model.fit(self.X_train)

    def _explain(self, test_item, n_cf=1):
        record, label, pred, proba, target = test_item
        record_bins = self.model.binning_process_.transform(record.to_frame().T, metric="bins").iloc[0]  # bin as string for all features

        self.cf_model.generate(
            query=record.to_frame().T,
            y=target,
            outcome_type="binary",
            n_cf=n_cf,
            max_changes=len(self.act_features),
            actionable_features=self.act_features
        )
        try:
            self.cf_model.display()  # bin as list for changes, original values elsewhere
        except CounterfactualsFoundWarning as e:
            return _empty_explanation_dict((record_bins, label, pred, proba, target))
        list_expl_changes = self.cf_model.display(show_only_changes=True, show_outcome=True)  # bin as list for changes, '-' for no change, outcome as last column
        list_new_probs = list_expl_changes['outcome']
        list_expl_changes = list_expl_changes.drop(columns=['outcome'])

        cfs_ = pd.DataFrame([record_bins]).reset_index(drop=True)
        cfs_changes_ = list_expl_changes.copy().reset_index(drop=True).map(str)
        mask = (cfs_changes_ != '-').reset_index(drop=True)
        cfs_ = cfs_.mask(mask, cfs_changes_)
        list_expl_full = cfs_

        expl_dict = prepare_output(record_bins, label, pred, proba, target, list_expl_full, list_new_probs)
        return expl_dict
