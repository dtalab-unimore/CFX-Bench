from explanation import ExplanationSet


class BaseExplainer:
    def _explain(self, test_item, n_cf=1) -> ExplanationSet:
        pass

    def explain(self, test_item, n_cf=1):
        expl_dict = self._explain(test_item, n_cf)
        expl_set = ExplanationSet(**expl_dict)
        return expl_set
