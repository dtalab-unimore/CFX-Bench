import pandas as pd


class ExplanationSet:
    def __init__(self, record: pd.Series, label, pred, proba, target, list_expl_full, list_expl_changes, list_new_probs,
                 id=None, costs=None):
        self.record = record
        self.label = label
        self.pred = pred
        self.proba = proba
        self.target = target
        self.list_expl_full = list_expl_full
        self.list_expl_changes = list_expl_changes
        self.list_new_probs = list_new_probs
        self.list_new_preds = list_new_probs > 0.5
        self.id = record.name if id is None else id
        self.costs = costs

    def get_pred(self):
        return self.pred

    def get_new_preds(self):
        return self.list_new_preds

    def get_target(self):
        return self.target

    def __len__(self):
        return len(self.list_expl_full)

    def get_list_expl_full(self):
        return self.list_expl_full

    def get_record(self):
        return self.record

    def to_json(self):

        def series_to_dict(s):
            s = s.to_dict()
            for k, v in s.items():
                if isinstance(v, float):
                    s[k] = round(v, 2)
            return {k: v for k, v in s.items() if not (isinstance(v, str) and v == '-')}

        cf_items = []
        for i in range(len(self)):
            cf_item = {
                'record': series_to_dict(self.list_expl_full.iloc[i]),
                'changes': series_to_dict(self.list_expl_changes.iloc[i]),
                'proba': round(self.list_new_probs.iloc[i], 4),
                'pred': int(self.list_new_preds.iloc[i])
            }
            if self.costs is not None:
                cf_item['cost'] = self.costs[i]
            cf_items.append(cf_item)

        data = {
            'id': self.id,
            'record': series_to_dict(self.record),
            'label': int(self.label),
            'pred': int(self.pred),
            'proba': round(self.proba, 4),
            'target': int(self.target),
            'list_cf': cf_items,
        }

        return data
