from explainers.global_explainers.utils_explainers import compute_bounds


class ModelWrapper:
    def __init__(self, model):
        self.model = model

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def decode(self, X, features_orig, features_oh,  num_features, binning_process):
        feat_dict = {}
        for feat in features_orig:
            feat_dict[feat] = []
            for col in features_oh:
                if col.startswith(feat):
                    feat_dict[feat].append(col)
        for cat, cols in feat_dict.items():
            if not cols:
                continue
            active = X[cols].idxmax(axis=1)[0]
            v = active.removeprefix(cat + "_")
            if cat in num_features:
                lb, ub = compute_bounds(v)
                if lb != '-inf' and ub != 'inf':
                    v = (lb + ub) / 2
                elif lb == '-inf' and ub != 'inf':
                    v = ub
                elif ub == 'inf' and lb != '-inf':
                    v = lb
            else:
                if v.count("'") > 2:
                    v = v.strip("[]").replace("\n", "").replace(",", "").split("' '")[0].strip("'")
                else:
                    v = v.strip("[]'")
            X[cat] = v
            X.drop(columns=cols, inplace=True)
        return X[binning_process.variable_names]