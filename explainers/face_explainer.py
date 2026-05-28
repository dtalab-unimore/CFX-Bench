import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from aux_models import OneHotDataClassifierAdapter
from explainers.base import BaseExplainer, prepare_output
from explainers.face import CFGenerator


class FaceExplainer(BaseExplainer):

    def _init(self):
        # CATEGORICAL FEATURES ONLY
        self.cat_features, self.num_features = self.features, []

        # encode categoric features (one-hot-encoding)
        ohe_sep = '§'  # use a separator that is not likely to appear in feature names or category names
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore", drop=None,
                                     feature_name_combiner=lambda a, b: f"{a}{ohe_sep}{b}")
        self.encoder.fit(self.X_train[self.cat_features])
        self.ohe_cat_columns = list(self.encoder.get_feature_names_out(self.cat_features))
        self.face_features = self.num_features + self.ohe_cat_columns

        df_face = self.scale_encode(self.X_train, self.encoder)

        # compute names of immutable features columns
        raw_cat_immutables = [f for f in self.cat_features if f not in self.act_features]
        ohe_immutables = []
        for raw_feat in raw_cat_immutables:
            cols = [c for c in self.ohe_cat_columns if c.startswith(raw_feat + ohe_sep)]
            ohe_immutables.extend(cols)
        self.immutables_idx = [self.face_features.index(f) for f in ohe_immutables]

        # define wrapper of the model using OneHotDataClassifierAdapter
        self.model_ohe = OneHotDataClassifierAdapter(
            model=self.model,
            ohe_feature_names=self.face_features,
            ohe_separator=ohe_sep,
        )
        self.model_ohe.fit(df_face[self.face_features])

        # pre-compute ranks for fast monotonicity checks
        self.category_rank = {
            feature: {cat: i for i, cat in enumerate(categories)}
            for feature, categories in self.monotonic_features.items()
        }
        self.cat_feature_idx = {feat: i for i, feat in enumerate(self.cat_features)}

        # to enforce constraints
        def edge_conditions(v0, v1):
            """
            Parameters
            ----------
            v0 : original instance
            v1 : counterfactual instance

            Returns
            -------

            """

            v0, v1 = v0.flatten(), v1.flatten()  # input vectors of shape (n_features, 1), convert to shape (n_features,)

            # 1. Fast immutable feature check
            if self.immutables_idx:
                if not np.array_equal(v0[self.immutables_idx],v1[self.immutables_idx]):
                    return False

            # 2. Fast monotonic constraint check using native inverse_transform
            if self.monotonic_features:
                bins = self.encoder.inverse_transform([v0, v1])
                row0, row1 = bins[0], bins[1]

                # 3. Fast monotonic constraint check
                for feature in self.monotonic_features:
                    idx = self.cat_feature_idx[feature]
                    val0 = row0[idx]
                    val1 = row1[idx]
                    if self.category_rank[feature][val0] > self.category_rank[feature][val1]:
                        return False

            return True

        self.cf = CFGenerator(
            predictor=self.model_ohe,
            method='kde',
            # method='knn',
            edge_conditions=edge_conditions,
            undirected=False,
            distance_threshold=np.log2(len(self.features)),  # rule of thumb
        )
        self.cf.fit(df_face[self.face_features].to_numpy(), self.y_train.to_numpy())

    def _explain(self, test_item, n_cf=1):
        record, label, pred, proba, target = test_item

        rec_face = self.scale_encode(record.to_frame().T, self.encoder)

        # generate counterfactuals
        paths = self.cf.compute_path(
            starting_point=rec_face[self.face_features].to_numpy(),
            target_class=target,
            plot=False
        )
        if not paths:
            expl_dict = {
                "record": record, "label": label, "pred": pred,
                "proba": proba, "target": target, "list_new_probs": pd.Series(dtype=float),
                "list_expl_full": pd.DataFrame(columns=self.features),
                "list_expl_changes": pd.DataFrame(columns=self.features),
            }
            return expl_dict

        # select the best counterfactual
        best_cf_vector = paths[0][1]

        cfs_raw = self.encoder.inverse_transform([best_cf_vector])
        cfs = pd.DataFrame(cfs_raw, columns=self.cat_features)
        list_new_probs = self.model.predict_proba(cfs)[:, 1]

        """list_expl_full = cfs.reset_index(drop=True)
        list_expl_changes = []
        for _, cf in cfs.iterrows():
            changes = []
            for col in cfs.columns:
                if cf[col] != record[col]:
                    changes.append(cf[col])
                else:
                    changes.append("-")
            list_expl_changes.append(changes)
        list_expl_changes = pd.DataFrame(list_expl_changes, columns=cfs.columns)
        return {
            "record": record, "label": label, "pred": pred, "proba": proba, "target": target, 
            "list_expl_full": list_expl_full,
            "list_expl_changes": list_expl_changes, "list_new_probs": pd.Series(list_new_probs),
        }"""
        expl_dict = prepare_output(*test_item, cfs, list_new_probs)
        return expl_dict

    # scale and encode the data
    def scale_encode(self, data: pd.DataFrame, encoder) -> pd.DataFrame:
        data_encoded = pd.DataFrame(
            encoder.transform(data[self.cat_features]),
            columns=self.ohe_cat_columns,
            index=data.index
        )
        return data_encoded
