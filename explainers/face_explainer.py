import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from aux_models import OneHotDataClassifierAdapter
from explainers.base import BaseExplainer, prepare_output
from explainers.FACE import CFGenerator


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

        # ------------------------------------------------------------------
        # Pre-compute everything needed to evaluate edge conditions in a fully
        # vectorized way, exploiting the one-hot structure of the binned data.
        #
        #  * Immutable check: the one-hot sub-vectors must be identical.
        #  * Monotonic check: the ordinal rank of a feature is simply the
        #    position of its "hot" column inside that feature's one-hot block.
        #    We therefore never need OneHotEncoder.inverse_transform at edge
        #    evaluation time -- we precompute, per monotonic feature, the column
        #    indices of its block (in encoder.categories_ order) and a lookup
        #    array mapping each in-block position to its monotonic rank.
        # ------------------------------------------------------------------
        self.immutables_idx_arr = np.asarray(self.immutables_idx, dtype=int)

        self.monotonic_blocks = []
        for feature in self.monotonic_features:
            cidx = self.cat_feature_idx[feature]
            # block columns appear in get_feature_names_out order == categories_ order
            block_cols = np.array(
                [i for i, c in enumerate(self.face_features)
                 if c.startswith(feature + ohe_sep)],
                dtype=int,
            )
            rank_lookup = np.array(
                [self.category_rank[feature][cat]
                 for cat in self.encoder.categories_[cidx]],
                dtype=int,
            )
            self.monotonic_blocks.append((block_cols, rank_lookup))

        # to enforce constraints (vectorized)
        def edge_conditions(V0, V1):
            """
            Parameters
            ----------
            V0 : np.ndarray of shape (M, n_features)
                Batch of original instances (M may be 1 for a single record).
            V1 : np.ndarray of shape (M, n_features)
                Batch of counterfactual instances. Edges are directional:
                V0[m] -> V1[m].

            Returns
            -------
            np.ndarray of shape (M,), dtype bool
                Mask of pairs that satisfy the immutability and monotonicity
                constraints.
            """
            ok = np.ones(V0.shape[0], dtype=bool)

            # 1. Immutable features: one-hot sub-vectors must match exactly
            if self.immutables_idx_arr.size:
                ok &= np.all(
                    V0[:, self.immutables_idx_arr] == V1[:, self.immutables_idx_arr],
                    axis=1,
                )

            # 2. Monotonic features: rank(V0) must not exceed rank(V1)
            for block_cols, rank_lookup in self.monotonic_blocks:
                r0 = rank_lookup[np.argmax(V0[:, block_cols], axis=1)]
                r1 = rank_lookup[np.argmax(V1[:, block_cols], axis=1)]
                ok &= (r0 <= r1)

            return ok

        self.cf = CFGenerator(
            predictor=self.model_ohe,
            method='kde',
            # kde_mode=2,  # use sigmoid instead of -log to avoid negative edges
            # method='knn',
            edge_conditions=edge_conditions,
            undirected=False,
            distance_threshold=np.sqrt(2)+0.1,
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
