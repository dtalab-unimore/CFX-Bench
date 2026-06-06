import time

import numpy as np
import pandas as pd
from optbinning import Scorecard
from sklearn.preprocessing import MinMaxScaler

from .FACEGroup.Feasibility import feasibility_consts
# The inductive subclass. Drop facegroup_inductive.py into the .FACEGroup package
# (next to FACEGroup.py) and have it do `from .FACEGroup import FACEGroup`, then:
from .FACEGroup.facegroup_inductive import FACEGroupInductive
from .FACEGroup.kernel import Kernel
from .FACEGroup.utils import GraphBuilder, get_subgraphs_by_group, get_normalized_group_identifier_value, \
    get_false_negatives_by_group
from .adapters import FaceGroupAdapter
from .cf_explainer import BaseExplainer, _empty_explanation_dict
from .utils_explainers import prepare_output, analyze_results, prepare_data_face

NAME = "facegroup"
EPSILON = 5
REPRESENTATION = 64
MAX_D = 5
COST_FUNCTION = "max_vector_distance"
K = 50
K_SELECTION_METHOD = "accross_all_ccs"
CFE_SELECTION_METHOD = "greedy"
ALG = "BINARY SEARCH"
GROUP_IDENTIFIER_GERMAN_CREDIT = "personal_status_sex_['female : divorced/separated/married']"
GROUP_IDENTIFIER_LENDING = "home_ownership_['MORTGAGE']"
GROUP_IDENTIFIER_COMPAS = "sex_['Female']"
GROUP_IDENTIFIER_ADULT = "sex_[' Female']"


class FaceGroupExplainer(BaseExplainer):
    """FaceGroup that separates explainer-construction data from inference data.

    BUILD (constructor): graph, kernel, candidate set, group-CFEs and the audit
    metrics are constructed on `df_train` ONLY. Because prepare_data_face / Kernel
    / GraphBuilder all fit on whatever frame is passed, train-only construction is
    achieved simply by passing a train-only `df_train` (and a model trained on
    train only). Nothing in the FACEGroup core changes.

    INFERENCE (`_explain`): a record that is NOT a build row is attached to the
    frozen train graph as a temporary node (FACEGroupInductive.compute_recourse_for_unseen),
    routed to a learned recourse target, then detached. Build rows keep the cheap
    precomputed group-CFE lookup. This is the descriptive(train) / operational(test)
    split made concrete at the method level.

    REQUIREMENTS ON THE CALLER (the external script you're allowed to change):
      * pass a genuine train-only frame as `df_train`;
      * pass `model` trained on that same train split only;
      * route held-out instances through `_explain` (the BaseExplainer eval loop
        already does this).
    """

    def __init__(self, model: Scorecard, df_train, features, cat_features, num_features, act_features, target,
                 dataset_name, **kwargs):
        self.model = model
        self.df_train = df_train
        self.features = features
        self.cat_features = cat_features
        self.num_features = num_features
        self.act_features = act_features
        self.target = target
        self.dataset_name = dataset_name

        # df_build: the frame the explainer is constructed on. Reset to a
        # RangeIndex so positional row i == graph node id i (the invariant the
        # group machinery and the inductive temp-node id both rely on).
        self.df_train = self.df_train.reset_index(drop=True)
        self.df_build = self.df_train

        # prepare data (fits ohe / binning on df_build only)
        (self.ohe, self.model_ohe, X_oh, self.binning_process, y, self.face_features,
         self.immutables, self.immutables_idx) = prepare_data_face(
            self.df_build, self.features, self.target, self.model,
            self.num_features, self.cat_features, self.act_features)

        self.X_oh = X_oh
        # the exact matrix the graph distances live in (face-feature space)
        self.face_matrix = X_oh[self.face_features].to_numpy()

        start = time.perf_counter()

        # kernel fit on the face-feature space of the build set
        kernel = Kernel(dataset_name, X_oh)
        kernel.fitKernel(X_oh.values)

        # inductive FACEGroup: _data kept in the ORIGINAL face-only layout so the
        # cost function (max_vector_distance) is unaffected; node-feature matrix
        # supplied explicitly so inductive distances match the graph exactly.
        self.facegroup = FACEGroupInductive(X_oh.to_numpy(), kernel, self.face_features,
                                            self.target, EPSILON, self.model_ohe)
        self.facegroup.set_node_features(self.face_matrix)

        # feasibility constraints
        self.feasibility_constraints_instance = feasibility_consts(self.face_features)
        for immutable in self.immutables:
            self.feasibility_constraints_instance.set_constraint(immutable, mutability=False, exact_match=True)

        # build graph on the build set only
        graph_builder = GraphBuilder(self.feasibility_constraints_instance, self.face_features,
                                     X_oh, kernel, exclude_columns=False)
        self.distances, self.graph, densities = graph_builder.compute_pairwise_distances_within_subgroups_and_graph(
            'dataset', X_oh[self.face_features], epsilon=EPSILON,
            feasibility_constraints_instance=self.feasibility_constraints_instance, representation=False)
        self.facegroup.set_graph(self.graph)

        # group identifier
        self.scaler = MinMaxScaler()
        self.scaler.fit(X_oh[self.face_features])
        attr_col_mapping = {col: i for i, col in enumerate(X_oh.columns)}
        group_identifier = {
            "german-credit": GROUP_IDENTIFIER_GERMAN_CREDIT,
            "lending": GROUP_IDENTIFIER_LENDING,
            "compas": GROUP_IDENTIFIER_COMPAS,
            "adult": GROUP_IDENTIFIER_ADULT,
        }.get(self.dataset_name, "")
        self.group_identifier = group_identifier
        group_identifier_value_normalized = get_normalized_group_identifier_value(
            group_identifier=group_identifier, group_identifier_value=1,
            min_max_scaler=self.scaler, data_df_copy=X_oh[self.face_features])

        # subgroups
        ohe_numeric_columns = []
        for raw_num_feat in self.num_features:
            cols = [c for c in self.face_features if c.startswith(raw_num_feat + "_")]
            ohe_numeric_columns.extend(cols)
        self.ohe_numeric_columns = ohe_numeric_columns
        subgroups = get_subgraphs_by_group(graph=self.graph, data_np=X_oh.to_numpy(), data=X_oh,
            attr_col_mapping=attr_col_mapping, group_identifier_column=group_identifier,
            group_identifier_value=group_identifier_value_normalized, numeric_columns=ohe_numeric_columns)

        # candidate CFs and factuals are TRAIN nodes (positional index == node id)
        candidate_counterfactuals = X_oh[self.model_ohe.predict(X_oh[self.face_features]) == 1]
        self.candidate_counterfactuals = {index: row.to_numpy() for index, row in candidate_counterfactuals.iterrows()}
        factuals_oh = X_oh[self.model_ohe.predict(X_oh[self.face_features]) == 0]
        factuals_oh = {index: row.to_numpy() for index, row in factuals_oh.iterrows()}
        self.facegroup.set_candidates(self.candidate_counterfactuals)

        gcfes, not_possible_to_cover_fns_group, time_gcfes = self.facegroup.compute_gcfes(subgroups,
            self.candidate_counterfactuals, factuals_oh, MAX_D, COST_FUNCTION, K,
            self.distances, K_SELECTION_METHOD, verbose=True, cfe_selection_method=CFE_SELECTION_METHOD)

        factuals_by_group = get_false_negatives_by_group(factuals_oh, group_identifier,
                            group_identifier_value_normalized, X_oh, ohe_numeric_columns)
        self.results = self.facegroup.apply_cfes(gcfes, factuals_by_group, self.distances,
            not_possible_to_cover_fns_group, K_SELECTION_METHOD, COST_FUNCTION, verbose=True)

        end = time.perf_counter()

        self.dict = analyze_results(self.results, self.facegroup, self.graph, self.feasibility_constraints_instance)

        # The compact set of LEARNED global recourse targets (the GCFE destinations).
        # These are train node ids; applying the global rule to an unseen instance
        # means routing it to one of these, not to any arbitrary candidate.
        self.gcfe_target_ids = sorted({
            t for v in self.dict.values()
            for t in (v if isinstance(v, (list, tuple, set, np.ndarray)) else [v])
        })

        self.training_efficiency = end - start
        self.adapter = FaceGroupAdapter(self.facegroup, subgroups, self.candidate_counterfactuals, factuals_oh,
                                   COST_FUNCTION, self.distances, K_SELECTION_METHOD, CFE_SELECTION_METHOD,
                                   factuals_by_group, ALG)

    # ------------------------------------------------------------------ #
    #  Transform one raw record into the face-feature vector.
    #  prepare_data_ohe fits ONE OneHotEncoder over all `features` (no separate
    #  binning is applied to X_oh; binning_process_ is grabbed only for compat).
    #  get_feature_names_out() order == face_features order, so a plain transform
    #  + ravel is already aligned. Reuses the FITTED encoder; never refits.
    # ------------------------------------------------------------------ #
    def _to_face_vector(self, record):
        import pandas as pd
        row = pd.DataFrame([record.values], columns=self.features)
        vec = np.asarray(self.ohe.transform(row)).ravel().astype(float)
        # defensive: guarantee column alignment to face_features
        if vec.shape[0] != len(self.face_features):
            s = pd.Series(vec, index=list(self.ohe.get_feature_names_out()))
            vec = s.reindex(self.face_features, fill_value=0.0).to_numpy()
        return vec

    def _explain(self, test_item, n_cf=1):
        record, label, pred, proba, target = test_item

        # Fast path: record IS a build (train) row -> use the precomputed group CFE.
        mask = (self.df_build[self.features] == record.values).all(axis=1)
        hits = self.df_build.index[mask]
        if len(hits) and hits[0] in self.dict:
            cfs_idx = self.dict[hits[0]]
            cfs = self.df_build.iloc[cfs_idx][self.features].to_frame().T \
                if np.isscalar(cfs_idx) else self.df_build.iloc[list(cfs_idx)][self.features]
            return prepare_output(self.model, cfs, test_item)

        # Inductive path: genuinely unseen instance.
        face_vec = self._to_face_vector(record)

        # Apply the LEARNED GLOBAL RULE: route to the GCFE destinations, not to
        # arbitrary candidates. Fall back to the full candidate set if no GCFEs.
        target_ids = self.gcfe_target_ids or list(self.candidate_counterfactuals.keys())
        self.facegroup.set_candidates({nid: self.X_oh.iloc[nid].to_numpy() for nid in target_ids})

        paths, min_target = self.facegroup.compute_recourse_for_unseen(
            face_vec, self.feasibility_constraints_instance, personalized=False)

        # restore the full candidate set for any subsequent group-level call
        self.facegroup.set_candidates(self.candidate_counterfactuals)

        if paths and (min_target in paths):
            # decode the REACHED node's one-hot vector back to original space,
            # matching FaceExplainer's OHE.inverse_transform pattern. This avoids
            # any node-id/df-row index assumption: the vector is taken straight
            # from FACEGroup's own path structure.
            cf_vec = np.asarray(paths[min_target]['vector'], dtype=float).reshape(1, -1)
            cfs = pd.DataFrame(self.ohe.inverse_transform(cf_vec), columns=self.features)
        else:
            return _empty_explanation_dict(test_item)

        return prepare_output(self.model, cfs, test_item)
