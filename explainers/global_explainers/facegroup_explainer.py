from explainers.global_explainers.cf_explainer import BaseExplainer
from optbinning import Scorecard
from sklearn.preprocessing import MinMaxScaler
from FACEGroup.src.FACEGroup import FACEGroup
from FACEGroup.src.utils import GraphBuilder, get_subgraphs_by_group, get_normalized_group_identifier_value, get_false_negatives_by_group
from FACEGroup.src.kernel import Kernel
from FACEGroup.src.Feasibility import feasibility_consts
from metrics_gcfes import (compute_metrics_global, compute_metrics_auc_global,
                           LOWER_LIMIT_RANGE_FOR_D, UPPER_LIMIT_RANGE_FOR_D, UPPER_LIMIT_FOR_K)
from utils_explainers import prepare_output, analyze_results, prepare_data_face
import time
from adapters import FaceGroupAdapter

NAME = "facegroup"
# EPSILON = 3.1
EPSILON = 5
REPRESENTATION = 64
# MAX_D = np.inf
MAX_D = 5
# COST_FUNCTION = "num_path_hops"
# COST_FUNCTION = "max_path_cost"
COST_FUNCTION = "max_vector_distance"
# K = 5
K = 50
# K = 100
K_SELECTION_METHOD = "accross_all_ccs"
# K_SELECTION_METHOD = "same_k_for_all_ccs"
# K_SELECTION_METHOD = "greeedy_accross_all_ccs"
CFE_SELECTION_METHOD = "greedy"
# CFE_SELECTION_METHOD = "mip"
# ALG = "MIP"
ALG = "BINARY SEARCH"
GROUP_IDENTIFIER_GERMAN_CREDIT = "personal_status_sex_['female : divorced/separated/married']"
GROUP_IDENTIFIER_LENDING = "home_ownership_['MORTGAGE']"
GROUP_IDENTIFIER_COMPAS = "sex_['Female']"
GROUP_IDENTIFIER_ADULT = "sex_[' Female']"


class FaceGroupExplainer(BaseExplainer):
    def __init__(self, model: Scorecard, df_train, features, cat_features, num_features, act_features, target, dataset_name, **kwargs):
        self.model = model
        self.df_train = df_train
        self.features = features
        self.cat_features = cat_features
        self.num_features = num_features
        self.act_features = act_features
        self.target = target
        self.dataset_name = dataset_name

        # prepare data
        self.ohe, self.model_ohe, X_oh, self.binning_process, y, self.face_features, self.immutables, self.immutables_idx = (
            prepare_data_face(self.df_train, self.features, self.target, self.model,
                              self.num_features, self.cat_features, self.act_features))

        # start timer to measure training efficiency
        start = time.perf_counter()

        # define and fit kernel
        kernel = Kernel('dataset', X_oh)
        kernel.fitKernel(X_oh)
        # define FACEGroup explainer
        self.facegroup = FACEGroup(X_oh.to_numpy(), kernel, self.face_features, self.target, EPSILON, self.model_ohe)

        # define feasibility constraints
        self.feasibility_constraints_instance = feasibility_consts(self.face_features)
        for immutable in self.immutables:
            self.feasibility_constraints_instance.set_constraint(immutable, mutability=False, exact_match=True)

        # build graph
        graph_builder = GraphBuilder(self.feasibility_constraints_instance, self.face_features,
                                     X_oh, kernel, exclude_columns=False)
        self.distances, self.graph, densities = graph_builder.compute_pairwise_distances_within_subgroups_and_graph(
            'dataset', X_oh[self.face_features], epsilon=EPSILON,
            feasibility_constraints_instance=self.feasibility_constraints_instance, representation=False)
        self.facegroup.set_graph(self.graph)

        # define group identifier
        self.scaler = MinMaxScaler()
        self.scaler.fit(X_oh[self.face_features])
        attr_col_mapping = {col: i for i, col in enumerate(X_oh.columns)}
        group_identifier = ""
        if self.dataset_name == "german-credit":
            group_identifier = GROUP_IDENTIFIER_GERMAN_CREDIT
        elif self.dataset_name == "lending":
            group_identifier = GROUP_IDENTIFIER_LENDING
        elif self.dataset_name == "compas":
            group_identifier = GROUP_IDENTIFIER_COMPAS
        elif self.dataset_name == "adult":
            group_identifier = GROUP_IDENTIFIER_ADULT
        group_identifier_value_normalized = get_normalized_group_identifier_value(group_identifier=group_identifier,
            group_identifier_value=1, min_max_scaler=self.scaler, data_df_copy=X_oh[self.face_features])

        # define subgroups
        ohe_numeric_columns = []
        for raw_num_feat in self.num_features:
            cols = [c for c in self.face_features if c.startswith(raw_num_feat + "_")]
            ohe_numeric_columns.extend(cols)
        subgroups = get_subgraphs_by_group(graph=self.graph, data_np=X_oh.to_numpy(), data=X_oh,
            attr_col_mapping=attr_col_mapping, group_identifier_column=group_identifier,
            group_identifier_value=group_identifier_value_normalized,numeric_columns=ohe_numeric_columns)

        # compute global counterfactuals
        candidate_counterfactuals = X_oh[self.model_ohe.predict(X_oh[self.face_features]) == 1]
        candidate_counterfactuals = {index: row.to_numpy() for index, row in candidate_counterfactuals.iterrows()}
        factuals_oh = X_oh[self.model_ohe.predict(X_oh[self.face_features]) == 0]
        factuals_oh = {index: row.to_numpy() for index, row in factuals_oh.iterrows()}
        self.facegroup.set_candidates(candidate_counterfactuals)
        gcfes, not_possible_to_cover_fns_group, time_gcfes = self.facegroup.compute_gcfes(subgroups,
            candidate_counterfactuals, factuals_oh, MAX_D, COST_FUNCTION, K, self.distances, K_SELECTION_METHOD,
            verbose=True, cfe_selection_method=CFE_SELECTION_METHOD)

        # compute the results
        factuals_by_group = get_false_negatives_by_group(factuals_oh, group_identifier,
                             group_identifier_value_normalized, X_oh, ohe_numeric_columns)
        self.results = self.facegroup.apply_cfes(gcfes, factuals_by_group, self.distances,
            not_possible_to_cover_fns_group, K_SELECTION_METHOD, COST_FUNCTION, verbose=True)

        # end timer to measure training efficiency
        end = time.perf_counter()

        # analyze and build a dictionary of results
        self.dict = analyze_results(self.results, self.facegroup, self.graph, self.feasibility_constraints_instance)

        # compute metrics
        name = self.dataset_name + "_" + NAME
        factuals = self.df_train[self.model.predict(self.df_train[self.features]) == 0]
        training_efficiency = end - start
        _, _ = compute_metrics_global(self.df_train, factuals, self.features, cat_features, num_features, self.target,
                                      self.model, self._explain, self.binning_process, training_efficiency, name, write=True)

        # compute metrics auc
        adapter = FaceGroupAdapter(self.facegroup, subgroups, candidate_counterfactuals, factuals_oh, COST_FUNCTION,
                                   self.distances, K_SELECTION_METHOD, CFE_SELECTION_METHOD,  factuals_by_group, ALG)
        _, _, _, _, _, _, _, _, _ = compute_metrics_auc_global(adapter, LOWER_LIMIT_RANGE_FOR_D, UPPER_LIMIT_FOR_K,
                                        UPPER_LIMIT_RANGE_FOR_D, name, self.distances, plot=True, write=True)

    def _explain(self, test_item, n_cf=1):
        # unpack test item
        record, label, pred, proba, target = test_item

        # find index of the factual
        mask = (self.df_train[self.features] == record.values).all(axis=1)
        factual_idx = self.df_train.index[mask][0]
        # extract cfs from results
        if factual_idx not in self.dict.keys():
            cfs = record.to_frame().T
            # raise RuntimeError("FACEGroup could not find a valid counterfactual.")
        else:
            cfs_idx = self.dict[factual_idx]
            cfs = self.df_train.iloc[cfs_idx][self.features].to_frame().T

        # prepare the output in the required format
        return prepare_output(self.model, cfs, test_item)