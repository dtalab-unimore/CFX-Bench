import pandas as pd
import math
from explainers.global_explainers.scorecard_one_hot import prepare_data_ohe
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
import numpy as np

CONT_FEATURES_GERMAN_CREDIT = ["duration", "credit_amount", "age"]
CONT_FEATURES_LENDING = ["emp_length", "annual_inc", "open_acc", "credit_years"]
CONT_FEATURES_COMPAS = ["age", "priors_count", "days_b_screening_arrest", "length_of_stay"]
CONT_FEATURES_ADULT = ["age", "hours-per-week"]


def prepare_output(model, cfs, test_item):
    """
    Prepare the explainer output in the required format.
    """

    # unpack test item
    record, label, pred, proba, target = test_item
    # prepare output
    list_new_probs = model.predict_proba(cfs)[:, 1]
    list_expl_full = cfs.reset_index(drop=True)
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
    }

def immutable_features(num_features, cat_features, act_features, face_features):
    """
    Compute names of immutable features columns.
    """

    immutables, immutables_idx = [], []
    raw_immutables = ([f for f in num_features if f not in act_features] +
                      [f for f in cat_features if f not in act_features])
    for raw_feat in raw_immutables:
        cols = [c for c in face_features if c.startswith(raw_feat + "_")]
        immutables.extend(cols)
    immutables_idx = [face_features.index(f) for f in immutables]
    return immutables, immutables_idx

def compute_nbins(features, binning_process):
    """
    Compute number of bins.
    """

    n_bins = 0
    for f in features:
        n_bins += len(binning_process._binned_variables[f].splits)
    n_bins /= len(features)
    return n_bins


def compute_bounds(v):
    """
    Compute bounds.
    """

    v = v.strip("[]()'")
    lb, ub = v.split(",", 1)
    lb = lb.strip()
    ub = ub.strip()
    if lb != '-inf':
        lb = math.ceil(float(lb))
    if ub != 'inf':
        ub = math.floor(float(ub))
    return lb, ub


def _rule_applies(x, outer_if, inner_if, num_features):
    """
    Check if rule applies.
    """

    x_cf = x.copy()
    for lit in (outer_if + inner_if):
        f, v = lit.split(" = ", 1)
        f, v = f.strip(), v.strip()
        if f in num_features:
            lb, ub = compute_bounds(v)
            if lb != '-inf' and ub != 'inf':
                if x_cf[f] < lb or x_cf[f] > ub:
                    return False
            elif lb == '-inf' and ub != 'inf':
                if x_cf[f] > ub:
                    return False
            elif ub == 'inf' and lb != '-inf':
                if x_cf[f] < lb:
                    return False
        else:
            if v.count("'") > 2:
                v_list = v.strip("[]").split("' '")
                flag = False
                for value in v_list:
                    if value.strip("'") == x_cf[f]:
                        flag = True
                if not flag:
                    return False
            else:
                v = v.strip("[]'")
                if v != x_cf[f]:
                    return False
    return True


def rule_applies(x_cf, outer_if, inner_if, num_features):
    """
    Check if rule applies.
    """
    x_cf = x_cf.copy()
    for lit in (outer_if + inner_if):
        f, v = lit.split(" = ", 1)
        f, v = f.strip(), v.strip()
        if x_cf[f] != v:
            return False
    return True


def _apply_then(x, then, num_features):
    """
    Apply then rule.
    """

    x_cf = x.copy()
    for lit in then:
        f, v = lit.split(" = ", 1)
        f, v = f.strip(), v.strip()
        if f in num_features:
            lb, ub = compute_bounds(v)
            if lb != '-inf' and ub != 'inf':
                if x_cf[f] < lb or x_cf[f] > ub:
                    v = (lb + ub) / 2
                else:
                    v = x_cf[f]
            elif lb == '-inf' and ub != 'inf':
                if x_cf[f] > ub:
                    v = ub
                else:
                    v = x_cf[f]
            elif ub == 'inf' and lb != '-inf':
                if x_cf[f] < lb:
                    v = lb
                else:
                    v = x_cf[f]
        else:
            if v.count("'") > 2:
                v = v.strip("[]").split("' '")[0].strip("'")
            else:
                v = v.strip("[]'")
        x_cf[f] = v
    return x_cf


def apply_then(x_cf, then, num_features):
    """
    Apply then rule.
    """
    x_cf = x_cf.copy()
    for lit in then:
        f, v = lit.split(" = ", 1)
        f, v = f.strip(), v.strip()
        x_cf[f] = v
    return x_cf


def bin_and_transform(x, binning_process, ohe):
    # x_bins = binning_process.transform(x, metric="bins")  # FIXME
    x_bins = x.copy()
    x_bins = x_bins.replace(r"np\.str_\((['\"].*?['\"])\)", r"\1", regex=True)
    x_oh = ohe.transform(x_bins)
    return pd.DataFrame(x_oh, columns=ohe.get_feature_names_out())


def analyze_results(results, facegroup, graph, feasibility_constraints_instance):
    """
    Analyze FACEGroup results.
    """

    dict = {}
    res = {str(key): value for key, value in results.items()}
    for key in res:
        if key in ["Node Connectivity", "Edge Connectivity", "Total coverage", "Graph Stats"]: continue
        res[key] = {str(k): v for k, v in res[key].items()}
        for factual in res[key]:
            if factual in ["Coverage", "Avg. distance", "Median distance", "Avg. path cost",
                           "Median path cost"]: continue
            shortest_paths_info, min_target_id = facegroup.compute_recourse(graph, int(factual),
                                                                            feasibility_constraints_instance, False)
            print("* Factual Point:", factual)
            if shortest_paths_info is not None and min_target_id is not None and shortest_paths_info != {}:
                print("	FACEGroup Info:")
                print(f"		CFE: {res[key][factual]['CFE_name']}")
                #print(f"		Shortest Path cost: {res[key][factual]['Shortest_path_cost']}")
                #print(f"		Shortest Path cost dist: {res[key][factual]['Shortest_paths_distance_cost']}")
                print(f"		Vector Distance: {res[key][factual]['Vector_distance']}")
            dict.update({int(factual): res[key][factual]['CFE_name']})
    return dict


def apply_rules_llm(record, rules, model):
    """
    Apply rules generated by the LLM and build the counterfactual.
    """

    cumulative_cfs, cumulative_cfs_final, cfs = record.copy(), [], []
    flag, flag1 = False, False
    for rule in rules:
        cfs = record.copy()
        for col, value in rule.items():
            if value != record[col]:
                cfs[col] = value
            if value != cumulative_cfs[col]:
                cumulative_cfs[col] = value
        if model.predict(cfs.to_frame().T)[0] == 1:
            flag = True
            break
        if model.predict(cumulative_cfs.to_frame().T)[0] == 1 and not flag1:
            cumulative_cfs_final = cumulative_cfs
            flag1 = True
    if flag:
        cfs = cfs
    else:
        if len(cumulative_cfs_final) == 0:
            cfs = cumulative_cfs
        else:
            cfs = cumulative_cfs_final
    return cfs


def prepare_data_face(df_train, features, target, model, num_features, cat_features, act_features):
    """
    Function to prepare data used in FACE and FACEGroup.
    """

    # prepare data
    ohe, model_ohe, X_oh, binning_process, y = prepare_data_ohe(df_train, features, target, model)
    # features passed to FACE and FACEGroup are all numeric
    face_features = list(ohe.get_feature_names_out())
    # define immutable features
    immutables, immutables_idx = immutable_features(num_features, cat_features, act_features, face_features)
    return ohe, model_ohe, X_oh, binning_process, y, face_features, immutables, immutables_idx


def prepare_data_ares_globece(features, act_features, df_train, target, model, dataset_name):
    """
    Function to prepare the data used in AReS and Globe-CE.
    """

    # define categorical, continuous and immutable features
    cat_features = features
    cont_features = []
    if dataset_name == "german-credit":
        cont_features = CONT_FEATURES_GERMAN_CREDIT
    elif dataset_name == "lending":
        cont_features = CONT_FEATURES_LENDING
    elif dataset_name == "compas":
        cont_features = CONT_FEATURES_COMPAS
    elif dataset_name == "adult":
        cont_features = CONT_FEATURES_ADULT
    immutables = list(set(features) - set(act_features))
    # prepare data
    ohe, model_ohe, X_oh, binning_process, y = prepare_data_ohe(df_train, features, target, model)
    # define n_bins, dataset loader and model wrapper
    n_bins = compute_nbins(features, binning_process)
    return cat_features, cont_features, immutables, ohe, model_ohe, X_oh, binning_process, y, n_bins


def compute_elbow(data, k_range):
    """
    Elbow method ro find the optimal k in KMeans.
    """

    inertias = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)
    return inertias


def select_diverse_points(X, n_select):
    """
    Select diverse points inside a cluster.
    """

    # start from a random point
    selected_indices = [np.random.randint(len(X))]
    # compute pairwise distances
    distances = pairwise_distances(X)
    for _ in range(1, n_select):
        min_dist_to_selected = np.min(distances[:, selected_indices], axis=1)
        # select index at maximum distance
        next_idx = np.argmax(min_dist_to_selected)
        selected_indices.append(next_idx)

    return selected_indices