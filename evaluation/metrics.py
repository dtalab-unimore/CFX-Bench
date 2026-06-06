import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import LocalOutlierFactor

from evaluation.utils import hybrid_distance, _aggregate


def num_valid_cf(expl_set):
    """
    Count the number of valid counterfactuals (i.e., if it matches the desired label).
    """
    new_preds = expl_set.get_new_preds()
    target = expl_set.get_target()
    return np.sum(new_preds == target)


def perc_valid_cf(expl_set, k=None):
    """
    Compute the percentage of valid counterfactuals.
    """
    n_val = num_valid_cf(expl_set=expl_set)
    k = len(expl_set) if k is None else k
    res = n_val / k
    return res


def num_actionable_cf(expl_set, variable_features):
    """
    Count the number of actionable counterfactuals.
    A counterfactual is considered actionable if all changes occur only
    in the features specified as variable (i.e., mutable).
    """
    num_actionable = 0
    record = expl_set.get_record()

    for expl in expl_set.get_list_expl_full():
        constraint_violated = False
        for j in range(len(expl)):
            # Check if any non-variable feature has been changed
            if expl[j] != record[j] and j not in variable_features:
                constraint_violated = True
                break
        if not constraint_violated:
            num_actionable += 1

    return num_actionable


def perc_actionable_cf(expl_set, variable_features, k=None):
    """
    Compute the percentage of actionable counterfactuals.
    """
    n_val = num_actionable_cf(expl_set, variable_features)
    k = len(expl_set) if k is None else k
    res = n_val / k
    return res


def num_valid_actionable_cf(expl_set, variable_features):
    """
    Count the number of valid and actionable counterfactuals.
    A counterfactual is valid if it matches the desired label,
    and actionable if it only changes allowed features.
    """
    new_preds = expl_set.get_new_preds()
    target = expl_set.get_target()
    validity_mask = new_preds == target
    record = expl_set.get_record()

    num_valid_actionable = 0
    for ix, expl in enumerate(expl_set.get_list_expl_full()):
        if not validity_mask[ix]:
            continue  # Skip non-valid counterfactuals
        constraint_violated = False
        for col in range(len(expl)):
            # Check if any non-variable feature has been changed
            if expl[col] != record[col] and col not in variable_features:
                constraint_violated = True
                break
        if not constraint_violated:
            num_valid_actionable += 1

    return num_valid_actionable


def perc_valid_actionable_cf(expl_set, variable_features, k=None):
    """
    Compute the percentage of counterfactuals that are both valid and actionable.
    """
    n_val = num_valid_actionable_cf(expl_set, variable_features)
    k = len(expl_set) if k is None else k
    res = n_val / k
    return res


def num_violations_per_cf(expl_set, variable_features):
    """
    Count the number of constraint violations for each counterfactual.
    A violation occurs when a non-variable feature is changed.
    """
    num_violations = np.zeros(len(expl_set))
    record = expl_set.get_record()

    for i, expl in enumerate(expl_set.get_list_expl_full()):
        for col in range(len(expl)):
            if expl[col] != record[col] and col not in variable_features:
                num_violations[i] += 1

    return num_violations


def avg_num_violations_per_cf(expl_set, variable_features):
    """
    Compute the average number of constraint violations per counterfactual.
    """
    return np.mean(num_violations_per_cf(expl_set, variable_features))


def avg_num_violations(expl_set, variable_features):
    """
    Compute the average number of violations per feature across all counterfactuals.
    """
    val = np.sum(num_violations_per_cf(expl_set, variable_features))
    num_cf, num_features = expl_set.get_list_expl_full().shape
    return val / (num_cf * num_features)


def num_changes_per_cf(expl_set, continuous_features):
    """
    Computes the number of feature changes between the original record and related counterfactuals.
    Continuous feature changes count as 1.0, while categorical feature changes count as 0.5.
    """
    record = expl_set.get_record()
    cf_list = expl_set.get_list_expl_full()

    nbr_changes = np.zeros(len(expl_set))
    for i, cf in enumerate(cf_list):
        for col in range(len(cf)):
            if cf[col] != record[col]:
                nbr_changes[i] += 1 if col in continuous_features else 0.5
    return nbr_changes


def avg_num_changes_per_cf(expl_set, continuous_features):
    """
    Computes the average number of feature changes across all counterfactuals.
    """
    return np.mean(num_changes_per_cf(expl_set, continuous_features))


def avg_num_changes(expl_set, nbr_features, continuous_features):
    """
    Computes the average number of changes per feature across all counterfactuals.
    """
    val = np.sum(num_changes_per_cf(expl_set, continuous_features))
    num_cf = len(expl_set)
    return val / (num_cf * nbr_features)


def count_diversity(expl_set, features, tot_features, continuous_features):
    """
    Counts the number of differing features between all pairs of counterfactuals.
    Continuous feature differences count as 1, categorical as 0.5.
    """
    cf_list = expl_set.get_list_expl_full()
    num_cf, _ = cf_list.shape

    num_changes = 0
    for i in range(num_cf):
        for j in range(i + 1, num_cf):
            for col in range(len(features)):
                if cf_list[i][col] != cf_list[j][col]:
                    num_changes += 1 if col in continuous_features else 0.5
    return num_changes / (num_cf * num_cf * tot_features)


def accuracy_knn_sklearn(expl_set, X_test, X_test_df,
                         model, continuous_features, categorical_features, test_size=5):
    """
    Computes the accuracy of a 1-NN classifier trained on the original instance and its counterfactuals.
    The classifier is tested on examples from the test set that are similar/dissimilar to the input `x`
    based on Euclidean-Jaccard distance over normalized features.
    """
    x = expl_set.get_record()
    cf_list = expl_set.get_list_expl_full()
    X_train = np.vstack([x.reshape(1, -1), cf_list])
    y_train = np.concatenate([np.reshape(expl_set.get_pred(), (-1,)), expl_set.get_new_preds()])

    clf = KNeighborsClassifier(n_neighbors=1)
    clf.fit(X_train, y_train)

    X_test_knn_index = _select_test_knn(
        x=x, y_val=expl_set.get_pred(),
        X_test=X_test, y_test=model.predict(X_test_df),
        continuous_features=continuous_features,
        categorical_features=categorical_features,
        test_size=test_size
    )
    y_test = model.predict(X_test_df.iloc[X_test_knn_index])
    y_pred = clf.predict(X_test[X_test_knn_index])

    return accuracy_score(y_test, y_pred)


def _select_test_knn(y_val, y_test, x, X_test, continuous_features, categorical_features, test_size=5):
    """
    Selects test instances that are most similar and dissimilar to a given instance `x`.
    The similarity is computed using Euclidean-Jaccard distance over normalized features. Returns
    an equal number of instances from both the same and different predicted classes.
    """
    # dist_f = euclidean_jaccard(x, X_test[y_test == y_val], continuous_features, categorical_features)
    # dist_cf = euclidean_jaccard(x, X_test[y_test != y_val], continuous_features, categorical_features)
    dist_f = hybrid_distance(
        x, X_test[y_test == y_val], continuous_features, categorical_features,
        cont_metric='euclidean', cate_metric='jaccard'
    )
    dist_cf = hybrid_distance(
        x, X_test[y_test != y_val], continuous_features, categorical_features,
        cont_metric='euclidean', cate_metric='jaccard'
    )
    index_f = np.argsort(dist_f)[0][:test_size].tolist()
    index_cf = np.argsort(dist_cf)[0][:test_size].tolist()
    index = np.array(index_f + index_cf)
    return index



def lof(expl_set, X_train):
    """
    Computes the average Local Outlier Factor (LOF) score of counterfactuals.
    Counterfactuals are evaluated as potential outliers with respect to the training data
    and the original instance.
    """
    record = expl_set.get_record()
    cf_list = expl_set.get_list_expl_full()

    X_train_full = np.vstack([record.reshape(1, -1), X_train])

    clf = LocalOutlierFactor(n_neighbors=3, novelty=True)
    clf.fit(X_train_full)

    lof_values = clf.predict(cf_list)
    return np.mean(np.abs(lof_values))


def delta_proba(expl_set, agg=None):
    """
    Computes the change in predicted probability between the original instance and its counterfactuals.
    """
    y_val = expl_set.proba
    y_cf = expl_set.list_new_probs
    deltas = np.abs(y_cf - y_val)

    # if agg is None:
    #     return deltas
    # if agg == 'mean':
    #     return np.mean(deltas)
    # if agg == 'max':
    #     return np.max(deltas)
    # if agg == 'min':
    #     return np.min(deltas)
    return _aggregate(deltas, agg)


def plausibility(expl_set, X_test, y_pred, continuous_features_all,
                 categorical_features_all, ratio_cont, cont_metric):
    """
    Computes the average distance between each counterfactual and the closest test
    sample that shares the same prediction.

    `cont_metric` is required: it must be a fully-formed metric (string accepted by
    scipy, or callable). Build data-dependent metrics like MAD-cityblock at the call
    site via `utils.make_mad_metric(X_train, features)` and pass the result in.
    The categorical metric is fixed to 'hamming' to match the original definition.

    Optimization: the record→X_test_y nearest-neighbor distance only depends on the
    CF's predicted class, not on the CF itself. The closest neighbor is cached per
    unique class label.
    """
    record = expl_set.get_record()
    cf_list = expl_set.get_list_expl_full()
    new_preds = expl_set.get_new_preds()

    # 1) For each unique class label appearing in new_preds, find the test row
    #    closest to `record` among rows where y_pred == that label. Cache it.
    closest_per_class = {}
    for y_cf_val in np.unique(new_preds):
        X_test_y = X_test[y_cf_val == y_pred]
        if len(X_test_y) == 0:
            closest_per_class[y_cf_val] = None
            continue
        neigh_dist = hybrid_distance(
            record.reshape(1, -1), X_test_y,
            continuous_features_all, categorical_features_all,
            ratio_cont=ratio_cont, cont_metric=cont_metric, cate_metric='hamming',
        )
        # neigh_dist may be a 2D array (when ratio_cont collapses to dist_cont) shaped (1, n).
        neigh_dist = np.asarray(neigh_dist)
        if neigh_dist.ndim == 2:
            neigh_dist = neigh_dist[0]
        idx_neigh = int(np.argsort(neigh_dist)[0])
        closest_per_class[y_cf_val] = X_test_y[idx_neigh]

    # 2) Sum d(cf_i, closest_for_its_class). Cheap per-CF call; the expensive first
    #    leg is now done at most once per unique class label above.
    sum_dist = 0.0
    for cf, y_cf_val in zip(cf_list, new_preds):
        closest = closest_per_class.get(y_cf_val)
        if closest is None:
            continue
        d = hybrid_distance(
            cf, closest.reshape(1, -1),
            continuous_features_all, categorical_features_all,
            ratio_cont=ratio_cont, cont_metric=cont_metric, cate_metric='hamming',
        )
        sum_dist += float(np.asarray(d).item())
    return sum_dist
