import numpy as np
from scipy.spatial.distance import cdist, pdist
from scipy.stats import median_abs_deviation


# ==========================================
# Private Helper Functions
# ==========================================

def _validate_vector(u, dtype=None):
    u = np.asarray(u, dtype=dtype, order='c')
    if u.ndim == 1:
        return u
    raise ValueError("Input vector should be 1-D.")


def _aggregate(dist, agg):
    if agg == 'mean': return np.mean(dist)
    if agg == 'max':  return np.max(dist)
    if agg == 'min':  return np.min(dist)
    return dist


def _combine_distances(dist_cont, dist_cate, num_features, num_cont, num_cate, ratio_cont=None):
    """
    Convex-combine a continuous and a categorical distance.

    `num_features`, `num_cont`, `num_cate` are expected to be expressed in the SAME
    units (variables, not one-hot columns). The implicit ratio when `ratio_cont` is
    None is then `num_cont / num_features` and `num_cate / num_features`.
    """
    ratio_continuous = ratio_cont if ratio_cont is not None else (num_cont / num_features)
    if np.isclose(ratio_continuous, 1):
        return dist_cont
    elif np.isclose(ratio_continuous, 0):
        return dist_cate
    ratio_categorical = 1.0 - ratio_continuous if ratio_cont is not None else (num_cate / num_features)
    return (ratio_continuous * dist_cont) + (ratio_categorical * dist_cate)


# ==========================================
# Core Distance/Diversity Math
# ==========================================

def mad_cityblock(u, v, mad):
    """Compute the L1 (Manhattan) distance between two vectors, normalized by MAD."""
    u = _validate_vector(u)
    v = _validate_vector(v)
    return (abs(u - v) / mad).sum()


def make_mad_metric(X_train, continuous_features=None):
    """
    Build a MAD-cityblock metric callable from training data.

    Parameters
    ----------
    X_train : array-like
        Either the full training matrix (then `continuous_features` selects columns)
        or a precomputed MAD vector (1-D), in which case `continuous_features` is ignored.
    continuous_features : sequence of int, optional
        Column indices of `X_train` to compute MAD over. Required when `X_train` is 2-D.

    Returns
    -------
    callable(u, v) -> float
        A metric suitable for passing to scipy.spatial.distance.cdist / pdist.
    """
    arr = np.asarray(X_train)
    if arr.ndim == 1:
        mad = arr  # already a MAD vector
    else:
        if continuous_features is None:
            raise ValueError("continuous_features is required when X_train is 2-D.")
        mad = median_abs_deviation(arr[:, continuous_features], axis=0)
    mad = np.where(np.asarray(mad) == 0, 1.0, mad)
    return lambda u, v: mad_cityblock(u, v, mad)


def continuous_distance(x, cf_list, continuous_features, metric='euclidean', agg=None, **kwargs):
    """
    Compute continuous distances. `metric` may be any value accepted by scipy's cdist:
    a string identifier (e.g. 'euclidean', 'cityblock') or a callable f(u, v) -> float.

    Any extra keyword arguments are forwarded to cdist.
    """
    dist = cdist(x.reshape(1, -1)[:, continuous_features],
                 cf_list[:, continuous_features],
                 metric=metric, **kwargs)
    return _aggregate(dist, agg)


def categorical_distance(x, cf_list, categorical_features, metric='jaccard', agg=None, **kwargs):
    """
    Compute categorical distances. `metric` may be a string or a callable; see
    `continuous_distance` for details.
    """
    dist = cdist(x.reshape(1, -1)[:, categorical_features],
                 cf_list[:, categorical_features],
                 metric=metric, **kwargs)
    return _aggregate(dist, agg)


def continuous_diversity(cf_list, continuous_features, metric='euclidean', agg=None, **kwargs):
    """
    Compute pairwise continuous diversity. `metric` may be a string or a callable.
    """
    dist = pdist(cf_list[:, continuous_features], metric=metric, **kwargs)
    return _aggregate(dist, agg)


def categorical_diversity(cf_list, categorical_features, metric='jaccard', agg=None, **kwargs):
    """
    Compute pairwise categorical diversity. `metric` may be a string or a callable.
    """
    dist = pdist(cf_list[:, categorical_features], metric=metric, **kwargs)
    return _aggregate(dist, agg)


# ==========================================
# Unified Hybrid Functions
# ==========================================

def hybrid_distance(x, cf_list, continuous_features, categorical_features,
                    cont_metric='euclidean', cate_metric='jaccard',
                    ratio_cont=None, agg=None):
    """
    Hybrid distance combining a continuous metric and a categorical metric.

    `cont_metric` / `cate_metric` may be strings or callables. For data-dependent
    metrics like MAD-cityblock, build the callable with `make_mad_metric` and pass it in.

    Note on units (bug-fix vs. earlier versions): the implicit `ratio_cont = n_cont / N`
    is now computed in *variable* units, not one-hot column counts. With the documented
    one-hot-without-drop assumption, `n_cate_vars = sum(x[categorical_features])`.
    """
    dist_cont = continuous_distance(x, cf_list, continuous_features,
                                    metric=cont_metric, agg=agg)
    dist_cate = categorical_distance(x, cf_list, categorical_features,
                                     metric=cate_metric, agg=agg)

    if len(continuous_features) > 0:
        dist_cont = dist_cont / len(continuous_features)  # normalized
    if cate_metric == 'hamming':
        dist_cate = dist_cate / 2  # for one-hot encoded data

    # one-hot without drop, no all-zero rows: each categorical variable contributes
    # exactly one active column, so the active-column count == variable count.
    n_cate_vars = int(np.sum(x[categorical_features])) if len(categorical_features) > 0 else 0
    n_total_vars = len(continuous_features) + n_cate_vars

    return _combine_distances(dist_cont, dist_cate, n_total_vars,
                              len(continuous_features), n_cate_vars, ratio_cont)


def hybrid_diversity(cf_list, continuous_features, categorical_features,
                     cont_metric='euclidean', cate_metric='jaccard',
                     ratio_cont=None, agg=None):
    """
    Hybrid pairwise diversity. `cont_metric` / `cate_metric` may be strings or callables.
    """
    dist_cont = continuous_diversity(cf_list, continuous_features,
                                     metric=cont_metric, agg=agg)
    dist_cate = categorical_diversity(cf_list, categorical_features,
                                      metric=cate_metric, agg=agg)

    n_total_vars = len(continuous_features) + len(categorical_features)
    return _combine_distances(dist_cont, dist_cate, n_total_vars,
                              len(continuous_features), len(categorical_features), ratio_cont)


# ==========================================
# Legacy shims (preserve old call sites)
# ==========================================

# Note: `distance_mh` and `diversity_mh` used to live here. They took `X_train` and
# called `make_mad_metric` internally — i.e. they constructed a data-dependent metric
# inside a module-level function. That violates the convention that metrics are
# already declared (string or callable) by the time they reach `hybrid_distance` /
# `hybrid_diversity`. Callers that need MAD-cityblock + Hamming should now do:
#
#     mad_metric = make_mad_metric(X_train, continuous_features)
#     hybrid_distance(x, cf_list, continuous_features, categorical_features,
#                     cont_metric=mad_metric, cate_metric='hamming', ...)


def distance_l2j(x, cf_list, continuous_features, categorical_features, ratio_cont=None, agg=None):
    return hybrid_distance(x, cf_list, continuous_features, categorical_features,
                           cont_metric='euclidean', cate_metric='jaccard',
                           ratio_cont=ratio_cont, agg=agg)


def diversity_l2j(cf_list, continuous_features, categorical_features, ratio_cont=None, agg=None):
    return hybrid_diversity(cf_list, continuous_features, categorical_features,
                            cont_metric='euclidean', cate_metric='jaccard',
                            ratio_cont=ratio_cont, agg=agg)
