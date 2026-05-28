from copy import deepcopy

import pandas as pd

# from evaluation._metrics import *
# from evaluation._utils import *
from evaluation.metrics import *
from evaluation.utils import *
from evaluation.utils import _combine_distances  # private; star-import skips it
from utils import OrdinalBinsEncoder


class ExplSetEvaluator:
    def __init__(self, model, X_train, X_test, cat_features, ord_features, num_features, act_features,
                 monotonic_features, mixed=False, binning_process=None):
        self.model = model
        # self.cat_features = cat_features
        # self.ord_features = ord_features
        # self.num_features = num_features
        self.act_features = act_features
        self.monotonic_features = monotonic_features
        self.X_train: pd.DataFrame = X_train
        self.X_test: pd.DataFrame = X_test
        self.mixed = mixed
        self.features = self.X_train.columns

        self.N_features = len(self.features)

        self.y_pred = self.model.predict(X_test)
        self.X_train_, self.X_test_ = self.X_train, self.X_test  # keep original dataframe

        if self.mixed:
            ...  # todo
            # In mixed mode the 'mh' family is MAD-cityblock + Hamming, so the
            # continuous metric is built from MAD on training data (see below).
            self._mh_cont_metric_kind = 'mad'
        else:  # binned
            # self.cat_features, self.num_features = [], []
            # self.ord_features = self.features
            # Use numerical distances on ordinal bins
            self.cat_features, self.ord_features = [], []
            self.num_features = list(range(len(self.features)))
            self.ratio_cont = 1.0
            if binning_process is None:
                raise ValueError("Binning process must be provided for binned evaluation.")
            else:
                self.binning_process = binning_process

            self.prep = OrdinalBinsEncoder(self.features, self.monotonic_features, self.binning_process)

            # In binned mode the 'mh' family is plain Cityblock + Hamming.
            self._mh_cont_metric_kind = 'cityblock'

        self.prep.fit(self.X_train)
        self.X_train = self.prep.transform(self.X_train)
        self.X_test = self.prep.transform(self.X_test)

        self.act_features = [i for i, f in enumerate(self.features) if f in act_features]

        # Build the metric callables ONCE.
        #
        # `_mad_metric`: used for `distance_mad` / `diversity_mad` and for `plausibility`
        #   (which always uses MAD-cityblock regardless of binned/mixed mode).
        # `_mh_cont_metric`: continuous metric for the 'mh' family; callable, so the
        #   binned-vs-mixed choice is captured by *which* callable we store here, not
        #   by a string we have to branch on later.
        if len(self.num_features) > 0:
            self._mad_metric = make_mad_metric(np.asarray(self.X_train), self.num_features)
        else:
            # No continuous features: a no-op callable. It will never be exercised
            # because cdist over zero columns short-circuits before calling it.
            self._mad_metric = lambda u, v: 0.0

        if self._mh_cont_metric_kind == 'mad':
            self._mh_cont_metric = self._mad_metric
        else:
            self._mh_cont_metric = self._mh_cont_metric_kind  # 'cityblock' string

    def evaluate(self, expl_set, max_num_cf):
        num_cf_ = len(expl_set)

        if num_cf_ > 0:
            expl_set = deepcopy(expl_set)
            x = expl_set.get_record()
            cf_list = expl_set.get_list_expl_full()

            x = x.to_frame().T

            x = self.prep.transform(x).astype(float)[0]
            cf_list = self.prep.transform(cf_list).astype(float)

            expl_set.record, expl_set.list_expl_full = x, cf_list
            expl_set.list_new_preds = expl_set.list_new_preds.values
            expl_set.list_new_probs = expl_set.list_new_probs.values
            act_features = self.act_features

            num_valid_cf_ = num_valid_cf(expl_set)
            perc_valid_cf_ = perc_valid_cf(expl_set, k=num_cf_)
            perc_valid_cf_all_ = perc_valid_cf(expl_set, k=max_num_cf)
            num_act_cf_ = num_actionable_cf(expl_set, act_features)
            perc_act_cf_ = perc_actionable_cf(expl_set, act_features, k=num_cf_)
            perc_act_cf_all_ = perc_actionable_cf(expl_set, act_features, k=max_num_cf)
            num_valid_act_cf_ = num_valid_actionable_cf(expl_set, act_features)
            perc_valid_act_cf_ = perc_valid_actionable_cf(expl_set, act_features, k=num_cf_)
            perc_valid_act_cf_all_ = perc_valid_actionable_cf(expl_set, act_features, k=max_num_cf)
            avg_num_violations_per_cf_ = avg_num_violations_per_cf(expl_set, act_features)
            avg_num_violations_ = avg_num_violations(expl_set, act_features)

            plaus_sum = plausibility(
                expl_set, self.X_test, self.y_pred, self.num_features,
                self.cat_features, self.ratio_cont, cont_metric=self._mad_metric,
            )
            plaus_max_num_cf_ = plaus_sum / max_num_cf
            plaus_num_cf_ = plaus_sum / num_cf_
            plaus_num_valid_cf_ = plaus_sum / num_valid_cf_ if num_valid_cf_ > 0 else plaus_sum
            plaus_num_act_cf_ = plaus_sum / num_act_cf_ if num_act_cf_ > 0 else plaus_sum
            plaus_num_valid_act_cf_ = plaus_sum / num_valid_act_cf_ if num_valid_act_cf_ > 0 else plaus_sum

            # ---- Distance block (refactored) ----------------------------------------
            # Each base distance matrix is computed once with agg=None; mean/min/max
            # are derived by post-hoc aggregation. The hybrid (l2j, mh) variants are
            # then built from the already-aggregated scalars and combined with the same
            # logic as hybrid_distance() so values match exactly.
            #
            # `self._mh_cont_metric` is a string in binned mode ('cityblock') or a
            # callable in mixed mode (MAD). We don't branch on it: cdist accepts both.
            # If it happens to be the same callable as `_mad_metric` (mixed mode),
            # we reuse D_mad.
            D_l2 = continuous_distance(x, cf_list, self.num_features, metric='euclidean', agg=None)
            D_mad = continuous_distance(x, cf_list, self.num_features, metric=self._mad_metric, agg=None)
            D_j = categorical_distance(x, cf_list, self.cat_features, metric='jaccard', agg=None)
            D_h = categorical_distance(x, cf_list, self.cat_features, metric='hamming', agg=None)

            if self._mh_cont_metric is self._mad_metric:
                D_mh_cont = D_mad
            else:
                D_mh_cont = continuous_distance(
                    x, cf_list, self.num_features, metric=self._mh_cont_metric, agg=None
                )

            distance_l2_, distance_l2_min_, distance_l2_max_ = D_l2.mean(), D_l2.min(), D_l2.max()
            distance_mad_, distance_mad_min_, distance_mad_max_ = D_mad.mean(), D_mad.min(), D_mad.max()
            distance_j_, distance_j_min_, distance_j_max_ = D_j.mean(), D_j.min(), D_j.max()
            distance_h_, distance_h_min_, distance_h_max_ = D_h.mean(), D_h.min(), D_h.max()
            mh_cont_mean, mh_cont_min, mh_cont_max = D_mh_cont.mean(), D_mh_cont.min(), D_mh_cont.max()

            # Hybrid distances: replicate hybrid_distance()'s scalar combination.
            # n_total_vars uses *variable* units (continuous + active categorical
            # one-hot count), matching the bug-fixed hybrid_distance.
            n_cont = len(self.num_features)
            num_cate_active = int(np.sum(x[self.cat_features])) if len(self.cat_features) > 0 else 0
            n_total_vars = n_cont + num_cate_active

            def _combine_dist_scalar(cont_scalar, cate_scalar, cate_metric):
                # Mirrors hybrid_distance: normalize continuous by feature count,
                # halve hamming for one-hot, then convex-combine via _combine_distances.
                c = cont_scalar / n_cont if n_cont > 0 else cont_scalar
                k = cate_scalar / 2 if cate_metric == 'hamming' else cate_scalar
                return _combine_distances(
                    c, k, n_total_vars, n_cont, num_cate_active, self.ratio_cont
                )

            distance_l2j_      = _combine_dist_scalar(distance_l2_,      distance_j_,      'jaccard')
            distance_l2j_min_  = _combine_dist_scalar(distance_l2_min_,  distance_j_min_,  'jaccard')
            distance_l2j_max_  = _combine_dist_scalar(distance_l2_max_,  distance_j_max_,  'jaccard')
            distance_mh_       = _combine_dist_scalar(mh_cont_mean,      distance_h_,      'hamming')
            distance_mh_min_   = _combine_dist_scalar(mh_cont_min,       distance_h_min_,  'hamming')
            distance_mh_max_   = _combine_dist_scalar(mh_cont_max,       distance_h_max_,  'hamming')

            avg_num_changes_per_cf_ = avg_num_changes_per_cf(expl_set, self.num_features)
            avg_num_changes_ = avg_num_changes(expl_set, self.N_features, self.num_features)

            delta_ = delta_proba(expl_set, agg='mean')
            delta_min_ = delta_proba(expl_set, agg='min')
            delta_max_ = delta_proba(expl_set, agg='max')

            if num_cf_ > 1:
                # ---- Diversity block (refactored) -----------------------------------
                # Same pattern as the distance block: compute each base pdist matrix
                # once, derive mean/min/max from it, and combine for hybrids.
                # Note on diversity_j_ / diversity_h_: the original code called
                # categorical_diversity without agg, returning an array (a latent issue
                # masked in binned mode where cat_features is empty). Here they become
                # proper scalars via .mean() — semantically what the dict expects.
                Dv_l2 = continuous_diversity(cf_list, self.num_features, metric='euclidean', agg=None)
                Dv_mad = continuous_diversity(cf_list, self.num_features, metric=self._mad_metric, agg=None)
                Dv_j = categorical_diversity(cf_list, self.cat_features, metric='jaccard', agg=None)
                Dv_h = categorical_diversity(cf_list, self.cat_features, metric='hamming', agg=None)

                if self._mh_cont_metric is self._mad_metric:
                    Dv_mh_cont = Dv_mad
                else:
                    Dv_mh_cont = continuous_diversity(
                        cf_list, self.num_features, metric=self._mh_cont_metric, agg=None
                    )

                diversity_l2_, diversity_l2_min_, diversity_l2_max_ = Dv_l2.mean(), Dv_l2.min(), Dv_l2.max()
                diversity_mad_, diversity_mad_min_, diversity_mad_max_ = Dv_mad.mean(), Dv_mad.min(), Dv_mad.max()
                diversity_j_, diversity_j_min_, diversity_j_max_ = Dv_j.mean(), Dv_j.min(), Dv_j.max()
                diversity_h_, diversity_h_min_, diversity_h_max_ = Dv_h.mean(), Dv_h.min(), Dv_h.max()
                dv_mh_cont_mean, dv_mh_cont_min, dv_mh_cont_max = (
                    Dv_mh_cont.mean(), Dv_mh_cont.min(), Dv_mh_cont.max()
                )

                # Hybrid diversities mirror hybrid_diversity(): no /len(cont) or /2
                # normalization, and num_cate slot uses len(cat_features), not active count.
                n_cat = len(self.cat_features)
                n_total_vars_div = n_cont + n_cat

                def _combine_div_scalar(cont_scalar, cate_scalar):
                    return _combine_distances(
                        cont_scalar, cate_scalar, n_total_vars_div, n_cont, n_cat, self.ratio_cont
                    )

                diversity_l2j_     = _combine_div_scalar(diversity_l2_,     diversity_j_)
                diversity_l2j_min_ = _combine_div_scalar(diversity_l2_min_, diversity_j_min_)
                diversity_l2j_max_ = _combine_div_scalar(diversity_l2_max_, diversity_j_max_)
                diversity_mh_      = _combine_div_scalar(dv_mh_cont_mean,    diversity_h_)
                diversity_mh_min_  = _combine_div_scalar(dv_mh_cont_min,     diversity_h_min_)
                diversity_mh_max_  = _combine_div_scalar(dv_mh_cont_max,     diversity_h_max_)

            else:
                diversity_l2_ = 0.0
                diversity_mad_ = 0.0
                diversity_j_ = 0.0
                diversity_h_ = 0.0
                diversity_l2j_ = 0.0
                diversity_mh_ = 0.0

                diversity_l2_min_ = 0.0
                diversity_mad_min_ = 0.0
                diversity_j_min_ = 0.0
                diversity_h_min_ = 0.0
                diversity_l2j_min_ = 0.0
                diversity_mh_min_ = 0.0

                diversity_l2_max_ = 0.0
                diversity_mad_max_ = 0.0
                diversity_j_max_ = 0.0
                diversity_h_max_ = 0.0
                diversity_l2j_max_ = 0.0
                diversity_mh_max_ = 0.0

            count_diversity_cont_ = count_diversity(
                expl_set, self.num_features, self.N_features, self.num_features
            )
            count_diversity_cate_ = count_diversity(
                expl_set, self.cat_features, self.N_features, self.num_features
            )
            count_diversity_all_ = count_diversity(
                expl_set, self.features, self.N_features, self.num_features
            )  # todo: not used if there is only one cf
            accuracy_knn_sklearn_ = accuracy_knn_sklearn(
                expl_set, self.X_test, self.X_test_, self.model, self.num_features,
                self.cat_features, test_size=5
            )

            # lof_ = lof(expl_set, self.X_train)
            lof_ = np.nan

            res = {
                'num_cf_': num_cf_,
                'num_valid_cf': num_valid_cf_,
                'perc_valid_cf': perc_valid_cf_,
                'perc_valid_cf_all': perc_valid_cf_all_,
                'num_act_cf': num_act_cf_,
                'perc_act_cf': perc_act_cf_,
                'perc_act_cf_all': perc_act_cf_all_,
                'num_valid_act_cf': num_valid_act_cf_,
                'perc_valid_act_cf': perc_valid_act_cf_,
                'perc_valid_act_cf_all': perc_valid_act_cf_all_,
                'avg_num_violations_per_cf': avg_num_violations_per_cf_,
                'avg_num_violations': avg_num_violations_,
                'distance_l2': distance_l2_,
                'distance_mad': distance_mad_,
                'distance_j': distance_j_,
                'distance_h': distance_h_,
                'distance_l2j': distance_l2j_,
                'distance_mh': distance_mh_,
                'avg_num_changes_per_cf': avg_num_changes_per_cf_,
                'avg_num_changes': avg_num_changes_,

                'distance_l2_min': distance_l2_min_,
                'distance_mad_min': distance_mad_min_,
                'distance_j_min': distance_j_min_,
                'distance_h_min': distance_h_min_,
                'distance_l2j_min': distance_l2j_min_,
                'distance_mh_min': distance_mh_min_,

                'distance_l2_max': distance_l2_max_,
                'distance_mad_max': distance_mad_max_,
                'distance_j_max': distance_j_max_,
                'distance_h_max': distance_h_max_,
                'distance_l2j_max': distance_l2j_max_,
                'distance_mh_max': distance_mh_max_,

                'diversity_l2': diversity_l2_,
                'diversity_mad': diversity_mad_,
                'diversity_j': diversity_j_,
                'diversity_h': diversity_h_,
                'diversity_l2j': diversity_l2j_,
                'diversity_mh': diversity_mh_,

                'diversity_l2_min': diversity_l2_min_,
                'diversity_mad_min': diversity_mad_min_,
                'diversity_j_min': diversity_j_min_,
                'diversity_h_min': diversity_h_min_,
                'diversity_l2j_min': diversity_l2j_min_,
                'diversity_mh_min': diversity_mh_min_,

                'diversity_l2_max': diversity_l2_max_,
                'diversity_mad_max': diversity_mad_max_,
                'diversity_j_max': diversity_j_max_,
                'diversity_h_max': diversity_h_max_,
                'diversity_l2j_max': diversity_l2j_max_,
                'diversity_mh_max': diversity_mh_max_,

                'count_diversity_cont': count_diversity_cont_,
                'count_diversity_cate': count_diversity_cate_,
                'count_diversity_all': count_diversity_all_,
                'accuracy_knn_sklearn': accuracy_knn_sklearn_,
                'lof': lof_,

                'delta': delta_,
                'delta_min': delta_min_,
                'delta_max': delta_max_,

                'plaus_sum': plaus_sum,
                'plaus_max_num_cf': plaus_max_num_cf_,
                'plaus_num_cf': plaus_num_cf_,
                'plaus_num_valid_cf': plaus_num_valid_cf_,
                'plaus_num_act_cf': plaus_num_act_cf_,
                'plaus_num_valid_act_cf': plaus_num_valid_act_cf_,
            }

        else:
            res = {
                'num_cf_': num_cf_,
                'num_valid_cf': 0.0,
                'perc_valid_cf': 0.0,
                'perc_valid_cf_all': 0.0,
                'num_act_cf': 0.0,
                'perc_act_cf': 0.0,
                'perc_act_cf_all': 0.0,
                'num_valid_act_cf': 0.0,
                'perc_valid_act_cf': 0.0,
                'perc_valid_act_cf_all': 0.0,
                'avg_num_violations_per_cf': np.nan,
                'avg_num_violations': np.nan,
                'distance_l2': np.nan,
                'distance_mad': np.nan,
                'distance_j': np.nan,
                'distance_h': np.nan,
                'distance_l2j': np.nan,
                'distance_mh': np.nan,
                'distance_l2_min': np.nan,
                'distance_mad_min': np.nan,
                'distance_j_min': np.nan,
                'distance_h_min': np.nan,
                'distance_l2j_min': np.nan,
                'distance_mh_min': np.nan,
                'distance_l2_max': np.nan,
                'distance_mad_max': np.nan,
                'distance_j_max': np.nan,
                'distance_h_max': np.nan,
                'distance_l2j_max': np.nan,
                'distance_mh_max': np.nan,
                'avg_num_changes_per_cf': np.nan,
                'avg_num_changes': np.nan,
                'diversity_l2': np.nan,
                'diversity_mad': np.nan,
                'diversity_j': np.nan,
                'diversity_h': np.nan,
                'diversity_l2j': np.nan,
                'diversity_mh': np.nan,
                'diversity_l2_min': np.nan,
                'diversity_mad_min': np.nan,
                'diversity_j_min': np.nan,
                'diversity_h_min': np.nan,
                'diversity_l2j_min': np.nan,
                'diversity_mh_min': np.nan,
                'diversity_l2_max': np.nan,
                'diversity_mad_max': np.nan,
                'diversity_j_max': np.nan,
                'diversity_h_max': np.nan,
                'diversity_l2j_max': np.nan,
                'diversity_mh_max': np.nan,
                'count_diversity_cont': np.nan,
                'count_diversity_cate': np.nan,
                'count_diversity_all': np.nan,
                'accuracy_knn_sklearn': 0.0,
                'accuracy_knn_dist': 0.0,
                'lof': np.nan,
                'delta': 0.0,
                'delta_min': 0.0,
                'delta_max': 0.0,

                'plaus_sum': 0.0,
                'plaus_max_num_cf': 0.0,
                'plaus_num_cf': 0.0,
                'plaus_num_valid_cf': 0.0,
                'plaus_num_act_cf': 0.0,
                'plaus_num_valid_act_cf': 0.0,
            }

        return res
