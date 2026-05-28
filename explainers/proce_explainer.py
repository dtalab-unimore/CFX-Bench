import os

import numpy as np
import pandas as pd
import torch
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.optimize import minimize
from pymoo.termination.default import DefaultMultiObjectiveTermination
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split

from explainers.base import BaseExplainer, _empty_explanation_dict, prepare_output
from explainers.proce.df_encoder.autoencoder import AutoEncoder
from explainers.proce.explainer.distance import Distance
from explainers.proce.explainer.prototype import get_pos_neg_latent, find_proto
from utils import OrdinalBinsEncoder


class ProCEProblemAE(Problem):
    # merged: ProCEProblemAE + its former superclass _ProCEProblem.
    # Methods overridden by ProCEProblemAE (_n_obj/_n_ieq_constr, _evaluate) keep the
    # subclass version; the rest is inherited from _ProCEProblem.

    _n_obj, _n_ieq_constr = 3, 0

    def __init__(
        self, *,
        x0,
        ord_enc,
        pred_model,
        dfencoder_model,
        vars_,
        xl,
        xu,
        proto=None,
        con_index=None,
        dict_cat_index=None,
    ):
        super().__init__(
            vars=vars_,
            xl=xl,
            xu=xu,
            vtype=int,
            n_obj=self._n_obj,
            n_ieq_constr=self._n_ieq_constr,
        )

        self.x0 = x0
        self.ord_enc = ord_enc
        self.pred_model = pred_model
        self.dfencoder_model = dfencoder_model
        self.proto = proto
        self.con_index = con_index
        self.dict_cat_index = dict_cat_index

        self.x0_bins = self._inverse_transform(self.x0)

    def _inverse_transform(self, X: np.ndarray) -> pd.DataFrame:
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        X_orig = pd.DataFrame(self.ord_enc.inverse_transform(X), columns=self.ord_enc.feature_names_in_)
        return X_orig

    def _evaluate(self, xcf, out, *args, **kwargs):
        x0_bins = self.x0_bins
        xcf_bins = self._inverse_transform(xcf)
        dict_cat_index = {k.replace('.', '_'): v for k, v in self.dict_cat_index.items()}

        y_loss, cat_dist, p_loss = [], [], []
        for i in range(xcf_bins.shape[0]):  # elementwise
            cf_bins = xcf_bins.iloc[i:i+1]
            dist = Distance(
                x0_bins.values[0], cf_bins.values[0], self.pred_model, self.dfencoder_model,
                con_index=self.con_index, dict_cat_index=dict_cat_index
            )

            yloss = dist.cross_entropy()
            catdist = dist.cat_representation_dist()

            zcf = self.dfencoder_model.get_representation(cf_bins.rename(columns=lambda x: x.replace('.', '_')))[0]
            ploss = dist.proto_loss(zcf, self.proto)

            y_loss.append(yloss)
            cat_dist.append(catdist)
            p_loss.append(ploss)

        out["F"] = np.column_stack([y_loss, cat_dist, p_loss])

    def _to_array(self, X) -> np.ndarray:
        if isinstance(X[0], dict):
            return pd.DataFrame.from_records(X).to_numpy()
        else:
            return np.asarray(X)


def _AutoEncoder(random_state=None, verbose=False, emb_size=512):
    return AutoEncoder(
        encoder_layers=[512, 512, emb_size],  # model architecture
        decoder_layers=[],  # decoder optional - you can create bottlenecks if you like
        activation='relu',
        swap_p=0.2,  # noise parameter
        lr=0.01,
        lr_decay=.99,
        # batch_size=512,
        batch_size=64,
        eval_batch_size=64,
        verbose=verbose,
        optimizer='sgd',
        scaler='gauss_rank',  # gauss rank scaling forces your numeric features into standard normal distribution
        random_state=random_state
    )


def _find_best_solution(solutions, x0, predict_fn, distance_fn):
    y_prediction = predict_fn(solutions)
    pos_index = np.where(y_prediction == 1)[0]
    filtered_arr = solutions[pos_index]
    if len(filtered_arr) == 0:
        return None
    x0, filtered_arr = x0.reshape(1, -1), filtered_arr
    # distance = cdist(x0, filtered_arr, 'euclidean')[0]
    distance = distance_fn(x0, filtered_arr)[0]
    min_index = np.where(distance == min(distance))[0]
    return filtered_arr[[min_index[0]]]


class ProCEAEBinnedExplainer(BaseExplainer):  # legacy implementation with autoencoder
    # merged: ProCEAEBinnedExplainer + its former superclasses ProCEBinnedExplainer and
    # ProCEExplainer.
    # - __init__, _init and _get_problem keep the ProCEAEBinnedExplainer (AE) version;
    #   the parts of ProCEBinnedExplainer reached via super() are inlined.
    # - _distance_fn, _weighted_distance_fn, _transform, _inverse_transform keep the
    #   ProCEAEBinnedExplainer version (the ProCEExplainer stubs are dropped).
    # - _adjust_bounds, _get_algorithm are inherited verbatim from ProCEBinnedExplainer.
    # - _explain is inherited verbatim from ProCEExplainer.

    def __init__(self, model, X_train: pd.DataFrame, features, act_features, target, ae_dir: str,
                 monotonic_features: dict = None, random_state: int = None, verbose=False, **kwargs):
        self.ae_dir = ae_dir

        # --- inlined from ProCEBinnedExplainer.__init__ ---
        self.random_state = random_state
        self.verbose = verbose
        super().__init__(
            model, X_train, None, features, features, None, act_features, target, monotonic_features
        )

    def _init(self):
        # --- inlined from ProCEBinnedExplainer._init ---
        X_bins = self.X_train.astype('category')

        self.ord_enc = OrdinalBinsEncoder(self.features, self.monotonic_features, self.model.binning_process_)

        X_ord = self.ord_enc.fit_transform(X_bins)
        X_ord = pd.DataFrame(X_ord, columns=X_bins.columns)
        self.xl, self.xu = X_ord.min().values, X_ord.max().values

        # --- ProCEAEBinnedExplainer-specific autoencoder setup ---
        X_bins = self.X_train.astype('category')
        y_pred = self.model.predict(self.X_train)

        categorical_index = {}
        for i, f in enumerate(X_bins.columns):
            if len(X_bins[f].unique()) > 2:  # categorical, not binary
                categorical_index[f] = i
        self.categorical_index = categorical_index

        ae_path = self.ae_dir + '/model.pth'
        self.rename_cols = lambda x: x.replace('.', '_')
        if os.path.exists(ae_path):
            print(f"Loading autoencoder model from {ae_path}")
            ae_model = torch.load(ae_path, weights_only=False)
            ae_model.to('cpu')
        else:
            ae_model = _AutoEncoder(random_state=self.random_state)
            ae_model.to('cpu')
            X_train = X_bins.copy().rename(columns=self.rename_cols)
            X_train, X_val = train_test_split(X_train, train_size=0.8, random_state=self.random_state)
            ae_model.fit(
                X_train, epochs=100, val=X_val
            )

            ae_dir = os.path.dirname(ae_path)
            if not os.path.exists(ae_dir):
                os.makedirs(ae_dir)
            print(f"Saving autoencoder model to {ae_path}")
            torch.save(ae_model, ae_path)
        self.ae_model = ae_model

        z_representation = self.ae_model.get_representation(X_bins.rename(columns=self.rename_cols))
        self.pos_z, self.neg_z = get_pos_neg_latent(torch.tensor(y_pred), z_representation)

    def _distance_fn(self, a, b):
        return cdist(a, b, 'euclidean')

    def _weighted_distance_fn(self, a, b, weights=None):
        dist = np.abs(a[:, np.newaxis, :] - b[np.newaxis, :, :])
        if weights is not None:
            dist *= weights
        dist = dist.sum(axis=-1)
        return dist

    def _transform(self, X: pd.DataFrame) -> np.ndarray:
        X = self.ord_enc.transform(X)
        return X

    def _inverse_transform(self, X: np.ndarray) -> pd.DataFrame:
        X = self.ord_enc.inverse_transform(X)
        X = pd.DataFrame(X, columns=self.ord_enc.feature_names_in_)
        return X

    def _adjust_bounds(self, x0):
        xl_, xu_ = self.xl.copy(), self.xu.copy()
        # fix immutable features by setting their lower and upper bounds to the original value
        for idx, feature in enumerate(self.features):
            if feature in self.act_features:
                if feature in self.monotonic_features:
                    # IMPORTANT: assume categories in ordinal encoder are in monotonic order, and monotonicity is increasing
                    xl_[idx] = x0[idx]
            else:
                xl_[idx] = x0[idx]
                xu_[idx] = x0[idx]
        return xl_, xu_

    def _get_problem(self, x0):
        ae_model = self.ae_model
        x0_bins = pd.DataFrame(self.ord_enc.inverse_transform([x0]), columns=self.ord_enc.feature_names_in_)
        z0 = ae_model.get_representation(x0_bins.rename(columns=self.rename_cols))
        K = 10  # fixme: add as parameter
        pos_proto, _ = find_proto(z0, self.pos_z, self.neg_z, K)

        vars_ = self.ord_enc.feature_names_in_.tolist()
        xl_, xu_ = self._adjust_bounds(x0)
        problem = ProCEProblemAE(
            x0=x0,
            ord_enc=self.ord_enc,
            pred_model=self.model,
            dfencoder_model=ae_model,
            vars_=vars_,
            xl=xl_,
            xu=xu_,
            proto=pos_proto,
            con_index=[],
            dict_cat_index=self.categorical_index
        )
        return problem

    def _get_algorithm(self):
        sampling = IntegerRandomSampling()
        crossover = SBX(vtype=float, repair=RoundingRepair())
        mutation = PM(vtype=float, repair=RoundingRepair())
        algorithm = NSGA2(
            pop_size=40,
            sampling=sampling,
            crossover=crossover,
            mutation=mutation,
            eliminate_duplicates=True
        )
        return algorithm

    # --- inherited verbatim from ProCEExplainer ---

    def _explain(self, test_item, n_cf=1):
        record, label, pred, proba, target = test_item
        x0_df = record.copy().to_frame().T
        x0 = self._transform(x0_df)[0]

        problem = self._get_problem(x0)

        algorithm = self._get_algorithm()

        # termination = None
        termination = DefaultMultiObjectiveTermination(
            period=10,
            n_max_gen=100,
            n_skip=0,
        )

        res = minimize(
            problem,
            algorithm,
            termination,
            save_history=True,
            seed=self.random_state,
            verbose=self.verbose
        )
        sols = res.X  # solutions
        if sols is None:
            return _empty_explanation_dict(test_item)
        if isinstance(sols, dict):
            sols = [sols]
        sols = problem._to_array(sols)
        if len(np.shape(sols)) == 1:
            sols = np.reshape(sols, (1, -1))
        if self.num_features:
            sols[:, self.num_ids] = sols[:, self.num_ids].astype(float) / 100  # percentile

        cfs = _find_best_solution(
            sols, x0,
            predict_fn=lambda x: self.model.predict(self._inverse_transform(x)),
            distance_fn=self._distance_fn
        )

        if cfs is None:
            return _empty_explanation_dict(test_item)

        list_expl_full = self._inverse_transform(cfs)
        list_new_probs = self.model.predict_proba(list_expl_full)[:, 1]
        '''list_expl_changes = []
        for i in range(list_expl_full.shape[0]):
            cf = list_expl_full.iloc[i]
            changes = []
            for col in list_expl_full.columns:
                if cf[col] != record[col]:
                    changes.append(cf[col])
                else:
                    changes.append('-')
                """if self.cat_features and col in self.cat_features and cf[col] != record[col]:
                    changes.append(cf[col])
                elif self.num_features and col in self.num_features and not np.isclose(cf[col], record[col]):
                    changes.append(cf[col])
                else:
                    changes.append('-')"""
            list_expl_changes.append(changes)
        list_expl_changes = pd.DataFrame(list_expl_changes, columns=list_expl_full.columns)

        expl_dict = {
            'record': record, 'label': label, 'pred': pred, 'proba': proba, 'target': target,
            'list_expl_full': list_expl_full, 'list_expl_changes': list_expl_changes,
            'list_new_probs': pd.Series(list_new_probs)
        }'''
        expl_dict = prepare_output(*test_item, list_expl_full, list_new_probs)
        return expl_dict
