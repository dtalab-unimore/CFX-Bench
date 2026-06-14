import logging
from typing import Dict, Optional, List, Callable

import numpy as np
import pandas as pd

from . import recourse as rs

logger = logging.getLogger(__name__)


class ActionableRecourse:
    """
    Implementation of Actionable Recourse from Ustun et.al. [1]
    without the Carla API dependencies.

    Parameters
    ----------
    predict_proba : Callable
        A function that takes a 2D array of instances and returns 
        the prediction probabilities (e.g., clf.predict_proba).
    X_train : pd.DataFrame
        Training data used to initialize the ActionSet and LIME explainer.
    y_train : pd.Series or np.ndarray
        Training labels used for LIME explainer.
    feature_names : List[str]
        Ordered list of feature names.
    continuous_features : List[str]
        List of continuous feature names.
    immutable_features : List[str]
        List of immutable feature names that cannot be changed.
    hyperparams : dict, optional
        Dictionary containing hyperparameters. 
    coeffs : np.ndarray, optional
        Global coefficients. Will be approximated locally by LIME if None.
    intercepts: np.ndarray, optional
        Global intercepts. Will be approximated locally by LIME if None.
    y_desired : int, default=1
        The desired target class for the counterfactuals.

    Notes
    -----
    - Hyperparams
        * "fs_size": int, default: 100 (Size of generated flipset)
        * "discretize": bool, default: False (Parameter for LIME sampling)
        * "sample": bool, default: True (LIME sampling around instance)

    .. [1] Berk Ustun, Alexander Spangher, and Y. Liu. 2019. Actionable Recourse in Linear Classification.
        In Proceedings of the Conference on Fairness, Accountability, and Transparency (FAT*)
    """

    _DEFAULT_HYPERPARAMS = {
        "fs_size": 100,
        "discretize": False,
        "sample": True,
    }

    def __init__(
            self,
            predict_proba: Callable,
            X_train: pd.DataFrame,
            y_train: pd.Series,
            feature_names: List[str],
            continuous_features: List[str],
            immutable_features: List[str],
            hyperparams: Optional[Dict] = None,
            coeffs: Optional[np.ndarray] = None,
            intercepts: Optional[np.ndarray] = None,
            y_desired: int = 1
    ) -> None:

        self.predict_proba = predict_proba
        self.X_train = X_train
        self.y_train = y_train
        self.feature_names = feature_names
        self.continuous_features = continuous_features
        self.immutable_features = immutable_features
        self.y_desired = y_desired

        # Merge hyperparams
        self.hyperparams = self._DEFAULT_HYPERPARAMS.copy()
        if hyperparams is not None:
            self.hyperparams.update(hyperparams)

        self._fs_size = self.hyperparams["fs_size"]
        self._discretize_continuous = self.hyperparams["discretize"]
        self._sample_around_instance = self.hyperparams["sample"]

        # Build ActionSet
        self.action_set = rs.ActionSet(
            X=self.X_train[self.feature_names],
            y_desired=self.y_desired
        )

        # Set immutable features
        for feature in self.immutable_features:
            if feature in self.action_set.name:
                self.action_set[feature].mutable = False
                self.action_set[feature].actionable = False

        self._coeffs = coeffs
        self._intercepts = intercepts

    def get_counterfactuals(self, factuals: pd.DataFrame) -> pd.DataFrame:
        """
        Generate counterfactual examples for given factual instances.
        """
        cfs = []
        coeffs = self._coeffs
        intercepts = self._intercepts

        # To keep matching indexes for iterrows and coeffs
        factuals = factuals.reset_index(drop=True)
        factuals = factuals[self.feature_names]

        # Check if we need LIME to build coefficients
        if (coeffs is None) and (intercepts is None):
            logger.info("Start generating LIME coefficients")
            coeffs, intercepts = self._get_lime_coefficients(factuals)
            logger.info("Finished generating LIME coefficients")
        else:
            # Broadcast global explanations to correct shape 
            coeffs = np.vstack([self._coeffs] * factuals.shape[0])
            intercepts = np.vstack([self._intercepts] * factuals.shape[0]).squeeze(axis=1)

        # Generate counterfactuals
        for index, row in factuals.iterrows():
            factual_enc_norm = row.values
            coeff = coeffs[index]
            intercept = intercepts[index]

            # Default counterfactual value if no action flips the prediction
            target_shape = factual_enc_norm.shape[0]
            counterfactual = np.empty(target_shape)
            counterfactual[:] = np.nan

            # Align action set to coefficients
            self.action_set.set_alignment(coefficients=coeff)

            # Build AR flipset
            fs = rs.Flipset(
                x=factual_enc_norm,
                action_set=self.action_set,
                coefficients=coeff,
                intercept=intercept,
            )

            try:
                fs_pop = fs.populate(total_items=self._fs_size, cost_type='total')
            except (ValueError, KeyError):
                logger.warning(
                    f"Actionable Recourse is not able to produce a counterfactual explanation for instance {index}"
                )
                logger.warning(row.values)
                cfs.append(counterfactual)
                continue

            # Get actions to flip predictions
            actions = fs_pop.actions

            for action in actions:
                candidate_cf = (factual_enc_norm + action).reshape((1, -1))

                # Check if candidate counterfactual really flips the prediction
                pred_cf = np.argmax(self.predict_proba(candidate_cf))
                pred_f = np.argmax(self.predict_proba(factual_enc_norm.reshape((1, -1))))

                if pred_cf != pred_f:
                    counterfactual = candidate_cf.squeeze()
                    break

            cfs.append(counterfactual)

        # Convert output into pandas DataFrame
        cfs = np.array(cfs)
        df_cfs = pd.DataFrame(cfs, columns=self.feature_names)

        # Append predicted probabilities/labels if useful, here just leaving as purely CF features
        return df_cfs
