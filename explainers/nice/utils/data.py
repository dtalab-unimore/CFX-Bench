import numpy as np
from typing import Optional, Union, List, Callable


class InstanceNICE:
    """
    Represents one explanation context:
    the specific instance x, its predicted and target classes,
    and the candidate subset drawn from the training data.
    """

    def __init__(
        self,
        x: np.ndarray,
        target_class: Union[str, int, List[int]],
        data_ref: "DataNICE",
    ):
        self.data_ref = data_ref
        self.x = self._as_float(x)
        self.x_score = self.data_ref.predict_fn(self.x)
        self.x_class = int(np.argmax(self.x_score))
        self.target_class = self._resolve_target_class(target_class)
        self.class_mask = np.isin(self.data_ref.X_train_class, self.target_class)
        self.mask = self.class_mask & self.data_ref.candidates_mask
        self.candidates_view = self.data_ref.X_train[self.mask, :]

    # ---------------------------------------------------------------------
    def _as_float(self, X: np.ndarray) -> np.ndarray:
        """Ensure numeric features are float64."""
        X = X.copy()
        X[:, self.data_ref.num_feat] = X[:, self.data_ref.num_feat].astype(np.float64)
        return X

    def _resolve_target_class(self, target_class) -> List[int]:
        n_classes = self.data_ref.n_classes
        if target_class == "other":
            return [i for i in range(n_classes) if i != self.x_class]
        elif isinstance(target_class, int):
            return [target_class]
        return list(target_class)


class DataNICE:
    """
    Immutable wrapper around the full training dataset and model interface.
    Creates per-instance contexts via `for_instance`.
    """

    def __init__(
        self,
        X_train: np.ndarray,
        y_train: Optional[np.ndarray],
        cat_feat: List[int],
        num_feat: Union[List[int], str],
        predict_fn: Callable[[np.ndarray], np.ndarray],
        justified_cf: bool = True,
        eps: float = 1e-11,
    ):
        self.cat_feat = cat_feat
        self.num_feat = self._infer_num_features(num_feat, X_train.shape[1], cat_feat)
        self.predict_fn = predict_fn
        self.justified_cf = justified_cf
        self.eps = eps

        # Normalize numeric columns
        self.X_train = self._as_float(X_train)
        self.y_train = y_train

        # Model predictions and derived training classes
        self.train_proba = predict_fn(self.X_train)
        self.n_classes = self.train_proba.shape[1]
        self.X_train_class = np.argmax(self.train_proba, axis=1)

        # Candidate mask for justified counterfactuals
        if self.justified_cf and y_train is not None:
            self.candidates_mask = self.y_train == self.X_train_class
        else:
            self.candidates_mask = np.ones(self.X_train.shape[0], dtype=bool)

    # ---------------------------------------------------------------------
    def _infer_num_features(self, num_feat, n_features: int, cat_feat: List[int]) -> List[int]:
        if num_feat == "auto":
            return [i for i in range(n_features) if i not in cat_feat]
        return num_feat

    def _as_float(self, X: np.ndarray) -> np.ndarray:
        X = X.copy()
        X[:, self.num_feat] = X[:, self.num_feat].astype(np.float64)
        return X

    # ---------------------------------------------------------------------
    def for_instance(self, x: np.ndarray, target_class: Union[str, int, List[int]]) -> InstanceNICE:
        """Return a new immutable per-instance context."""
        return InstanceNICE(x, target_class, self)
