import numpy as np
from abc import ABC, abstractmethod


# -------------------------------------------------------------------------
# Abstract base classes
# -------------------------------------------------------------------------
class NumericDistance(ABC):
    @abstractmethod
    def measure(self, x, X):
        pass


class DistanceMetric(ABC):
    @abstractmethod
    def measure(self, x, X):
        pass


# -------------------------------------------------------------------------
# Numeric distances
# -------------------------------------------------------------------------
class StandardDistance(NumericDistance):
    def __init__(self, X_train: np.ndarray, num_feat: list, eps: float):
        self.num_feat = num_feat
        self.scale = X_train[:, num_feat].std(axis=0, dtype=np.float64)
        self.scale[self.scale < eps] = eps

    def measure(self, x: np.ndarray, X: np.ndarray) -> np.ndarray:
        diff = np.abs(X[:, self.num_feat] - x[0, self.num_feat]) / self.scale
        return np.sum(diff, axis=1)


class MinMaxDistance(NumericDistance):
    def __init__(self, X_train: np.ndarray, num_feat: list, eps: float):
        self.num_feat = num_feat
        self.scale = X_train[:, num_feat].max(axis=0) - X_train[:, num_feat].min(axis=0)
        self.scale[self.scale < eps] = eps

    def measure(self, x: np.ndarray, X: np.ndarray) -> np.ndarray:
        diff = np.abs(X[:, self.num_feat] - x[0, self.num_feat]) / self.scale
        return np.sum(diff, axis=1)


# -------------------------------------------------------------------------
# Mixed-type distance
# -------------------------------------------------------------------------
class HEOM(DistanceMetric):
    """
    Heterogeneous Euclidean–Overlap Metric.
    Combines numeric (normalized) and categorical dissimilarities.
    """

    def __init__(self, data_ref, numeric_distance_cls):
        self.cat_feat = data_ref.cat_feat
        self.numeric_distance = numeric_distance_cls(
            data_ref.X_train, data_ref.num_feat, data_ref.eps
        )

    def measure(self, x: np.ndarray, X: np.ndarray) -> np.ndarray:
        num_distance = self.numeric_distance.measure(x, X)
        cat_distance = np.sum(
            X[:, self.cat_feat] != x[0, self.cat_feat], axis=1
        )
        return num_distance + cat_distance


# -------------------------------------------------------------------------
# Nearest neighbour search
# -------------------------------------------------------------------------
class NearestNeighbour:
    def __init__(self, distance_metric: DistanceMetric):
        self.distance_metric = distance_metric

    def find_neighbour(self, instance) -> np.ndarray:
        """
        Find the nearest neighbour within the candidate subset
        for the given instance.
        """
        distances = self.distance_metric.measure(instance.x, instance.candidates_view)
        min_idx = distances.argmin()
        return instance.candidates_view[min_idx, :].copy()[np.newaxis, :]
