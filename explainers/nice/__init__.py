from typing import Optional, Union, List, Callable
import numpy as np

from explainers.nice.utils.data import DataNICE
from explainers.nice.utils.distance import HEOM, StandardDistance, MinMaxDistance, NearestNeighbour
from explainers.nice.utils.optimization.heuristic import BestFirst
from explainers.nice.utils.optimization.reward import SparsityReward, ProximityReward, PlausibilityReward


CRITERIA_DIS = {"HEOM": HEOM}
CRITERIA_NRM = {"std": StandardDistance, "minmax": MinMaxDistance}
CRITERIA_REW = {
    "sparsity": SparsityReward,
    "proximity": ProximityReward,
    "plausibility": PlausibilityReward,
}


class NICE:
    """
    Neighbourhood-based Interpretable Counterfactual Explanation (NICE).
    """

    def __init__(
        self,
        predict_fn: Callable[[np.ndarray], np.ndarray],
        X_train: np.ndarray,
        cat_feat: List[int],
        num_feat: Union[List[int], str] = "auto",
        y_train: Optional[np.ndarray] = None,
        optimization: str = "sparsity",
        justified_cf: bool = True,
        distance_metric: str = "HEOM",
        num_normalization: str = "minmax",
        auto_encoder: Optional[object] = None,
    ):
        self.optimization = optimization
        self.data = DataNICE(X_train, y_train, cat_feat, num_feat, predict_fn, justified_cf)
        distance_cls = CRITERIA_DIS[distance_metric]
        norm_cls = CRITERIA_NRM[num_normalization]
        self.distance_metric = distance_cls(self.data, norm_cls)
        self.nearest_neighbour = NearestNeighbour(self.distance_metric)
        self.auto_encoder = auto_encoder

    # -------------------------------------------------------------------------
    def explain(self, x: np.ndarray, target_class: Union[str, int, List[int]] = "other") -> np.ndarray:
        instance = self.data.for_instance(x, target_class)
        nn = self.nearest_neighbour.find_neighbour(instance)

        if self.optimization == "none":
            return nn

        reward_cls = CRITERIA_REW[self.optimization]
        reward_fn = reward_cls(instance, distance_metric=self.distance_metric, auto_encoder=self.auto_encoder)
        optimizer = BestFirst(instance, reward_fn)
        return optimizer.optimize(nn)
