from abc import ABC, abstractmethod
import numpy as np


class Optimization(ABC):
    @abstractmethod
    def optimize(self, NN):
        pass


class BestFirst(Optimization):
    """
    Simple best-first search optimization based on feature substitutions.
    """

    def __init__(self, instance, reward_function):
        self.instance = instance
        self.reward_function = reward_function
        self.predict_fn = instance.data_ref.predict_fn

    def optimize(self, NN: np.ndarray) -> np.ndarray:
        cf_candidate = self.instance.x.copy()
        target_class = self.instance.target_class

        while True:
            diff = np.where(cf_candidate != NN)[1]
            if len(diff) == 0:
                return cf_candidate  # No further change possible

            X_prune = np.tile(cf_candidate, (len(diff), 1))
            for r, c in enumerate(diff):
                X_prune[r, c] = NN[0, c]

            cf_candidate = self.reward_function.calculate_reward(X_prune, cf_candidate)
            pred_class = self.predict_fn(cf_candidate).argmax()
            if pred_class in target_class:
                return cf_candidate
