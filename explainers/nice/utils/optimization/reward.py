from abc import ABC, abstractmethod
import numpy as np


class RewardFunction(ABC):
    def __init__(self, instance, **kwargs):
        self.instance = instance
        self.data_ref = instance.data_ref

    @abstractmethod
    def calculate_reward(self, X_prune, previous_cf):
        pass


class SparsityReward(RewardFunction):
    def calculate_reward(self, X_prune, previous_cf):
        predict_fn = self.data_ref.predict_fn
        score_prune = -predict_fn(previous_cf) + predict_fn(X_prune)
        score_diff = score_prune[:, self.instance.target_class]
        idx_max = np.argmax(score_diff.max(axis=1))
        return X_prune[idx_max:idx_max + 1, :]


class ProximityReward(RewardFunction):
    def __init__(self, instance, distance_metric, **kwargs):
        super().__init__(instance)
        self.distance_metric = distance_metric

    def calculate_reward(self, X_prune, previous_cf):
        predict_fn = self.data_ref.predict_fn
        data = self.instance
        eps = self.data_ref.eps

        score_diff = (
            predict_fn(X_prune)[:, data.target_class]
            - predict_fn(previous_cf)[:, data.target_class]
        )
        distance = (
            self.distance_metric.measure(data.x, X_prune)
            - self.distance_metric.measure(data.x, previous_cf)
        )
        idx_max = np.argmax(score_diff / (distance + eps)[:, np.newaxis])
        return X_prune[idx_max:idx_max + 1, :]


class PlausibilityReward(RewardFunction):
    def __init__(self, instance, auto_encoder, **kwargs):
        super().__init__(instance)
        self.auto_encoder = auto_encoder

    def calculate_reward(self, X_prune, previous_cf):
        predict_fn = self.data_ref.predict_fn
        data = self.instance

        score_diff = (
            predict_fn(X_prune)[:, data.target_class]
            - predict_fn(previous_cf)[:, data.target_class]
        )
        ae_loss_diff = self.auto_encoder(previous_cf) - self.auto_encoder(X_prune)
        idx_max = np.argmax(score_diff * ae_loss_diff[:, np.newaxis])
        return X_prune[idx_max:idx_max + 1, :]
