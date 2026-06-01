from ..base import ClusteringMethod
from sklearn.mixture import GaussianMixture
import pandas as pd
import numpy as np

class GaussianMixtureMethod(ClusteringMethod):
    """
    Clustering method using Gaussian Mixture Models.
    """

    def __init__(self, num_components=2, random_seed=None):
        self.num_components = num_components
        self.random_seed = random_seed
        self.model = GaussianMixture()

    def fit(self, data: pd.DataFrame):
        self.model = GaussianMixture(
            n_components=self.num_components, random_state=self.random_seed
        )
        self.model.fit(data)

    def predict(self, instances: pd.DataFrame) -> np.ndarray:
        return self.model.predict(instances)
