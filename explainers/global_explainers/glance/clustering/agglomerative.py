from ..base import ClusteringMethod
from sklearn.cluster import AgglomerativeClustering
import pandas as pd
import numpy as np

class AgglomerativeMethod(ClusteringMethod):
    """
    Clustering method using Agglomerative Clustering.
    """

    def __init__(self, num_clusters):
        self.num_clusters = num_clusters
        self.model = AgglomerativeClustering()

    def fit(self, data: pd.DataFrame):
        self.model = AgglomerativeClustering(n_clusters=self.num_clusters)
        self.model.fit(data)

    def predict(self, instances: pd.DataFrame) -> np.ndarray:
        # AgglomerativeClustering doesn't support predict; use fit_predict as workaround
        return self.model.fit_predict(instances)