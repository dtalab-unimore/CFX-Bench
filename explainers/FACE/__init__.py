# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 14:33:03 2019

@author: rp13102
"""

from copy import deepcopy

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.pyplot import cm
from scipy.sparse import csr_matrix
# from sklearn.metrics import euclidean_distances

from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity, NearestNeighbors


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def plot_decision_boundary(X, y, func, method):
    h = 0.1
    xmin, ymin = np.min(X, axis=0)
    xmax, ymax = np.max(X, axis=0)

    xx, yy = np.meshgrid(
        np.arange(xmin, xmax, h),
        np.arange(ymin, ymax, h)
    )

    cm = plt.cm.RdBu
    cm_bright = mpl.colors.ListedColormap(['#FF0000', '#0000FF'])

    newx = np.c_[xx.ravel(), yy.ravel()]

    fig, ax = plt.subplots()
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    # Z = clf.predict_proba(newx)[:, 1]
    Z = func(newx)
    Z = Z.reshape(xx.shape)

    v = contour_plot = ax.contourf(
        xx, yy,
        Z,
        levels=20,
        cmap=cm,
        alpha=.8)

    ax.scatter(X[:, 0], X[:, 1],
               c=y,
               cmap=cm_bright,
               edgecolors='k',
               zorder=1)

    ax.grid(color='k',
            linestyle='-',
            linewidth=0.50,
            alpha=0.75)

    plt.colorbar(v, ax=ax)
    return ax


def plot_path(X, path, ax, color='lightgreen', extra_point=None):
    if X.shape[1] != 2:
        return 0

    n_nodes = len(path)
    if isinstance(extra_point, np.ndarray):
        ax.plot([X[-1, 0], extra_point[0]],
                [X[-1, 1], extra_point[1]],
                'k', alpha=0.50)
        ax.scatter(extra_point[0], extra_point[1],
                   color='k',
                   marker='o',
                   facecolors='lightyellow',
                   edgecolors='lightyellow',
                   alpha=0.80,
                   zorder=1,
                   s=250)

    args = {'color': 'lightgreen',
            'marker': 'x',
            's': 100}

    for idx in range(n_nodes - 1):
        i = int(path[idx])
        j = int(path[idx + 1])
        ax.plot(X[[i, j], 0], X[[i, j], 1], 'k', alpha=0.50)

    ax.scatter(X[path[-1], 0], X[path[-1], 1],
               color='k',
               marker='o',
               facecolors=color,
               edgecolors=color,
               alpha=0.50,
               zorder=2,
               s=150)


def plot_paths(X, method, howmanypaths, ax, all_paths):
    counter = 0
    colors = cm.Greens(np.linspace(0, 1, howmanypaths))

    for idx, item in enumerate(all_paths):
        if counter > howmanypaths - 1:
            break
        path = item[-1]
        if method in ['kde']:
            plot_path(X, path, ax, colors[counter])
        else:
            plot_path(X, path, ax, colors[counter])
        counter += 1


def plot_density(self, ax):
    newx = self.prepare_grid()
    if (self.n_features == 2):
        Z = np.exp(self.density_estimator.score_samples(newx))
        self.plot_density_scores(Z, ax)


class CFGenerator(object):
    def __init__(
            self,
            predictor,
            method=None,
            weight_function=None,
            prediction_threshold=None,
            density_threshold=None,
            K=None,
            radius_limit=None,
            kde_mode=None,
            n_neighbours=None,
            epsilon=None,
            distance_threshold=None,
            edge_conditions=None,
            howmanypaths=None,
            undirected=True
    ):

        self.edge_conditions = edge_conditions
        self.undirected = undirected
        self.chunk_size=2_000_000

        if method in ['knn', 'kde', 'egraph']:
            self.method = method
        else:
            raise ValueError('Unknown method')

        if howmanypaths is None:
            self.howmanypaths = 5
        else:
            self.howmanypaths = howmanypaths

        if weight_function is None:
            self.weight_function = lambda x: -np.log(x)
        else:
            self.weight_function = weight_function

        if not hasattr(predictor, 'predict_proba'):
            raise ValueError('Predictor needs to have attribute: \'predict proba\'')
        else:
            self.predictor = predictor

        if prediction_threshold is None:
            self.prediction_threshold = 0.60
        else:
            self.prediction_threshold = prediction_threshold

        if density_threshold is None:
            self.density_threshold = 1e-5
        else:
            self.density_threshold = density_threshold

        if K is None:
            self.K = 10
        else:
            self.K = K

        if epsilon is None:
            self.epsilon = 0.75
        else:
            self.epsilon = epsilon
            self.distance_threshold = distance_threshold

        if radius_limit is None:
            self.radius_limit = 1.10
        else:
            self.radius_limit = radius_limit

        self.kde_mode = 1 if kde_mode is None else kde_mode

        if n_neighbours is None:
            self.n_neighbours = 20
        else:
            self.n_neighbours = n_neighbours

        if distance_threshold is None:
            self.distance_threshold = np.inf
        else:
            self.distance_threshold = distance_threshold

    def check_conditions(self, V0, V1):
        """Vectorized edge-condition check.

        Parameters
        ----------
        V0, V1 : np.ndarray of shape (M, n_features)
            Batches of candidate endpoints. A single record is simply the
            M == 1 case, i.e. shape (1, n_features). Edges are interpreted
            directionally as V0[m] -> V1[m].

        Returns
        -------
        np.ndarray of shape (M,), dtype bool
            Mask of pairs that satisfy the edge conditions. When no edge
            condition callable was supplied, every pair is accepted.
        """
        if self.edge_conditions is None:
            return np.ones(V0.shape[0], dtype=bool)
        return np.asarray(self.edge_conditions(V0, V1), dtype=bool)

    def _radius_edges(self, radius, sort_results=False):
        """Return directed candidate edges (i, j, dist) with i != j and
        dist > 0, found via a sparse radius query instead of a dense n x n
        distance matrix.

        radius_neighbors is inclusive of the boundary (dist <= radius),
        matching the previous `dist_matrix <= threshold` semantics. Zero-distance
        duplicates and self-loops are excluded, matching the previous
        `fill_diagonal(inf)` + `isclose(dist, 0) = inf`.

        With sort_results=True each row's neighbours are returned in ascending
        distance order (used by the kNN top-k selection).
        """
        nn = NearestNeighbors(metric='euclidean', radius=radius)
        nn.fit(self.X)
        dists, indices = nn.radius_neighbors(
            self.X, return_distance=True, sort_results=sort_results
        )

        # Flatten the jagged per-row arrays into flat edge lists.
        n_per_row = np.fromiter((len(ind) for ind in indices), dtype=np.intp,
                                count=self.n_samples)
        valid_i = np.repeat(np.arange(self.n_samples), n_per_row)
        valid_j = np.concatenate(indices) if self.n_samples else np.array([], dtype=np.intp)
        valid_d = np.concatenate(dists) if self.n_samples else np.array([], dtype=float)

        # Exclude self-loops and zero-distance duplicates (order-preserving,
        # so the per-row ascending-distance ordering from sort_results survives).
        keep = (valid_i != valid_j) & ~np.isclose(valid_d, 0)
        return valid_i[keep], valid_j[keep], valid_d[keep]

    def _condition_mask(self, valid_i, valid_j):
        """Batched edge-condition mask, evaluated in memory-bounded chunks so the
        fancy-indexed (chunk, n_features) arrays never blow up peak memory."""
        n_edges = valid_i.size
        mask = np.empty(n_edges, dtype=bool)
        iterator = range(0, n_edges, self.chunk_size)
        if n_edges > self.chunk_size:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc="check_conditions (chunked)")
        for start in iterator:
            stop = start + self.chunk_size
            ci = valid_i[start:stop]
            cj = valid_j[start:stop]
            mask[start:stop] = self.check_conditions(self.X[ci], self.X[cj])
        return mask

    def _empty_kernel(self):
        return csr_matrix((self.n_samples, self.n_samples), dtype=float)

    def get_weights_kde(self, density_scorer, mode):
        # 1. & 3. Sparse radius search replaces the dense matrix + threshold scan.
        valid_i, valid_j, valid_d = self._radius_edges(self.distance_threshold)
        if valid_i.size == 0:
            return self._empty_kernel()

        # 4. Vectorized, chunked edge-condition filtering.
        mask = self._condition_mask(valid_i, valid_j)
        final_i = valid_i[mask]
        final_j = valid_j[mask]
        final_d = valid_d[mask]
        if final_i.size == 0:
            return self._empty_kernel()

        # 5./6./7. Batch midpoints, densities, and weights.
        midpoints = (self.X[final_i] + self.X[final_j]) / 2.0
        densities = density_scorer(midpoints)
        if mode == 1:
            weights = self.weight_function(np.exp(densities)) * final_d
        else:
            weights = self.weight_function(sigmoid(densities)) * final_d
        weights = np.maximum(weights, 0.0)  # FACE cost is a non-negative traversal cost

        # 8. Build the sparse kernel. radius_neighbors yields unique (i, j) pairs,
        #    so no duplicate entries are summed by the COO->CSR construction.
        return csr_matrix(
            (weights, (final_i, final_j)),
            shape=(self.n_samples, self.n_samples),
        )

    def get_weights_knn(self):
        # Distance-sorted neighbours per row (ascending), within the threshold.
        # NOTE ON TIES: when more than n_neighbours candidates are equidistant at
        # the cut-off (common on one-hot data, where every single-feature change
        # is at distance sqrt(2)), the *set* of kept weights is well-defined but
        # *which* tied neighbours are retained depends on the neighbour ordering.
        # This matches the original (which used an unstable argsort) in keeping
        # exactly k edges with the k smallest distances, but tie resolution is
        # not guaranteed to pick the identical subset.
        valid_i, valid_j, valid_d = self._radius_edges(
            self.distance_threshold, sort_results=True
        )
        if valid_i.size == 0:
            return self._empty_kernel()

        # Edge-condition filter (order-preserving -> rows stay distance-sorted).
        mask = self._condition_mask(valid_i, valid_j)
        valid_i = valid_i[mask]
        valid_j = valid_j[mask]
        valid_d = valid_d[mask]
        if valid_i.size == 0:
            return self._empty_kernel()

        # Keep the closest n_neighbours per row. valid_i is non-decreasing and
        # each row's entries are ascending in distance, so an edge's rank within
        # its row is just its offset from that row's first occurrence.
        rows, starts, counts = np.unique(valid_i, return_index=True, return_counts=True)
        within_row_rank = np.arange(valid_i.size) - np.repeat(starts, counts)
        keep = within_row_rank < self.n_neighbours

        kept_i = valid_i[keep]
        kept_j = valid_j[keep]
        weights = self.weight_function(valid_d[keep])
        return csr_matrix(
            (weights, (kept_i, kept_j)),
            shape=(self.n_samples, self.n_samples),
        )

    def get_weights_e(self):
        # epsilon-ball radius search (undirected graph).
        valid_i, valid_j, valid_d = self._radius_edges(self.epsilon)
        if valid_i.size == 0:
            return self._empty_kernel()

        # Lower triangle only (i > j), matching the original `for j in range(i)`,
        # condition checked in the i -> j direction, then mirrored.
        lower = valid_i > valid_j
        i_idx = valid_i[lower]
        j_idx = valid_j[lower]
        d_idx = valid_d[lower]
        if i_idx.size == 0:
            return self._empty_kernel()

        mask = self._condition_mask(i_idx, j_idx)
        i_idx = i_idx[mask]
        j_idx = j_idx[mask]
        d_idx = d_idx[mask]
        if i_idx.size == 0:
            return self._empty_kernel()

        vals = self.weight_function(d_idx)
        # Symmetric: emit both (i, j) and (j, i).
        rows = np.concatenate([i_idx, j_idx])
        cols = np.concatenate([j_idx, i_idx])
        data = np.concatenate([vals, vals])
        return csr_matrix(
            (data, (rows, cols)),
            shape=(self.n_samples, self.n_samples),
        )

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.n_samples, self.n_features = self.X.shape
        if self.n_samples != self.y.shape[0]:
            raise ValueError('Inconsistent dimensions')
        self.predictions = self.predictor.predict_proba(X)
        self.kernel = self.get_kernel()
        self.fit_graph()

    def get_kernel(self):
        if self.method == 'kde':
            self.get_kde()
            # Cache the training-set densities once: reused by `condition` and by
            # `compute_path` instead of re-scoring on every call.
            self.train_density = np.exp(self.density_estimator.score_samples(self.X))
            density_scorer = self.density_estimator.score_samples
            kernel = self.get_weights_kde(density_scorer, self.kde_mode)

        elif self.method == 'knn':
            kernel = self.get_weights_knn()

        elif self.method == 'egraph':
            kernel = self.get_weights_e()

        self.kernel = kernel
        return kernel

    def fit_graph(self):
        # Choose Directed or Undirected based on your monotonicity needs
        graph_type = nx.Graph if self.undirected else nx.DiGraph

        # Build the graph directly from the sparse kernel. Entry (i, j) becomes
        # a directed edge i -> j (for DiGraph) with attribute 'weight'.
        self.graph = nx.from_scipy_sparse_array(self.kernel, create_using=graph_type, edge_attribute='weight')

    def condition(self, item):
        pred = self.predictions[item, self.y[item]]
        if (self.y[item] == self.target_class
                and pred >= self.prediction_threshold):
            if self.method == 'kde':
                kde = self.train_density[item]
                if kde >= self.density_threshold:
                    return (pred, kde), True
            elif self.method in ['knn', 'egraph']:
                return (pred), True
        return 0, False

    def get_kde(self):
        bandwidths = np.logspace(-2, 0, 20)
        # bandwidths = [0.65]
        # bandwidths = [0.5]
        grid = GridSearchCV(
            KernelDensity(kernel='gaussian'),
            {'bandwidth': bandwidths},
            n_jobs=-1,  # parallel across the bandwidth grid; selection is deterministic
            verbose=2
        )
        grid.fit(deepcopy(self.X))
        self.density_estimator = grid.best_estimator_
        print("Best bandwidth:", grid.best_params_['bandwidth'])

    def compute_path(
            self,
            starting_point,
            target_class,
            plot=False
    ):
        self.target_class = target_class

        # Assign a unique ID to the temporary node (using the dataset size is safe)
        temp_node_id = self.n_samples

        # Ensure starting_point is a flat 1D array
        starting_point = starting_point.flatten()

        # 1. Find valid candidate targets based on prediction and density thresholds
        t0 = np.where(self.predictions[:, self.target_class] >= self.prediction_threshold)[0]
        t1 = np.where(self.y == self.target_class)[0]

        if self.method == 'kde':
            t2 = np.where(self.train_density >= self.density_threshold)[0]
            self.candidate_targets = list(set(t0).intersection(set(t1)).intersection(set(t2)))
        else:
            self.candidate_targets = list(set(t0).intersection(set(t1)))

        if not self.candidate_targets:
            return []

        # 2. Compute distances from the temporary node to all existing points
        dists = np.linalg.norm(self.X - starting_point, axis=1)

        # 3. Filter by distance threshold
        valid_neighbors = np.where(dists <= self.distance_threshold)[0]

        # 4. Evaluate conditions in one batched call (direction: starting_point -> X[j])
        if valid_neighbors.size:
            mask = self.check_conditions(
                np.broadcast_to(starting_point, (valid_neighbors.size, starting_point.size)),
                self.X[valid_neighbors],
            )
            valid_neighbors = valid_neighbors[mask]

        if valid_neighbors.size == 0:
            # The starting point is isolated and cannot legally connect to the graph
            return []

        # Compute the appropriate weights for the surviving edges, batched.
        if self.method == 'kde':
            midpoints = (starting_point[None, :] + self.X[valid_neighbors]) / 2.0
            densities = self.density_estimator.score_samples(midpoints)
            # weights = self.weight_function(np.exp(densities)) * dists[valid_neighbors]
            transformed = np.exp(densities) if self.kde_mode == 1 else sigmoid(densities)
            weights = self.weight_function(transformed) * dists[valid_neighbors]
        else:
            weights = self.weight_function(dists[valid_neighbors])
        weights = np.maximum(weights, 0.0)

        new_edges = [
            (temp_node_id, int(j), float(w))
            for j, w in zip(valid_neighbors, weights)
        ]

        # 5. Temporarily inject the node and its valid edges into the graph
        self.graph.add_weighted_edges_from(new_edges)

        # 6. Compute shortest paths using NetworkX
        try:
            lengths, paths = nx.single_source_dijkstra(
                self.graph,
                temp_node_id,
                weight='weight'
            )
        except nx.NetworkXNoPath:
            lengths, paths = {}, {}

        # 7. CLEANUP: Immediately remove the temporary node so it doesn't pollute the graph
        self.graph.remove_node(temp_node_id)

        # 8. Filter the results for our valid targets
        all_paths = []
        for target in self.candidate_targets:
            if target in lengths:
                value, satisfied = self.condition(target)
                if satisfied:
                    # Remove the temporary node ID from the returned path sequence
                    # (Optional: keep it if you want to explicitly show the injected starting state)
                    final_path = paths[target][1:] if len(paths[target]) > 1 else paths[target]

                    all_paths.append((
                        target,
                        self.X[target, :],
                        lengths[target],
                        value,
                        final_path
                    ))

        # Sort paths by shortest distance
        all_paths = sorted(all_paths, key=lambda x: x[2])

        return all_paths
