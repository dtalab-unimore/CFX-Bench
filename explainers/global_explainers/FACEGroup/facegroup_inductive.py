"""
facegroup_inductive.py  —  PURE EXTENSION. Does not modify any file in src/.

Adds inductive querying to FACEGroup: generate a counterfactual / recourse path
for an instance that was NOT a node when the graph was built, by temporarily
attaching it to the frozen graph (FACE-style), running the EXISTING
compute_recourse, then detaching it.

Reuses, without redefining:
  - self._data          (node coordinates; features are columns :-1)
  - self._epsilon       (edge radius)
  - self._kernel_obj    (fitted KDE; kernelKDE(xi, xj, dist) is a pure function)
  - self.compute_recourse(...)         (unchanged)
  - self.get_personalized_candidates   (unchanged)

The only formula touched is reused verbatim from the kernel object:
    wij = kernelKDE(xi, xj, dist) = (1 / (density_at_mean + EPSILON)) * dist
and the edge also carries 'distance' = dist, matching utils.pairwise_distances_and_graph.
"""

import numpy as np
from .FACEGroup import FACEGroup


class FACEGroupInductive(FACEGroup):
    # Optional: explicit matrix of node feature vectors, one row per graph node,
    # in EXACTLY the columns/space the graph distances were built on. Set this
    # when self._data is not laid out as [features | target] (e.g. face-only
    # _data, or extra trailing columns). When None we fall back to _data[:, :-1].
    _node_features = None

    def set_node_features(self, matrix):
        self._node_features = np.asarray(matrix)

    def _candidate_edges_for_query(self, x_query, feasibility_constraints,
                                   directed=True):
        """Build (wij, distance) edges from a query point to every in-radius,
        feasibility-satisfying existing node. Pure read of frozen state."""
        feats = self._node_features if self._node_features is not None else self._data[:, :-1]  # node feature vectors
        x = np.asarray(x_query, dtype=feats.dtype).ravel()

        # distances query -> all nodes (same Euclidean metric as the builder)
        dists = np.linalg.norm(feats - x, axis=1)

        within = np.where(dists <= self._epsilon)[0]   # epsilon radius filter
        edges = []
        for j in within:
            xj = feats[j]
            d = float(dists[j])
            # direction query -> node j (recourse moves AWAY from the source)
            if feasibility_constraints.check_constraints(x, xj):
                wij = self._kernel_obj.kernelKDE(x, xj, d)
                if isinstance(wij, np.ndarray):
                    wij = wij.item()
                edges.append((int(j), {'distance': d, 'wij': float(wij)}))
        return edges

    def compute_recourse_for_unseen(self, x_query, feasibility_constraints,
                                    personalized=True):
        """Inductive analogue of FACE's compute_path.

        Returns the same structure as FACEGroup.compute_recourse:
            (shortest_paths_info, min_target_id)
        or (None, None) if the query cannot legally connect to the graph.

        Side-effect-free w.r.t. the stored graph and self._data: the temporary
        node and its row are removed before returning.
        """
        assert self._Graph is not None, "Graph must be built/loaded first."

        temp_id = max(self._Graph.nodes) + 1 if len(self._Graph.nodes) else 0

        edges = self._candidate_edges_for_query(
            x_query, feasibility_constraints)
        if not edges:
            return None, None      # isolated query: no feasible in-radius node

        # --- attach -------------------------------------------------------
        # 1) graph edges (DiGraph in this codebase -> add source->node edges)
        self._Graph.add_node(temp_id)
        for j, attrs in edges:
            self._Graph.add_edge(temp_id, j, **attrs)

        # 2) get_personalized_candidates reads self._data[source, :-1], so the
        #    query must occupy row `temp_id` in _data for that call to work.
        #    Make the appended row match self._data's width exactly, whatever the
        #    layout ([features|target], face-only, or with extra columns).
        x_full = np.asarray(x_query, dtype=float).ravel()
        w = self._data.shape[1]
        if x_full.shape[0] < w:                       # pad trailing cols (e.g. target)
            x_full = np.concatenate([x_full, np.zeros(w - x_full.shape[0])])
        elif x_full.shape[0] > w:                     # trim if query carried extras
            x_full = x_full[:w]
        data_backup = self._data
        self._data = np.vstack([self._data, x_full[None, :]])

        try:
            result = self.compute_recourse(
                self._Graph, temp_id, feasibility_constraints,
                personalized=personalized)
        finally:
            # --- detach (always, even on exception) -----------------------
            self._Graph.remove_node(temp_id)     # also drops its edges
            self._data = data_backup

        return result
