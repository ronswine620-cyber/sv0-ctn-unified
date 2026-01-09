import numpy as np
from .utils import pairwise_distances, UnionFind

def betti0_persistence(X: np.ndarray, epsilons: np.ndarray | None = None) -> dict:
    """Approximate Betti-0 (connected components) persistence.
    We sweep ε from small to large; at each ε we connect edges with D_ij <= ε
    and track the number of components. We record 'lifetimes' of components
    as they merge.

    Returns:
      {
        'epsilons': list of thresholds,
        'components': list of component counts,
        'merge_events': [(epsilon, new_component_count) ...]
      }
    """
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    if n == 0:
        return {'epsilons': [], 'components': [], 'merge_events': []}

    D = pairwise_distances(X)
    iu = np.triu_indices(n, k=1)
    edges = list(zip(iu[0].tolist(), iu[1].tolist(), D[iu].tolist()))
    edges.sort(key=lambda e: e[2])

    if epsilons is None:
        # Unique sorted distances as thresholds
        epsilons = np.unique(np.array([w for _,_,w in edges], dtype=float))
    else:
        epsilons = np.sort(np.asarray(epsilons, dtype=float))

    uf = UnionFind(n)
    comps_over_eps = []
    merges = []
    edge_idx = 0

    for eps in epsilons:
        # Union all edges with weight <= eps
        while edge_idx < len(edges) and edges[edge_idx][2] <= eps:
            i, j, _ = edges[edge_idx]
            before = uf.components
            merged = uf.union(i, j)
            after = uf.components
            if merged and after != before:
                merges.append((float(eps), int(after)))
            edge_idx += 1
        comps_over_eps.append(int(uf.components))

    return {
        'epsilons': [float(e) for e in epsilons.tolist()],
        'components': comps_over_eps,
        'merge_events': merges
    }
