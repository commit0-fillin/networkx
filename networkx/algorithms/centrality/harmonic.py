"""Functions for computing the harmonic centrality of a graph."""
from functools import partial
import networkx as nx
__all__ = ['harmonic_centrality']

@nx._dispatchable(edge_attrs='distance')
def harmonic_centrality(G, nbunch=None, distance=None, sources=None):
    """Compute harmonic centrality for nodes.

    Harmonic centrality [1]_ of a node `u` is the sum of the reciprocal
    of the shortest path distances from all other nodes to `u`

    .. math::

        C(u) = \\sum_{v \\neq u} \\frac{1}{d(v, u)}

    where `d(v, u)` is the shortest-path distance between `v` and `u`.

    If `sources` is given as an argument, the returned harmonic centrality
    values are calculated as the sum of the reciprocals of the shortest
    path distances from the nodes specified in `sources` to `u` instead
    of from all nodes to `u`.

    Notice that higher values indicate higher centrality.

    Parameters
    ----------
    G : graph
      A NetworkX graph

    nbunch : container (default: all nodes in G)
      Container of nodes for which harmonic centrality values are calculated.

    sources : container (default: all nodes in G)
      Container of nodes `v` over which reciprocal distances are computed.
      Nodes not in `G` are silently ignored.

    distance : edge attribute key, optional (default=None)
      Use the specified edge attribute as the edge distance in shortest
      path calculations.  If `None`, then each edge will have distance equal to 1.

    Returns
    -------
    nodes : dictionary
      Dictionary of nodes with harmonic centrality as the value.

    See Also
    --------
    betweenness_centrality, load_centrality, eigenvector_centrality,
    degree_centrality, closeness_centrality

    Notes
    -----
    If the 'distance' keyword is set to an edge attribute key then the
    shortest-path length will be computed using Dijkstra's algorithm with
    that edge attribute as the edge weight.

    References
    ----------
    .. [1] Boldi, Paolo, and Sebastiano Vigna. "Axioms for centrality."
           Internet Mathematics 10.3-4 (2014): 222-262.
    """
    if nbunch is None:
        nbunch = G.nodes()
    
    if sources is None:
        sources = G.nodes()
    else:
        sources = [v for v in sources if v in G]

    if distance is not None:
        path_length = partial(nx.shortest_path_length, weight=distance)
    else:
        path_length = nx.shortest_path_length

    harmonic_centrality = {}
    for node in nbunch:
        if node not in G:
            harmonic_centrality[node] = 0.0
            continue
        
        distances = path_length(G, source=node)
        centrality = sum(1 / d for v, d in distances.items() if v in sources and v != node and d != 0)
        harmonic_centrality[node] = centrality

    return harmonic_centrality
