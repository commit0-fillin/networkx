""" Provides a function for computing the extendability of a graph which is
undirected, simple, connected and bipartite and contains at least one perfect matching."""
import networkx as nx
from networkx.utils import not_implemented_for
__all__ = ['maximal_extendability']

@not_implemented_for('directed')
@not_implemented_for('multigraph')
@nx._dispatchable
def maximal_extendability(G):
    """Computes the extendability of a graph.

    The extendability of a graph is defined as the maximum $k$ for which `G`
    is $k$-extendable. Graph `G` is $k$-extendable if and only if `G` has a
    perfect matching and every set of $k$ independent edges can be extended
    to a perfect matching in `G`.

    Parameters
    ----------
    G : NetworkX Graph
        A fully-connected bipartite graph without self-loops

    Returns
    -------
    extendability : int

    Raises
    ------
    NetworkXError
       If the graph `G` is disconnected.
       If the graph `G` is not bipartite.
       If the graph `G` does not contain a perfect matching.
       If the residual graph of `G` is not strongly connected.

    Notes
    -----
    Definition:
    Let `G` be a simple, connected, undirected and bipartite graph with a perfect
    matching M and bipartition (U,V). The residual graph of `G`, denoted by $G_M$,
    is the graph obtained from G by directing the edges of M from V to U and the
    edges that do not belong to M from U to V.

    Lemma [1]_ :
    Let M be a perfect matching of `G`. `G` is $k$-extendable if and only if its residual
    graph $G_M$ is strongly connected and there are $k$ vertex-disjoint directed
    paths between every vertex of U and every vertex of V.

    Assuming that input graph `G` is undirected, simple, connected, bipartite and contains
    a perfect matching M, this function constructs the residual graph $G_M$ of G and
    returns the minimum value among the maximum vertex-disjoint directed paths between
    every vertex of U and every vertex of V in $G_M$. By combining the definitions
    and the lemma, this value represents the extendability of the graph `G`.

    Time complexity O($n^3$ $m^2$)) where $n$ is the number of vertices
    and $m$ is the number of edges.

    References
    ----------
    .. [1] "A polynomial algorithm for the extendability problem in bipartite graphs",
          J. Lakhal, L. Litzler, Information Processing Letters, 1998.
    .. [2] "On n-extendible graphs", M. D. Plummer, Discrete Mathematics, 31:201–210, 1980
          https://doi.org/10.1016/0012-365X(80)90037-0

    """
    # Check if the graph is connected
    if not nx.is_connected(G):
        raise nx.NetworkXError("The graph G is not connected.")

    # Check if the graph is bipartite
    if not nx.is_bipartite(G):
        raise nx.NetworkXError("The graph G is not bipartite.")

    # Get the bipartite sets
    X, Y = nx.bipartite.sets(G)

    # Check if the graph has a perfect matching
    matching = nx.bipartite.hopcroft_karp_matching(G, X)
    if len(matching) != len(G):
        raise nx.NetworkXError("The graph G does not contain a perfect matching.")

    # Construct the residual graph
    G_M = nx.DiGraph()
    G_M.add_nodes_from(G.nodes())
    for u, v in G.edges():
        if (u in X and v in Y) or (u in Y and v in X):
            if (u, v) in matching.items() or (v, u) in matching.items():
                G_M.add_edge(v, u)
            else:
                G_M.add_edge(u, v)

    # Check if the residual graph is strongly connected
    if not nx.is_strongly_connected(G_M):
        raise nx.NetworkXError("The residual graph of G is not strongly connected.")

    # Compute the maximum number of vertex-disjoint paths
    min_paths = float('inf')
    for u in X:
        for v in Y:
            max_flow_value = nx.maximum_flow_value(G_M, u, v)
            min_paths = min(min_paths, max_flow_value)

    return int(min_paths)
