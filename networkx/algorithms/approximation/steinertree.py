from itertools import chain
import networkx as nx
from networkx.utils import not_implemented_for, pairwise
__all__ = ['metric_closure', 'steiner_tree']

@not_implemented_for('directed')
@nx._dispatchable(edge_attrs='weight', returns_graph=True)
def metric_closure(G, weight='weight'):
    """Return the metric closure of a graph.

    The metric closure of a graph *G* is the complete graph in which each edge
    is weighted by the shortest path distance between the nodes in *G* .

    Parameters
    ----------
    G : NetworkX graph

    Returns
    -------
    NetworkX graph
        Metric closure of the graph `G`.

    """
    M = nx.Graph()
    M.add_nodes_from(G)
    for u in G.nodes:
        # We compute shortest paths from each node to all other nodes
        length = nx.single_source_dijkstra_path_length(G, u, weight=weight)
        # Add weighted edges for all pairs
        M.add_weighted_edges_from((u, v, d) for v, d in length.items() if u != v)
    return M
def _kou_steiner_tree(G, terminal_nodes, weight='weight'):
    # Compute the metric closure of G
    M = metric_closure(G, weight=weight)
    
    # Create a subgraph with only the terminal nodes
    S = M.subgraph(terminal_nodes)
    
    # Find the minimum spanning tree of the subgraph
    mst = nx.minimum_spanning_tree(S, weight=weight)
    
    # Initialize the Steiner tree
    T = nx.Graph()
    
    # For each edge in the MST, find the corresponding shortest path in G
    for u, v in mst.edges():
        path = nx.shortest_path(G, u, v, weight=weight)
        nx.add_path(T, path, weight=weight)
    
    # Remove non-terminal leaves
    while True:
        leaves = [node for node in T.nodes() if T.degree(node) == 1]
        non_terminal_leaves = set(leaves) - set(terminal_nodes)
        if not non_terminal_leaves:
            break
        T.remove_nodes_from(non_terminal_leaves)
    
    return T

def _mehlhorn_steiner_tree(G, terminal_nodes, weight='weight'):
    # Find the closest terminal node for each non-terminal node
    closest_terminal = {}
    for node in G:
        if node not in terminal_nodes:
            distances = [(t, nx.shortest_path_length(G, node, t, weight=weight)) for t in terminal_nodes]
            closest_terminal[node] = min(distances, key=lambda x: x[1])[0]
    
    # Create a complete graph of terminal nodes
    M = nx.Graph()
    for u in terminal_nodes:
        for v in terminal_nodes:
            if u != v:
                path = nx.shortest_path(G, u, v, weight=weight)
                distance = sum(G[path[i]][path[i+1]].get(weight, 1) for i in range(len(path)-1))
                M.add_edge(u, v, weight=distance, path=path)
    
    # Find the minimum spanning tree of M
    mst = nx.minimum_spanning_tree(M, weight=weight)
    
    # Initialize the Steiner tree
    T = nx.Graph()
    
    # For each edge in the MST, add the corresponding path to T
    for u, v in mst.edges():
        path = M[u][v]['path']
        nx.add_path(T, path, weight=weight)
    
    # Add non-terminal nodes that are closest to a terminal in T
    for node, terminal in closest_terminal.items():
        if terminal in T:
            path = nx.shortest_path(G, node, terminal, weight=weight)
            nx.add_path(T, path, weight=weight)
    
    # Remove non-terminal leaves
    while True:
        leaves = [node for node in T.nodes() if T.degree(node) == 1]
        non_terminal_leaves = set(leaves) - set(terminal_nodes)
        if not non_terminal_leaves:
            break
        T.remove_nodes_from(non_terminal_leaves)
    
    return T

ALGORITHMS = {'kou': _kou_steiner_tree, 'mehlhorn': _mehlhorn_steiner_tree}

@not_implemented_for('directed')
@nx._dispatchable(preserve_all_attrs=True, returns_graph=True)
def steiner_tree(G, terminal_nodes, weight='weight', method=None):
    """Return an approximation to the minimum Steiner tree of a graph.

    The minimum Steiner tree of `G` w.r.t a set of `terminal_nodes` (also *S*)
    is a tree within `G` that spans those nodes and has minimum size (sum of
    edge weights) among all such trees.

    The approximation algorithm is specified with the `method` keyword
    argument. All three available algorithms produce a tree whose weight is
    within a ``(2 - (2 / l))`` factor of the weight of the optimal Steiner tree,
    where ``l`` is the minimum number of leaf nodes across all possible Steiner
    trees.

    * ``"kou"`` [2]_ (runtime $O(|S| |V|^2)$) computes the minimum spanning tree of
      the subgraph of the metric closure of *G* induced by the terminal nodes,
      where the metric closure of *G* is the complete graph in which each edge is
      weighted by the shortest path distance between the nodes in *G*.

    * ``"mehlhorn"`` [3]_ (runtime $O(|E|+|V|\\log|V|)$) modifies Kou et al.'s
      algorithm, beginning by finding the closest terminal node for each
      non-terminal. This data is used to create a complete graph containing only
      the terminal nodes, in which edge is weighted with the shortest path
      distance between them. The algorithm then proceeds in the same way as Kou
      et al..

    Parameters
    ----------
    G : NetworkX graph

    terminal_nodes : list
         A list of terminal nodes for which minimum steiner tree is
         to be found.

    weight : string (default = 'weight')
        Use the edge attribute specified by this string as the edge weight.
        Any edge attribute not present defaults to 1.

    method : string, optional (default = 'mehlhorn')
        The algorithm to use to approximate the Steiner tree.
        Supported options: 'kou', 'mehlhorn'.
        Other inputs produce a ValueError.

    Returns
    -------
    NetworkX graph
        Approximation to the minimum steiner tree of `G` induced by
        `terminal_nodes` .

    Raises
    ------
    NetworkXNotImplemented
        If `G` is directed.

    ValueError
        If the specified `method` is not supported.

    Notes
    -----
    For multigraphs, the edge between two nodes with minimum weight is the
    edge put into the Steiner tree.


    References
    ----------
    .. [1] Steiner_tree_problem on Wikipedia.
           https://en.wikipedia.org/wiki/Steiner_tree_problem
    .. [2] Kou, L., G. Markowsky, and L. Berman. 1981.
           ‘A Fast Algorithm for Steiner Trees’.
           Acta Informatica 15 (2): 141–45.
           https://doi.org/10.1007/BF00288961.
    .. [3] Mehlhorn, Kurt. 1988.
           ‘A Faster Approximation Algorithm for the Steiner Problem in Graphs’.
           Information Processing Letters 27 (3): 125–28.
           https://doi.org/10.1016/0020-0190(88)90066-X.
    """
    pass
