"""Functions for computing and verifying matchings in a graph."""
from collections import Counter
from itertools import combinations, repeat
import networkx as nx
from networkx.utils import not_implemented_for
__all__ = ['is_matching', 'is_maximal_matching', 'is_perfect_matching', 'max_weight_matching', 'min_weight_matching', 'maximal_matching']

@not_implemented_for('multigraph')
@not_implemented_for('directed')
@nx._dispatchable
def maximal_matching(G):
    """Find a maximal matching in the graph.

    A matching is a subset of edges in which no node occurs more than once.
    A maximal matching cannot add more edges and still be a matching.

    Parameters
    ----------
    G : NetworkX graph
        Undirected graph

    Returns
    -------
    matching : set
        A maximal matching of the graph.

    Examples
    --------
    >>> G = nx.Graph([(1, 2), (1, 3), (2, 3), (2, 4), (3, 5), (4, 5)])
    >>> sorted(nx.maximal_matching(G))
    [(1, 2), (3, 5)]

    Notes
    -----
    The algorithm greedily selects a maximal matching M of the graph G
    (i.e. no superset of M exists). It runs in $O(|E|)$ time.
    """
    matching = set()
    nodes = set(G.nodes())

    for u, v in G.edges():
        if u not in nodes or v not in nodes:
            continue
        matching.add((u, v))
        nodes.remove(u)
        nodes.remove(v)

    return matching

def matching_dict_to_set(matching):
    """Converts matching dict format to matching set format

    Converts a dictionary representing a matching (as returned by
    :func:`max_weight_matching`) to a set representing a matching (as
    returned by :func:`maximal_matching`).

    In the definition of maximal matching adopted by NetworkX,
    self-loops are not allowed, so the provided dictionary is expected
    to never have any mapping from a key to itself. However, the
    dictionary is expected to have mirrored key/value pairs, for
    example, key ``u`` with value ``v`` and key ``v`` with value ``u``.

    """
    return set(tuple(sorted((u, v))) for u, v in matching.items() if u < v)

@nx._dispatchable
def is_matching(G, matching):
    """Return True if ``matching`` is a valid matching of ``G``

    A *matching* in a graph is a set of edges in which no two distinct
    edges share a common endpoint. Each node is incident to at most one
    edge in the matching. The edges are said to be independent.

    Parameters
    ----------
    G : NetworkX graph

    matching : dict or set
        A dictionary or set representing a matching. If a dictionary, it
        must have ``matching[u] == v`` and ``matching[v] == u`` for each
        edge ``(u, v)`` in the matching. If a set, it must have elements
        of the form ``(u, v)``, where ``(u, v)`` is an edge in the
        matching.

    Returns
    -------
    bool
        Whether the given set or dictionary represents a valid matching
        in the graph.

    Raises
    ------
    NetworkXError
        If the proposed matching has an edge to a node not in G.
        Or if the matching is not a collection of 2-tuple edges.

    Examples
    --------
    >>> G = nx.Graph([(1, 2), (1, 3), (2, 3), (2, 4), (3, 5), (4, 5)])
    >>> nx.is_maximal_matching(G, {1: 3, 2: 4})  # using dict to represent matching
    True

    >>> nx.is_matching(G, {(1, 3), (2, 4)})  # using set to represent matching
    True

    """
    if isinstance(matching, dict):
        matching = matching_dict_to_set(matching)

    if not all(len(edge) == 2 for edge in matching):
        raise nx.NetworkXError("Matching is not a collection of 2-tuple edges.")

    nodes = set()
    for u, v in matching:
        if u not in G or v not in G:
            raise nx.NetworkXError("Matching contains an edge to a node not in G.")
        if u in nodes or v in nodes:
            return False
        if not G.has_edge(u, v):
            return False
        nodes.add(u)
        nodes.add(v)

    return True

@nx._dispatchable
def is_maximal_matching(G, matching):
    """Return True if ``matching`` is a maximal matching of ``G``

    A *maximal matching* in a graph is a matching in which adding any
    edge would cause the set to no longer be a valid matching.

    Parameters
    ----------
    G : NetworkX graph

    matching : dict or set
        A dictionary or set representing a matching. If a dictionary, it
        must have ``matching[u] == v`` and ``matching[v] == u`` for each
        edge ``(u, v)`` in the matching. If a set, it must have elements
        of the form ``(u, v)``, where ``(u, v)`` is an edge in the
        matching.

    Returns
    -------
    bool
        Whether the given set or dictionary represents a valid maximal
        matching in the graph.

    Examples
    --------
    >>> G = nx.Graph([(1, 2), (1, 3), (2, 3), (3, 4), (3, 5)])
    >>> nx.is_maximal_matching(G, {(1, 2), (3, 4)})
    True

    """
    if not is_matching(G, matching):
        return False

    if isinstance(matching, dict):
        matching = matching_dict_to_set(matching)

    matched_nodes = set(node for edge in matching for node in edge)
    for u, v in G.edges():
        if u not in matched_nodes and v not in matched_nodes:
            return False

    return True

@nx._dispatchable
def is_perfect_matching(G, matching):
    """Return True if ``matching`` is a perfect matching for ``G``

    A *perfect matching* in a graph is a matching in which exactly one edge
    is incident upon each vertex.

    Parameters
    ----------
    G : NetworkX graph

    matching : dict or set
        A dictionary or set representing a matching. If a dictionary, it
        must have ``matching[u] == v`` and ``matching[v] == u`` for each
        edge ``(u, v)`` in the matching. If a set, it must have elements
        of the form ``(u, v)``, where ``(u, v)`` is an edge in the
        matching.

    Returns
    -------
    bool
        Whether the given set or dictionary represents a valid perfect
        matching in the graph.

    Examples
    --------
    >>> G = nx.Graph([(1, 2), (1, 3), (2, 3), (2, 4), (3, 5), (4, 5), (4, 6)])
    >>> my_match = {1: 2, 3: 5, 4: 6}
    >>> nx.is_perfect_matching(G, my_match)
    True

    """
    if not is_matching(G, matching):
        return False

    if isinstance(matching, dict):
        matching = matching_dict_to_set(matching)

    matched_nodes = set(node for edge in matching for node in edge)
    return len(matched_nodes) == len(G)

@not_implemented_for('multigraph')
@not_implemented_for('directed')
@nx._dispatchable(edge_attrs='weight')
def min_weight_matching(G, weight='weight'):
    """Computing a minimum-weight maximal matching of G.

    Use the maximum-weight algorithm with edge weights subtracted
    from the maximum weight of all edges.

    A matching is a subset of edges in which no node occurs more than once.
    The weight of a matching is the sum of the weights of its edges.
    A maximal matching cannot add more edges and still be a matching.
    The cardinality of a matching is the number of matched edges.

    This method replaces the edge weights with 1 plus the maximum edge weight
    minus the original edge weight.

    new_weight = (max_weight + 1) - edge_weight

    then runs :func:`max_weight_matching` with the new weights.
    The max weight matching with these new weights corresponds
    to the min weight matching using the original weights.
    Adding 1 to the max edge weight keeps all edge weights positive
    and as integers if they started as integers.

    You might worry that adding 1 to each weight would make the algorithm
    favor matchings with more edges. But we use the parameter
    `maxcardinality=True` in `max_weight_matching` to ensure that the
    number of edges in the competing matchings are the same and thus
    the optimum does not change due to changes in the number of edges.

    Read the documentation of `max_weight_matching` for more information.

    Parameters
    ----------
    G : NetworkX graph
      Undirected graph

    weight: string, optional (default='weight')
       Edge data key corresponding to the edge weight.
       If key not found, uses 1 as weight.

    Returns
    -------
    matching : set
        A minimal weight matching of the graph.

    See Also
    --------
    max_weight_matching
    """
    # Find the maximum weight
    max_weight = max(d.get(weight, 1) for u, v, d in G.edges(data=True))

    # Create a new graph with modified weights
    H = G.copy()
    for u, v, d in H.edges(data=True):
        d[weight] = (max_weight + 1) - d.get(weight, 1)

    # Run max_weight_matching with the new weights
    matching = max_weight_matching(H, maxcardinality=True, weight=weight)

    # Convert the result to a set of edges
    return matching_dict_to_set(matching)

@not_implemented_for('multigraph')
@not_implemented_for('directed')
@nx._dispatchable(edge_attrs='weight')
def max_weight_matching(G, maxcardinality=False, weight='weight'):
    """Compute a maximum-weighted matching of G.

    A matching is a subset of edges in which no node occurs more than once.
    The weight of a matching is the sum of the weights of its edges.
    A maximal matching cannot add more edges and still be a matching.
    The cardinality of a matching is the number of matched edges.

    Parameters
    ----------
    G : NetworkX graph
      Undirected graph

    maxcardinality: bool, optional (default=False)
       If maxcardinality is True, compute the maximum-cardinality matching
       with maximum weight among all maximum-cardinality matchings.

    weight: string, optional (default='weight')
       Edge data key corresponding to the edge weight.
       If key not found, uses 1 as weight.


    Returns
    -------
    matching : set
        A maximal matching of the graph.

     Examples
    --------
    >>> G = nx.Graph()
    >>> edges = [(1, 2, 6), (1, 3, 2), (2, 3, 1), (2, 4, 7), (3, 5, 9), (4, 5, 3)]
    >>> G.add_weighted_edges_from(edges)
    >>> sorted(nx.max_weight_matching(G))
    [(2, 4), (5, 3)]

    Notes
    -----
    If G has edges with weight attributes the edge data are used as
    weight values else the weights are assumed to be 1.

    This function takes time O(number_of_nodes ** 3).

    If all edge weights are integers, the algorithm uses only integer
    computations.  If floating point weights are used, the algorithm
    could return a slightly suboptimal matching due to numeric
    precision errors.

    This method is based on the "blossom" method for finding augmenting
    paths and the "primal-dual" method for finding a matching of maximum
    weight, both methods invented by Jack Edmonds [1]_.

    Bipartite graphs can also be matched using the functions present in
    :mod:`networkx.algorithms.bipartite.matching`.

    References
    ----------
    .. [1] "Efficient Algorithms for Finding Maximum Matching in Graphs",
       Zvi Galil, ACM Computing Surveys, 1986.
    """
    from networkx.algorithms.matching import INFINITY, maximal_matching

    if not G:
        return set()

    # Initialize matching and dual variables
    matching = {}
    dual = {v: 0 for v in G}
    allowance = {(u, v): 0 for u, v in G.edges()}

    def slack(u, v):
        return dual[u] + dual[v] - G[u][v].get(weight, 1)

    def update_allowance(u, v):
        allowance[(u, v)] = allowance[(v, u)] = slack(u, v)

    for u, v in G.edges():
        update_allowance(u, v)

    def find_augmenting_path(v):
        seen = {}
        def recurse(v):
            for u in G[v]:
                if u not in seen:
                    seen[u] = v
                    if u not in matching or recurse(matching[u]):
                        matching[u] = v
                        matching[v] = u
                        return True
            return False
        return recurse(v)

    # Find initial matching
    for v in G:
        if v not in matching:
            find_augmenting_path(v)

    while True:
        # Find a vertex with maximum dual variable
        v = max(G, key=lambda x: dual[x])
        if dual[v] == 0:
            break

        # Find augmenting path
        if not find_augmenting_path(v):
            # Update dual variables
            delta = min(slack(u, v) for u, v in G.edges() if u != v)
            for u in G:
                dual[u] -= delta
            for u, v in G.edges():
                update_allowance(u, v)

    return matching_dict_to_set(matching)
