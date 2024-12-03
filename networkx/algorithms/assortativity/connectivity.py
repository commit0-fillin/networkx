from collections import defaultdict
import networkx as nx
__all__ = ['average_degree_connectivity']

@nx._dispatchable(edge_attrs='weight')
def average_degree_connectivity(G, source='in+out', target='in+out', nodes=None, weight=None):
    """Compute the average degree connectivity of graph.

    The average degree connectivity is the average nearest neighbor degree of
    nodes with degree k. For weighted graphs, an analogous measure can
    be computed using the weighted average neighbors degree defined in
    [1]_, for a node `i`, as

    .. math::

        k_{nn,i}^{w} = \\frac{1}{s_i} \\sum_{j \\in N(i)} w_{ij} k_j

    where `s_i` is the weighted degree of node `i`,
    `w_{ij}` is the weight of the edge that links `i` and `j`,
    and `N(i)` are the neighbors of node `i`.

    Parameters
    ----------
    G : NetworkX graph

    source :  "in"|"out"|"in+out" (default:"in+out")
       Directed graphs only. Use "in"- or "out"-degree for source node.

    target : "in"|"out"|"in+out" (default:"in+out"
       Directed graphs only. Use "in"- or "out"-degree for target node.

    nodes : list or iterable (optional)
        Compute neighbor connectivity for these nodes. The default is all
        nodes.

    weight : string or None, optional (default=None)
       The edge attribute that holds the numerical value used as a weight.
       If None, then each edge has weight 1.

    Returns
    -------
    d : dict
       A dictionary keyed by degree k with the value of average connectivity.

    Raises
    ------
    NetworkXError
        If either `source` or `target` are not one of 'in',
        'out', or 'in+out'.
        If either `source` or `target` is passed for an undirected graph.

    Examples
    --------
    >>> G = nx.path_graph(4)
    >>> G.edges[1, 2]["weight"] = 3
    >>> nx.average_degree_connectivity(G)
    {1: 2.0, 2: 1.5}
    >>> nx.average_degree_connectivity(G, weight="weight")
    {1: 2.0, 2: 1.75}

    See Also
    --------
    average_neighbor_degree

    References
    ----------
    .. [1] A. Barrat, M. Barthélemy, R. Pastor-Satorras, and A. Vespignani,
       "The architecture of complex weighted networks".
       PNAS 101 (11): 3747–3752 (2004).
    """
    if G.is_directed() and (source not in {'in', 'out', 'in+out'} or target not in {'in', 'out', 'in+out'}):
        raise nx.NetworkXError("Invalid source or target value. Must be 'in', 'out', or 'in+out'.")
    if not G.is_directed() and (source != 'in+out' or target != 'in+out'):
        raise nx.NetworkXError("source and target arguments are only for directed graphs.")

    if nodes is None:
        nodes = G.nodes()

    if weight is None:
        weight_func = lambda x: 1
    else:
        weight_func = lambda x: x.get(weight, 1)

    degree_dict = defaultdict(float)
    total_weight_dict = defaultdict(float)

    for n in nodes:
        if G.is_directed():
            source_degree = G.in_degree(n) if source == 'in' else G.out_degree(n) if source == 'out' else G.degree(n)
            target_degree = G.in_degree if target == 'in' else G.out_degree if target == 'out' else G.degree
            neighbors = G.predecessors(n) if target == 'in' else G.successors(n) if target == 'out' else G.neighbors(n)
        else:
            source_degree = G.degree(n)
            target_degree = G.degree
            neighbors = G.neighbors(n)

        for nbr in neighbors:
            w = weight_func(G[n][nbr])
            degree_dict[source_degree] += w * target_degree(nbr)
            total_weight_dict[source_degree] += w

    return {k: v / total_weight_dict[k] if total_weight_dict[k] > 0 else 0
            for k, v in degree_dict.items()}
