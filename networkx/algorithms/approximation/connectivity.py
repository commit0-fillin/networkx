""" Fast approximation for node connectivity
"""
import itertools
from operator import itemgetter
import networkx as nx
__all__ = ['local_node_connectivity', 'node_connectivity', 'all_pairs_node_connectivity']

@nx._dispatchable(name='approximate_local_node_connectivity')
def local_node_connectivity(G, source, target, cutoff=None):
    """Compute node connectivity between source and target.

    Pairwise or local node connectivity between two distinct and nonadjacent
    nodes is the minimum number of nodes that must be removed (minimum
    separating cutset) to disconnect them. By Menger's theorem, this is equal
    to the number of node independent paths (paths that share no nodes other
    than source and target). Which is what we compute in this function.

    This algorithm is a fast approximation that gives an strict lower
    bound on the actual number of node independent paths between two nodes [1]_.
    It works for both directed and undirected graphs.

    Parameters
    ----------

    G : NetworkX graph

    source : node
        Starting node for node connectivity

    target : node
        Ending node for node connectivity

    cutoff : integer
        Maximum node connectivity to consider. If None, the minimum degree
        of source or target is used as a cutoff. Default value None.

    Returns
    -------
    k: integer
       pairwise node connectivity

    Examples
    --------
    >>> # Platonic octahedral graph has node connectivity 4
    >>> # for each non adjacent node pair
    >>> from networkx.algorithms import approximation as approx
    >>> G = nx.octahedral_graph()
    >>> approx.local_node_connectivity(G, 0, 5)
    4

    Notes
    -----
    This algorithm [1]_ finds node independents paths between two nodes by
    computing their shortest path using BFS, marking the nodes of the path
    found as 'used' and then searching other shortest paths excluding the
    nodes marked as used until no more paths exist. It is not exact because
    a shortest path could use nodes that, if the path were longer, may belong
    to two different node independent paths. Thus it only guarantees an
    strict lower bound on node connectivity.

    Note that the authors propose a further refinement, losing accuracy and
    gaining speed, which is not implemented yet.

    See also
    --------
    all_pairs_node_connectivity
    node_connectivity

    References
    ----------
    .. [1] White, Douglas R., and Mark Newman. 2001 A Fast Algorithm for
        Node-Independent Paths. Santa Fe Institute Working Paper #01-07-035
        http://eclectic.ss.uci.edu/~drwhite/working.pdf

    """
    if cutoff is None:
        cutoff = min(G.degree(source), G.degree(target))

    if G.is_directed():
        G = G.to_undirected()

    exclude = set()
    paths = 0

    while True:
        try:
            path = _bidirectional_shortest_path(G, source, target, exclude)
            exclude.update(set(path) - {source, target})
            paths += 1
            if paths == cutoff:
                break
        except nx.NetworkXNoPath:
            break

    return paths

@nx._dispatchable(name='approximate_node_connectivity')
def node_connectivity(G, s=None, t=None):
    """Returns an approximation for node connectivity for a graph or digraph G.

    Node connectivity is equal to the minimum number of nodes that
    must be removed to disconnect G or render it trivial. By Menger's theorem,
    this is equal to the number of node independent paths (paths that
    share no nodes other than source and target).

    If source and target nodes are provided, this function returns the
    local node connectivity: the minimum number of nodes that must be
    removed to break all paths from source to target in G.

    This algorithm is based on a fast approximation that gives an strict lower
    bound on the actual number of node independent paths between two nodes [1]_.
    It works for both directed and undirected graphs.

    Parameters
    ----------
    G : NetworkX graph
        Undirected graph

    s : node
        Source node. Optional. Default value: None.

    t : node
        Target node. Optional. Default value: None.

    Returns
    -------
    K : integer
        Node connectivity of G, or local node connectivity if source
        and target are provided.

    Examples
    --------
    >>> # Platonic octahedral graph is 4-node-connected
    >>> from networkx.algorithms import approximation as approx
    >>> G = nx.octahedral_graph()
    >>> approx.node_connectivity(G)
    4

    Notes
    -----
    This algorithm [1]_ finds node independents paths between two nodes by
    computing their shortest path using BFS, marking the nodes of the path
    found as 'used' and then searching other shortest paths excluding the
    nodes marked as used until no more paths exist. It is not exact because
    a shortest path could use nodes that, if the path were longer, may belong
    to two different node independent paths. Thus it only guarantees an
    strict lower bound on node connectivity.

    See also
    --------
    all_pairs_node_connectivity
    local_node_connectivity

    References
    ----------
    .. [1] White, Douglas R., and Mark Newman. 2001 A Fast Algorithm for
        Node-Independent Paths. Santa Fe Institute Working Paper #01-07-035
        http://eclectic.ss.uci.edu/~drwhite/working.pdf

    """
    if s is not None and t is not None:
        return local_node_connectivity(G, s, t)
    
    if G.is_directed():
        G = G.to_undirected()
    
    if len(G) < 3:
        return min(G.degree())

    connectivity = float('inf')
    for s, t in itertools.combinations(G, 2):
        connectivity = min(connectivity, local_node_connectivity(G, s, t))
        if connectivity == 1:
            return 1
    
    return connectivity

@nx._dispatchable(name='approximate_all_pairs_node_connectivity')
def all_pairs_node_connectivity(G, nbunch=None, cutoff=None):
    """Compute node connectivity between all pairs of nodes.

    Pairwise or local node connectivity between two distinct and nonadjacent
    nodes is the minimum number of nodes that must be removed (minimum
    separating cutset) to disconnect them. By Menger's theorem, this is equal
    to the number of node independent paths (paths that share no nodes other
    than source and target). Which is what we compute in this function.

    This algorithm is a fast approximation that gives an strict lower
    bound on the actual number of node independent paths between two nodes [1]_.
    It works for both directed and undirected graphs.


    Parameters
    ----------
    G : NetworkX graph

    nbunch: container
        Container of nodes. If provided node connectivity will be computed
        only over pairs of nodes in nbunch.

    cutoff : integer
        Maximum node connectivity to consider. If None, the minimum degree
        of source or target is used as a cutoff in each pair of nodes.
        Default value None.

    Returns
    -------
    K : dictionary
        Dictionary, keyed by source and target, of pairwise node connectivity

    Examples
    --------
    A 3 node cycle with one extra node attached has connectivity 2 between all
    nodes in the cycle and connectivity 1 between the extra node and the rest:

    >>> G = nx.cycle_graph(3)
    >>> G.add_edge(2, 3)
    >>> import pprint  # for nice dictionary formatting
    >>> pprint.pprint(nx.all_pairs_node_connectivity(G))
    {0: {1: 2, 2: 2, 3: 1},
     1: {0: 2, 2: 2, 3: 1},
     2: {0: 2, 1: 2, 3: 1},
     3: {0: 1, 1: 1, 2: 1}}

    See Also
    --------
    local_node_connectivity
    node_connectivity

    References
    ----------
    .. [1] White, Douglas R., and Mark Newman. 2001 A Fast Algorithm for
        Node-Independent Paths. Santa Fe Institute Working Paper #01-07-035
        http://eclectic.ss.uci.edu/~drwhite/working.pdf
    """
    if nbunch is None:
        nbunch = G.nodes()
    else:
        nbunch = set(nbunch)

    connectivity = {}
    for u in nbunch:
        connectivity[u] = {}
        for v in nbunch:
            if u == v:
                continue
            connectivity[u][v] = local_node_connectivity(G, u, v, cutoff=cutoff)

    return connectivity

def _bidirectional_shortest_path(G, source, target, exclude):
    """Returns shortest path between source and target ignoring nodes in the
    container 'exclude'.

    Parameters
    ----------

    G : NetworkX graph

    source : node
        Starting node for path

    target : node
        Ending node for path

    exclude: container
        Container for nodes to exclude from the search for shortest paths

    Returns
    -------
    path: list
        Shortest path between source and target ignoring nodes in 'exclude'

    Raises
    ------
    NetworkXNoPath
        If there is no path or if nodes are adjacent and have only one path
        between them

    Notes
    -----
    This function and its helper are originally from
    networkx.algorithms.shortest_paths.unweighted and are modified to
    accept the extra parameter 'exclude', which is a container for nodes
    already used in other paths that should be ignored.

    References
    ----------
    .. [1] White, Douglas R., and Mark Newman. 2001 A Fast Algorithm for
        Node-Independent Paths. Santa Fe Institute Working Paper #01-07-035
        http://eclectic.ss.uci.edu/~drwhite/working.pdf

    """
    if source == target:
        return [source]
    
    # Initialize forward and backward frontiers
    forward_fringe = [source]
    backward_fringe = [target]
    
    # Initialize forward and backward paths
    forward_paths = {source: [source]}
    backward_paths = {target: [target]}
    
    while forward_fringe and backward_fringe:
        # Forward pass
        this_level = forward_fringe
        forward_fringe = []
        for v in this_level:
            for w in G.neighbors(v):
                if w not in exclude and w not in forward_paths:
                    forward_paths[w] = forward_paths[v] + [w]
                    if w in backward_paths:
                        return forward_paths[w][:-1] + backward_paths[w][::-1]
                    forward_fringe.append(w)
        
        # Backward pass
        this_level = backward_fringe
        backward_fringe = []
        for v in this_level:
            for w in G.neighbors(v):
                if w not in exclude and w not in backward_paths:
                    backward_paths[w] = [w] + backward_paths[v]
                    if w in forward_paths:
                        return forward_paths[w] + backward_paths[w][-2::-1]
                    backward_fringe.append(w)
    
    raise nx.NetworkXNoPath(f"No path between {source} and {target}.")
