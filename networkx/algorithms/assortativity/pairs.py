"""Generators of  x-y pairs of node data."""
import networkx as nx
__all__ = ['node_attribute_xy', 'node_degree_xy']

@nx._dispatchable(node_attrs='attribute')
def node_attribute_xy(G, attribute, nodes=None):
    """Returns iterator of node-attribute pairs for all edges in G.

    Parameters
    ----------
    G: NetworkX graph

    attribute: key
       The node attribute key.

    nodes: list or iterable (optional)
        Use only edges that are incident to specified nodes.
        The default is all nodes.

    Returns
    -------
    (x, y): 2-tuple
        Generates 2-tuple of (attribute, attribute) values.

    Examples
    --------
    >>> G = nx.DiGraph()
    >>> G.add_node(1, color="red")
    >>> G.add_node(2, color="blue")
    >>> G.add_edge(1, 2)
    >>> list(nx.node_attribute_xy(G, "color"))
    [('red', 'blue')]

    Notes
    -----
    For undirected graphs each edge is produced twice, once for each edge
    representation (u, v) and (v, u), with the exception of self-loop edges
    which only appear once.
    """
    if nodes is None:
        nodes = G.nodes()
    else:
        nodes = set(nodes)

    for u, v in G.edges():
        if u in nodes or v in nodes:
            yield (G.nodes[u].get(attribute), G.nodes[v].get(attribute))

    if not G.is_directed():
        for u, v in G.edges():
            if u != v and (u in nodes or v in nodes):
                yield (G.nodes[v].get(attribute), G.nodes[u].get(attribute))

@nx._dispatchable(edge_attrs='weight')
def node_degree_xy(G, x='out', y='in', weight=None, nodes=None):
    """Generate node degree-degree pairs for edges in G.

    Parameters
    ----------
    G: NetworkX graph

    x: string ('in','out')
       The degree type for source node (directed graphs only).

    y: string ('in','out')
       The degree type for target node (directed graphs only).

    weight: string or None, optional (default=None)
       The edge attribute that holds the numerical value used
       as a weight.  If None, then each edge has weight 1.
       The degree is the sum of the edge weights adjacent to the node.

    nodes: list or iterable (optional)
        Use only edges that are adjacency to specified nodes.
        The default is all nodes.

    Returns
    -------
    (x, y): 2-tuple
        Generates 2-tuple of (degree, degree) values.


    Examples
    --------
    >>> G = nx.DiGraph()
    >>> G.add_edge(1, 2)
    >>> list(nx.node_degree_xy(G, x="out", y="in"))
    [(1, 1)]
    >>> list(nx.node_degree_xy(G, x="in", y="out"))
    [(0, 0)]

    Notes
    -----
    For undirected graphs each edge is produced twice, once for each edge
    representation (u, v) and (v, u), with the exception of self-loop edges
    which only appear once.
    """
    if nodes is None:
        nodes = G.nodes()
    else:
        nodes = set(nodes)

    if G.is_directed():
        degree_func = {
            'out': G.out_degree,
            'in': G.in_degree
        }
        x_degree = degree_func[x]
        y_degree = degree_func[y]
    else:
        x_degree = y_degree = G.degree

    for u, v in G.edges():
        if u in nodes or v in nodes:
            yield (x_degree(u, weight=weight), y_degree(v, weight=weight))

    if not G.is_directed():
        for u, v in G.edges():
            if u != v and (u in nodes or v in nodes):
                yield (x_degree(v, weight=weight), y_degree(u, weight=weight))
