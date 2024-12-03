"""Routines to find the boundary of a set of nodes.

An edge boundary is a set of edges, each of which has exactly one
endpoint in a given set of nodes (or, in the case of directed graphs,
the set of edges whose source node is in the set).

A node boundary of a set *S* of nodes is the set of (out-)neighbors of
nodes in *S* that are outside *S*.

"""
from itertools import chain
import networkx as nx
__all__ = ['edge_boundary', 'node_boundary']

@nx._dispatchable(edge_attrs={'data': 'default'}, preserve_edge_attrs='data')
def edge_boundary(G, nbunch1, nbunch2=None, data=False, keys=False, default=None):
    """Returns the edge boundary of `nbunch1`.

    The *edge boundary* of a set *S* with respect to a set *T* is the
    set of edges (*u*, *v*) such that *u* is in *S* and *v* is in *T*.
    If *T* is not specified, it is assumed to be the set of all nodes
    not in *S*.

    Parameters
    ----------
    G : NetworkX graph

    nbunch1 : iterable
        Iterable of nodes in the graph representing the set of nodes
        whose edge boundary will be returned. (This is the set *S* from
        the definition above.)

    nbunch2 : iterable
        Iterable of nodes representing the target (or "exterior") set of
        nodes. (This is the set *T* from the definition above.) If not
        specified, this is assumed to be the set of all nodes in `G`
        not in `nbunch1`.

    keys : bool
        This parameter has the same meaning as in
        :meth:`MultiGraph.edges`.

    data : bool or object
        This parameter has the same meaning as in
        :meth:`MultiGraph.edges`.

    default : object
        This parameter has the same meaning as in
        :meth:`MultiGraph.edges`.

    Returns
    -------
    iterator
        An iterator over the edges in the boundary of `nbunch1` with
        respect to `nbunch2`. If `keys`, `data`, or `default`
        are specified and `G` is a multigraph, then edges are returned
        with keys and/or data, as in :meth:`MultiGraph.edges`.

    Examples
    --------
    >>> G = nx.wheel_graph(6)

    When nbunch2=None:

    >>> list(nx.edge_boundary(G, (1, 3)))
    [(1, 0), (1, 2), (1, 5), (3, 0), (3, 2), (3, 4)]

    When nbunch2 is given:

    >>> list(nx.edge_boundary(G, (1, 3), (2, 0)))
    [(1, 0), (1, 2), (3, 0), (3, 2)]

    Notes
    -----
    Any element of `nbunch` that is not in the graph `G` will be
    ignored.

    `nbunch1` and `nbunch2` are usually meant to be disjoint, but in
    the interest of speed and generality, that is not required here.

    """
    nset1 = set(n for n in nbunch1 if n in G)
    if nbunch2 is None:
        nset2 = set(G) - nset1
    else:
        nset2 = set(n for n in nbunch2 if n in G)
    
    bdy_edges = ((n1, n2) for n1 in nset1 for n2 in G[n1] if n2 in nset2)
    
    if G.is_multigraph():
        if keys:
            bdy_edges = ((n1, n2, k) for n1, n2 in bdy_edges for k in G[n1][n2])
        if data:
            if data is True:
                if default is None:
                    bdy_edges = ((u, v, k, d) for u, v, k in bdy_edges
                                 for d in [G[u][v][k]])
                else:
                    bdy_edges = ((u, v, k, G[u][v][k].get(data, default))
                                 for u, v, k in bdy_edges)
            else:
                bdy_edges = ((u, v, k, G[u][v][k].get(data, default))
                             for u, v, k in bdy_edges)
    elif data:
        if data is True:
            bdy_edges = ((u, v, G[u][v]) for u, v in bdy_edges)
        else:
            bdy_edges = ((u, v, G[u][v].get(data, default)) for u, v in bdy_edges)
    
    return bdy_edges

@nx._dispatchable
def node_boundary(G, nbunch1, nbunch2=None):
    """Returns the node boundary of `nbunch1`.

    The *node boundary* of a set *S* with respect to a set *T* is the
    set of nodes *v* in *T* such that for some *u* in *S*, there is an
    edge joining *u* to *v*. If *T* is not specified, it is assumed to
    be the set of all nodes not in *S*.

    Parameters
    ----------
    G : NetworkX graph

    nbunch1 : iterable
        Iterable of nodes in the graph representing the set of nodes
        whose node boundary will be returned. (This is the set *S* from
        the definition above.)

    nbunch2 : iterable
        Iterable of nodes representing the target (or "exterior") set of
        nodes. (This is the set *T* from the definition above.) If not
        specified, this is assumed to be the set of all nodes in `G`
        not in `nbunch1`.

    Returns
    -------
    set
        The node boundary of `nbunch1` with respect to `nbunch2`.

    Examples
    --------
    >>> G = nx.wheel_graph(6)

    When nbunch2=None:

    >>> list(nx.node_boundary(G, (3, 4)))
    [0, 2, 5]

    When nbunch2 is given:

    >>> list(nx.node_boundary(G, (3, 4), (0, 1, 5)))
    [0, 5]

    Notes
    -----
    Any element of `nbunch` that is not in the graph `G` will be
    ignored.

    `nbunch1` and `nbunch2` are usually meant to be disjoint, but in
    the interest of speed and generality, that is not required here.

    """
    nset1 = set(n for n in nbunch1 if n in G)
    if nbunch2 is None:
        nset2 = set(G) - nset1
    else:
        nset2 = set(n for n in nbunch2 if n in G)
    
    bdy_nodes = set()
    for n1 in nset1:
        bdy_nodes.update(n2 for n2 in G[n1] if n2 in nset2)
    
    return bdy_nodes
