"""Routines to calculate the broadcast time of certain graphs.

Broadcasting is an information dissemination problem in which a node in a graph,
called the originator, must distribute a message to all other nodes by placing
a series of calls along the edges of the graph. Once informed, other nodes aid
the originator in distributing the message.

The broadcasting must be completed as quickly as possible subject to the
following constraints:
- Each call requires one unit of time.
- A node can only participate in one call per unit of time.
- Each call only involves two adjacent nodes: a sender and a receiver.
"""
import networkx as nx
from networkx import NetworkXError
from networkx.utils import not_implemented_for
__all__ = ['tree_broadcast_center', 'tree_broadcast_time']

@not_implemented_for('directed')
@not_implemented_for('multigraph')
@nx._dispatchable
def tree_broadcast_center(G):
    """Return the Broadcast Center of the tree `G`.

    The broadcast center of a graph G denotes the set of nodes having
    minimum broadcast time [1]_. This is a linear algorithm for determining
    the broadcast center of a tree with ``N`` nodes, as a by-product it also
    determines the broadcast time from the broadcast center.

    Parameters
    ----------
    G : undirected graph
        The graph should be an undirected tree

    Returns
    -------
    BC : (int, set) tuple
        minimum broadcast number of the tree, set of broadcast centers

    Raises
    ------
    NetworkXNotImplemented
        If the graph is directed or is a multigraph.

    References
    ----------
    .. [1] Slater, P.J., Cockayne, E.J., Hedetniemi, S.T,
       Information dissemination in trees. SIAM J.Comput. 10(4), 692–701 (1981)
    """
    if not nx.is_tree(G):
        raise NetworkXError("The graph G must be a tree.")

    # If the tree has only one node, it is the broadcast center
    if len(G) == 1:
        return 0, set(G.nodes())

    # Find the center of the tree
    center = nx.center(G)

    # Start from any center node
    root = center[0]

    # Perform a BFS to get the levels and find the deepest level
    levels = nx.single_source_shortest_path_length(G, root)
    max_level = max(levels.values())

    # The broadcast time is the deepest level
    broadcast_time = max_level

    # Find nodes that achieve this broadcast time
    broadcast_centers = {node for node in G.nodes() if max(nx.single_source_shortest_path_length(G, node).values()) == broadcast_time}

    return broadcast_time, broadcast_centers

@not_implemented_for('directed')
@not_implemented_for('multigraph')
@nx._dispatchable
def tree_broadcast_time(G, node=None):
    """Return the Broadcast Time of the tree `G`.

    The minimum broadcast time of a node is defined as the minimum amount
    of time required to complete broadcasting starting from the
    originator. The broadcast time of a graph is the maximum over
    all nodes of the minimum broadcast time from that node [1]_.
    This function returns the minimum broadcast time of `node`.
    If `node` is None the broadcast time for the graph is returned.

    Parameters
    ----------
    G : undirected graph
        The graph should be an undirected tree
    node: int, optional
        index of starting node. If `None`, the algorithm returns the broadcast
        time of the tree.

    Returns
    -------
    BT : int
        Broadcast Time of a node in a tree

    Raises
    ------
    NetworkXNotImplemented
        If the graph is directed or is a multigraph.

    References
    ----------
    .. [1] Harutyunyan, H. A. and Li, Z.
        "A Simple Construction of Broadcast Graphs."
        In Computing and Combinatorics. COCOON 2019
        (Ed. D. Z. Du and C. Tian.) Springer, pp. 240-253, 2019.
    """
    if not nx.is_tree(G):
        raise NetworkXError("The graph G must be a tree.")

    if node is None:
        # If node is None, return the broadcast time of the tree
        return tree_broadcast_center(G)[0]
    else:
        # If node is specified, return its broadcast time
        if node not in G:
            raise NetworkXError(f"Node {node} is not in the graph.")
        
        # Perform BFS from the specified node
        levels = nx.single_source_shortest_path_length(G, node)
        
        # The broadcast time is the maximum level (depth) from the node
        return max(levels.values())
