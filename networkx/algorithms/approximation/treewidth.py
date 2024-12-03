"""Functions for computing treewidth decomposition.

Treewidth of an undirected graph is a number associated with the graph.
It can be defined as the size of the largest vertex set (bag) in a tree
decomposition of the graph minus one.

`Wikipedia: Treewidth <https://en.wikipedia.org/wiki/Treewidth>`_

The notions of treewidth and tree decomposition have gained their
attractiveness partly because many graph and network problems that are
intractable (e.g., NP-hard) on arbitrary graphs become efficiently
solvable (e.g., with a linear time algorithm) when the treewidth of the
input graphs is bounded by a constant [1]_ [2]_.

There are two different functions for computing a tree decomposition:
:func:`treewidth_min_degree` and :func:`treewidth_min_fill_in`.

.. [1] Hans L. Bodlaender and Arie M. C. A. Koster. 2010. "Treewidth
      computations I.Upper bounds". Inf. Comput. 208, 3 (March 2010),259-275.
      http://dx.doi.org/10.1016/j.ic.2009.03.008

.. [2] Hans L. Bodlaender. "Discovering Treewidth". Institute of Information
      and Computing Sciences, Utrecht University.
      Technical Report UU-CS-2005-018.
      http://www.cs.uu.nl

.. [3] K. Wang, Z. Lu, and J. Hicks *Treewidth*.
      https://web.archive.org/web/20210507025929/http://web.eecs.utk.edu/~cphill25/cs594_spring2015_projects/treewidth.pdf

"""
import itertools
import sys
from heapq import heapify, heappop, heappush
import networkx as nx
from networkx.utils import not_implemented_for
__all__ = ['treewidth_min_degree', 'treewidth_min_fill_in']

@not_implemented_for('directed')
@not_implemented_for('multigraph')
@nx._dispatchable(returns_graph=True)
def treewidth_min_degree(G):
    """Returns a treewidth decomposition using the Minimum Degree heuristic.

    The heuristic chooses the nodes according to their degree, i.e., first
    the node with the lowest degree is chosen, then the graph is updated
    and the corresponding node is removed. Next, a new node with the lowest
    degree is chosen, and so on.

    Parameters
    ----------
    G : NetworkX graph

    Returns
    -------
    Treewidth decomposition : (int, Graph) tuple
          2-tuple with treewidth and the corresponding decomposed tree.
    """
    return treewidth_decomp(G, heuristic=MinDegreeHeuristic)

@not_implemented_for('directed')
@not_implemented_for('multigraph')
@nx._dispatchable(returns_graph=True)
def treewidth_min_fill_in(G):
    """Returns a treewidth decomposition using the Minimum Fill-in heuristic.

    The heuristic chooses a node from the graph, where the number of edges
    added turning the neighborhood of the chosen node into clique is as
    small as possible.

    Parameters
    ----------
    G : NetworkX graph

    Returns
    -------
    Treewidth decomposition : (int, Graph) tuple
        2-tuple with treewidth and the corresponding decomposed tree.
    """
    return treewidth_decomp(G, heuristic=min_fill_in_heuristic)

class MinDegreeHeuristic:
    """Implements the Minimum Degree heuristic.

    The heuristic chooses the nodes according to their degree
    (number of neighbors), i.e., first the node with the lowest degree is
    chosen, then the graph is updated and the corresponding node is
    removed. Next, a new node with the lowest degree is chosen, and so on.
    """

    def __init__(self, graph):
        self._graph = graph
        self._update_nodes = []
        self._degreeq = []
        self.count = itertools.count()
        for n in graph:
            self._degreeq.append((len(graph[n]), next(self.count), n))
        heapify(self._degreeq)

def min_fill_in_heuristic(graph):
    """Implements the Minimum Fill-in heuristic.

    Returns the node from the graph, where the number of edges added when
    turning the neighborhood of the chosen node into clique is as small as
    possible. This algorithm chooses the nodes using the Minimum Fill-In
    heuristic. The running time of the algorithm is :math:`O(V^3)` and it uses
    additional constant memory."""
    min_fill = float('inf')
    min_node = None
    for node in graph:
        fill = 0
        neighbors = list(graph.neighbors(node))
        for i, n1 in enumerate(neighbors):
            for n2 in neighbors[i + 1:]:
                if not graph.has_edge(n1, n2):
                    fill += 1
        if fill < min_fill:
            min_fill = fill
            min_node = node
    return min_node

@nx._dispatchable(returns_graph=True)
def treewidth_decomp(G, heuristic=min_fill_in_heuristic):
    """Returns a treewidth decomposition using the passed heuristic.

    Parameters
    ----------
    G : NetworkX graph
    heuristic : heuristic function

    Returns
    -------
    Treewidth decomposition : (int, Graph) tuple
        2-tuple with treewidth and the corresponding decomposed tree.
    """
    H = G.copy()
    elimination_order = []
    tree = nx.Graph()
    max_bag_size = 0
    
    while H:
        node = heuristic(H) if callable(heuristic) else heuristic.pop(H)
        neighbors = list(H.neighbors(node))
        elimination_order.append((node, set(neighbors)))
        
        # Create a clique with the neighbors
        for i, n1 in enumerate(neighbors):
            for n2 in neighbors[i + 1:]:
                H.add_edge(n1, n2)
        
        H.remove_node(node)
    
    # Build the tree decomposition
    for i, (node, neighbors) in enumerate(elimination_order):
        bag = {node} | neighbors
        max_bag_size = max(max_bag_size, len(bag))
        tree.add_node(i, bag=bag)
        
        # Connect to previous bags containing neighbors
        for j in range(i):
            if neighbors & tree.nodes[j]['bag']:
                tree.add_edge(i, j)
    
    return max_bag_size - 1, tree
