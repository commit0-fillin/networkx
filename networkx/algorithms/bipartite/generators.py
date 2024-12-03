"""
Generators and functions for bipartite graphs.
"""
import math
import numbers
from functools import reduce
import networkx as nx
from networkx.utils import nodes_or_number, py_random_state
__all__ = ['configuration_model', 'havel_hakimi_graph', 'reverse_havel_hakimi_graph', 'alternating_havel_hakimi_graph', 'preferential_attachment_graph', 'random_graph', 'gnmk_random_graph', 'complete_bipartite_graph']

@nx._dispatchable(graphs=None, returns_graph=True)
@nodes_or_number([0, 1])
def complete_bipartite_graph(n1, n2, create_using=None):
    """Returns the complete bipartite graph `K_{n_1,n_2}`.

    The graph is composed of two partitions with nodes 0 to (n1 - 1)
    in the first and nodes n1 to (n1 + n2 - 1) in the second.
    Each node in the first is connected to each node in the second.

    Parameters
    ----------
    n1, n2 : integer or iterable container of nodes
        If integers, nodes are from `range(n1)` and `range(n1, n1 + n2)`.
        If a container, the elements are the nodes.
    create_using : NetworkX graph instance, (default: nx.Graph)
       Return graph of this type.

    Notes
    -----
    Nodes are the integers 0 to `n1 + n2 - 1` unless either n1 or n2 are
    containers of nodes. If only one of n1 or n2 are integers, that
    integer is replaced by `range` of that integer.

    The nodes are assigned the attribute 'bipartite' with the value 0 or 1
    to indicate which bipartite set the node belongs to.

    This function is not imported in the main namespace.
    To use it use nx.bipartite.complete_bipartite_graph
    """
    if create_using is None:
        G = nx.Graph()
    else:
        G = nx.empty_graph(0, create_using)

    if isinstance(n1, numbers.Integral):
        n1 = range(n1)
    if isinstance(n2, numbers.Integral):
        n2 = range(len(n1), len(n1) + n2)

    G.add_nodes_from(n1, bipartite=0)
    G.add_nodes_from(n2, bipartite=1)
    G.add_edges_from((u, v) for u in n1 for v in n2)
    return G

@py_random_state(3)
@nx._dispatchable(name='bipartite_configuration_model', graphs=None, returns_graph=True)
def configuration_model(aseq, bseq, create_using=None, seed=None):
    """Returns a random bipartite graph from two given degree sequences.

    Parameters
    ----------
    aseq : list
       Degree sequence for node set A.
    bseq : list
       Degree sequence for node set B.
    create_using : NetworkX graph instance, optional
       Return graph of this type.
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    The graph is composed of two partitions. Set A has nodes 0 to
    (len(aseq) - 1) and set B has nodes len(aseq) to (len(bseq) - 1).
    Nodes from set A are connected to nodes in set B by choosing
    randomly from the possible free stubs, one in A and one in B.

    Notes
    -----
    The sum of the two sequences must be equal: sum(aseq)=sum(bseq)
    If no graph type is specified use MultiGraph with parallel edges.
    If you want a graph with no parallel edges use create_using=Graph()
    but then the resulting degree sequences might not be exact.

    The nodes are assigned the attribute 'bipartite' with the value 0 or 1
    to indicate which bipartite set the node belongs to.

    This function is not imported in the main namespace.
    To use it use nx.bipartite.configuration_model
    """
    if create_using is None:
        create_using = nx.MultiGraph()
    elif not create_using.is_multigraph():
        raise nx.NetworkXError("create_using must be a multigraph")

    G = nx.empty_graph(0, create_using)
    n = len(aseq)
    m = len(bseq)
    
    if sum(aseq) != sum(bseq):
        raise nx.NetworkXError("Degree sequences must have equal sums")

    # Add nodes with bipartite attribute
    G.add_nodes_from(range(n), bipartite=0)
    G.add_nodes_from(range(n, n + m), bipartite=1)

    # Create stubs
    astubs = list(nx.utils.flatten([n] * d for n, d in enumerate(aseq)))
    bstubs = list(nx.utils.flatten([n] * d for n, d in enumerate(bseq, start=n)))

    # Shuffle stubs
    seed = nx.utils.create_random_state(seed)
    seed.shuffle(astubs)
    seed.shuffle(bstubs)

    # Add edges
    G.add_edges_from(zip(astubs, bstubs))

    return G

@nx._dispatchable(name='bipartite_havel_hakimi_graph', graphs=None, returns_graph=True)
def havel_hakimi_graph(aseq, bseq, create_using=None):
    """Returns a bipartite graph from two given degree sequences using a
    Havel-Hakimi style construction.

    The graph is composed of two partitions. Set A has nodes 0 to
    (len(aseq) - 1) and set B has nodes len(aseq) to (len(bseq) - 1).
    Nodes from the set A are connected to nodes in the set B by
    connecting the highest degree nodes in set A to the highest degree
    nodes in set B until all stubs are connected.

    Parameters
    ----------
    aseq : list
       Degree sequence for node set A.
    bseq : list
       Degree sequence for node set B.
    create_using : NetworkX graph instance, optional
       Return graph of this type.

    Notes
    -----
    The sum of the two sequences must be equal: sum(aseq)=sum(bseq)
    If no graph type is specified use MultiGraph with parallel edges.
    If you want a graph with no parallel edges use create_using=Graph()
    but then the resulting degree sequences might not be exact.

    The nodes are assigned the attribute 'bipartite' with the value 0 or 1
    to indicate which bipartite set the node belongs to.

    This function is not imported in the main namespace.
    To use it use nx.bipartite.havel_hakimi_graph
    """
    if create_using is None:
        create_using = nx.MultiGraph()
    elif not create_using.is_multigraph():
        raise nx.NetworkXError("create_using must be a multigraph")

    G = nx.empty_graph(0, create_using)
    n = len(aseq)
    m = len(bseq)

    if sum(aseq) != sum(bseq):
        raise nx.NetworkXError("Degree sequences must have equal sums")

    # Add nodes with bipartite attribute
    G.add_nodes_from(range(n), bipartite=0)
    G.add_nodes_from(range(n, n + m), bipartite=1)

    # Sort degree sequences in descending order
    aseq = sorted(enumerate(aseq), key=lambda x: x[1], reverse=True)
    bseq = sorted(enumerate(bseq, start=n), key=lambda x: x[1], reverse=True)

    stubs_left = sum(aseq[i][1] for i in range(n))

    while stubs_left > 0:
        node_a, deg_a = aseq[0]
        for i in range(deg_a):
            if len(bseq) == 0:
                raise nx.NetworkXError("Not enough stubs in set B")
            node_b, deg_b = bseq[0]
            G.add_edge(node_a, node_b)
            bseq[0] = (node_b, deg_b - 1)
            if bseq[0][1] == 0:
                bseq.pop(0)
            else:
                bseq = sorted(bseq, key=lambda x: x[1], reverse=True)
        aseq.pop(0)
        stubs_left -= deg_a

    return G

@nx._dispatchable(graphs=None, returns_graph=True)
def reverse_havel_hakimi_graph(aseq, bseq, create_using=None):
    """Returns a bipartite graph from two given degree sequences using a
    Havel-Hakimi style construction.

    The graph is composed of two partitions. Set A has nodes 0 to
    (len(aseq) - 1) and set B has nodes len(aseq) to (len(bseq) - 1).
    Nodes from set A are connected to nodes in the set B by connecting
    the highest degree nodes in set A to the lowest degree nodes in
    set B until all stubs are connected.

    Parameters
    ----------
    aseq : list
       Degree sequence for node set A.
    bseq : list
       Degree sequence for node set B.
    create_using : NetworkX graph instance, optional
       Return graph of this type.

    Notes
    -----
    The sum of the two sequences must be equal: sum(aseq)=sum(bseq)
    If no graph type is specified use MultiGraph with parallel edges.
    If you want a graph with no parallel edges use create_using=Graph()
    but then the resulting degree sequences might not be exact.

    The nodes are assigned the attribute 'bipartite' with the value 0 or 1
    to indicate which bipartite set the node belongs to.

    This function is not imported in the main namespace.
    To use it use nx.bipartite.reverse_havel_hakimi_graph
    """
    pass

@nx._dispatchable(graphs=None, returns_graph=True)
def alternating_havel_hakimi_graph(aseq, bseq, create_using=None):
    """Returns a bipartite graph from two given degree sequences using
    an alternating Havel-Hakimi style construction.

    The graph is composed of two partitions. Set A has nodes 0 to
    (len(aseq) - 1) and set B has nodes len(aseq) to (len(bseq) - 1).
    Nodes from the set A are connected to nodes in the set B by
    connecting the highest degree nodes in set A to alternatively the
    highest and the lowest degree nodes in set B until all stubs are
    connected.

    Parameters
    ----------
    aseq : list
       Degree sequence for node set A.
    bseq : list
       Degree sequence for node set B.
    create_using : NetworkX graph instance, optional
       Return graph of this type.

    Notes
    -----
    The sum of the two sequences must be equal: sum(aseq)=sum(bseq)
    If no graph type is specified use MultiGraph with parallel edges.
    If you want a graph with no parallel edges use create_using=Graph()
    but then the resulting degree sequences might not be exact.

    The nodes are assigned the attribute 'bipartite' with the value 0 or 1
    to indicate which bipartite set the node belongs to.

    This function is not imported in the main namespace.
    To use it use nx.bipartite.alternating_havel_hakimi_graph
    """
    pass

@py_random_state(3)
@nx._dispatchable(graphs=None, returns_graph=True)
def preferential_attachment_graph(aseq, p, create_using=None, seed=None):
    """Create a bipartite graph with a preferential attachment model from
    a given single degree sequence.

    The graph is composed of two partitions. Set A has nodes 0 to
    (len(aseq) - 1) and set B has nodes starting with node len(aseq).
    The number of nodes in set B is random.

    Parameters
    ----------
    aseq : list
       Degree sequence for node set A.
    p :  float
       Probability that a new bottom node is added.
    create_using : NetworkX graph instance, optional
       Return graph of this type.
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    References
    ----------
    .. [1] Guillaume, J.L. and Latapy, M.,
       Bipartite graphs as models of complex networks.
       Physica A: Statistical Mechanics and its Applications,
       2006, 371(2), pp.795-813.
    .. [2] Jean-Loup Guillaume and Matthieu Latapy,
       Bipartite structure of all complex networks,
       Inf. Process. Lett. 90, 2004, pg. 215-221
       https://doi.org/10.1016/j.ipl.2004.03.007

    Notes
    -----
    The nodes are assigned the attribute 'bipartite' with the value 0 or 1
    to indicate which bipartite set the node belongs to.

    This function is not imported in the main namespace.
    To use it use nx.bipartite.preferential_attachment_graph
    """
    pass

@py_random_state(3)
@nx._dispatchable(graphs=None, returns_graph=True)
def random_graph(n, m, p, seed=None, directed=False):
    """Returns a bipartite random graph.

    This is a bipartite version of the binomial (Erdős-Rényi) graph.
    The graph is composed of two partitions. Set A has nodes 0 to
    (n - 1) and set B has nodes n to (n + m - 1).

    Parameters
    ----------
    n : int
        The number of nodes in the first bipartite set.
    m : int
        The number of nodes in the second bipartite set.
    p : float
        Probability for edge creation.
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.
    directed : bool, optional (default=False)
        If True return a directed graph

    Notes
    -----
    The bipartite random graph algorithm chooses each of the n*m (undirected)
    or 2*nm (directed) possible edges with probability p.

    This algorithm is $O(n+m)$ where $m$ is the expected number of edges.

    The nodes are assigned the attribute 'bipartite' with the value 0 or 1
    to indicate which bipartite set the node belongs to.

    This function is not imported in the main namespace.
    To use it use nx.bipartite.random_graph

    See Also
    --------
    gnp_random_graph, configuration_model

    References
    ----------
    .. [1] Vladimir Batagelj and Ulrik Brandes,
       "Efficient generation of large random networks",
       Phys. Rev. E, 71, 036113, 2005.
    """
    pass

@py_random_state(3)
@nx._dispatchable(graphs=None, returns_graph=True)
def gnmk_random_graph(n, m, k, seed=None, directed=False):
    """Returns a random bipartite graph G_{n,m,k}.

    Produces a bipartite graph chosen randomly out of the set of all graphs
    with n top nodes, m bottom nodes, and k edges.
    The graph is composed of two sets of nodes.
    Set A has nodes 0 to (n - 1) and set B has nodes n to (n + m - 1).

    Parameters
    ----------
    n : int
        The number of nodes in the first bipartite set.
    m : int
        The number of nodes in the second bipartite set.
    k : int
        The number of edges
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.
    directed : bool, optional (default=False)
        If True return a directed graph

    Examples
    --------
    from nx.algorithms import bipartite
    G = bipartite.gnmk_random_graph(10,20,50)

    See Also
    --------
    gnm_random_graph

    Notes
    -----
    If k > m * n then a complete bipartite graph is returned.

    This graph is a bipartite version of the `G_{nm}` random graph model.

    The nodes are assigned the attribute 'bipartite' with the value 0 or 1
    to indicate which bipartite set the node belongs to.

    This function is not imported in the main namespace.
    To use it use nx.bipartite.gnmk_random_graph
    """
    pass
