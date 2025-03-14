"""Katz centrality."""
import math
import networkx as nx
from networkx.utils import not_implemented_for
__all__ = ['katz_centrality', 'katz_centrality_numpy']

@not_implemented_for('multigraph')
@nx._dispatchable(edge_attrs='weight')
def katz_centrality(G, alpha=0.1, beta=1.0, max_iter=1000, tol=1e-06, nstart=None, normalized=True, weight=None):
    """Compute the Katz centrality for the nodes of the graph G.

    Katz centrality computes the centrality for a node based on the centrality
    of its neighbors. It is a generalization of the eigenvector centrality. The
    Katz centrality for node $i$ is

    .. math::

        x_i = \\alpha \\sum_{j} A_{ij} x_j + \\beta,

    where $A$ is the adjacency matrix of graph G with eigenvalues $\\lambda$.

    The parameter $\\beta$ controls the initial centrality and

    .. math::

        \\alpha < \\frac{1}{\\lambda_{\\max}}.

    Katz centrality computes the relative influence of a node within a
    network by measuring the number of the immediate neighbors (first
    degree nodes) and also all other nodes in the network that connect
    to the node under consideration through these immediate neighbors.

    Extra weight can be provided to immediate neighbors through the
    parameter $\\beta$.  Connections made with distant neighbors
    are, however, penalized by an attenuation factor $\\alpha$ which
    should be strictly less than the inverse largest eigenvalue of the
    adjacency matrix in order for the Katz centrality to be computed
    correctly. More information is provided in [1]_.

    Parameters
    ----------
    G : graph
      A NetworkX graph.

    alpha : float, optional (default=0.1)
      Attenuation factor

    beta : scalar or dictionary, optional (default=1.0)
      Weight attributed to the immediate neighborhood. If not a scalar, the
      dictionary must have a value for every node.

    max_iter : integer, optional (default=1000)
      Maximum number of iterations in power method.

    tol : float, optional (default=1.0e-6)
      Error tolerance used to check convergence in power method iteration.

    nstart : dictionary, optional
      Starting value of Katz iteration for each node.

    normalized : bool, optional (default=True)
      If True normalize the resulting values.

    weight : None or string, optional (default=None)
      If None, all edge weights are considered equal.
      Otherwise holds the name of the edge attribute used as weight.
      In this measure the weight is interpreted as the connection strength.

    Returns
    -------
    nodes : dictionary
       Dictionary of nodes with Katz centrality as the value.

    Raises
    ------
    NetworkXError
       If the parameter `beta` is not a scalar but lacks a value for at least
       one node

    PowerIterationFailedConvergence
        If the algorithm fails to converge to the specified tolerance
        within the specified number of iterations of the power iteration
        method.

    Examples
    --------
    >>> import math
    >>> G = nx.path_graph(4)
    >>> phi = (1 + math.sqrt(5)) / 2.0  # largest eigenvalue of adj matrix
    >>> centrality = nx.katz_centrality(G, 1 / phi - 0.01)
    >>> for n, c in sorted(centrality.items()):
    ...     print(f"{n} {c:.2f}")
    0 0.37
    1 0.60
    2 0.60
    3 0.37

    See Also
    --------
    katz_centrality_numpy
    eigenvector_centrality
    eigenvector_centrality_numpy
    :func:`~networkx.algorithms.link_analysis.pagerank_alg.pagerank`
    :func:`~networkx.algorithms.link_analysis.hits_alg.hits`

    Notes
    -----
    Katz centrality was introduced by [2]_.

    This algorithm it uses the power method to find the eigenvector
    corresponding to the largest eigenvalue of the adjacency matrix of ``G``.
    The parameter ``alpha`` should be strictly less than the inverse of largest
    eigenvalue of the adjacency matrix for the algorithm to converge.
    You can use ``max(nx.adjacency_spectrum(G))`` to get $\\lambda_{\\max}$ the largest
    eigenvalue of the adjacency matrix.
    The iteration will stop after ``max_iter`` iterations or an error tolerance of
    ``number_of_nodes(G) * tol`` has been reached.

    For strongly connected graphs, as $\\alpha \\to 1/\\lambda_{\\max}$, and $\\beta > 0$,
    Katz centrality approaches the results for eigenvector centrality.

    For directed graphs this finds "left" eigenvectors which corresponds
    to the in-edges in the graph. For out-edges Katz centrality,
    first reverse the graph with ``G.reverse()``.

    References
    ----------
    .. [1] Mark E. J. Newman:
       Networks: An Introduction.
       Oxford University Press, USA, 2010, p. 720.
    .. [2] Leo Katz:
       A New Status Index Derived from Sociometric Index.
       Psychometrika 18(1):39–43, 1953
       https://link.springer.com/content/pdf/10.1007/BF02289026.pdf
    """
    from networkx.utils import not_implemented_for

    @not_implemented_for('multigraph')
    def _katz_centrality(G, alpha, beta, max_iter, tol, nstart, weight):
        if len(G) == 0:
            return {}

        if isinstance(beta, dict):
            if set(beta) != set(G):
                raise nx.NetworkXError('beta dictionary must have a value for every node')
            b = beta
        else:
            b = dict((n, beta) for n in G)

        if nstart is None:
            x = dict((n, 0) for n in G)
        else:
            x = nstart

        try:
            b_length = len(b)
        except TypeError:
            b_length = 1

        for _ in range(max_iter):
            xlast = x
            x = dict((n, 0) for n in G)
            for n in x:
                for nbr in G[n]:
                    w = G[n][nbr].get(weight, 1) if weight else 1
                    x[n] += xlast[nbr] * w
                x[n] = alpha * x[n] + b[n]

            # check convergence
            err = sum(abs(x[n] - xlast[n]) for n in x)
            if err < b_length * tol:
                if normalized:
                    # normalize to sum = 1
                    s = sum(x.values())
                    x = dict((k, v / s) for k, v in x.items())
                return x
        raise nx.PowerIterationFailedConvergence(max_iter)

    return _katz_centrality(G, alpha, beta, max_iter, tol, nstart, weight)

@not_implemented_for('multigraph')
@nx._dispatchable(edge_attrs='weight')
def katz_centrality_numpy(G, alpha=0.1, beta=1.0, normalized=True, weight=None):
    """Compute the Katz centrality for the graph G.

    Katz centrality computes the centrality for a node based on the centrality
    of its neighbors. It is a generalization of the eigenvector centrality. The
    Katz centrality for node $i$ is

    .. math::

        x_i = \\alpha \\sum_{j} A_{ij} x_j + \\beta,

    where $A$ is the adjacency matrix of graph G with eigenvalues $\\lambda$.

    The parameter $\\beta$ controls the initial centrality and

    .. math::

        \\alpha < \\frac{1}{\\lambda_{\\max}}.

    Katz centrality computes the relative influence of a node within a
    network by measuring the number of the immediate neighbors (first
    degree nodes) and also all other nodes in the network that connect
    to the node under consideration through these immediate neighbors.

    Extra weight can be provided to immediate neighbors through the
    parameter $\\beta$.  Connections made with distant neighbors
    are, however, penalized by an attenuation factor $\\alpha$ which
    should be strictly less than the inverse largest eigenvalue of the
    adjacency matrix in order for the Katz centrality to be computed
    correctly. More information is provided in [1]_.

    Parameters
    ----------
    G : graph
      A NetworkX graph

    alpha : float
      Attenuation factor

    beta : scalar or dictionary, optional (default=1.0)
      Weight attributed to the immediate neighborhood. If not a scalar the
      dictionary must have an value for every node.

    normalized : bool
      If True normalize the resulting values.

    weight : None or string, optional
      If None, all edge weights are considered equal.
      Otherwise holds the name of the edge attribute used as weight.
      In this measure the weight is interpreted as the connection strength.

    Returns
    -------
    nodes : dictionary
       Dictionary of nodes with Katz centrality as the value.

    Raises
    ------
    NetworkXError
       If the parameter `beta` is not a scalar but lacks a value for at least
       one node

    Examples
    --------
    >>> import math
    >>> G = nx.path_graph(4)
    >>> phi = (1 + math.sqrt(5)) / 2.0  # largest eigenvalue of adj matrix
    >>> centrality = nx.katz_centrality_numpy(G, 1 / phi)
    >>> for n, c in sorted(centrality.items()):
    ...     print(f"{n} {c:.2f}")
    0 0.37
    1 0.60
    2 0.60
    3 0.37

    See Also
    --------
    katz_centrality
    eigenvector_centrality_numpy
    eigenvector_centrality
    :func:`~networkx.algorithms.link_analysis.pagerank_alg.pagerank`
    :func:`~networkx.algorithms.link_analysis.hits_alg.hits`

    Notes
    -----
    Katz centrality was introduced by [2]_.

    This algorithm uses a direct linear solver to solve the above equation.
    The parameter ``alpha`` should be strictly less than the inverse of largest
    eigenvalue of the adjacency matrix for there to be a solution.
    You can use ``max(nx.adjacency_spectrum(G))`` to get $\\lambda_{\\max}$ the largest
    eigenvalue of the adjacency matrix.

    For strongly connected graphs, as $\\alpha \\to 1/\\lambda_{\\max}$, and $\\beta > 0$,
    Katz centrality approaches the results for eigenvector centrality.

    For directed graphs this finds "left" eigenvectors which corresponds
    to the in-edges in the graph. For out-edges Katz centrality,
    first reverse the graph with ``G.reverse()``.

    References
    ----------
    .. [1] Mark E. J. Newman:
       Networks: An Introduction.
       Oxford University Press, USA, 2010, p. 173.
    .. [2] Leo Katz:
       A New Status Index Derived from Sociometric Index.
       Psychometrika 18(1):39–43, 1953
       https://link.springer.com/content/pdf/10.1007/BF02289026.pdf
    """
    import numpy as np
    from networkx.utils import not_implemented_for

    @not_implemented_for('multigraph')
    def _katz_centrality_numpy(G, alpha, beta, normalized, weight):
        if len(G) == 0:
            return {}

        try:
            nodelist = list(G)
            A = nx.to_numpy_array(G, nodelist=nodelist, weight=weight)
            n, m = A.shape
        except AttributeError as e:
            msg = "Katz centrality requires NumPy: http://scipy.org/"
            raise AttributeError(msg) from e

        if isinstance(beta, dict):
            b = np.array([beta.get(n, 0) for n in nodelist])
        else:
            b = np.ones(n) * beta

        I = np.eye(n, m)
        x = np.linalg.solve(I - alpha * A.T, b)

        if normalized:
            x = x / x.sum()

        return dict(zip(nodelist, map(float, x)))

    return _katz_centrality_numpy(G, alpha, beta, normalized, weight)
