"""Functions for computing eigenvector centrality."""
import math
import numpy as np
import scipy as sp
import scipy.sparse
import networkx as nx
from networkx.utils import not_implemented_for
__all__ = ['eigenvector_centrality', 'eigenvector_centrality_numpy']

@not_implemented_for('multigraph')
@nx._dispatchable(edge_attrs='weight')
def eigenvector_centrality(G, max_iter=100, tol=1e-06, nstart=None, weight=None):
    """Compute the eigenvector centrality for the graph G.

    Eigenvector centrality computes the centrality for a node by adding
    the centrality of its predecessors. The centrality for node $i$ is the
    $i$-th element of a left eigenvector associated with the eigenvalue $\\lambda$
    of maximum modulus that is positive. Such an eigenvector $x$ is
    defined up to a multiplicative constant by the equation

    .. math::

         \\lambda x^T = x^T A,

    where $A$ is the adjacency matrix of the graph G. By definition of
    row-column product, the equation above is equivalent to

    .. math::

        \\lambda x_i = \\sum_{j\\to i}x_j.

    That is, adding the eigenvector centralities of the predecessors of
    $i$ one obtains the eigenvector centrality of $i$ multiplied by
    $\\lambda$. In the case of undirected graphs, $x$ also solves the familiar
    right-eigenvector equation $Ax = \\lambda x$.

    By virtue of the Perron–Frobenius theorem [1]_, if G is strongly
    connected there is a unique eigenvector $x$, and all its entries
    are strictly positive.

    If G is not strongly connected there might be several left
    eigenvectors associated with $\\lambda$, and some of their elements
    might be zero.

    Parameters
    ----------
    G : graph
      A networkx graph.

    max_iter : integer, optional (default=100)
      Maximum number of power iterations.

    tol : float, optional (default=1.0e-6)
      Error tolerance (in Euclidean norm) used to check convergence in
      power iteration.

    nstart : dictionary, optional (default=None)
      Starting value of power iteration for each node. Must have a nonzero
      projection on the desired eigenvector for the power method to converge.
      If None, this implementation uses an all-ones vector, which is a safe
      choice.

    weight : None or string, optional (default=None)
      If None, all edge weights are considered equal. Otherwise holds the
      name of the edge attribute used as weight. In this measure the
      weight is interpreted as the connection strength.

    Returns
    -------
    nodes : dictionary
       Dictionary of nodes with eigenvector centrality as the value. The
       associated vector has unit Euclidean norm and the values are
       nonegative.

    Examples
    --------
    >>> G = nx.path_graph(4)
    >>> centrality = nx.eigenvector_centrality(G)
    >>> sorted((v, f"{c:0.2f}") for v, c in centrality.items())
    [(0, '0.37'), (1, '0.60'), (2, '0.60'), (3, '0.37')]

    Raises
    ------
    NetworkXPointlessConcept
        If the graph G is the null graph.

    NetworkXError
        If each value in `nstart` is zero.

    PowerIterationFailedConvergence
        If the algorithm fails to converge to the specified tolerance
        within the specified number of iterations of the power iteration
        method.

    See Also
    --------
    eigenvector_centrality_numpy
    :func:`~networkx.algorithms.link_analysis.pagerank_alg.pagerank`
    :func:`~networkx.algorithms.link_analysis.hits_alg.hits`

    Notes
    -----
    Eigenvector centrality was introduced by Landau [2]_ for chess
    tournaments. It was later rediscovered by Wei [3]_ and then
    popularized by Kendall [4]_ in the context of sport ranking. Berge
    introduced a general definition for graphs based on social connections
    [5]_. Bonacich [6]_ reintroduced again eigenvector centrality and made
    it popular in link analysis.

    This function computes the left dominant eigenvector, which corresponds
    to adding the centrality of predecessors: this is the usual approach.
    To add the centrality of successors first reverse the graph with
    ``G.reverse()``.

    The implementation uses power iteration [7]_ to compute a dominant
    eigenvector starting from the provided vector `nstart`. Convergence is
    guaranteed as long as `nstart` has a nonzero projection on a dominant
    eigenvector, which certainly happens using the default value.

    The method stops when the change in the computed vector between two
    iterations is smaller than an error tolerance of ``G.number_of_nodes()
    * tol`` or after ``max_iter`` iterations, but in the second case it
    raises an exception.

    This implementation uses $(A + I)$ rather than the adjacency matrix
    $A$ because the change preserves eigenvectors, but it shifts the
    spectrum, thus guaranteeing convergence even for networks with
    negative eigenvalues of maximum modulus.

    References
    ----------
    .. [1] Abraham Berman and Robert J. Plemmons.
       "Nonnegative Matrices in the Mathematical Sciences."
       Classics in Applied Mathematics. SIAM, 1994.

    .. [2] Edmund Landau.
       "Zur relativen Wertbemessung der Turnierresultate."
       Deutsches Wochenschach, 11:366–369, 1895.

    .. [3] Teh-Hsing Wei.
       "The Algebraic Foundations of Ranking Theory."
       PhD thesis, University of Cambridge, 1952.

    .. [4] Maurice G. Kendall.
       "Further contributions to the theory of paired comparisons."
       Biometrics, 11(1):43–62, 1955.
       https://www.jstor.org/stable/3001479

    .. [5] Claude Berge
       "Théorie des graphes et ses applications."
       Dunod, Paris, France, 1958.

    .. [6] Phillip Bonacich.
       "Technique for analyzing overlapping memberships."
       Sociological Methodology, 4:176–185, 1972.
       https://www.jstor.org/stable/270732

    .. [7] Power iteration:: https://en.wikipedia.org/wiki/Power_iteration

    """
    if len(G) == 0:
        raise nx.NetworkXPointlessConcept('Cannot compute eigenvector centrality for the null graph.')

    # Initialize the centrality dictionary
    x = nstart if nstart is not None else dict.fromkeys(G, 1.0)

    # Normalize the initial vector
    s = 1.0 / sum(x.values())
    for k in x:
        x[k] *= s

    nnodes = G.number_of_nodes()

    # Power iteration: make up to max_iter iterations
    for _ in range(max_iter):
        xlast = x
        x = dict.fromkeys(xlast, 0)
        # Do the matrix multiplication y = Ax
        for n, nbrs in G.adjacency():
            for nbr, w in nbrs.items():
                x[n] += xlast[nbr] * (w if weight is None else w.get(weight, 1))
        # Normalize the vector
        try:
            s = 1.0 / math.sqrt(sum(v * v for v in x.values()))
        except ZeroDivisionError:
            s = 1.0
        for n in x:
            x[n] *= s
        # Check for convergence
        err = sum(abs(x[n] - xlast[n]) for n in x)
        if err < nnodes * tol:
            return x

    raise nx.PowerIterationFailedConvergence(max_iter)

@nx._dispatchable(edge_attrs='weight')
def eigenvector_centrality_numpy(G, weight=None, max_iter=50, tol=0):
    """Compute the eigenvector centrality for the graph G.

    Eigenvector centrality computes the centrality for a node by adding
    the centrality of its predecessors. The centrality for node $i$ is the
    $i$-th element of a left eigenvector associated with the eigenvalue $\\lambda$
    of maximum modulus that is positive. Such an eigenvector $x$ is
    defined up to a multiplicative constant by the equation

    .. math::

         \\lambda x^T = x^T A,

    where $A$ is the adjacency matrix of the graph G. By definition of
    row-column product, the equation above is equivalent to

    .. math::

        \\lambda x_i = \\sum_{j\\to i}x_j.

    That is, adding the eigenvector centralities of the predecessors of
    $i$ one obtains the eigenvector centrality of $i$ multiplied by
    $\\lambda$. In the case of undirected graphs, $x$ also solves the familiar
    right-eigenvector equation $Ax = \\lambda x$.

    By virtue of the Perron–Frobenius theorem [1]_, if G is strongly
    connected there is a unique eigenvector $x$, and all its entries
    are strictly positive.

    If G is not strongly connected there might be several left
    eigenvectors associated with $\\lambda$, and some of their elements
    might be zero.

    Parameters
    ----------
    G : graph
      A networkx graph.

    max_iter : integer, optional (default=50)
      Maximum number of Arnoldi update iterations allowed.

    tol : float, optional (default=0)
      Relative accuracy for eigenvalues (stopping criterion).
      The default value of 0 implies machine precision.

    weight : None or string, optional (default=None)
      If None, all edge weights are considered equal. Otherwise holds the
      name of the edge attribute used as weight. In this measure the
      weight is interpreted as the connection strength.

    Returns
    -------
    nodes : dictionary
       Dictionary of nodes with eigenvector centrality as the value. The
       associated vector has unit Euclidean norm and the values are
       nonegative.

    Examples
    --------
    >>> G = nx.path_graph(4)
    >>> centrality = nx.eigenvector_centrality_numpy(G)
    >>> print([f"{node} {centrality[node]:0.2f}" for node in centrality])
    ['0 0.37', '1 0.60', '2 0.60', '3 0.37']

    Raises
    ------
    NetworkXPointlessConcept
        If the graph G is the null graph.

    ArpackNoConvergence
        When the requested convergence is not obtained. The currently
        converged eigenvalues and eigenvectors can be found as
        eigenvalues and eigenvectors attributes of the exception object.

    See Also
    --------
    :func:`scipy.sparse.linalg.eigs`
    eigenvector_centrality
    :func:`~networkx.algorithms.link_analysis.pagerank_alg.pagerank`
    :func:`~networkx.algorithms.link_analysis.hits_alg.hits`

    Notes
    -----
    Eigenvector centrality was introduced by Landau [2]_ for chess
    tournaments. It was later rediscovered by Wei [3]_ and then
    popularized by Kendall [4]_ in the context of sport ranking. Berge
    introduced a general definition for graphs based on social connections
    [5]_. Bonacich [6]_ reintroduced again eigenvector centrality and made
    it popular in link analysis.

    This function computes the left dominant eigenvector, which corresponds
    to adding the centrality of predecessors: this is the usual approach.
    To add the centrality of successors first reverse the graph with
    ``G.reverse()``.

    This implementation uses the
    :func:`SciPy sparse eigenvalue solver<scipy.sparse.linalg.eigs>` (ARPACK)
    to find the largest eigenvalue/eigenvector pair using Arnoldi iterations
    [7]_.

    References
    ----------
    .. [1] Abraham Berman and Robert J. Plemmons.
       "Nonnegative Matrices in the Mathematical Sciences."
       Classics in Applied Mathematics. SIAM, 1994.

    .. [2] Edmund Landau.
       "Zur relativen Wertbemessung der Turnierresultate."
       Deutsches Wochenschach, 11:366–369, 1895.

    .. [3] Teh-Hsing Wei.
       "The Algebraic Foundations of Ranking Theory."
       PhD thesis, University of Cambridge, 1952.

    .. [4] Maurice G. Kendall.
       "Further contributions to the theory of paired comparisons."
       Biometrics, 11(1):43–62, 1955.
       https://www.jstor.org/stable/3001479

    .. [5] Claude Berge
       "Théorie des graphes et ses applications."
       Dunod, Paris, France, 1958.

    .. [6] Phillip Bonacich.
       "Technique for analyzing overlapping memberships."
       Sociological Methodology, 4:176–185, 1972.
       https://www.jstor.org/stable/270732

    .. [7] Arnoldi iteration:: https://en.wikipedia.org/wiki/Arnoldi_iteration

    """
    import numpy as np
    import scipy as sp
    import scipy.sparse  # call as sp.sparse

    if len(G) == 0:
        raise nx.NetworkXPointlessConcept('Cannot compute eigenvector centrality for the null graph.')

    A = nx.to_scipy_sparse_array(G, nodelist=list(G), weight=weight, dtype=float)
    if A.shape[0] == 1:
        return {G.nodes(): 1}
    
    # Add self-loops to dangling nodes
    A = A + sp.sparse.eye(A.shape[0])
    
    # Compute the eigenvector
    eigenvalues, eigenvectors = sp.sparse.linalg.eigs(A.T, k=1, which='LR', maxiter=max_iter, tol=tol)
    largest = eigenvectors.flatten().real
    
    # Normalize the eigenvector
    norm = np.sign(largest.sum()) * np.linalg.norm(largest)
    centrality = dict(zip(G, map(float, largest / norm)))
    
    return centrality
