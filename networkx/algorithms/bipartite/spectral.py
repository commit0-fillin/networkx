"""
Spectral bipartivity measure.
"""
import networkx as nx
__all__ = ['spectral_bipartivity']

@nx._dispatchable(edge_attrs='weight')
def spectral_bipartivity(G, nodes=None, weight='weight'):
    """Returns the spectral bipartivity.

    Parameters
    ----------
    G : NetworkX graph

    nodes : list or container  optional(default is all nodes)
      Nodes to return value of spectral bipartivity contribution.

    weight : string or None  optional (default = 'weight')
      Edge data key to use for edge weights. If None, weights set to 1.

    Returns
    -------
    sb : float or dict
       A single number if the keyword nodes is not specified, or
       a dictionary keyed by node with the spectral bipartivity contribution
       of that node as the value.

    Examples
    --------
    >>> from networkx.algorithms import bipartite
    >>> G = nx.path_graph(4)
    >>> bipartite.spectral_bipartivity(G)
    1.0

    Notes
    -----
    This implementation uses Numpy (dense) matrices which are not efficient
    for storing large sparse graphs.

    See Also
    --------
    color

    References
    ----------
    .. [1] E. Estrada and J. A. Rodríguez-Velázquez, "Spectral measures of
       bipartivity in complex networks", PhysRev E 72, 046105 (2005)
    """
    import numpy as np
    from scipy import linalg

    if G.number_of_nodes() == 0:
        return 0.0

    # Create adjacency matrix
    A = nx.to_numpy_array(G, weight=weight)
    
    # Compute the eigenvalues
    w = linalg.eigvals(A)
    
    # Calculate the spectral bipartivity
    expw = np.exp(w)
    expmw = np.exp(-w)
    sb = np.sum(expw + expmw) / (2 * np.sum(expw))

    if nodes is None:
        return sb
    else:
        # Compute eigenvectors
        w, v = linalg.eig(A)
        
        # Calculate node contributions
        expw = np.exp(w)
        expmw = np.exp(-w)
        
        # Normalize eigenvectors
        v = v / np.linalg.norm(v, axis=0)
        
        sb_dict = {}
        for node in nodes:
            i = list(G.nodes()).index(node)
            sb_dict[node] = np.sum((expw + expmw) * (v[i, :] ** 2)) / (2 * np.sum(expw))
        
        return sb_dict
