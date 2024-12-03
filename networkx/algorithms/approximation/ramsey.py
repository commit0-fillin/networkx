"""
Ramsey numbers.
"""
import networkx as nx
from networkx.utils import not_implemented_for
from ...utils import arbitrary_element
__all__ = ['ramsey_R2']

@not_implemented_for('directed')
@not_implemented_for('multigraph')
@nx._dispatchable
def ramsey_R2(G):
    """Compute the largest clique and largest independent set in `G`.

    This can be used to estimate bounds for the 2-color
    Ramsey number `R(2;s,t)` for `G`.

    This is a recursive implementation which could run into trouble
    for large recursions. Note that self-loop edges are ignored.

    Parameters
    ----------
    G : NetworkX graph
        Undirected graph

    Returns
    -------
    max_pair : (set, set) tuple
        Maximum clique, Maximum independent set.

    Raises
    ------
    NetworkXNotImplemented
        If the graph is directed or is a multigraph.
    """
    if len(G) == 0:
        return set(), set()
    
    v = arbitrary_element(G)
    G_v = G.subgraph(set(G) - {v} - set(G[v]))
    clique_v, indep_v = ramsey_R2(G_v)
    clique_with_v = {v} | set(G[v]) & clique_v
    indep_with_v = {v} | (indep_v - set(G[v]))
    
    return max(
        (clique_with_v, indep_v),
        (clique_v, indep_with_v),
        key=lambda pair: len(pair[0]) + len(pair[1])
    )
