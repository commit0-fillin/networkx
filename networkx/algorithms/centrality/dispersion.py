from itertools import combinations
import networkx as nx
__all__ = ['dispersion']

@nx._dispatchable
def dispersion(G, u=None, v=None, normalized=True, alpha=1.0, b=0.0, c=0.0):
    """Calculate dispersion between `u` and `v` in `G`.

    A link between two actors (`u` and `v`) has a high dispersion when their
    mutual ties (`s` and `t`) are not well connected with each other.

    Parameters
    ----------
    G : graph
        A NetworkX graph.
    u : node, optional
        The source for the dispersion score (e.g. ego node of the network).
    v : node, optional
        The target of the dispersion score if specified.
    normalized : bool
        If True (default) normalize by the embeddedness of the nodes (u and v).
    alpha, b, c : float
        Parameters for the normalization procedure. When `normalized` is True,
        the dispersion value is normalized by::

            result = ((dispersion + b) ** alpha) / (embeddedness + c)

        as long as the denominator is nonzero.

    Returns
    -------
    nodes : dictionary
        If u (v) is specified, returns a dictionary of nodes with dispersion
        score for all "target" ("source") nodes. If neither u nor v is
        specified, returns a dictionary of dictionaries for all nodes 'u' in the
        graph with a dispersion score for each node 'v'.

    Notes
    -----
    This implementation follows Lars Backstrom and Jon Kleinberg [1]_. Typical
    usage would be to run dispersion on the ego network $G_u$ if $u$ were
    specified.  Running :func:`dispersion` with neither $u$ nor $v$ specified
    can take some time to complete.

    References
    ----------
    .. [1] Romantic Partnerships and the Dispersion of Social Ties:
        A Network Analysis of Relationship Status on Facebook.
        Lars Backstrom, Jon Kleinberg.
        https://arxiv.org/pdf/1310.6753v1.pdf

    """
    def calc_dispersion(G, u, v):
        """Calculate dispersion for a single pair of nodes."""
        common_neighbors = set(G[u]) & set(G[v])
        if len(common_neighbors) < 2:
            return 0
        
        dispersion = 0
        for s, t in combinations(common_neighbors, 2):
            if s not in G[t]:
                dispersion += 1
        
        return dispersion

    def calc_embeddedness(G, u, v):
        """Calculate embeddedness for a single pair of nodes."""
        return len(set(G[u]) & set(G[v]))

    if u is None and v is None:
        # Calculate dispersion for all pairs of nodes
        result = {n: {} for n in G}
        for n1, n2 in combinations(G, 2):
            disp = calc_dispersion(G, n1, n2)
            if normalized:
                emb = calc_embeddedness(G, n1, n2)
                if emb + c != 0:
                    disp = ((disp + b) ** alpha) / (emb + c)
                else:
                    disp = 0
            result[n1][n2] = result[n2][n1] = disp
        return result
    elif u is not None and v is None:
        # Calculate dispersion from u to all other nodes
        result = {}
        for n in G:
            if n != u:
                disp = calc_dispersion(G, u, n)
                if normalized:
                    emb = calc_embeddedness(G, u, n)
                    if emb + c != 0:
                        disp = ((disp + b) ** alpha) / (emb + c)
                    else:
                        disp = 0
                result[n] = disp
        return result
    elif u is None and v is not None:
        # Calculate dispersion from all nodes to v
        result = {}
        for n in G:
            if n != v:
                disp = calc_dispersion(G, n, v)
                if normalized:
                    emb = calc_embeddedness(G, n, v)
                    if emb + c != 0:
                        disp = ((disp + b) ** alpha) / (emb + c)
                    else:
                        disp = 0
                result[n] = disp
        return result
    else:
        # Calculate dispersion between u and v
        disp = calc_dispersion(G, u, v)
        if normalized:
            emb = calc_embeddedness(G, u, v)
            if emb + c != 0:
                disp = ((disp + b) ** alpha) / (emb + c)
            else:
                disp = 0
        return disp
