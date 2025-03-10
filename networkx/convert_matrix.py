"""Functions to convert NetworkX graphs to and from common data containers
like numpy arrays, scipy sparse arrays, and pandas DataFrames.

The preferred way of converting data to a NetworkX graph is through the
graph constructor.  The constructor calls the `~networkx.convert.to_networkx_graph`
function which attempts to guess the input type and convert it automatically.

Examples
--------
Create a 10 node random graph from a numpy array

>>> import numpy as np
>>> rng = np.random.default_rng()
>>> a = rng.integers(low=0, high=2, size=(10, 10))
>>> DG = nx.from_numpy_array(a, create_using=nx.DiGraph)

or equivalently:

>>> DG = nx.DiGraph(a)

which calls `from_numpy_array` internally based on the type of ``a``.

See Also
--------
nx_agraph, nx_pydot
"""
import itertools
from collections import defaultdict
import networkx as nx
from networkx.utils import not_implemented_for
__all__ = ['from_pandas_adjacency', 'to_pandas_adjacency', 'from_pandas_edgelist', 'to_pandas_edgelist', 'from_scipy_sparse_array', 'to_scipy_sparse_array', 'from_numpy_array', 'to_numpy_array']

@nx._dispatchable(edge_attrs='weight')
def to_pandas_adjacency(G, nodelist=None, dtype=None, order=None, multigraph_weight=sum, weight='weight', nonedge=0.0):
    """Returns the graph adjacency matrix as a Pandas DataFrame.

    Parameters
    ----------
    G : graph
        The NetworkX graph used to construct the Pandas DataFrame.

    nodelist : list, optional
       The rows and columns are ordered according to the nodes in `nodelist`.
       If `nodelist` is None, then the ordering is produced by G.nodes().

    multigraph_weight : {sum, min, max}, optional
        An operator that determines how weights in multigraphs are handled.
        The default is to sum the weights of the multiple edges.

    weight : string or None, optional
        The edge attribute that holds the numerical value used for
        the edge weight.  If an edge does not have that attribute, then the
        value 1 is used instead.

    nonedge : float, optional
        The matrix values corresponding to nonedges are typically set to zero.
        However, this could be undesirable if there are matrix values
        corresponding to actual edges that also have the value zero. If so,
        one might prefer nonedges to have some other value, such as nan.

    Returns
    -------
    df : Pandas DataFrame
       Graph adjacency matrix

    Notes
    -----
    For directed graphs, entry i,j corresponds to an edge from i to j.

    The DataFrame entries are assigned to the weight edge attribute. When
    an edge does not have a weight attribute, the value of the entry is set to
    the number 1.  For multiple (parallel) edges, the values of the entries
    are determined by the 'multigraph_weight' parameter.  The default is to
    sum the weight attributes for each of the parallel edges.

    When `nodelist` does not contain every node in `G`, the matrix is built
    from the subgraph of `G` that is induced by the nodes in `nodelist`.

    The convention used for self-loop edges in graphs is to assign the
    diagonal matrix entry value to the weight attribute of the edge
    (or the number 1 if the edge has no weight attribute).  If the
    alternate convention of doubling the edge weight is desired the
    resulting Pandas DataFrame can be modified as follows::

        >>> import pandas as pd
        >>> G = nx.Graph([(1, 1), (2, 2)])
        >>> df = nx.to_pandas_adjacency(G)
        >>> df
             1    2
        1  1.0  0.0
        2  0.0  1.0
        >>> diag_idx = list(range(len(df)))
        >>> df.iloc[diag_idx, diag_idx] *= 2
        >>> df
             1    2
        1  2.0  0.0
        2  0.0  2.0

    Examples
    --------
    >>> G = nx.MultiDiGraph()
    >>> G.add_edge(0, 1, weight=2)
    0
    >>> G.add_edge(1, 0)
    0
    >>> G.add_edge(2, 2, weight=3)
    0
    >>> G.add_edge(2, 2)
    1
    >>> nx.to_pandas_adjacency(G, nodelist=[0, 1, 2], dtype=int)
       0  1  2
    0  0  2  0
    1  1  0  0
    2  0  0  4

    """
    import pandas as pd
    import numpy as np

    if nodelist is None:
        nodelist = list(G.nodes())

    nodeset = set(nodelist)
    if len(nodelist) != len(nodeset):
        raise nx.NetworkXError("Duplicate node names detected.")

    A = nx.to_numpy_array(G, nodelist=nodelist, dtype=dtype, order=order,
                          multigraph_weight=multigraph_weight,
                          weight=weight, nonedge=nonedge)

    df = pd.DataFrame(A, index=nodelist, columns=nodelist)
    return df

@nx._dispatchable(graphs=None, returns_graph=True)
def from_pandas_adjacency(df, create_using=None):
    """Returns a graph from Pandas DataFrame.

    The Pandas DataFrame is interpreted as an adjacency matrix for the graph.

    Parameters
    ----------
    df : Pandas DataFrame
      An adjacency matrix representation of a graph

    create_using : NetworkX graph constructor, optional (default=nx.Graph)
       Graph type to create. If graph instance, then cleared before populated.

    Notes
    -----
    For directed graphs, explicitly mention create_using=nx.DiGraph,
    and entry i,j of df corresponds to an edge from i to j.

    If `df` has a single data type for each entry it will be converted to an
    appropriate Python data type.

    If you have node attributes stored in a separate dataframe `df_nodes`,
    you can load those attributes to the graph `G` using the following code:

    ```
    df_nodes = pd.DataFrame({"node_id": [1, 2, 3], "attribute1": ["A", "B", "C"]})
    G.add_nodes_from((n, dict(d)) for n, d in df_nodes.iterrows())
    ```

    If `df` has a user-specified compound data type the names
    of the data fields will be used as attribute keys in the resulting
    NetworkX graph.

    See Also
    --------
    to_pandas_adjacency

    Examples
    --------
    Simple integer weights on edges:

    >>> import pandas as pd
    >>> pd.options.display.max_columns = 20
    >>> df = pd.DataFrame([[1, 1], [2, 1]])
    >>> df
       0  1
    0  1  1
    1  2  1
    >>> G = nx.from_pandas_adjacency(df)
    >>> G.name = "Graph from pandas adjacency matrix"
    >>> print(G)
    Graph named 'Graph from pandas adjacency matrix' with 2 nodes and 3 edges
    """
    import pandas as pd
    import numpy as np

    if not isinstance(df, pd.DataFrame):
        raise nx.NetworkXError("Input is not a pandas DataFrame")

    if create_using is None:
        create_using = nx.Graph

    G = nx.empty_graph(0, create_using)

    if G.is_directed() and not G.is_multigraph():
        G.add_weighted_edges_from(
            (u, v, df.loc[u, v])
            for u, v in zip(*df.values.nonzero())
            if u != v or G.allows_self_loops()
        )
    else:
        G.add_weighted_edges_from(
            (df.index[u], df.columns[v], df.iloc[u, v])
            for u, v in zip(*np.triu(df.values.nonzero(), k=0))
            if u != v or G.allows_self_loops()
        )

    return G

@nx._dispatchable(preserve_edge_attrs=True)
def to_pandas_edgelist(G, source='source', target='target', nodelist=None, dtype=None, edge_key=None):
    """Returns the graph edge list as a Pandas DataFrame.

    Parameters
    ----------
    G : graph
        The NetworkX graph used to construct the Pandas DataFrame.

    source : str or int, optional
        A valid column name (string or integer) for the source nodes (for the
        directed case).

    target : str or int, optional
        A valid column name (string or integer) for the target nodes (for the
        directed case).

    nodelist : list, optional
       Use only nodes specified in nodelist

    dtype : dtype, default None
        Use to create the DataFrame. Data type to force.
        Only a single dtype is allowed. If None, infer.

    edge_key : str or int or None, optional (default=None)
        A valid column name (string or integer) for the edge keys (for the
        multigraph case). If None, edge keys are not stored in the DataFrame.

    Returns
    -------
    df : Pandas DataFrame
       Graph edge list

    Examples
    --------
    >>> G = nx.Graph(
    ...     [
    ...         ("A", "B", {"cost": 1, "weight": 7}),
    ...         ("C", "E", {"cost": 9, "weight": 10}),
    ...     ]
    ... )
    >>> df = nx.to_pandas_edgelist(G, nodelist=["A", "C"])
    >>> df[["source", "target", "cost", "weight"]]
      source target  cost  weight
    0      A      B     1       7
    1      C      E     9      10

    >>> G = nx.MultiGraph([("A", "B", {"cost": 1}), ("A", "B", {"cost": 9})])
    >>> df = nx.to_pandas_edgelist(G, nodelist=["A", "C"], edge_key="ekey")
    >>> df[["source", "target", "cost", "ekey"]]
      source target  cost  ekey
    0      A      B     1     0
    1      A      B     9     1

    """
    import pandas as pd

    if nodelist is None:
        edgelist = G.edges(data=True)
    else:
        edgelist = G.edges(nodelist, data=True)

    source_nodes = []
    target_nodes = []
    edge_data = defaultdict(list)

    if G.is_multigraph():
        for u, v, key, data in edgelist:
            source_nodes.append(u)
            target_nodes.append(v)
            if edge_key is not None:
                edge_data[edge_key].append(key)
            for k, v in data.items():
                edge_data[k].append(v)
    else:
        for u, v, data in edgelist:
            source_nodes.append(u)
            target_nodes.append(v)
            for k, v in data.items():
                edge_data[k].append(v)

    edge_data[source] = source_nodes
    edge_data[target] = target_nodes

    df = pd.DataFrame(edge_data, dtype=dtype)
    return df

@nx._dispatchable(graphs=None, returns_graph=True)
def from_pandas_edgelist(df, source='source', target='target', edge_attr=None, create_using=None, edge_key=None):
    """Returns a graph from Pandas DataFrame containing an edge list.

    The Pandas DataFrame should contain at least two columns of node names and
    zero or more columns of edge attributes. Each row will be processed as one
    edge instance.

    Note: This function iterates over DataFrame.values, which is not
    guaranteed to retain the data type across columns in the row. This is only
    a problem if your row is entirely numeric and a mix of ints and floats. In
    that case, all values will be returned as floats. See the
    DataFrame.iterrows documentation for an example.

    Parameters
    ----------
    df : Pandas DataFrame
        An edge list representation of a graph

    source : str or int
        A valid column name (string or integer) for the source nodes (for the
        directed case).

    target : str or int
        A valid column name (string or integer) for the target nodes (for the
        directed case).

    edge_attr : str or int, iterable, True, or None
        A valid column name (str or int) or iterable of column names that are
        used to retrieve items and add them to the graph as edge attributes.
        If `True`, all of the remaining columns will be added.
        If `None`, no edge attributes are added to the graph.

    create_using : NetworkX graph constructor, optional (default=nx.Graph)
        Graph type to create. If graph instance, then cleared before populated.

    edge_key : str or None, optional (default=None)
        A valid column name for the edge keys (for a MultiGraph). The values in
        this column are used for the edge keys when adding edges if create_using
        is a multigraph.

    If you have node attributes stored in a separate dataframe `df_nodes`,
    you can load those attributes to the graph `G` using the following code:

    ```
    df_nodes = pd.DataFrame({"node_id": [1, 2, 3], "attribute1": ["A", "B", "C"]})
    G.add_nodes_from((n, dict(d)) for n, d in df_nodes.iterrows())
    ```

    See Also
    --------
    to_pandas_edgelist

    Examples
    --------
    Simple integer weights on edges:

    >>> import pandas as pd
    >>> pd.options.display.max_columns = 20
    >>> import numpy as np
    >>> rng = np.random.RandomState(seed=5)
    >>> ints = rng.randint(1, 11, size=(3, 2))
    >>> a = ["A", "B", "C"]
    >>> b = ["D", "A", "E"]
    >>> df = pd.DataFrame(ints, columns=["weight", "cost"])
    >>> df[0] = a
    >>> df["b"] = b
    >>> df[["weight", "cost", 0, "b"]]
       weight  cost  0  b
    0       4     7  A  D
    1       7     1  B  A
    2      10     9  C  E
    >>> G = nx.from_pandas_edgelist(df, 0, "b", ["weight", "cost"])
    >>> G["E"]["C"]["weight"]
    10
    >>> G["E"]["C"]["cost"]
    9
    >>> edges = pd.DataFrame(
    ...     {
    ...         "source": [0, 1, 2],
    ...         "target": [2, 2, 3],
    ...         "weight": [3, 4, 5],
    ...         "color": ["red", "blue", "blue"],
    ...     }
    ... )
    >>> G = nx.from_pandas_edgelist(edges, edge_attr=True)
    >>> G[0][2]["color"]
    'red'

    Build multigraph with custom keys:

    >>> edges = pd.DataFrame(
    ...     {
    ...         "source": [0, 1, 2, 0],
    ...         "target": [2, 2, 3, 2],
    ...         "my_edge_key": ["A", "B", "C", "D"],
    ...         "weight": [3, 4, 5, 6],
    ...         "color": ["red", "blue", "blue", "blue"],
    ...     }
    ... )
    >>> G = nx.from_pandas_edgelist(
    ...     edges,
    ...     edge_key="my_edge_key",
    ...     edge_attr=["weight", "color"],
    ...     create_using=nx.MultiGraph(),
    ... )
    >>> G[0][2]
    AtlasView({'A': {'weight': 3, 'color': 'red'}, 'D': {'weight': 6, 'color': 'blue'}})


    """
    import pandas as pd

    if not isinstance(df, pd.DataFrame):
        raise nx.NetworkXError("Input is not a pandas DataFrame")

    if create_using is None:
        create_using = nx.Graph

    g = nx.empty_graph(0, create_using)

    if edge_attr is None:
        g.add_edges_from(zip(df[source], df[target]))
        return g

    if edge_attr is True:
        cols = list(df.columns)
        cols.remove(source)
        cols.remove(target)
        if edge_key is not None:
            cols.remove(edge_key)
        edge_attr = cols

    if g.is_multigraph():
        if edge_key is not None:
            for row in df.itertuples(index=False):
                s, t = getattr(row, source), getattr(row, target)
                key = getattr(row, edge_key)
                g.add_edge(s, t, key=key, **{k: getattr(row, k) for k in edge_attr})
        else:
            for row in df.itertuples(index=False):
                s, t = getattr(row, source), getattr(row, target)
                g.add_edge(s, t, **{k: getattr(row, k) for k in edge_attr})
    else:
        for row in df.itertuples(index=False):
            s, t = getattr(row, source), getattr(row, target)
            g.add_edge(s, t, **{k: getattr(row, k) for k in edge_attr})

    return g

@nx._dispatchable(edge_attrs='weight')
def to_scipy_sparse_array(G, nodelist=None, dtype=None, weight='weight', format='csr'):
    """Returns the graph adjacency matrix as a SciPy sparse array.

    Parameters
    ----------
    G : graph
        The NetworkX graph used to construct the sparse matrix.

    nodelist : list, optional
       The rows and columns are ordered according to the nodes in `nodelist`.
       If `nodelist` is None, then the ordering is produced by G.nodes().

    dtype : NumPy data-type, optional
        A valid NumPy dtype used to initialize the array. If None, then the
        NumPy default is used.

    weight : string or None   optional (default='weight')
        The edge attribute that holds the numerical value used for
        the edge weight.  If None then all edge weights are 1.

    format : str in {'bsr', 'csr', 'csc', 'coo', 'lil', 'dia', 'dok'}
        The type of the matrix to be returned (default 'csr').  For
        some algorithms different implementations of sparse matrices
        can perform better.  See [1]_ for details.

    Returns
    -------
    A : SciPy sparse array
       Graph adjacency matrix.

    Notes
    -----
    For directed graphs, matrix entry i,j corresponds to an edge from i to j.

    The matrix entries are populated using the edge attribute held in
    parameter weight. When an edge does not have that attribute, the
    value of the entry is 1.

    For multiple edges the matrix values are the sums of the edge weights.

    When `nodelist` does not contain every node in `G`, the adjacency matrix
    is built from the subgraph of `G` that is induced by the nodes in
    `nodelist`.

    The convention used for self-loop edges in graphs is to assign the
    diagonal matrix entry value to the weight attribute of the edge
    (or the number 1 if the edge has no weight attribute).  If the
    alternate convention of doubling the edge weight is desired the
    resulting SciPy sparse array can be modified as follows:

    >>> G = nx.Graph([(1, 1)])
    >>> A = nx.to_scipy_sparse_array(G)
    >>> print(A.todense())
    [[1]]
    >>> A.setdiag(A.diagonal() * 2)
    >>> print(A.toarray())
    [[2]]

    Examples
    --------
    >>> G = nx.MultiDiGraph()
    >>> G.add_edge(0, 1, weight=2)
    0
    >>> G.add_edge(1, 0)
    0
    >>> G.add_edge(2, 2, weight=3)
    0
    >>> G.add_edge(2, 2)
    1
    >>> S = nx.to_scipy_sparse_array(G, nodelist=[0, 1, 2])
    >>> print(S.toarray())
    [[0 2 0]
     [1 0 0]
     [0 0 4]]

    References
    ----------
    .. [1] Scipy Dev. References, "Sparse Matrices",
       https://docs.scipy.org/doc/scipy/reference/sparse.html
    """
    import scipy as sp
    import numpy as np

    if nodelist is None:
        nodelist = list(G)
    nlen = len(nodelist)
    index = {nodelist[i]: i for i in range(nlen)}

    if nlen == 0:
        raise nx.NetworkXError("Graph has no nodes or edges")

    if G.is_multigraph():
        entries = ((index[u], index[v], d.get(weight, 1))
                   for u, v, d in G.edges(nodelist, data=True)
                   if u in index and v in index)
    else:
        entries = ((index[u], index[v], d.get(weight, 1))
                   for u, v, d in G.edges(nodelist, data=True)
                   if u in index and v in index)

    if G.is_directed():
        matrix = sp.sparse.coo_array((nlen, nlen), dtype=dtype)
    else:
        matrix = sp.sparse.coo_array((nlen, nlen), dtype=dtype)

    row, col, data = zip(*entries) if entries else ([], [], [])
    matrix.row = np.array(row)
    matrix.col = np.array(col)
    matrix.data = np.array(data)
    matrix.sum_duplicates()

    if not G.is_directed():
        matrix = matrix + matrix.T

    return matrix.asformat(format)

def _csr_gen_triples(A):
    """Converts a SciPy sparse array in **Compressed Sparse Row** format to
    an iterable of weighted edge triples.

    """
    nrows = A.shape[0]
    data, indices, indptr = A.data, A.indices, A.indptr
    for i in range(nrows):
        for j in range(indptr[i], indptr[i + 1]):
            yield i, indices[j], data[j]

def _csc_gen_triples(A):
    """Converts a SciPy sparse array in **Compressed Sparse Column** format to
    an iterable of weighted edge triples.

    """
    ncols = A.shape[1]
    data, indices, indptr = A.data, A.indices, A.indptr
    for i in range(ncols):
        for j in range(indptr[i], indptr[i + 1]):
            yield indices[j], i, data[j]

def _coo_gen_triples(A):
    """Converts a SciPy sparse array in **Coordinate** format to an iterable
    of weighted edge triples.

    """
    return zip(A.row, A.col, A.data)

def _dok_gen_triples(A):
    """Converts a SciPy sparse array in **Dictionary of Keys** format to an
    iterable of weighted edge triples.

    """
    return ((r, c, v) for (r, c), v in A.items())

def _generate_weighted_edges(A):
    """Returns an iterable over (u, v, w) triples, where u and v are adjacent
    vertices and w is the weight of the edge joining u and v.

    `A` is a SciPy sparse array (in any format).

    """
    if A.format == 'csr':
        return _csr_gen_triples(A)
    elif A.format == 'csc':
        return _csc_gen_triples(A)
    elif A.format == 'coo':
        return _coo_gen_triples(A)
    elif A.format == 'dok':
        return _dok_gen_triples(A)
    else:
        raise nx.NetworkXError(f"Unknown sparse matrix format: {A.format}")

@nx._dispatchable(graphs=None, returns_graph=True)
def from_scipy_sparse_array(A, parallel_edges=False, create_using=None, edge_attribute='weight'):
    """Creates a new graph from an adjacency matrix given as a SciPy sparse
    array.

    Parameters
    ----------
    A: scipy.sparse array
      An adjacency matrix representation of a graph

    parallel_edges : Boolean
      If this is True, `create_using` is a multigraph, and `A` is an
      integer matrix, then entry *(i, j)* in the matrix is interpreted as the
      number of parallel edges joining vertices *i* and *j* in the graph.
      If it is False, then the entries in the matrix are interpreted as
      the weight of a single edge joining the vertices.

    create_using : NetworkX graph constructor, optional (default=nx.Graph)
       Graph type to create. If graph instance, then cleared before populated.

    edge_attribute: string
       Name of edge attribute to store matrix numeric value. The data will
       have the same type as the matrix entry (int, float, (real,imag)).

    Notes
    -----
    For directed graphs, explicitly mention create_using=nx.DiGraph,
    and entry i,j of A corresponds to an edge from i to j.

    If `create_using` is :class:`networkx.MultiGraph` or
    :class:`networkx.MultiDiGraph`, `parallel_edges` is True, and the
    entries of `A` are of type :class:`int`, then this function returns a
    multigraph (constructed from `create_using`) with parallel edges.
    In this case, `edge_attribute` will be ignored.

    If `create_using` indicates an undirected multigraph, then only the edges
    indicated by the upper triangle of the matrix `A` will be added to the
    graph.

    Examples
    --------
    >>> import scipy as sp
    >>> A = sp.sparse.eye(2, 2, 1)
    >>> G = nx.from_scipy_sparse_array(A)

    If `create_using` indicates a multigraph and the matrix has only integer
    entries and `parallel_edges` is False, then the entries will be treated
    as weights for edges joining the nodes (without creating parallel edges):

    >>> A = sp.sparse.csr_array([[1, 1], [1, 2]])
    >>> G = nx.from_scipy_sparse_array(A, create_using=nx.MultiGraph)
    >>> G[1][1]
    AtlasView({0: {'weight': 2}})

    If `create_using` indicates a multigraph and the matrix has only integer
    entries and `parallel_edges` is True, then the entries will be treated
    as the number of parallel edges joining those two vertices:

    >>> A = sp.sparse.csr_array([[1, 1], [1, 2]])
    >>> G = nx.from_scipy_sparse_array(A, parallel_edges=True, create_using=nx.MultiGraph)
    >>> G[1][1]
    AtlasView({0: {'weight': 1}, 1: {'weight': 1}})

    """
    import scipy as sp
    import numpy as np

    G = nx.empty_graph(0, create_using)
    n, m = A.shape
    if n != m:
        raise nx.NetworkXError(f"Adjacency matrix not square: nx,ny={A.shape}")
    
    if parallel_edges and G.is_multigraph():
        # In this case, we assume integer entries in the matrix are
        # counts of parallel edges in the graph.
        triples = ((u, v, {'weight': int(d)})
                   for (u, v, d) in _generate_weighted_edges(A)
                   if d != 0)
        if G.is_directed():
            G.add_edges_from(triples)
        else:
            # This is an undirected graph, so we only add the upper triangle.
            G.add_edges_from((u, v, d) for u, v, d in triples if u <= v)
    else:
        # In this case, we assume integer entries in the matrix are
        # weights of single edges in the graph.
        triples = ((u, v, {edge_attribute: d})
                   for (u, v, d) in _generate_weighted_edges(A)
                   if d != 0)
        if G.is_directed():
            G.add_edges_from(triples)
        else:
            # This is an undirected graph, so we only add the upper triangle.
            G.add_edges_from((u, v, d) for u, v, d in triples if u <= v)

    return G

@nx._dispatchable(edge_attrs='weight')
def to_numpy_array(G, nodelist=None, dtype=None, order=None, multigraph_weight=sum, weight='weight', nonedge=0.0):
    """Returns the graph adjacency matrix as a NumPy array.

    Parameters
    ----------
    G : graph
        The NetworkX graph used to construct the NumPy array.

    nodelist : list, optional
        The rows and columns are ordered according to the nodes in `nodelist`.
        If `nodelist` is ``None``, then the ordering is produced by ``G.nodes()``.

    dtype : NumPy data type, optional
        A NumPy data type used to initialize the array. If None, then the NumPy
        default is used. The dtype can be structured if `weight=None`, in which
        case the dtype field names are used to look up edge attributes. The
        result is a structured array where each named field in the dtype
        corresponds to the adjacency for that edge attribute. See examples for
        details.

    order : {'C', 'F'}, optional
        Whether to store multidimensional data in C- or Fortran-contiguous
        (row- or column-wise) order in memory. If None, then the NumPy default
        is used.

    multigraph_weight : callable, optional
        An function that determines how weights in multigraphs are handled.
        The function should accept a sequence of weights and return a single
        value. The default is to sum the weights of the multiple edges.

    weight : string or None optional (default = 'weight')
        The edge attribute that holds the numerical value used for
        the edge weight. If an edge does not have that attribute, then the
        value 1 is used instead. `weight` must be ``None`` if a structured
        dtype is used.

    nonedge : array_like (default = 0.0)
        The value used to represent non-edges in the adjacency matrix.
        The array values corresponding to nonedges are typically set to zero.
        However, this could be undesirable if there are array values
        corresponding to actual edges that also have the value zero. If so,
        one might prefer nonedges to have some other value, such as ``nan``.

    Returns
    -------
    A : NumPy ndarray
        Graph adjacency matrix

    Raises
    ------
    NetworkXError
        If `dtype` is a structured dtype and `G` is a multigraph
    ValueError
        If `dtype` is a structured dtype and `weight` is not `None`

    See Also
    --------
    from_numpy_array

    Notes
    -----
    For directed graphs, entry ``i, j`` corresponds to an edge from ``i`` to ``j``.

    Entries in the adjacency matrix are given by the `weight` edge attribute.
    When an edge does not have a weight attribute, the value of the entry is
    set to the number 1.  For multiple (parallel) edges, the values of the
    entries are determined by the `multigraph_weight` parameter. The default is
    to sum the weight attributes for each of the parallel edges.

    When `nodelist` does not contain every node in `G`, the adjacency matrix is
    built from the subgraph of `G` that is induced by the nodes in `nodelist`.

    The convention used for self-loop edges in graphs is to assign the
    diagonal array entry value to the weight attribute of the edge
    (or the number 1 if the edge has no weight attribute). If the
    alternate convention of doubling the edge weight is desired the
    resulting NumPy array can be modified as follows:

    >>> import numpy as np
    >>> G = nx.Graph([(1, 1)])
    >>> A = nx.to_numpy_array(G)
    >>> A
    array([[1.]])
    >>> A[np.diag_indices_from(A)] *= 2
    >>> A
    array([[2.]])

    Examples
    --------
    >>> G = nx.MultiDiGraph()
    >>> G.add_edge(0, 1, weight=2)
    0
    >>> G.add_edge(1, 0)
    0
    >>> G.add_edge(2, 2, weight=3)
    0
    >>> G.add_edge(2, 2)
    1
    >>> nx.to_numpy_array(G, nodelist=[0, 1, 2])
    array([[0., 2., 0.],
           [1., 0., 0.],
           [0., 0., 4.]])

    When `nodelist` argument is used, nodes of `G` which do not appear in the `nodelist`
    and their edges are not included in the adjacency matrix. Here is an example:

    >>> G = nx.Graph()
    >>> G.add_edge(3, 1)
    >>> G.add_edge(2, 0)
    >>> G.add_edge(2, 1)
    >>> G.add_edge(3, 0)
    >>> nx.to_numpy_array(G, nodelist=[1, 2, 3])
    array([[0., 1., 1.],
           [1., 0., 0.],
           [1., 0., 0.]])

    This function can also be used to create adjacency matrices for multiple
    edge attributes with structured dtypes:

    >>> G = nx.Graph()
    >>> G.add_edge(0, 1, weight=10)
    >>> G.add_edge(1, 2, cost=5)
    >>> G.add_edge(2, 3, weight=3, cost=-4.0)
    >>> dtype = np.dtype([("weight", int), ("cost", float)])
    >>> A = nx.to_numpy_array(G, dtype=dtype, weight=None)
    >>> A["weight"]
    array([[ 0, 10,  0,  0],
           [10,  0,  1,  0],
           [ 0,  1,  0,  3],
           [ 0,  0,  3,  0]])
    >>> A["cost"]
    array([[ 0.,  1.,  0.,  0.],
           [ 1.,  0.,  5.,  0.],
           [ 0.,  5.,  0., -4.],
           [ 0.,  0., -4.,  0.]])

    As stated above, the argument "nonedge" is useful especially when there are
    actually edges with weight 0 in the graph. Setting a nonedge value different than 0,
    makes it much clearer to differentiate such 0-weighted edges and actual nonedge values.

    >>> G = nx.Graph()
    >>> G.add_edge(3, 1, weight=2)
    >>> G.add_edge(2, 0, weight=0)
    >>> G.add_edge(2, 1, weight=0)
    >>> G.add_edge(3, 0, weight=1)
    >>> nx.to_numpy_array(G, nonedge=-1.0)
    array([[-1.,  2., -1.,  1.],
           [ 2., -1.,  0., -1.],
           [-1.,  0., -1.,  0.],
           [ 1., -1.,  0., -1.]])
    """
    import numpy as np

    if nodelist is None:
        nodelist = list(G)
    nodeset = set(nodelist)
    if len(nodelist) != len(nodeset):
        raise nx.NetworkXError("Duplicate node names detected.")

    nlen = len(nodelist)
    index = dict(zip(nodelist, range(nlen)))

    if G.is_multigraph():
        if weight is not None:
            edges = G.edges(nodelist, data=weight, keys=True)
            edge_weights = defaultdict(list)
            for u, v, k, d in edges:
                edge_weights[(u, v)].append(d)
        else:
            edges = G.edges(nodelist, data=True, keys=True)
            edge_weights = defaultdict(list)
            for u, v, k, d in edges:
                edge_weights[(u, v)].append(d)
    else:
        if weight is not None:
            edges = G.edges(nodelist, data=weight)
        else:
            edges = G.edges(nodelist, data=True)
        edge_weights = defaultdict(list)
        for u, v, d in edges:
            edge_weights[(u, v)].append(d)

    if dtype is None:
        dtype = float

    if weight is None and isinstance(dtype, np.dtype) and dtype.names:
        labels = dtype.names
        A = np.full((nlen, nlen), nonedge, dtype=dtype)
        for (u, v), attrs in edge_weights.items():
            i, j = index[u], index[v]
            for label in labels:
                values = [attr.get(label, nonedge) for attr in attrs]
                A[i, j][label] = multigraph_weight(values)
                if not G.is_directed():
                    A[j, i][label] = A[i, j][label]
    else:
        A = np.full((nlen, nlen), nonedge, dtype=dtype)
        for (u, v), weights in edge_weights.items():
            i, j = index[u], index[v]
            w = multigraph_weight(weights) if len(weights) > 1 else weights[0]
            A[i, j] = w
            if not G.is_directed():
                A[j, i] = w

    A = np.asarray(A, dtype=dtype, order=order)
    return A

@nx._dispatchable(graphs=None, returns_graph=True)
def from_numpy_array(A, parallel_edges=False, create_using=None, edge_attr='weight'):
    """Returns a graph from a 2D NumPy array.

    The 2D NumPy array is interpreted as an adjacency matrix for the graph.

    Parameters
    ----------
    A : a 2D numpy.ndarray
        An adjacency matrix representation of a graph

    parallel_edges : Boolean
        If this is True, `create_using` is a multigraph, and `A` is an
        integer array, then entry *(i, j)* in the array is interpreted as the
        number of parallel edges joining vertices *i* and *j* in the graph.
        If it is False, then the entries in the array are interpreted as
        the weight of a single edge joining the vertices.

    create_using : NetworkX graph constructor, optional (default=nx.Graph)
       Graph type to create. If graph instance, then cleared before populated.

    edge_attr : String, optional (default="weight")
        The attribute to which the array values are assigned on each edge. If
        it is None, edge attributes will not be assigned.

    Notes
    -----
    For directed graphs, explicitly mention create_using=nx.DiGraph,
    and entry i,j of A corresponds to an edge from i to j.

    If `create_using` is :class:`networkx.MultiGraph` or
    :class:`networkx.MultiDiGraph`, `parallel_edges` is True, and the
    entries of `A` are of type :class:`int`, then this function returns a
    multigraph (of the same type as `create_using`) with parallel edges.

    If `create_using` indicates an undirected multigraph, then only the edges
    indicated by the upper triangle of the array `A` will be added to the
    graph.

    If `edge_attr` is Falsy (False or None), edge attributes will not be
    assigned, and the array data will be treated like a binary mask of
    edge presence or absence. Otherwise, the attributes will be assigned
    as follows:

    If the NumPy array has a single data type for each array entry it
    will be converted to an appropriate Python data type.

    If the NumPy array has a user-specified compound data type the names
    of the data fields will be used as attribute keys in the resulting
    NetworkX graph.

    See Also
    --------
    to_numpy_array

    Examples
    --------
    Simple integer weights on edges:

    >>> import numpy as np
    >>> A = np.array([[1, 1], [2, 1]])
    >>> G = nx.from_numpy_array(A)
    >>> G.edges(data=True)
    EdgeDataView([(0, 0, {'weight': 1}), (0, 1, {'weight': 2}), (1, 1, {'weight': 1})])

    If `create_using` indicates a multigraph and the array has only integer
    entries and `parallel_edges` is False, then the entries will be treated
    as weights for edges joining the nodes (without creating parallel edges):

    >>> A = np.array([[1, 1], [1, 2]])
    >>> G = nx.from_numpy_array(A, create_using=nx.MultiGraph)
    >>> G[1][1]
    AtlasView({0: {'weight': 2}})

    If `create_using` indicates a multigraph and the array has only integer
    entries and `parallel_edges` is True, then the entries will be treated
    as the number of parallel edges joining those two vertices:

    >>> A = np.array([[1, 1], [1, 2]])
    >>> temp = nx.MultiGraph()
    >>> G = nx.from_numpy_array(A, parallel_edges=True, create_using=temp)
    >>> G[1][1]
    AtlasView({0: {'weight': 1}, 1: {'weight': 1}})

    User defined compound data type on edges:

    >>> dt = [("weight", float), ("cost", int)]
    >>> A = np.array([[(1.0, 2)]], dtype=dt)
    >>> G = nx.from_numpy_array(A)
    >>> G.edges()
    EdgeView([(0, 0)])
    >>> G[0][0]["cost"]
    2
    >>> G[0][0]["weight"]
    1.0

    """
    import numpy as np

    kind_to_python_type = {
        'f': float,
        'i': int,
        'u': int,
        'b': bool,
        'c': complex,
        'S': str,
        'U': str,
        'V': 'void'
    }

    G = nx.empty_graph(0, create_using)
    n, m = A.shape
    if n != m:
        raise nx.NetworkXError(f"Adjacency matrix not square: nx,ny={A.shape}")

    dt = A.dtype
    try:
        python_type = kind_to_python_type[dt.kind]
    except Exception:
        raise TypeError(f"Unknown numpy data type: {dt}")

    # Make sure we get even the isolated nodes of the graph.
    G.add_nodes_from(range(n))
    # Get a list of all the entries in the array with nonzero entries. These
    # coordinates become edges in the graph. (convert to int from np.int64)
    edges = ((int(e[0]), int(e[1])) for e in zip(*np.asarray(A).nonzero()))

    # handle multigraphs
    if parallel_edges and G.is_multigraph():
        edges = ((u, v, k) for (u, v) in edges for k in range(int(A[u, v])))

    # update the graph
    if edge_attr is None:
        G.add_edges_from(edges)
    else:
        if python_type is 'void':
            # We have a structured array. Convert it to a list of tuples.
            edge_data = {e[0]: tuple(e) for e in A[np.nonzero(A)]}
            G.add_edges_from((u, v, edge_data[(u, v)]) for (u, v) in edges)
        else:
            G.add_edges_from((u, v, {edge_attr: python_type(A[u, v])}) for (u, v) in edges)

    return G
