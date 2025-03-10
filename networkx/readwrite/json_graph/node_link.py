from itertools import chain, count
import networkx as nx
__all__ = ['node_link_data', 'node_link_graph']
_attrs = {'source': 'source', 'target': 'target', 'name': 'id', 'key': 'key', 'link': 'links'}

def _to_tuple(x):
    """Converts lists to tuples, including nested lists.

    All other non-list inputs are passed through unmodified. This function is
    intended to be used to convert potentially nested lists from json files
    into valid nodes.

    Examples
    --------
    >>> _to_tuple([1, 2, [3, 4]])
    (1, 2, (3, 4))
    """
    pass

def node_link_data(G, *, source='source', target='target', name='id', key='key', link='links'):
    """Returns data in node-link format that is suitable for JSON serialization
    and use in JavaScript documents.

    Parameters
    ----------
    G : NetworkX graph
    source : string
        A string that provides the 'source' attribute name for storing NetworkX-internal graph data.
    target : string
        A string that provides the 'target' attribute name for storing NetworkX-internal graph data.
    name : string
        A string that provides the 'name' attribute name for storing NetworkX-internal graph data.
    key : string
        A string that provides the 'key' attribute name for storing NetworkX-internal graph data.
    link : string
        A string that provides the 'link' attribute name for storing NetworkX-internal graph data.

    Returns
    -------
    data : dict
       A dictionary with node-link formatted data.

    Raises
    ------
    NetworkXError
        If the values of 'source', 'target' and 'key' are not unique.

    Examples
    --------
    >>> G = nx.Graph([("A", "B")])
    >>> data1 = nx.node_link_data(G)
    >>> data1
    {'directed': False, 'multigraph': False, 'graph': {}, 'nodes': [{'id': 'A'}, {'id': 'B'}], 'links': [{'source': 'A', 'target': 'B'}]}

    To serialize with JSON

    >>> import json
    >>> s1 = json.dumps(data1)
    >>> s1
    '{"directed": false, "multigraph": false, "graph": {}, "nodes": [{"id": "A"}, {"id": "B"}], "links": [{"source": "A", "target": "B"}]}'

    A graph can also be serialized by passing `node_link_data` as an encoder function. The two methods are equivalent.

    >>> s1 = json.dumps(G, default=nx.node_link_data)
    >>> s1
    '{"directed": false, "multigraph": false, "graph": {}, "nodes": [{"id": "A"}, {"id": "B"}], "links": [{"source": "A", "target": "B"}]}'

    The attribute names for storing NetworkX-internal graph data can
    be specified as keyword options.

    >>> H = nx.gn_graph(2)
    >>> data2 = nx.node_link_data(H, link="edges", source="from", target="to")
    >>> data2
    {'directed': True, 'multigraph': False, 'graph': {}, 'nodes': [{'id': 0}, {'id': 1}], 'edges': [{'from': 1, 'to': 0}]}

    Notes
    -----
    Graph, node, and link attributes are stored in this format.  Note that
    attribute keys will be converted to strings in order to comply with JSON.

    Attribute 'key' is only used for multigraphs.

    To use `node_link_data` in conjunction with `node_link_graph`,
    the keyword names for the attributes must match.


    See Also
    --------
    node_link_graph, adjacency_data, tree_data
    """
    pass

@nx._dispatchable(graphs=None, returns_graph=True)
def node_link_graph(data, directed=False, multigraph=True, *, source='source', target='target', name='id', key='key', link='links'):
    """Returns graph from node-link data format.
    Useful for de-serialization from JSON.

    Parameters
    ----------
    data : dict
        node-link formatted graph data

    directed : bool
        If True, and direction not specified in data, return a directed graph.

    multigraph : bool
        If True, and multigraph not specified in data, return a multigraph.

    source : string
        A string that provides the 'source' attribute name for storing NetworkX-internal graph data.
    target : string
        A string that provides the 'target' attribute name for storing NetworkX-internal graph data.
    name : string
        A string that provides the 'name' attribute name for storing NetworkX-internal graph data.
    key : string
        A string that provides the 'key' attribute name for storing NetworkX-internal graph data.
    link : string
        A string that provides the 'link' attribute name for storing NetworkX-internal graph data.

    Returns
    -------
    G : NetworkX graph
        A NetworkX graph object

    Examples
    --------

    Create data in node-link format by converting a graph.

    >>> G = nx.Graph([("A", "B")])
    >>> data = nx.node_link_data(G)
    >>> data
    {'directed': False, 'multigraph': False, 'graph': {}, 'nodes': [{'id': 'A'}, {'id': 'B'}], 'links': [{'source': 'A', 'target': 'B'}]}

    Revert data in node-link format to a graph.

    >>> H = nx.node_link_graph(data)
    >>> print(H.edges)
    [('A', 'B')]

    To serialize and deserialize a graph with JSON,

    >>> import json
    >>> d = json.dumps(node_link_data(G))
    >>> H = node_link_graph(json.loads(d))
    >>> print(G.edges, H.edges)
    [('A', 'B')] [('A', 'B')]


    Notes
    -----
    Attribute 'key' is only used for multigraphs.

    To use `node_link_data` in conjunction with `node_link_graph`,
    the keyword names for the attributes must match.

    See Also
    --------
    node_link_data, adjacency_data, tree_data
    """
    pass