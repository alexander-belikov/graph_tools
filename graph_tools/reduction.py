from networkx import DiGraph, Graph, from_pandas_adjacency
from numpy import array, dot
import logging


def project_to_nodes(g, nodes):
    h = Graph()
    edges = g.edges(nodes, data=True)
    h.add_edges_from(edges)
    return h


def reduce_digraphs(ga, gb):
    """
    ga : a -> r, DiGraph
    gb : b -> r, DiGraph
    Gout : a -> b, DiGraph
    """

    # identify non empty neighborhoods of ga, i.e. a
    a_left = [left for left in ga.nodes() if ga.neighbors(left)]
    # identify non empty neighborhoods of ga, i.e. a
    b_left = [left for left in gb.nodes() if gb.neighbors(left)]

    gout = DiGraph()
    gout.add_nodes_from(a_left)
    gout.add_nodes_from(b_left)

    for a in a_left:
        r_ga = ga.neighbors(a)
        for b in b_left:
            r_gb = gb.neighbors(b)
            r_intersection = list(set(r_ga) & set(r_gb))
            ar_vec = array([ga[a][r]['weight'] for r in r_intersection])
            br_vec = array([gb[b][r]['weight'] for r in r_intersection])
            gout.add_edge(a, b, {'weight': dot(ar_vec, br_vec)})
    return gout


def reduce_bigraphs(ga, gb, out_labels=None, ga_labels=None, gb_labels=None):
    """
    ga : a -> r, BiGraph
    gb : b -> r, BiGraph
    Gout : a -> b, BiGraph
    """

    # check whether ga and gb are bipartite
    try:
        ga_types = set([item[0] for item in ga.nodes()])
        gb_types = set([item[0] for item in gb.nodes()])
    except:
        logging.error(' in reduce_bigraphs() : ga or gb are not typified '
                      '(nodes should be tuples, e.g. (type, value))')
        raise

    if len(ga_types) != 2 or len(gb_types) != 2:
        logging.error(' in reduce_bigraphs() : one of the graphs '
                      'is not bipartite')
        raise ValueError(' in reduce_bigraphs() : one of the graphs is not bipartite')

    if not ga_labels and not gb_labels:
        try:
            r = list(ga_types & gb_types)[0]
            a = list(ga_types - gb_types)[0]
            b = list(gb_types - ga_types)[0]
        except:
            logging.error(' in reduce_bigraphs() : ga and gb '
                          'do not share a type')
            raise

    if out_labels:
        out_left, out_right = out_labels
    elif ga_labels and gb_labels:
        out_left, out_right = ga_labels[0], gb_labels[0]
    else:
        out_left, out_right = a, b

    if ga_labels:
        ga_left, ga_right = ga_labels
    else:
        ga_left, ga_right = (a, r)

    if gb_labels:
        gb_left, gb_right = gb_labels
    else:
        gb_left, gb_right = (b, r)

    # identify left neighborhoods of ga, i.e. a
    a_left = [left for left in ga.nodes() if left[0] == ga_left]
    # identify left neighborhoods of ga, i.e. a
    b_left = [left for left in gb.nodes() if left[0] == gb_left]
    gout = Graph()

    for a in a_left:
        r_ga = ga.neighbors(a)
        r_ga_ = list(map(lambda x: x[1], r_ga))
        for b in b_left:
            r_gb = gb.neighbors(b)
            r_gb_ = list(map(lambda x: x[1], r_gb))
            r_intersection = list(set(r_ga_) & set(r_gb_))
            if r_intersection:
                ar_vec = array([ga[a][(ga_right, r_)]['weight'] for r_ in r_intersection])
                br_vec = array([gb[b][(gb_right, r_)]['weight'] for r_ in r_intersection])
                a_ = (out_left, a[1])
                b_ = (out_right, b[1])
                gout.add_edge(a_, b_, {'weight': dot(ar_vec, br_vec)})
    return gout


def update_edges(ga, gb):
    for e in gb.edges(data=True):
        u, v, w = e
        if ga.has_edge(u, v):
            ga[u][v]['weight'] += w['weight']
        else:
            ga.add_edge(u, v, w)


def describe_graph(g):
    g_nodes = g.nodes()
    response = ''
    g_types = list(set(map(lambda x: x[0], g_nodes)))
    r_list = list(map(lambda y: len(list(filter(lambda x: x[0] == y, g_nodes))), g_types))
    pairs = zip(g_types, r_list)
    pairs_to_str = map(lambda x: 'type {0}: {1}'.format(x[0], x[1]), pairs)
    nodes_msg = 'edges : {0}'.format(len(g.edges()))
    response += ' description : {0}. {1}'.format('; '.join(pairs_to_str), nodes_msg)
    return response


def project_graph_return_adj(g, nodes, transpose=False):
    """

    :param g:
    :param nodes:
    :param transpose:
    :return:
    NB: nodes should be sorted in most cases
    """
    projection = project_to_nodes(g, nodes)
    adj_ = from_pandas_adjacency(projection)
    columns = sorted(list(filter(lambda x: x[0] != nodes[0][0], adj_.columns)), key=lambda x: x[1])
    adj = adj_.loc[nodes, columns]
    if transpose:
        adj = adj.T
    return adj