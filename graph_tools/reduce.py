from networkx import DiGraph, Graph
from numpy import array, dot


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
        raise(' in reduce_bigraphs() : ga or gb are not typified '
              '(nodes should be tuples, e.g. (type, value))')

    if len(ga_types) != 2 or len(gb_types) != 2:
        raise(' in reduce_bigraphs() : one of the graphs '
              'is not bipartite')

    if not ga_labels and gb_labels:
        try:
            r = list(ga_types & gb_types)[0]
            a = list(ga_types - gb_types)[0]
            b = list(gb_types - ga_types)[0]
        except:
            raise(' in reduce_bigraphs() : ga and gb '
                  'do not share a type')

    if out_labels:
        out_left, out_right = out_labels
    else:
        out_left, out_right = a, b

    if ga_labels and ga_labels == (a, r):
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
        for b in b_left:
            r_gb = gb.neighbors(b)
            r_intersection = list(set(r_ga) & set(r_gb))
            if r_intersection:
                ar_vec = array([ga[a][r]['weight'] for r in r_intersection])
                br_vec = array([gb[b][r]['weight'] for r in r_intersection])
                a_ = (out_left, a[1])
                b_ = (out_right, b[1])
                gout.add_edge(a_, b_, {'weight': dot(ar_vec, br_vec)})
    return gout


def update_edges(ga, gb):
    for e in gb.edges():
        if e in ga.edges():
            ga[e[0]][e[1]]['weight'] += gb[e[0]][e[1]]['weight']
        else:
            ga.add_edge(e[0], e[1], {'weight': gb[e[0]][e[1]]['weight']})
    return ga