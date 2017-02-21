from networkx import DiGraph
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
