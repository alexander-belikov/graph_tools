from numpy import array, sum, ones, multiply, append, all
from networkx import dijkstra_path_length, all_pairs_dijkstra_path_length, Graph
from networkx import dijkstra_path, number_connected_components, connected_components, all_pairs_dijkstra_path
from cvxpy import Variable, Problem, Minimize, sum_entries, mul_elemwise
import pathos.multiprocessing as mp
from functools import partial

# Ollivier Ricci Graph curvature


def compute_measure(g, x, x_nei, alpha, mode='weight'):
    """
    mode : 'weight' or 'freq'
    """
    if mode == 'weight':
        x_weights = array([1./g[x][xp]['weight'] for xp in x_nei])
    elif mode == 'freq':
        x_weights = array([g[x][xp]['weight'] for xp in x_nei])
    else:
        x_weights = array([1./len(x_nei)]*len(x_nei))
    x_weight_norm = sum(x_weights)
    mx = (1. - alpha)*x_weights/x_weight_norm
    mx = append(mx, alpha)
    return mx


def compute_edge_curv(edge, g, alpha=0.0, dist_global=None, mode=None, verbose=False):
    """
    a weighted version inspired by https://github.com/saibalmars/GraphRicciCurvature

    :param edge:
    :param g:
    :param alpha:
    :param dist_global:
    :param mode:
    :param verbose:
    :return:
    """

    x, y = edge
    if x == y:
        raise ValueError('x == y')

    x_nei = list(g[x].keys())
    y_nei = list(g[y].keys())
    if x not in y_nei or y not in x_nei:
        raise ValueError('x and y are not neighbours')

    if verbose:
        print(len(x_nei), len(y_nei), len(set(x_nei) & set(y_nei)))

    if verbose:
        print(x_nei, y_nei)

    x_nei_ext = x_nei + [x]
    y_nei_ext = y_nei + [y]

    if dist_global:
        dist = array([[dist_global[xp][yp] for yp in y_nei_ext] for xp in x_nei_ext])
    else:
        dist = array([[dijkstra_path_length(g, xp, yp) for yp in y_nei_ext] for xp in x_nei_ext])

    mx = compute_measure(g, x, x_nei, alpha, mode)
    my = compute_measure(g, y, y_nei, alpha, mode)

    if verbose:
        print(mx.shape, my.shape, dist.shape)
        print(mx, my)

    plan = Variable(len(x_nei_ext), len(y_nei_ext))
    m_trans = multiply(mx[:, None], dist)
    obj = Minimize(sum_entries(mul_elemwise(m_trans, plan)))
    plan_i = sum_entries(plan, axis=1)
    my_constraints = (mx*plan).T
    if verbose:
        print(my_constraints.size)
    constraints = [my_constraints == my,
                   plan >= 0,
                   plan <= 1,
                   plan_i == ones(len(x_nei_ext))[:, None]]
    problem = Problem(obj, constraints)
    wd = problem.solve()
    curv = 1. - wd/dist_global[x][y]
    # perhaps assign curv as an attribute to an edge?
    return curv


def compute_graph_curv(g, edges=None, dict_dist=None, alpha=0.0, n_processes=1, mode=None):

    if not dict_dist:
        dict_dist = dict(all_pairs_dijkstra_path_length(g))
    func = partial(compute_edge_curv, g=g, dist_global=dict_dist, alpha=alpha, mode=mode)

    if not edges:
        edges = g.edges()

    if n_processes and n_processes > 1:
        with mp.Pool(n_processes) as p:
            curvs = array(p.map(func, edges))
    else:
        curvs = array([func(x) for x in edges])

    edges_curv_sorted = sorted(list(zip(edges, curvs)), key=lambda x: x[1])
    return curvs


def compute_boundary(g, ddist):
    """

    :param g: graph
    :param ddist: dijkstra dists : dict(nx.all_pairs_dijkstra_path_length(g))
    :return:
    """

    bnd_nodes = []
    for inode in g.node():
        node_candidates = list(set(g.nodes()) - {inode} - set(bnd_nodes))
        for jnode in node_candidates:
            cur_dist = ddist[inode][jnode]
            nei_dists = array([ddist[inode][n] for n in g[jnode].keys()])
            if all(cur_dist >= nei_dists):
                bnd_nodes.append(jnode)
    return bnd_nodes


def compute_nnns_paths(g, verbose=False):
    set_nn_paths = set()
    if verbose:
        n_nodes = len(g.nodes())
        n_ed_max = n_nodes*(n_nodes - 1) // 2
        print(n_nodes, len(g.edges()), n_ed_max)
    for e in g.nodes():
        ns = g.neighbors(e)
        set_nn_paths.update([(e, v) for v in ns])
        for n in g.neighbors(e):
            nns = g.neighbors(n)
            set_nn_paths.update([(e, v) for v in nns])
            for nn in g.neighbors(n):
                nnns = g.neighbors(nn)
                set_nn_paths.update([(e, v) for v in nnns])

    set_final = set()
    for p in iter(set_nn_paths):
        if p[0] < p[1]:
            set_final.update([p])
    return set_final


def subpath_in_path(path, subpath, directed=False):
    """
    return True if path contains subpath
    :param path: list of vertices denoting a path
    :param subpath: list of vertices denoting a subpath
    :param directed:
    :return:
    """
    l = len(subpath)
    antipath = subpath[::-1]
    if l > len(path):
        return False
    else:
        chunked_path = [tuple(path[i:i + l]) for i in range(len(path) - l + 1)]
        flag = any([subpath == c for c in chunked_path])
        if not directed:
            reverse_subpath_in_path = any([antipath == c for c in chunked_path])
            flag |= reverse_subpath_in_path
        return flag


def project_paths_on_subgraph(subgraph, paths, dists, edges_curv_sorted):
    nodes = subgraph.nodes()
    proj_paths = {k: {kk: vv for kk, vv in v.items() if kk in nodes} for k, v in paths.items() if k in nodes}
    proj_dists = {k: {kk: vv for kk, vv in v.items() if kk in nodes} for k, v in dists.items() if k in nodes}
    edges = subgraph.edges()
    proj_curvs = [(e, c) for e, c in edges_curv_sorted if e in iter(edges)]
    return proj_paths, proj_dists, proj_curvs


def update_paths_on_subgraph(subgraph, paths, dists, curvs, removed_edges, alpha=0.2, n_processes=4, verbose=True):
    nodes = subgraph.nodes()

    paths2update = {k: [kk for kk, vv in v.items() if (set(vv) - set(nodes)) or
                        any([subpath_in_path(vv, edge) for edge in removed_edges])] for k, v in paths.items()}
    # clean from empty lists
    paths2update2 = {k: v for k, v in paths2update.items() if v}

    # leave only in and out nodes
    edges_recompute = [[(k, kk) for kk in v] for k, v in paths2update2.items()]

    # flatten the list
    edges_recompute_flat = [x for sublist in edges_recompute for x in sublist]
    if verbose:
        print('recompute {0} paths in distance matrix'.format(len(edges_recompute_flat)))
    updated_paths = {k: {kk: dijkstra_path(subgraph, k, kk) for kk in v} for k, v in paths2update2.items()}
    updated_dists = {k: {kk: dijkstra_path_length(subgraph, k, kk) for kk in v} for k, v in paths2update2.items()}
    update_curv_edges_mask = [array([[(n1, n2) in edges_recompute_flat or (n2, n1) in edges_recompute_flat
                                         for n1 in subgraph.neighbors(w1)]
                                        for n2 in subgraph.neighbors(w2)]).any() for w1, w2 in subgraph.edges()]

    paths_new = {k: {**v, **updated_paths[k]} if k in updated_paths.keys() else v for k, v in paths.items()}
    dists_new = {k: {**v, **updated_dists[k]} if k in updated_dists.keys() else v for k, v in dists.items()}
    update_edges = [e for e, flag in zip(subgraph.edges(), update_curv_edges_mask) if flag]

    updated_curvs = compute_graph_curv(subgraph, update_edges, dists, alpha, n_processes)

    updated_edges_curvs = list(zip(update_edges, updated_curvs))

    edges_old = [(e, c) for e, c in curvs if (e in subgraph.edges()) and (e not in update_edges)]

    new_edges_curv_sorted = sorted(edges_old + updated_edges_curvs, key=lambda x: x[1])

    return paths_new, dists_new, new_edges_curv_sorted


def remove_edge(gn, paths, dists, edges_curv_sorted, alpha=0.0, n_processes=1, verbose=False):

    removed_edge, removed_curv = edges_curv_sorted[0]

    # paths that contain the edge
    paths2update = {k: {kk: vv for kk, vv in v.items() if subpath_in_path(vv, removed_edge)} for k, v in
                    paths.items()}

    # clean from empty lists
    paths2update2 = {k: v for k, v in paths2update.items() if v}

    # leave only in and out nodes
    edges_recompute = [[(k, kk) for kk, vv in v.items()] for k, v in paths2update2.items()]

    # leave only in and out nodes
    edges_recompute_dict = {k: list(v.keys()) for k, v in paths2update2.items()}

    # flatten the list
    edges_recompute_flat = [x for sublist in edges_recompute for x in sublist]

    gn.remove_edge(*removed_edge)

    if verbose:
        print('After deletion of edge {0}, the number of connected components is {1}'.format(removed_edge,
              number_connected_components(gn)))

    updated_paths = {k: {kk: dijkstra_path(gn, k, kk) for kk in v} for k, v in edges_recompute_dict.items()}
    updated_dists = {k: {kk: dijkstra_path_length(gn, k, kk) for kk in v} for k, v in edges_recompute_dict.items()}

    update_curv_edges_mask = [array([[(n1, n2) in edges_recompute_flat or (n2, n1) in edges_recompute_flat
                                      for n1 in gn.neighbors(w1)]
                                     for n2 in gn.neighbors(w2)]).any() for w1, w2 in gn.edges()]

    paths_new = {k: {**v, **updated_paths[k]} if k in updated_paths.keys() else v for k, v in paths.items()}
    dists_new = {k: {**v, **updated_dists[k]} if k in updated_dists.keys() else v for k, v in dists.items()}

    update_edges = [e for e, flag in zip(gn.edges(), update_curv_edges_mask) if flag]

    updated_curvs = compute_graph_curv(gn, update_edges[:], dists, alpha, n_processes)
    updated_edges_curv_dict = dict(zip(update_edges, updated_curvs))

    new_edges_curv_sorted = sorted([(e, curvv) if e not in updated_edges_curv_dict.keys()
                                    else (e, updated_edges_curv_dict[e])
                                    for e, curvv in edges_curv_sorted[1:]], key=lambda x: x[1])
    return gn, paths_new, dists_new, new_edges_curv_sorted


def trim_negative_edges_from_graph(g, curvs, alpha, n_proc=1, verbose=False):
    # takes a connected graph and deletes negative curvature edges
    # until either there are two connected components or there are no negative edges
    gn = g.copy()
    edges_curv_sorted = sorted(list(zip(gn.edges(), curvs)), key=lambda x: x[1])
    number_negative_edges = sum([x[1] < 0 for x in edges_curv_sorted])
    nc = 1
    if number_negative_edges > 0:
        number_edges_removed = 0
        while edges_curv_sorted \
                and nc == 1 \
                and number_edges_removed <= number_negative_edges:
            current_edge = edges_curv_sorted.pop(0)
            gn.remove_edge(*current_edge[0])
            number_edges_removed += 1
            nc = number_connected_components(gn)
        if nc > 1:
            return True, gn
        else:
            dists = dict(all_pairs_dijkstra_path_length(gn))
            curvs = compute_graph_curv(gn, None, dists, alpha, n_proc, None)
            return trim_negative_edges_from_graph(gn, curvs, alpha, n_proc, verbose)
    else:
        return False, gn


def reduce_graph(gn, paths, dists, edges_curv_sorted, alpha=0.0, n_processes=1,
                 mode='both', max_components=None, n_components=1, verbose=True):
    """

    :param gn:
    :param paths:
    :param dists:
    :param edges_curv_sorted:
    :param alpha:
    :param n_processes:
    :param mode: 'both', 'boundary' or 'curvature'
    :param max_components:
    :param n_components:
    :param verbose:
    :return:
    """

    number_negative_edges = sum([x[1] < 0 for x in edges_curv_sorted])

    bnd = compute_boundary(gn, dists)
    g_bnd = gn.subgraph(bnd)
    n_connected_bnd = number_connected_components(g_bnd)

    if mode == 'curvature':
        split_condition = (number_negative_edges >= 0)
    elif mode == 'boundary':
        split_condition = (n_connected_bnd > 1)
    else:
        # split_condition = (number_negative_edges >= 0) and (n_connected_bnd > 1)
        split_condition = (number_negative_edges >= 0) or (n_connected_bnd > 1)

    if (not max_components or (max_components and n_components < max_components)) and split_condition:
        number_edges_removed = 0
        removed_edges = []
        while edges_curv_sorted \
                and number_connected_components(gn) == 1 \
                and number_edges_removed <= number_negative_edges:
            current_edge = edges_curv_sorted.pop(0)
            gn.remove_edge(*current_edge[0])
            removed_edges += [current_edge[0]]
            number_edges_removed += 1

        if verbose:
            print('After deletion of {0} edges {1}, '
                  'the number of connected components is {2}'.format(number_edges_removed,
                                                                     removed_edges,
                                                                     number_connected_components(gn)))

        cc_g = list(connected_components(gn))
        if len(cc_g) > 1:
            ga, gb = gn.subgraph(cc_g[0]), gn.subgraph(cc_g[1])

            pp_a, dd_a, cc_a = project_paths_on_subgraph(ga, paths, dists, edges_curv_sorted)
            pp_a2, dd_a2, cc_a2 = update_paths_on_subgraph(ga, pp_a, dd_a, cc_a, removed_edges)
            ga = Graph(ga)

            pp_b, dd_b, cc_b = project_paths_on_subgraph(gb, paths, dists, edges_curv_sorted)
            pp_b2, dd_b2, cc_b2 = update_paths_on_subgraph(gb, pp_b, dd_b, cc_b, removed_edges)
            gb = Graph(gb)

            gn_list_a = reduce_graph(ga, pp_a2, dd_a2, cc_a2, alpha, n_processes, mode,
                                     max_components, n_components+1, verbose)
            gn_list_b = reduce_graph(gb, pp_b2, dd_b2, cc_b2, alpha, n_processes, mode,
                                     max_components, n_components+1, verbose)
            return [*gn_list_a, *gn_list_b]
        else:
            return [gn]
    else:
        return [gn]


def reduce_graph_simple(gn, dists=None, curvs=None, alpha=0.0, n_processes=1,
                        mode='curvature', max_components=None, n_components=1, verbose=True):
    """

    :param gn:
    :param dists:
    :param curvs:
    :param alpha:
    :param n_processes:
    :param mode: 'both', 'boundary' or 'curvature'
    :param max_components:
    :param n_components:
    :param verbose:
    :return:
    """

    gn = gn.copy()
    if not dists:
        dists = dict(all_pairs_dijkstra_path_length(gn))
    if not curvs:
        curvs = compute_graph_curv(gn, None, dists, alpha, n_processes, None)
    # edges_curv_sorted = sorted(list(zip(gn.edges(), curvs)), key=lambda x: x[1])
    # number_negative_edges = sum([x[1] < 0 for x in edges_curv_sorted])
    # if verbose:
    #     print('negative_edges: {0}'.format(number_negative_edges))
    #
    # split_condition = (number_negative_edges > 0)
    #
    # if (not max_components or (max_components and n_components < max_components)) and split_condition:
    flag, gn = trim_negative_edges_from_graph(gn, curvs, alpha, n_processes, curvs)
    if flag:
        cc_g = list(connected_components(gn))
        ga, gb = gn.subgraph(cc_g[0]), gn.subgraph(cc_g[1])
        gn_list_a = reduce_graph_simple(ga, None, None, alpha, n_processes, mode,
                                        max_components, n_components + 1, verbose)
        gn_list_b = reduce_graph_simple(gb, None, None, alpha, n_processes, mode,
                                        max_components, n_components + 1, verbose)
        return [*gn_list_a, *gn_list_b]
    else:
        return [gn]
