import networkx as nx
import igraph as ig
import numpy as np
from cvxpy import Variable, Parameter, Minimize, Problem
from cvxpy import multiply as cvx_mul
from cvxpy import sum as cvx_sum


def create_nx_graph(edges, edge_weights, directed=True):
    if directed:
        g = nx.DiGraph()
    else:
        g = nx.Graph()
    nodes = sorted(set([u for u, _ in edges]) | set([u for v, u in edges]))
    inodes = list(np.arange(0, len(nodes)))

    vmap = dict(zip(nodes, inodes))
    vmap_inv = dict(zip(inodes, nodes))

    iedges = [(vmap[u], vmap[v]) for u, v in edges]
    edge_weights_dict = dict(zip(iedges, edge_weights))

    g.add_nodes_from(inodes)
    g.add_edges_from(iedges)

    nx.set_node_attributes(g, vmap_inv, 'name')
    nx.set_edge_attributes(g, edge_weights_dict, 'weight')
    return g


def create_ig_graph(edges, edge_weights, directed=True):
    if directed:
        g = ig.Graph().as_directed()
    else:
        g = ig.Graph().as_undirected()
    nodes = sorted(set([u for u, _ in edges]) | set([u for v, u in edges]))
    inodes = list(np.arange(0, len(nodes)))

    vmap = dict(zip(list(nodes), inodes))

    iedges = [(vmap[u], vmap[v]) for u, v in edges]

    g.add_vertices(inodes)

    g.vs['name0'] = nodes
    g.add_edges(iedges)
    g.es['weight'] = edge_weights
    return g


class GraphWrapper:
    def __init__(self, g, mode='ig', directed=False,
                 edge_attr_count='cnt', edge_attr_dist='dist', vertex_attr_weight='vweight'):
        """
        
        :param g: nx or ig graph as in input
        :param mode: specify nx or ig
        :param directed: consider the graph directed or undirected
        :param edge_attr_count: attribute name to denote counts on edges
        :param edge_attr_dist: attribute name to denote distance on edges (derived from counts) 
        :param vertex_attr_weight: 
        """
        self.mode = mode
        self.directed = directed
        self.v_weight_attr = vertex_attr_weight
        self.v_weight_attr_in = vertex_attr_weight + '_IN'
        self.v_weight_attr_out = vertex_attr_weight + '_OUT'

        self.v_weights = [self.v_weight_attr, self.v_weight_attr_in, self.v_weight_attr_out]
        self.dirs = ['IN', 'OUT', 'ALL']

        if mode == 'ig':
            if self.directed:
                self.g = g.as_directed()
            else:
                self.g = g.as_undirected()
        elif mode == 'nx':
            if self.directed:
                self.g = g.to_directed()
            else:
                self.g = g.to_undirected()
        else:
            raise NotImplemented(f'mode {mode} not implemented')

        self.graphs = []

        if mode == 'ig':
            if edge_attr_count in g.es.attribute_names():
                self.count_col = edge_attr_count
                self.dist_col = edge_attr_dist
                # NB: undirected to directed cast messes up the edges
                if not directed and g.is_directed():
                    for e in self.g.es:
                        e_orig = g.get_eid(e.source, e.target, False)
                        e[self.count_col] = g.es[e_orig][self.count_col]
                        # NB: check that count_cnt is > 0
                for e in self.g.es:
                    if e[self.count_col] > 0:
                        e[self.dist_col] = 1. / e[self.count_col]
                    else:
                        raise ValueError('Count weight should be positive')

                # NB check well : OUT
                for v in self.g.vs:
                    v[self.v_weight_attr] = sum([self.g.es[x][self.count_col] for x in self.g.incident(v, 'ALL')])
                    if self.directed:
                        v[self.v_weight_attr_in] = sum([self.g.es[x][self.count_col] for x in self.g.incident(v, 'IN')])
                        v[self.v_weight_attr_out] = sum([self.g.es[x][self.count_col] for x in self.g.incident(v, 'OUT')])
            else:
                self.count_col = None
        elif mode == 'nx':
            if all([edge_attr_count in props for a, b, props in self.g.edges(data=True)]):
                self.count_col = edge_attr_count
                self.dist_col = edge_attr_dist
            for e in self.g.edges(data=True):
                if e[-1][self.count_col] > 0:
                    e[-1][self.dist_col] = 1. / e[-1][self.count_col]
                else:
                    raise ValueError('Count weight should be positive')
                if self.directed:
                    for ii, props in self.g.nodes(data=True):
                        props[self.v_weight_attr_in] = sum([x[-1][self.count_col] for x in self.g.in_edges(ii, data=True)])
                        props[self.v_weight_attr_out] = sum([x[-1][self.count_col] for x in self.g.out_edges(ii, data=True)])
                        props[self.v_weight_attr] = props[self.v_weight_attr_out] + props[self.v_weight_attr_in]
                else:
                    for ii, props in self.g.nodes(data=True):
                        props[self.v_weight_attr_in] = sum([x[-1][self.count_col] for x in self.g.edges(ii, data=True)])
            pass
        else:
            raise NotImplemented(f'mode {mode} not implemented')

    def degree(self, vertex, direction=None, count_col=None):
        if self.mode == 'ig':
            if count_col and count_col == self.count_col:
                return sum([self.g.es[e][self.count_col] for e in self.g.incident(vertex, direction)])
            else:
                return len(self.g.incident(vertex, direction))
        else:
            pass

    def measure(self, vertex, direction=None,
                alpha=0.1,
                vertex_weight=None, measure_dir=None,
                count_col=None,
                verbose=False):
        if self.mode == 'ig':
            if direction in self.dirs or direction is None:
                neighbors = sorted([x for x in self.g.neighbors(vertex, direction)])
            else:
                raise ValueError(f'Incorrect direction value. Can be from {self.dirs} or None')
            if vertex_weight:
                if vertex_weight in self.v_weights:
                    measure = [self.g.vs[v][vertex_weight] for v in neighbors]
                else:
                    raise ValueError(f'Incorrect vertex_weight value. Can be from {self.v_weights}')
            else:
                if measure_dir:
                    if measure_dir in self.dirs:
                        measure = [self.g.vs[v].degree(measure_dir) for v in neighbors]
                    else:
                        raise ValueError(f'Incorrect measure_dir value. Can be from {self.dirs}')
                else:
                    measure = [1 for v in neighbors]

        elif self.mode == 'nx':
            if self.directed:
                if direction == 'OUT':
                    neighbors = sorted(self.g.successors(vertex))
                elif direction == 'IN':
                    neighbors = sorted(self.g.predecessors(vertex))
                elif direction is None or direction == 'ALL':
                    neighbors = sorted(set(self.g.successors(vertex)) | set(self.g.predecessors(vertex)))
                else:
                    raise ValueError(f'Incorrect direction value. Can be from {self.dirs} or None')
            else:
                neighbors = sorted(self.g.neighbors(vertex))
            if vertex_weight:
                # measure define by vertex weights
                if vertex_weight in self.v_weights:
                    measure = [self.g.nodes[v][vertex_weight] for v in neighbors]
                else:
                    raise ValueError(f'Incorrect vertex_weight value. Can be from {self.v_weights}')
            else:
                # measure defined by neighbour degrees
                if measure_dir:
                    if self.directed:
                        if measure_dir == 'OUT':
                            measure = [self.g.out_degree[v] for v in neighbors]
                        elif measure_dir == 'IN':
                            measure = [self.g.in_degree[v] for v in neighbors]
                        elif measure_dir == 'ALL':
                            measure = [(self.g.out_degree[v] + self.g.in_degree[v]) for v in neighbors]
                        else:
                            raise ValueError(f'Incorrect direction value. Can be from {direction}')

                    else:
                        if measure_dir == 'ALL':
                            measure = [self.g.degree[v] for v in neighbors]
                        else:
                            raise ValueError(f'Incorrect measure_dir value. Can be ALL for directed graph')
                else:
                    measure = [1 for v in neighbors]
        else:
            raise NotImplemented(f'self.mode {self.mode} not yet implemented')

        measure = np.array(measure)
        measure_norm = sum(measure)
        measure = measure/measure_norm
        if 0 < alpha < 1:
            measure = (1. - alpha) * measure
            measure = np.append(measure, alpha)
            neighbors += [vertex]
        return neighbors, measure

    def distance_matrix(self, vset_a, vset_b, dist_dir='OUT', weighted=False):
        """
        warning for directed graph nx dist_dir can only be effectively OUT

        :param vset_a:
        :param vset_b:
        :param dist_dir:
        :param weighted:
        :return:
        """
        if weighted:
            dist_col = self.dist_col
        else:
            dist_col = None
        if self.mode == 'ig':
            try:
                d = self.g.shortest_paths(vset_a, vset_b, dist_col, mode=dist_dir)
            except:
                # vset_a_ = [gtmp.vs[u]['name'] for u in vset_a]
                print(f'seta {vset_a}, setb {vset_b}')
                raise ValueError(f'no go encountered')
        else:
            d = [[nx.dijkstra_path_length(self.g, xp, yp,  weight=dist_col) for yp in vset_b] for xp in vset_a]
        return np.array(d)

    def components(self, how='STRONG', verbose=False):
        """

        :return:
        """
        if self.mode == 'ig':
            comps = self.g.components(mode=how)
            comps = [x for x in comps if x]
            graphs = [self.g.subgraph(x, implementation="create_from_scratch") for x in comps]
            self.graphs = [g for g in graphs if len(g.es()) > 0]
        elif self.mode == 'nx':
            if how == 'STRONG':
                comps = nx.strongly_connected_components(self.g)
            else:
                comps = nx.connected_components(self.g)
            comps = [x for x in comps if x]
            graphs = [self.g.subgraph(x) for x in comps]
            self.graphs = graphs
        if verbose:
            print(f'found {len(self.graphs)} {how} components')

    def compute_edge_curv(self, vertex_a, vertex_b,
                          direction=None, alpha=0.0,
                          vertex_weight=None, measure_dir=None,
                          dist_dir='OUT', weighted_distance=False,
                          solver=None, solver_options={}, verbose=False):
        """
        :param vertex_a:
        :param vertex_b:
        :param direction:
        :param alpha:
        :param vertex_weight:
        :param measure_dir:
        :param solver:
        :param solver_options:
        :param verbose:
        :return:
        """

        if vertex_a == vertex_b:
            return 1.

        nei_a, mx = self.measure(vertex_a, direction, alpha, vertex_weight, measure_dir)
        nei_b, my = self.measure(vertex_b, direction, alpha, vertex_weight, measure_dir)

        if verbose:
            print(f'a: {vertex_a}, nei: {nei_a}, measure: {mx}')
            print(f'b: {vertex_b}, nei: {nei_b}, measure: {my}')

        if (vertex_a not in nei_b) and (vertex_b not in nei_a):
            print(f'a: {vertex_a}, nei: {nei_a}, measure: {mx}')
            print(f'b: {vertex_b}, nei: {nei_b}, measure: {my}')
            raise ValueError('x and y are not neighbours')

        if verbose:
            print(len(nei_a), len(nei_b), len(set(nei_a) & set(nei_b)))

        dist = self.distance_matrix(nei_a, nei_b, dist_dir, weighted=weighted_distance)
        dist0 = self.distance_matrix([vertex_a], [vertex_b], dist_dir, weighted=weighted_distance)

        if verbose:
            print(dist)

        plan = Variable((len(nei_a), len(nei_b)))
        mx_trans = mx.reshape(-1, 1)*dist
        mu_trans_param = Parameter(mx_trans.shape, value=mx_trans)
        obj = Minimize(cvx_sum(cvx_mul(plan, mu_trans_param)))
        plan_i = cvx_sum(plan, axis=1)
        my_constraint = mx * plan
        constraints = [my_constraint == my,
                       plan >= 0, plan <= 1,
                       plan_i == np.ones(len(nei_a))]
        problem = Problem(obj, constraints)
        wd = problem.solve(solver=solver, **solver_options)
        curv = 1. - wd/dist0[0, 0]
        return curv

    def compute_curv_components(self, direction=None, alpha=0.0, vertex_weight=None,
                                measure_dir=None, dist_dir='OUT', weighted_distance=False,
                                solver=None, solver_options={}, verbose=False):
        agg = []
        cnt = 0
        cnt_delta = 1000
        for j, gtmp in enumerate(self.graphs):
            print(f'component {j}: number of edges: {len(gtmp.es())}')
            gw_gtmp = GraphWrapper(gtmp, mode=self.mode, directed=self.directed,
                                   edge_attr_count=self.count_col, edge_attr_dist=self.dist_col)
            if self.mode == 'nx':
                for e in gtmp.edges(data=True):
                    u, v, prop = e
                    curv = gw_gtmp.compute_edge_curv(u, v,
                                                     direction, alpha,
                                                     vertex_weight, measure_dir,
                                                     dist_dir, weighted_distance,
                                                     solver, solver_options, verbose
                                                     )
                    agg += [(gtmp.nodes[u]['name0'], gtmp.nodes[v]['name0'], curv)]
                    cnt += 1
                    if cnt % cnt_delta == 0:
                        print(f'{cnt} edges processessed...')
            elif self.mode == 'ig':
                for e in gtmp.es():
                    u, v = e.source, e. target
                    curv = gw_gtmp.compute_edge_curv(u, v,
                                                     direction, alpha,
                                                     vertex_weight, measure_dir,
                                                     dist_dir, weighted_distance,
                                                     solver, solver_options, verbose)
                    agg += [(gtmp.vs[u]['name0'], gtmp.vs[v]['name0'], curv)]
                    cnt += 1
                    if cnt % cnt_delta == 0:
                        print(f'{cnt} edges processessed...')
        return agg
