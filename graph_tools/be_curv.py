import networkx as nx
from cvxpy import Variable, Problem, Minimize, norm
import numpy as np


def compute_node_be_curvature(g, n=None, distances=None, verbose=True):
    if n is not None:
        s1 = list(g.neighbors(n))

        # distance two neighbours
        s2 = []
        for n_ in s1:
            for y in g.neighbors(n_):
                if y not in s1 and y not in s2:
                    s2.append(y)

        s3 = []
        for n_ in s2:
            for y in g.neighbors(n_):
                if y not in s2 and y not in s3 and y not in s1:
                    s3.append(y)

        # compute gamma xx
        dx = g.degree[n]
        gamma_xx = np.array([3*dx - dx**2], ndmin=2)

        # compute gamma x S1
        outer_b2 = list(set(s1) | set(s2))
        if not distances:
            order_two_dists = {}
            for t in outer_b2:
                order_two_dists[t] = nx.dijkstra_path_length(g, n, t)
        else:
            order_two_dists = distances[n]

        dy_plus = [len([nn for nn in g.neighbors(n_) if order_two_dists[nn] > order_two_dists[n_]]) for n_ in s1]
        gamma_xs1 = np.array([-3 - dx - z for z in dy_plus], ndmin=2)

        # compute gamma S1 S1
        dy_zero = [len([nn for nn in g.neighbors(n_) if order_two_dists[nn] == order_two_dists[n_]]) for n_ in s1]
        diag = np.array([5 - dx + 3 * a + 4 * b for a, b in zip(dy_plus, dy_zero)])
        gamma_s1s1 = np.zeros((len(s1), len(s1)))
        for i in range(len(s1)):
            for j in range(i):
                u, v = s1[i], s1[j]
                gamma_s1s1[i, j] = 2 - 4 * int(u in g.neighbors(v))
        gamma_s1s1 += gamma_s1s1.T
        gamma_s1s1[range(len(s1)), range(len(s1))] = diag

        # compute gamma x S2
        xs2_and_s2_neighbours = list(set(s1) | set(s2) | set(s3))

        if not distances:
            dists_s2_neighbours = {}
            for t in xs2_and_s2_neighbours:
                dists_s2_neighbours[t] = nx.dijkstra_path_length(g, n, t)
        else:
            dists_s2_neighbours = distances[n]

        dz_minus = [len([nn for nn in g.neighbors(n_) if dists_s2_neighbours[nn] < dists_s2_neighbours[n_]]) for n_ in s2]
        gamma_xs2 = np.array(dz_minus, ndmin=2)

        gamma_s2s2 = np.zeros((len(s2), len(s2)))
        gamma_s2s2[range(len(s2)), range(len(s2))] = dz_minus

        gamma_s1s2 = np.zeros((len(s1), len(s2)))
        for i in range(len(s1)):
            for j in range(len(s2)):
                u, v = s1[i], s2[j]
                gamma_s1s2[i, j] = -2 * int(u in g.neighbors(v))

        dgamma2 = np.block([
            [gamma_xx, gamma_xs1, gamma_xs2],
            [gamma_xs1.T, gamma_s1s1, gamma_s1s2],
            [gamma_xs2.T, gamma_s1s2.T, gamma_s2s2],
        ])

        ext_dim = 1 + len(s1) + len(s2)
        one_dim = 1 + len(s1)

        gammax = np.zeros((ext_dim, ext_dim))

        gammax[range(1, one_dim), range(1, one_dim)] = 1.0
        gammax[0, 0] = g.degree(n)
        gammax[0, range(1, one_dim)] = -1.
        gammax[range(1, one_dim), 0] = -1.

        x = Variable()
        objective = Minimize(norm(dgamma2 + x * gammax))
        constraints = []
        prob = Problem(objective, constraints)

        curv_value = prob.solve()
        r = {n: curv_value}
    else:
        r = {n: compute_node_be_curvature(g, n, distances) for n in g.nodes()}
    return r


