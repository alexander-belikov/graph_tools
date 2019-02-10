import networkx as nx
from cvxpy import Variable, Problem, Maximize, Parameter
import numpy as np

# networkx/cvxpy based implementation of Barky-Emery curvature calculation

def construct_dgamma2(g, n, verbose=False):
    if n is not None:
        s1 = list(g.neighbors(n))
        if verbose:
            print('S1: {0}'.format(s1))

        # distance two neighbours
        s2 = []
        for n_ in s1:
            for y in g.neighbors(n_):
                if y not in s1 and y not in s2 and y != n:
                    s2.append(y)

        if verbose:
            print('S2: {0}'.format(s2))

        s3 = []
        for n_ in s2:
            for y in g.neighbors(n_):
                if y not in s2 and y not in s3 and y not in s1:
                    s3.append(y)

        if verbose:
            print('S3: {0}'.format(s3))

        # compute gamma xx
        dx = g.degree[n]
        gamma_xx = np.array([3*dx + dx**2], ndmin=2)

        if verbose:
            print('degree of {0} : {1}'.format(n, dx))

        # compute gamma x S1
        dy_plus = [len([nn for nn in g.neighbors(n_) if nn in s2]) for n_ in s1]
        if verbose:
            print(' d^plus_y:')
            print(dy_plus)
        gamma_xs1 = np.array([-3 - dx - z for z in dy_plus], ndmin=2)

        # compute gamma S1 S1
        dy_zero = [len([nn for nn in g.neighbors(n_) if nn in s1]) for n_ in s1]
        if verbose:
            print(' d^zero_y:')
            print(dy_zero)
        diag = np.array([5 - dx + 3 * a + 4 * b for a, b in zip(dy_plus, dy_zero)])
        gamma_s1s1 = np.zeros((len(s1), len(s1)))
        for i in range(len(s1)):
            for j in range(i):
                u, v = s1[i], s1[j]
                gamma_s1s1[i, j] = 2 - 4 * int(u in g.neighbors(v))
        gamma_s1s1 += gamma_s1s1.T
        gamma_s1s1[range(len(s1)), range(len(s1))] = diag

        if verbose:
            print('4 gamma2 S1S1')
            print(gamma_s1s1)

        dz_minus = [len([nn for nn in g.neighbors(n_) if nn in s1]) for n_ in s2]
        if verbose:
            print(' d^minus_z:')
            print(dz_minus)

        gamma_xs2 = np.array(dz_minus, ndmin=2)

        gamma_s2s2 = np.zeros((len(s2), len(s2)))
        gamma_s2s2[range(len(s2)), range(len(s2))] = dz_minus

        gamma_s1s2 = np.zeros((len(s1), len(s2)))
        for i in range(len(s1)):
            for j in range(len(s2)):
                u, v = s1[i], s2[j]
                gamma_s1s2[i, j] = int(u in g.neighbors(v))

        gamma_s1s2 *= -2

        if verbose:
            print('4gamma_S1S2:')
            print(gamma_s1s2)

        dgamma2 = 0.25*np.block([
            [gamma_xx, gamma_xs1, gamma_xs2],
            [gamma_xs1.T, gamma_s1s1, gamma_s1s2],
            [gamma_xs2.T, gamma_s1s2.T, gamma_s2s2],
        ])
    else:
        return np.array((0, 0))
    return dgamma2


def construct_gammax(g, n, verbose=False):
    dx = g.degree[n]
    dim_b1 = 1 + dx
    gammax = np.zeros((dim_b1, dim_b1))

    gammax[range(1, dim_b1), range(1, dim_b1)] = 1.0
    gammax[0, 0] = g.degree[n]
    gammax[0, range(1, dim_b1)] = -1.
    gammax[range(1, dim_b1), 0] = -1.

    gammax *= 0.5
    return gammax


def compute_node_be_curvature(g, n=None, distances=None, verbose=True):
    if n is not None:
        dgamma2 = construct_dgamma2(g, n, verbose)
        gammax = construct_gammax(g, n)

        #extend dimension of gammax to b2
        dim_b1 = gammax.shape[0]
        dim_b2 = dgamma2.shape[0]
        dim_s2 = dgamma2.shape[0] - gammax.shape[0]
        gammax_ext = np.block([
            [gammax, np.zeros((dim_b1, dim_s2))],
            [np.zeros((dim_b1, dim_s2)).T, np.zeros((dim_s2, dim_s2))],
        ])

        a = Parameter((dim_b2, dim_b2), value=dgamma2)
        b = Parameter((dim_b2, dim_b2), value=gammax_ext)
        kappa = Variable()
        constraints = [(a - kappa * b >> 0)]

        objective = Maximize(kappa)
        prob = Problem(objective, constraints)
        prob.solve()
        return prob.value
    else:
        r = {n: compute_node_be_curvature(g, n, distances) for n in g.nodes()}
    return r


