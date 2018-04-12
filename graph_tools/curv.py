from numpy import array, sum, ones, multiply, append
from networkx import dijkstra_path_length
from cvxpy import Variable, Problem, Minimize, sum_entries, mul_elemwise

# a weighted version based on https://github.com/saibalmars/GraphRicciCurvature
# Ollivier Ricci Graph curvature

def compute_measure(g, x, x_nei, alpha, mode='weight'):
    """
    mode : 'weight' or 'freq'
    """
    if mode == 'weight':
        x_weights = array([1./g[x][xp]['weight'] for xp in x_nei])
    else:
        x_weights = array([g[x][xp]['weight'] for xp in x_nei])
    x_weight_norm = sum(x_weights)
    mx = (1. - alpha)*x_weights/x_weight_norm
    mx = append(mx, alpha)
    return mx


def compute_edge_curv(g, x, y, alpha=0.0, dist_global=None, verbose=False):
    if x == y:
        raise ValueError('x == y')

    x_nei = g.neighbors(x)
    y_nei = g.neighbors(y)
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

    mx = compute_measure(g, x, x_nei, alpha, 'weight')
    my = compute_measure(g, y, y_nei, alpha, 'weight')

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
    curv = 1. - wd/dist[-1, -1]
    return curv