from numpy import fill_diagonal, zeros, ones, repeat
from numpy import where, dot, sum


def calc_eigen_vec(z, a, alpha=0.85, eps=1e-6):
    """

    :param z: np.array n x n
        adjacency matrix for directed weighted edge graph
    :param a: np.array
        normalization vector
    :param alpha: float

    :param eps: float

    :return:
    """
    # TODO: check that z is square, the shapes of a and z are compatible

    n = z.shape[0]
    fill_diagonal(z, 0)
    z_j = sum(z, axis=0)

    dangle_vec = zeros(shape=n)
    dangle_mask = (z_j == 0.)
    dangle_vec[dangle_mask] = 1.0

    # avoid dividing by zero
    z_j_prime = z_j.copy()
    z_j_prime[z_j_prime == 0.] = 1.0

    h = z/z_j_prime
    pi_0 = ones(shape=z_j.shape)/n
    # h_prime = h.copy()

    # mm = repeat(dangle_mask.reshape((1, n)), n, axis=0)
    # aa = repeat(a.reshape((n, 1)), n, axis=1)
    # h_prime = where(mm, aa, h)

    # p = alpha * h_prime + (1 - alpha) * aa

    pi = pi_0.copy()

    norm = 1.0
    k = 0
    while norm > eps:
        pi_next = alpha * dot(h, pi) + (alpha * dot(pi, dangle_vec) + (1 - alpha) * sum(pi)) * a
        norm = sum((pi_next - pi) ** 2) ** 0.5
        pi = pi_next
        #         print k, norm
        k += 1
        if norm > 1e1:
            print('norm is too high')
            break
    ef_aux = dot(h, pi)
    ef = 1e2 * ef_aux / sum(ef_aux)
    # avoid zero by zero div
    # in practice it's never the case
    a_prime = a.copy()
    zero_mask = (a == 0.)
    a_prime[zero_mask] = 1.0
    af = 0.01 * ef / a_prime
    #     return h_prime, dangle_mask, mm, h, aa, dangle_vec, pi, pi_0
    return ef, af
