#TODO fix imports in test
from unittest import TestCase, main
from ..graph_tools.ef import calc_eigen_vec

from numpy import array, sum, float


class TestTrim_data(TestCase):
    a_vec = array([3, 2, 5, 1, 2, 1], dtype=float)
    a_vec_norm = a_vec / sum(a_vec)
    z_raw = array([
        [1, 0, 2, 0, 4, 3],
        [3, 0, 1, 1, 0, 0],
        [2, 0, 4, 0, 1, 0],
        [0, 0, 1, 0, 0, 1],
        [8, 0, 3, 0, 5, 2],
        [0, 0, 0, 0, 0, 0]
    ], dtype=float)
    rr = calc_eigen_vec(z_raw, a_vec_norm)
    ef_ans = array([34.05091253, 17.20376627, 12.17544751, 3.65316603, 32.91670767, 0.])

    def test_trim_data(self):
        ans = calc_eigen_vec(self.z_raw, self.a_vec_norm)
        diff = (sum((ans[0] - self.ef_ans) ** 2)) ** 0.5
        self.assertAlmostEqual(diff, 0, 6)

if __name__ == '__main__':
    main()
