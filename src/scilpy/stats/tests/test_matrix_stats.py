# -*- coding: utf-8 -*-
import numpy as np

from scilpy.stats.matrix_stats import omega_sigma


def test_ttest_two_matrices():
    # toDo
    pass


def test_omega_sigma():
    # Two disconnected components: bct.distance_wei() sets unreachable
    # node pairs to inf. omega_sigma() must average over finite (reachable)
    # distances only, not let inf silently propagate into NaN.
    rng = np.random.RandomState(0)

    def make_component(n, density=0.4):
        w = rng.rand(n, n)
        w = (w + w.T) / 2
        np.fill_diagonal(w, 0)
        w[w < (1 - density)] = 0
        return w

    n1, n2 = 10, 10
    matrix = np.zeros((n1 + n2, n1 + n2))
    matrix[:n1, :n1] = make_component(n1)
    matrix[n1:, n1:] = make_component(n2)

    omega, sigma = omega_sigma(matrix)
    assert np.isfinite(omega)
    assert np.isfinite(sigma)
