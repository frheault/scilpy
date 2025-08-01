import numpy as np
from numpy.testing import assert_array_equal, assert_almost_equal
from scilpy.connectivity.matrix_tools import (
    compute_olo,
    apply_olo,
    apply_reordering,
    evaluate_graph_measures,
    normalize_matrix_from_values,
    normalize_matrix_from_parcel,
)


def test_compute_olo():
    matrix = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    perm = compute_olo(matrix)
    assert len(perm) == 3


def test_apply_olo():
    matrix = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    perm = compute_olo(matrix)
    reordered_matrix = apply_olo(matrix, perm)
    assert reordered_matrix.shape == (3, 3)


def test_apply_reordering():
    matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    ordering = [[0, 2], [0, 2]]
    reordered_matrix = apply_reordering(matrix, ordering)
    assert_array_equal(reordered_matrix, [[1, 3], [7, 9]])


def test_evaluate_graph_measures():
    pass


def test_normalize_matrix_from_values():
    matrix = np.array([[1, 2], [3, 4]], dtype=float)
    norm_factor = np.array([[2, 2], [2, 2]])
    normalized_matrix = normalize_matrix_from_values(matrix, norm_factor, False)
    assert_array_equal(normalized_matrix, [[2, 4], [6, 8]])

    normalized_matrix = normalize_matrix_from_values(matrix, norm_factor, True)
    assert_array_equal(normalized_matrix, [[1., 2.], [3., 4.]])


def test_normalize_matrix_from_parcel():
    pass
