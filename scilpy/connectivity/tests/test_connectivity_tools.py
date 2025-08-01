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


import pytest
from nibabel import Nifti1Image
from dipy.utils.optpkg import optional_package

bct, have_bct, _ = optional_package('bct')


@pytest.mark.skipif(not have_bct, reason="BCT not installed")
def test_evaluate_graph_measures():
    conn_matrix = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
    len_matrix = np.array([[0, 0.5, 0], [0.5, 0, 0.2], [0, 0.2, 0]],
                          dtype=float)

    # Test with avg_node_wise=True
    measures = evaluate_graph_measures(conn_matrix, len_matrix, True, False)
    assert isinstance(measures, dict)
    for key, value in measures.items():
        assert isinstance(value, float)

    # Test with avg_node_wise=False
    measures = evaluate_graph_measures(conn_matrix, len_matrix, False, False)
    assert isinstance(measures, dict)
    for key, value in measures.items():
        if key not in ['modularity', 'assortativity', 'global_efficiency',
                        'density', 'rich_club']:
            assert isinstance(value, list)

    # Test with small_world=True
    measures = evaluate_graph_measures(conn_matrix, len_matrix, True, True)
    assert 'omega' in measures
    assert 'sigma' in measures


def test_normalize_matrix_from_values():
    matrix = np.array([[1, 2], [3, 4]], dtype=float)
    norm_factor = np.array([[2, 2], [2, 2]])
    normalized_matrix = normalize_matrix_from_values(matrix.copy(),
                                                     norm_factor, False)
    assert_array_equal(normalized_matrix, [[2, 4], [6, 8]])

    matrix = np.array([[1, 2], [3, 4]], dtype=float)
    normalized_matrix = normalize_matrix_from_values(matrix.copy(),
                                                     norm_factor, True)
    assert_array_equal(normalized_matrix, [[0.5, 1.], [1.5, 2.]])


def test_normalize_matrix_from_parcel():
    matrix = np.ones((3, 3))
    labels_list = np.array([1, 2, 3])
    affine = np.eye(4)

    # Test with isotropic data
    data = np.zeros((10, 10, 10))
    data[2:4, 2:4, 2:4] = 1
    data[4:6, 4:6, 4:6] = 2
    data[6:8, 6:8, 6:8] = 3
    atlas_img = Nifti1Image(data, affine)

    # Test with parcel_from_volume=True
    norm_matrix = normalize_matrix_from_parcel(matrix.copy(), atlas_img,
                                               labels_list, True)
    assert norm_matrix.shape == matrix.shape

    # Test with parcel_from_volume=False
    norm_matrix = normalize_matrix_from_parcel(matrix.copy(), atlas_img,
                                               labels_list, False)
    assert norm_matrix.shape == matrix.shape

    # Test with non-isotropic data
    affine_aniso = np.diag([2, 1, 1, 1])
    atlas_img_aniso = Nifti1Image(data, affine_aniso)
    with pytest.raises(ValueError):
        normalize_matrix_from_parcel(matrix.copy(), atlas_img_aniso,
                                     labels_list, True)

    # Test with wrong labels_list size
    with pytest.raises(ValueError):
        normalize_matrix_from_parcel(matrix.copy(), atlas_img,
                                     np.array([1, 2]), True)
