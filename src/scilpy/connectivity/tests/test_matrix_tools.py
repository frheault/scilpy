# -*- coding: utf-8 -*-
import numpy as np

from scilpy.connectivity.matrix_tools import (apply_reordering,
                                              evaluate_graph_measures)


def test_compute_olo():
    # Simple and basically only using hierarchy's function. Not testing.
    pass


def test_apply_reordering():
    conn_matrix = np.asarray([[1, 2, 3, 4],
                              [5, 6, 7, 8],
                              [9, 10, 11, 12],
                              [13, 14, 15, 16]])
    output = apply_reordering(conn_matrix, [[0, 1, 3, 2],
                                            [1, 2, 3, 0]])

    # Changing rows 2 and 3 + permuting columns
    expected_out = np.asarray([[2, 3, 4, 1],
                              [6, 7, 8, 5],
                              [14, 15, 16, 13],
                              [10, 11, 12, 9]])
    assert np.array_equal(output, expected_out)


def test_evaluate_graph_measures():
    # Symmetric 5x5 test matrix
    conn_matrix = np.asarray([
        [0, 10, 20, 0, 5],
        [10, 0, 15, 30, 0],
        [20, 15, 0, 25, 10],
        [0, 30, 25, 0, 12],
        [5, 0, 10, 12, 0]
    ], dtype=float)

    len_matrix = np.asarray([
        [0, 35, 45, 0, 20],
        [35, 0, 40, 60, 0],
        [45, 40, 0, 55, 30],
        [0, 60, 55, 0, 25],
        [20, 0, 30, 25, 0]
    ], dtype=float)

    # 1. Test without len_matrix (measures must compute from conn_matrix)
    res = evaluate_graph_measures(conn_matrix, None, avg_node_wise=True,
                                  small_world=False)
    assert 'local_efficiency' in res
    assert 'global_efficiency' in res
    assert 'betweenness_centrality' in res
    assert 'path_length' in res
    assert 'edge_count' in res
    assert 'clustering' in res
    assert 'modularity' in res
    assert 'nodal_strength' in res
    assert 'anatomical_path_length' not in res

    # Efficiency and betweenness must be properly bounded in [0, 1]
    assert 0.0 <= res['local_efficiency'] <= 1.0
    assert 0.0 <= res['global_efficiency'] <= 1.0
    assert 0.0 <= res['betweenness_centrality'] <= 1.0

    # 2. Test with len_matrix (anatomical path measures added)
    res_len = evaluate_graph_measures(conn_matrix, len_matrix,
                                      avg_node_wise=True, small_world=False)
    assert 'anatomical_path_length' in res_len
    assert 'anatomical_edge_count' in res_len
    assert res_len['anatomical_path_length'] > 0

    # wiring_cost must be a connection-strength-weighted mean of the
    # physical edge lengths, so it has to fall within [min, max] of the
    # existing (nonzero) edge lengths, expressed in the same unit (mm).
    triu_idx = np.triu_indices(len(conn_matrix), k=1)
    len_triu = len_matrix[triu_idx]
    w_triu = conn_matrix[triu_idx] / np.max(conn_matrix)
    expected_wiring_cost = np.sum(w_triu * len_triu) / np.sum(w_triu)
    assert 'wiring_cost' in res_len
    assert np.isclose(res_len['wiring_cost'], expected_wiring_cost)
    assert len_triu[len_triu > 0].min() <= res_len['wiring_cost'] \
        <= len_triu.max()

    # 3. Test node-wise (avg_node_wise=False)
    res_nodewise = evaluate_graph_measures(conn_matrix, len_matrix,
                                           avg_node_wise=False,
                                           small_world=False)
    assert len(res_nodewise['betweenness_centrality']) == 5
    assert len(res_nodewise['local_efficiency']) == 5
    assert all(0.0 <= x <= 1.0 for x in res_nodewise['local_efficiency'])
    assert all(0.0 <= x <= 1.0 for x in res_nodewise['betweenness_centrality'])

    # 4. Test cost_model='length_over_weight'
    res_low = evaluate_graph_measures(conn_matrix, len_matrix,
                                      avg_node_wise=True,
                                      small_world=False,
                                      cost_model='length_over_weight')
    assert 0.0 <= res_low['local_efficiency'] <= 1.0
    assert 0.0 <= res_low['global_efficiency'] <= 1.0
    assert 0.0 <= res_low['betweenness_centrality'] <= 1.0
    assert 'wiring_cost' in res_low

    # 5. Test cost_model='anatomical_only'
    res_anat = evaluate_graph_measures(conn_matrix, len_matrix,
                                       avg_node_wise=True,
                                       small_world=False,
                                       cost_model='anatomical_only')
    assert 0.0 <= res_anat['local_efficiency'] <= 1.0
    assert 0.0 <= res_anat['global_efficiency'] <= 1.0
    assert 0.0 <= res_anat['betweenness_centrality'] <= 1.0

    # 6. Test error when length_over_weight is called without len_matrix
    try:
        evaluate_graph_measures(conn_matrix, None,
                                cost_model='length_over_weight')
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_normalize_matrix_from_values():
    pass


def test_normalize_matrix_from_parcel():
    pass
