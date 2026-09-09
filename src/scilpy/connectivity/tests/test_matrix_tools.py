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


def test_evaluate_graph_measures_mismatched_edges():
    # conn_matrix has an edge (0, 2) with no matching length in len_matrix
    # (len_matrix[0, 2] == 0), simulating inconsistent inputs. Edges present
    # in only one of the two matrices must be excluded from length-based
    # cost models ('length_over_weight', 'anatomical_only') and from
    # wiring_cost, instead of being silently treated as valid connections
    # with a length of 0.
    conn_matrix = np.asarray([
        [0, 10, 20, 0, 5],
        [10, 0, 15, 30, 0],
        [20, 15, 0, 25, 10],
        [0, 30, 25, 0, 12],
        [5, 0, 10, 12, 0]
    ], dtype=float)
    len_matrix = np.asarray([
        [0, 35, 0, 0, 20],
        [35, 0, 40, 60, 0],
        [0, 40, 0, 55, 30],
        [0, 60, 55, 0, 25],
        [20, 0, 30, 25, 0]
    ], dtype=float)

    N = len(conn_matrix)
    triu_idx = np.triu_indices(N, k=1)
    w_triu = conn_matrix[triu_idx] / np.max(conn_matrix)
    len_triu = len_matrix[triu_idx]
    valid = (w_triu > 0) & (len_triu > 0)
    expected_wiring_cost = (np.sum(w_triu[valid] * len_triu[valid]) /
                            np.sum(w_triu[valid]))
    # The naive (buggy) computation that includes the mismatched edge with
    # length 0 would deflate the result - make sure the two differ, so this
    # test would actually fail without the fix.
    naive_wiring_cost = np.sum(w_triu * len_triu) / np.sum(w_triu)
    assert not np.isclose(expected_wiring_cost, naive_wiring_cost)

    res = evaluate_graph_measures(conn_matrix, len_matrix,
                                  avg_node_wise=True, small_world=False,
                                  cost_model='anatomical_only')
    assert np.isclose(res['wiring_cost'], expected_wiring_cost)


def test_evaluate_graph_measures_wiring_cost_always_present():
    # 'wiring_cost' must always be in the output whenever a len_matrix is
    # given, even when it cannot be computed (here: len_matrix has no
    # length data at all). Otherwise, in a --append_json batch over many
    # subjects, a key that is only sometimes present either misaligns the
    # per-subject lists (if the "good" subject comes first) or raises a
    # KeyError (if the "bad" subject comes first).
    conn_matrix = np.asarray([[0, 10, 20],
                              [10, 0, 15],
                              [20, 15, 0]], dtype=float)
    len_matrix_empty = np.zeros((3, 3))

    res = evaluate_graph_measures(conn_matrix, len_matrix_empty,
                                  avg_node_wise=True, small_world=False)
    assert 'wiring_cost' in res
    assert res['wiring_cost'] == -1.0


def test_evaluate_graph_measures_empty_conn_matrix_raises():
    # An all-zero connectivity matrix (e.g. a failed subject, or one fully
    # zeroed out by --filtering_mask) must raise a clear, actionable error
    # instead of crashing deep inside bct.modularity_louvain_und with a
    # cryptic "Modularity Infinite Loop Style B" BCTParamError.
    empty_conn_matrix = np.zeros((5, 5))
    try:
        evaluate_graph_measures(empty_conn_matrix, None, avg_node_wise=True,
                                small_world=False)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_normalize_matrix_from_values():
    pass


def test_normalize_matrix_from_parcel():
    pass
