# -*- coding: utf-8 -*-
import itertools
import logging
import warnings
from warnings import simplefilter

import bct
from dipy.utils.optpkg import optional_package
import numpy as np
from scipy.cluster import hierarchy

from scilpy.image.labels import get_data_as_labels
from scilpy.stats.matrix_stats import omega_sigma
from scilpy.tractanalysis.reproducibility_measures import \
    approximate_surface_node

simplefilter("ignore", hierarchy.ClusterWarning)

cl, have_bct, _ = optional_package('bct')


def compute_olo(array):
    """
    Optimal Leaf Ordering permutes a weighted matrix that has a
    symmetric sparsity pattern using hierarchical clustering.

    Parameters
    ----------
    array: ndarray (NxN)
        Connectivity matrix.

    Returns
    -------
    perm: ndarray (N,)
        Output permutations for rows and columns.
    """
    if array.ndim != 2:
        raise ValueError('RCM can only be applied to 2D array.')

    Z = hierarchy.ward(array)
    perm = hierarchy.leaves_list(
        hierarchy.optimal_leaf_ordering(Z, array))

    return perm


def apply_reordering(array, ordering):
    """
    Apply a non-symmetric array ordering that support non-square output.
    The ordering can contain duplicated or discarded rows/columns.

    Parameters
    ----------
    array: ndarray (NxN)
        Sparse connectivity matrix.
    ordering: list of lists
        First elements of the list is the permutation to apply to the rows.
        First elements of the list is the permutation to apply to the columns.

    Returns
    -------
    tmp_array: ndarray (N,N)
        Reordered array.
    """
    if array.ndim != 2:
        raise ValueError('RCM can only be applied to 2D array.')
    if not isinstance(ordering, list) or len(ordering) != 2:
        raise ValueError('Ordering should be a list of lists.\n'
                         '[[x1, x2,..., xn], [y1, y2,..., yn]]')
    ind_1, ind_2 = ordering
    if (np.array(ind_1) > array.shape[0]).any() \
            or (ind_2 > np.array(array.shape[1])).any():
        raise ValueError('Indices from configuration are larger than the'
                         'matrix size, maybe you need a labels list?')
    tmp_array = array[tuple(ind_1), :]
    tmp_array = tmp_array[:, tuple(ind_2)]

    return tmp_array


def evaluate_functional_graph_measures(conn_matrix, conn_threshold,
                                       avg_node_wise):
    """
    Parameters
    ----------
    conn_matrix: np.ndarray
        2D matrix of functional connectivity weights
    conn_threshold: float
        2D matrix of bundle lengths.
    avg_node_wise: bool
        If true, return a single value for node-wise measures.
    """
    if not have_bct:
        raise RuntimeError("bct is not installed. Please install to use "
                           "this connectivity script.")

    def avg_cast(_input):
        return float(np.average(_input))

    def list_cast(_input):
        if isinstance(_input, np.ndarray):
            if _input.ndim == 2:
                return np.average(_input, axis=1).astype(np.float32).tolist()
            return _input.astype(np.float32).tolist()
        return float(_input)

    if avg_node_wise:
        func_cast = avg_cast
    else:
        func_cast = list_cast

    # Taking the absolute value and thresholding matrix
    print("Keep only positive correlations above threshold:", conn_threshold)
    Wp = np.copy(conn_matrix)
    Wp = np.abs(conn_matrix)
    Wp[Wp <= conn_threshold] = 0

    gtm_dict = {}
    ci, gtm_dict['modularity'] = bct.modularity_louvain_und(Wp, seed=0)
    gtm_dict['assortativity'] = bct.assortativity_wei(Wp, flag=0)
    gtm_dict['participation'] = func_cast(
        bct.participation_coef_sign(Wp, ci)[0])
    gtm_dict['clustering'] = func_cast(bct.clustering_coef_wu(Wp))

    gtm_dict['nodal_strength'] = func_cast(bct.strengths_und(Wp))
    gtm_dict['density'] = func_cast(bct.density_und(Wp)[0])

    # Rich club always gives an error for the matrix rank and gives NaN
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tmp_rich_club = bct.rich_club_wu(Wp)
    gtm_dict['rich_club'] = func_cast(tmp_rich_club[~np.isnan(tmp_rich_club)])

    return gtm_dict


def evaluate_graph_measures(conn_matrix, len_matrix=None,
                            avg_node_wise=False, small_world=False,
                            cost_model='inverse_weight'):
    """
    Parameters
    ----------
    conn_matrix: np.ndarray
        2D matrix of connectivity weights. Typically a streamline count matrix.
    len_matrix: np.ndarray
        2D matrix of bundle lengths (in mm). Optional.
    avg_node_wise: bool
        If true, return a single value for node-wise measures.
    small_world: bool
        If true, compute measure related to small worldness (omega and sigma).
        This option is much slower.
    cost_model: str, optional
        Cost model for shortest path, centrality, and efficiency:
        - 'inverse_weight': Cost = 1 / W (default BCT).
        - 'length_over_weight': Cost = Length / W (hybrid model).
        - 'anatomical_only': Cost = Length (physical distance in mm).
    """
    if not have_bct:
        raise RuntimeError("bct is not installed. Please install to use "
                           "this connectivity script.")
    N = conn_matrix.shape[0]

    valid_models = ['inverse_weight', 'length_over_weight', 'anatomical_only']
    if cost_model not in valid_models:
        raise ValueError(
            f"Unknown cost_model: {cost_model}. Must be one of {valid_models}")

    if cost_model in ['length_over_weight', 'anatomical_only'] and \
            len_matrix is None:
        raise ValueError(
            f"cost_model '{cost_model}' requires a length matrix.")

    if len_matrix is not None:
        off_diag = ~np.eye(N, dtype=bool)

        len_diag = np.diagonal(len_matrix)
        if np.any(len_diag != 0):
            n_diag = int(np.count_nonzero(len_diag))
            max_diag = float(np.max(np.abs(len_diag)))
            logging.warning(
                f"Length matrix has a non-zero diagonal ({n_diag} "
                f"self-connection(s), max={max_diag:.3g}). "
                "Self-connections are set to 0 before computing anything, "
                "for the same reason as the connectivity matrix's diagonal "
                "(see below).")

        if np.any(len_matrix < 0):
            n_neg = int(np.sum(len_matrix < 0))
            logging.warning(
                f"Length matrix contains {n_neg} negative value(s); "
                "lengths should be non-negative physical distances.")

        nonzero_lengths = len_matrix[(len_matrix != 0) & off_diag]
        if nonzero_lengths.size > 0 and np.max(nonzero_lengths) < 1:
            max_len = float(np.max(nonzero_lengths))
            logging.warning(
                f"Length matrix's largest nonzero value is {max_len:.3g}, "
                "unexpectedly small for a physical bundle-length matrix "
                "expected in millimeters. Check that this is truly a "
                "length matrix and not, e.g., a normalized weight matrix.")

        conn_edges_raw = (conn_matrix != 0) & off_diag
        len_edges = (len_matrix != 0) & off_diag
        mismatch = np.count_nonzero(conn_edges_raw != len_edges) // 2
        if mismatch > 0:
            logging.warning(
                f"{mismatch} edge(s) differ in presence between the "
                "connectivity and length matrices (nonzero in one but not "
                "the other). These edges are excluded from length-based "
                "cost models ('length_over_weight', 'anatomical_only').")

    def avg_cast(_input):
        if isinstance(_input, np.ndarray):
            if _input.ndim == 2:
                masked = _input.copy().astype(float)
                np.fill_diagonal(masked, np.nan)
                masked[np.isinf(masked)] = np.nan
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    val = np.nanmean(masked)
                return float(val) if not np.isnan(val) else -1.0
            valid = _input[~np.isinf(_input) & ~np.isnan(_input)]
            if len(valid) == 0:
                return -1.0
            return float(np.mean(valid))
        return float(_input)

    def list_cast(_input):
        if isinstance(_input, np.ndarray):
            if _input.ndim == 2:
                masked = _input.copy().astype(float)
                np.fill_diagonal(masked, np.nan)
                masked[np.isinf(masked)] = np.nan
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    row_means = np.nanmean(masked, axis=1)
                row_means[np.isnan(row_means)] = -1.0
                return row_means.astype(np.float32).tolist()
            return _input.astype(np.float32).tolist()
        return float(_input)

    if avg_node_wise:
        func_cast = avg_cast
    else:
        func_cast = list_cast

    gtm_dict = {}

    # Ensure zero diagonal
    conn_matrix = conn_matrix.copy().astype(float)
    conn_diag = np.diagonal(conn_matrix)
    if np.any(conn_diag != 0):
        n_diag = int(np.count_nonzero(conn_diag))
        max_diag = float(np.max(np.abs(conn_diag)))
        logging.warning(
            f"Connectivity matrix has a non-zero diagonal ({n_diag} "
            f"self-connection(s), max={max_diag:.3g}). Self-connections "
            "are not meaningful for graph-theoretic measures (they would "
            "inflate nodal_strength/density and are undefined for "
            "shortest-path measures); the diagonal is set to 0 before "
            "computing anything.")
    if np.any(conn_matrix < 0):
        n_neg = int(np.sum(conn_matrix < 0))
        logging.warning(
            f"Connectivity matrix contains {n_neg} negative value(s); "
            "BCT measures assume non-negative weights and results may "
            "be invalid.")
    np.fill_diagonal(conn_matrix, 0)

    off_diag = ~np.eye(N, dtype=bool)
    conn_edges = (conn_matrix != 0) & off_diag
    n_possible_edges = N * (N - 1)
    density = (np.count_nonzero(conn_edges) / n_possible_edges
               if n_possible_edges else 0.0)
    if density < 0.05:
        logging.warning(
            f"Connectivity matrix is very sparse (density={density:.1%}); "
            "path-length-based measures (path_length, edge_count, "
            "betweenness_centrality, efficiency) may be unstable or not "
            "meaningful on very sparse graphs.")
    isolated_idx = np.where(~conn_edges.any(axis=1))[0]
    if len(isolated_idx) > 0:
        logging.warning(
            f"{len(isolated_idx)} node(s) have zero connections (indices: "
            f"{isolated_idx.tolist()}); their path_length/edge_count/"
            "anatomical_* values will be reported as -1.")

    # Normalize weights to [0, 1] for BCT measures (efficiency, clustering)
    max_val = np.max(np.abs(conn_matrix))
    if max_val > 1:
        logging.warning(
            f"Connectivity matrix values reach {max_val:.3g}, outside "
            "the [0, 1] range expected by BCT for weighted measures (e.g. "
            "raw streamline counts). It is automatically rescaled by "
            "dividing by this maximum before computing graph measures; "
            "this only affects the internal computation, not your input "
            "file.")
    if max_val > 0:
        norm_conn_matrix = conn_matrix / max_val
    else:
        norm_conn_matrix = conn_matrix.copy()

    # Determine effective weight matrix (W_eff) and cost matrix (L_cost)
    if cost_model == 'inverse_weight':
        W_eff = norm_conn_matrix
        L_cost = bct.weight_conversion(W_eff, 'lengths')

    elif cost_model == 'length_over_weight':
        len_mat = len_matrix.copy().astype(float)
        np.fill_diagonal(len_mat, 0)
        max_len = np.max(len_mat)

        W_eff = np.zeros_like(norm_conn_matrix)
        mask = (norm_conn_matrix > 0) & (len_mat > 0)
        if max_len > 0 and np.any(mask):
            W_eff[mask] = norm_conn_matrix[mask] / (len_mat[mask] / max_len)
            max_eff = np.max(W_eff)
            if max_eff > 0:
                W_eff /= max_eff
        L_cost = bct.weight_conversion(W_eff, 'lengths')

    elif cost_model == 'anatomical_only':
        len_mat = len_matrix.copy().astype(float)
        np.fill_diagonal(len_mat, 0)
        L_cost = len_mat.copy()

        W_eff = np.zeros_like(len_mat)
        mask = len_mat > 0
        if np.any(mask):
            W_eff[mask] = 1.0 / len_mat[mask]
            max_eff = np.max(W_eff)
            if max_eff > 0:
                W_eff /= max_eff

    # Centrality & Efficiency
    betweenness_centrality = bct.betweenness_wei(
        L_cost) / ((N - 1) * (N - 2)) if N > 2 else np.zeros((N,))
    gtm_dict['betweenness_centrality'] = func_cast(betweenness_centrality)
    gtm_dict['local_efficiency'] = func_cast(
        bct.efficiency_wei(W_eff, local=True))
    gtm_dict['global_efficiency'] = func_cast(
        bct.efficiency_wei(W_eff))

    # Path length and edge count on selected cost model
    path_length_tuple = bct.distance_wei(L_cost)
    gtm_dict['path_length'] = func_cast(path_length_tuple[0])
    gtm_dict['edge_count'] = func_cast(path_length_tuple[1])

    if small_world:
        gtm_dict['omega'], gtm_dict['sigma'] = omega_sigma(W_eff)

    ci, gtm_dict['modularity'] = bct.modularity_louvain_und(norm_conn_matrix,
                                                            seed=0)
    gtm_dict['assortativity'] = bct.assortativity_wei(norm_conn_matrix,
                                                      flag=0)
    gtm_dict['participation'] = func_cast(bct.participation_coef_sign(
        norm_conn_matrix, ci)[0])
    gtm_dict['clustering'] = func_cast(
        bct.clustering_coef_wu(norm_conn_matrix))

    gtm_dict['nodal_strength'] = func_cast(bct.strengths_und(conn_matrix))
    gtm_dict['density'] = func_cast(bct.density_und(conn_matrix)[0])

    # Rich club
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tmp_rich_club = bct.rich_club_wu(norm_conn_matrix)
    gtm_dict['rich_club'] = func_cast(tmp_rich_club[~np.isnan(tmp_rich_club)])

    # Physical / anatomical length measures if len_matrix is provided
    if len_matrix is not None:
        len_mat = len_matrix.copy().astype(float)
        np.fill_diagonal(len_mat, 0)
        anat_path_tuple = bct.distance_wei(len_mat)
        gtm_dict['anatomical_path_length'] = func_cast(anat_path_tuple[0])
        gtm_dict['anatomical_edge_count'] = func_cast(anat_path_tuple[1])

        # Wiring cost (Bullmore & Sporns 2012): connection-strength-weighted
        # mean physical length of the network's edges (in mm).
        triu_idx = np.triu_indices(N, k=1)
        len_triu = len_mat[triu_idx]
        w_triu = norm_conn_matrix[triu_idx]
        sum_w = np.sum(w_triu)
        if sum_w > 0:
            gtm_dict['wiring_cost'] = float(
                np.sum(w_triu * len_triu) / sum_w)

    return gtm_dict


def normalize_matrix_from_values(matrix, norm_factor, inverse):
    """
    Parameters
    ----------
    matrix: np.ndarray
        Connectivity matrix
    norm_factor: np.ndarray of shape ?
        Matrix used for edge-wise multiplication. Ex: length or volume of the
        bundles.
    inverse: bool
        If true, divide by the matrix rather than multiply.
    """
    where_above0 = norm_factor > 0
    if inverse:
        matrix[where_above0] /= norm_factor[where_above0]
    else:
        matrix[where_above0] *= norm_factor[where_above0]
    return matrix


def normalize_matrix_from_parcel(matrix, atlas_img, labels_list,
                                 parcel_from_volume):
    """
    Parameters
    ----------
    matrix: np.ndarray
        Connectivity matrix
    atlas_img: nib.Nifti1Image
        Atlas for edge-wise division.
    labels_list: np.ndarray
        The list of labels of interest for edge-wise division.
    parcel_from_volume: bool
        If true, parcel from volume. Else, parcel from surface.
    """
    atlas_data = get_data_as_labels(atlas_img)

    voxels_size = atlas_img.header.get_zooms()[:3]
    if voxels_size[0] != voxels_size[1] \
            or voxels_size[0] != voxels_size[2]:
        raise ValueError('Atlas must have an isotropic resolution.')

    voxels_vol = np.prod(atlas_img.header.get_zooms()[:3])
    voxels_sur = np.prod(atlas_img.header.get_zooms()[:2])

    if len(labels_list) != matrix.shape[0] \
            and len(labels_list) != matrix.shape[1]:
        raise ValueError('labels_list should have the same number of label as '
                         'the input matrix.')

    pos_list = range(len(labels_list))
    all_comb = list(itertools.combinations(pos_list, r=2))
    all_comb.extend(zip(pos_list, pos_list))

    # Prevent useless computations for approximate_surface_node()
    factor_list = []
    for label in labels_list:
        if parcel_from_volume:
            factor_list.append(
                np.count_nonzero(atlas_data == label) * voxels_vol)
        else:
            if np.count_nonzero(atlas_data == label):
                roi = np.zeros(atlas_data.shape)
                roi[atlas_data == label] = 1
                factor_list.append(
                    approximate_surface_node(roi) * voxels_sur)
            else:
                factor_list.append(0)

    for pos_1, pos_2 in all_comb:
        factor = factor_list[pos_1] + factor_list[pos_2]
        if abs(factor) > 0.001:
            matrix[pos_1, pos_2] /= factor
            matrix[pos_2, pos_1] /= factor

    return matrix
