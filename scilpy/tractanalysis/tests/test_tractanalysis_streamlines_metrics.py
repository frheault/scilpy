# -*- coding: utf-8 -*-
import numpy as np
from scilpy.tractanalysis.streamlines_metrics import compute_tract_counts_map


def test_compute_tract_counts_map():
    # Test with a single horizontal streamline
    streamlines = [np.array([[0.5, 0.5, 0.5], [1.5, 0.5, 0.5],
                               [2.5, 0.5, 0.5]], dtype=np.float32)]
    vol_dims = (3, 2, 2)
    counts_map = compute_tract_counts_map(streamlines, vol_dims)
    expected_map = np.zeros(vol_dims, dtype=int)
    expected_map[0, 0, 0] = 1
    expected_map[1, 0, 0] = 1
    expected_map[2, 0, 0] = 1
    assert np.array_equal(counts_map, expected_map)

    # Test with a diagonal streamline
    streamlines = [np.array([[0.5, 0.5, 0.5], [1.5, 1.5, 1.5]],
                              dtype=np.float32)]
    vol_dims = (2, 2, 2)
    counts_map = compute_tract_counts_map(streamlines, vol_dims)
    expected_map = np.zeros(vol_dims, dtype=int)
    expected_map[0, 0, 0] = 1
    expected_map[1, 1, 1] = 1
    assert np.array_equal(counts_map, expected_map)

    # Test with multiple streamlines
    streamlines = [
        np.array([[0.5, 0.5, 0.5], [1.5, 0.5, 0.5]], dtype=np.float32),
        np.array([[0.5, 0.5, 0.5], [0.5, 1.5, 0.5]], dtype=np.float32)
    ]
    vol_dims = (2, 2, 2)
    counts_map = compute_tract_counts_map(streamlines, vol_dims)
    expected_map = np.zeros(vol_dims, dtype=int)
    expected_map[0, 0, 0] = 2  # Both streamlines start here
    expected_map[1, 0, 0] = 1
    expected_map[0, 1, 0] = 1
    assert np.array_equal(counts_map, expected_map)

    # Test with empty streamlines
    streamlines = []
    vol_dims = (2, 2, 2)
    counts_map = compute_tract_counts_map(streamlines, vol_dims)
    assert np.array_equal(counts_map, np.zeros(vol_dims))
