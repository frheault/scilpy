# -*- coding: utf-8 -*-
import numpy as np
from nibabel.streamlines.array_sequence import ArraySequence
from scilpy.tractograms.uncompress import uncompress


def test_uncompress():
    # Test with a single horizontal streamline
    streamlines = [np.array([[0.5, 0.5, 0.5], [0.6, 0.5, 0.5],
                               [1.5, 0.5, 0.5]], dtype=np.float32)]
    arr_seq = ArraySequence(streamlines)
    uncompressed_seq = uncompress(arr_seq)

    # Expected: voxel (0,0,0) and (1,0,0)
    expected_indices = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.uint16)
    assert np.array_equal(uncompressed_seq[0], expected_indices)

    # Test with return_mapping
    uncompressed_seq, mapping = uncompress(arr_seq, return_mapping=True)
    assert np.array_equal(uncompressed_seq[0], expected_indices)

    # Points 1 and 2 are in voxel (0,0,0), point 3 is in voxel (1,0,0)
    # The mapping should be [0, 0, 1]
    expected_mapping = np.array([0, 0, 1], dtype=np.uint16)
    assert np.array_equal(mapping[0], expected_mapping)

    # Test with a diagonal streamline
    streamlines = [np.array([[0.5, 0.5, 0.5], [1.5, 1.5, 1.5]],
                              dtype=np.float32)]
    arr_seq = ArraySequence(streamlines)
    uncompressed_seq = uncompress(arr_seq)

    # Expected voxels: (0,0,0) and (1,1,1)
    expected_indices = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.uint16)
    assert np.array_equal(uncompressed_seq[0], expected_indices)

    # Test with empty streamlines
    arr_seq = ArraySequence([])
    uncompressed_seq = uncompress(arr_seq)
    assert len(uncompressed_seq) == 0
