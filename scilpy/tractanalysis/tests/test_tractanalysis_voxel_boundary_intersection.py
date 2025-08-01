# -*- coding: utf-8 -*-
import numpy as np
from nibabel.streamlines.array_sequence import ArraySequence
from scilpy.tractanalysis.voxel_boundary_intersection import \
    subdivide_streamlines_at_voxel_faces


def test_subdivide_streamlines_at_voxel_faces():
    # Test with a single horizontal streamline crossing one boundary
    streamlines = [np.array([[0.5, 0.5, 0.5], [1.5, 0.5, 0.5]],
                              dtype=np.float32)]
    arr_seq = ArraySequence(streamlines)
    subdivided_arr_seq = subdivide_streamlines_at_voxel_faces(arr_seq)

    # Expected: original points + one intersection point at x=1.0
    expected_streamline = np.array([[0.5, 0.5, 0.5],
                                      [1.0, 0.5, 0.5],
                                      [1.5, 0.5, 0.5]], dtype=np.float32)
    assert np.allclose(subdivided_arr_seq[0], expected_streamline, atol=1e-6)

    # Test with a diagonal streamline
    streamlines = [np.array([[0.5, 0.5, 0.5], [2.5, 2.5, 2.5]],
                              dtype=np.float32)]
    arr_seq = ArraySequence(streamlines)
    subdivided_arr_seq = subdivide_streamlines_at_voxel_faces(arr_seq)

    # Expected intersection points at x=1, y=1, z=1 and x=2, y=2, z=2
    # The order of intersections should be preserved along the streamline
    expected_streamline = np.array([[0.5, 0.5, 0.5],
                                      [1.0, 1.0, 1.0],
                                      [2.0, 2.0, 2.0],
                                      [2.5, 2.5, 2.5]], dtype=np.float32)
    assert subdivided_arr_seq[0].shape[0] == 4
    assert np.allclose(subdivided_arr_seq[0], expected_streamline, atol=1e-6)

    # Test with a streamline that lies on a boundary
    streamlines = [np.array([[0.5, 1.0, 0.5], [1.5, 1.0, 0.5]],
                              dtype=np.float32)]
    arr_seq = ArraySequence(streamlines)
    subdivided_arr_seq = subdivide_streamlines_at_voxel_faces(arr_seq)
    # The jittering in the function should handle this.
    # The number of points should be 3 (start, intersection, end)
    assert subdivided_arr_seq[0].shape[0] == 3

    # Test with empty streamlines
    arr_seq = ArraySequence([])
    subdivided_arr_seq = subdivide_streamlines_at_voxel_faces(arr_seq)
    assert len(subdivided_arr_seq) == 0
