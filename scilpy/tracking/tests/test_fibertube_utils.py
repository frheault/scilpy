import numpy as np
from numpy.testing import assert_array_equal
from scilpy.tracking.fibertube_utils import streamlines_to_segments


def test_streamlines_to_segments():
    streamlines = [np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])]
    centers, indices, max_length = streamlines_to_segments(streamlines)
    assert centers.shape == (2, 3)
    assert indices.shape == (2, 2)
    assert max_length > 0
