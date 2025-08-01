import numpy as np
import nibabel as nib
from numpy.testing import assert_array_almost_equal
from scilpy.image.reslice import reslice


def test_reslice():
    data = np.zeros((10, 10, 10))
    affine = np.eye(4)
    zooms = (1.0, 1.0, 1.0)
    new_zooms = (2.0, 2.0, 2.0)
    data2, affine2 = reslice(data, affine, zooms, new_zooms)
    assert data2.shape == (5, 5, 5)
    expected_affine = np.array([[2., 0., 0., 0.5],
                                [0., 2., 0., 0.5],
                                [0., 0., 2., 0.5],
                                [0., 0., 0., 1.]])
    assert_array_almost_equal(affine2, expected_affine)
