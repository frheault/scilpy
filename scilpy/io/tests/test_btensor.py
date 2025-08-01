import os
import tempfile
import numpy as np
import nibabel as nib
from numpy.testing import assert_array_equal, assert_almost_equal
from scilpy.io.btensor import (
    convert_bshape_to_bdelta,
    convert_bdelta_to_bshape,
    generate_btensor_input,
)


def test_convert_bshape_to_bdelta():
    b_shapes = np.array(['LTE', 'PTE', 'STE', 'CTE'])
    b_deltas = convert_bshape_to_bdelta(b_shapes)
    assert_array_equal(b_deltas, [1, -0.5, 0, 0.5])


def test_convert_bdelta_to_bshape():
    b_deltas = np.array([1, -0.5, 0, 0.5])
    b_shapes = convert_bdelta_to_bshape(b_deltas)
    assert_array_equal(b_shapes, ['LTE', 'PTE', 'STE', 'CTE'])


def test_generate_btensor_input():
    with tempfile.TemporaryDirectory() as d:
        dwi_path = os.path.join(d, 'dwi.nii.gz')
        bval_path = os.path.join(d, 'dwi.bval')
        bvec_path = os.path.join(d, 'dwi.bvec')

        data = np.zeros((2, 2, 2, 2))
        nib.save(nib.Nifti1Image(data, np.eye(4)), dwi_path)
        np.savetxt(bval_path, [0, 1000], fmt='%d')
        bvecs = np.array([[0, 0, 0], [1, 0, 0]]).T
        np.savetxt(bvec_path, bvecs)

        gtab, data_full, ubvals, ub_deltas = generate_btensor_input(
            [dwi_path], [bval_path], [bvec_path], [1.0])

        assert gtab.bvals.shape == (2,)
        assert data_full.shape == (2, 2, 2, 2)
        assert ubvals.shape == (2,)
        assert ub_deltas.shape == (2,)
