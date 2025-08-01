import os
import tempfile
import numpy as np
from numpy.testing import assert_array_equal
from scilpy.io.gradients import (
    fsl2mrtrix,
    mrtrix2fsl,
    save_gradient_sampling_mrtrix,
    save_gradient_sampling_fsl,
)


def test_fsl2mrtrix():
    with tempfile.TemporaryDirectory() as d:
        bval_filename = os.path.join(d, "test.bval")
        bvec_filename = os.path.join(d, "test.bvec")
        mrtrix_filename = os.path.join(d, "test.b")
        with open(bval_filename, 'w') as f:
            f.write("1000 2000")
        with open(bvec_filename, 'w') as f:
            f.write("1 0 0\n0 1 0")
        fsl2mrtrix(bval_filename, bvec_filename, mrtrix_filename)
        with open(mrtrix_filename, 'r') as f:
            lines = f.readlines()
            assert len(lines) == 2
            # Note: fsl2mrtrix transposes the bvecs
            assert lines[0].strip() == "1.00000000 0.00000000 0.00000000 1000"
            assert lines[1].strip() == "0.00000000 1.00000000 0.00000000 2000"


def test_mrtrix2fsl():
    with tempfile.TemporaryDirectory() as d:
        mrtrix_filename = os.path.join(d, "test.b")
        fsl_filename = os.path.join(d, "test")
        with open(mrtrix_filename, 'w') as f:
            f.write("1 0 0 1000\n0 1 0 2000")
        mrtrix2fsl(mrtrix_filename, fsl_filename)
        assert os.path.exists(fsl_filename + ".bval")
        assert os.path.exists(fsl_filename + ".bvec")
        read_bvals = np.loadtxt(fsl_filename + ".bval")
        read_bvecs = np.loadtxt(fsl_filename + ".bvec")
        assert_array_equal(read_bvals, [1000, 2000])
        assert_array_equal(read_bvecs, np.array([[1, 0], [0, 1], [0, 0]]))


def test_save_gradient_sampling_mrtrix():
    with tempfile.TemporaryDirectory() as d:
        filename = os.path.join(d, "test.b")
        bvecs = np.array([[1, 0, 0], [0, 1, 0]]).T
        shell_idx = np.array([0, 1])
        bvals = np.array([1000, 2000])
        save_gradient_sampling_mrtrix(bvecs, shell_idx, bvals, filename)
        with open(filename, 'r') as f:
            lines = f.readlines()
            assert len(lines) == 2
            assert lines[0].strip() == "1.00000000 0.00000000 0.00000000 1000"
            assert lines[1].strip() == "0.00000000 1.00000000 0.00000000 2000"


def test_save_gradient_sampling_fsl():
    with tempfile.TemporaryDirectory() as d:
        bval_filename = os.path.join(d, "test.bval")
        bvec_filename = os.path.join(d, "test.bvec")
        bvecs = np.array([[1, 0, 0], [0, 1, 0]]).T
        shell_idx = np.array([0, 1])
        bvals = np.array([1000, 2000])
        save_gradient_sampling_fsl(
            bvecs, shell_idx, bvals, bval_filename, bvec_filename)
        assert os.path.exists(bval_filename)
        assert os.path.exists(bvec_filename)
        read_bvals = np.loadtxt(bval_filename)
        read_bvecs = np.loadtxt(bvec_filename)
        assert_array_equal(read_bvals, [1000, 2000])
        assert_array_equal(read_bvecs, bvecs)
