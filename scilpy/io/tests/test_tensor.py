import numpy as np
from numpy.testing import assert_array_equal
from scilpy.io.tensor import (
    convert_tensor_to_dipy_format,
    convert_tensor_from_dipy_format,
    convert_tensor_format,
)


def test_convert_tensor_to_dipy_format():
    # FSL format
    tensor_fsl = np.array([1, 2, 3, 4, 5, 6])
    tensor_dipy = convert_tensor_to_dipy_format(tensor_fsl, 'fsl')
    assert_array_equal(tensor_dipy, [1, 2, 4, 3, 5, 6])

    # MRtrix format
    tensor_mrtrix = np.array([1, 2, 3, 4, 5, 6])
    tensor_dipy = convert_tensor_to_dipy_format(tensor_mrtrix, 'mrtrix')
    assert_array_equal(tensor_dipy, [1, 4, 2, 5, 6, 3])


def test_convert_tensor_from_dipy_format():
    tensor_dipy = np.array([1, 2, 3, 4, 5, 6])
    # FSL format
    tensor_fsl = convert_tensor_from_dipy_format(tensor_dipy, 'fsl')
    assert_array_equal(tensor_fsl, [1, 2, 4, 3, 5, 6])

    # MRtrix format
    tensor_mrtrix = convert_tensor_from_dipy_format(tensor_dipy, 'mrtrix')
    assert_array_equal(tensor_mrtrix, [1, 3, 6, 2, 4, 5])


def test_convert_tensor_format():
    tensor_fsl = np.array([1, 2, 3, 4, 5, 6])
    tensor_mrtrix = convert_tensor_format(tensor_fsl, 'fsl', 'mrtrix')
    assert_array_equal(tensor_mrtrix, [1, 4, 6, 2, 3, 5])
