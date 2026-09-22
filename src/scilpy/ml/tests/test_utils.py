import numpy as np
import torch

from scilpy.ml.utils import to_numpy


def test_to_numpy_half_precision():
    bfloat_tensor = torch.ones((3, 3), dtype=torch.bfloat16)
    arr = to_numpy(bfloat_tensor)
    assert arr.dtype == np.float32
    assert np.allclose(arr, 1.0)

    half_tensor = torch.ones((3, 3), dtype=torch.float16)
    arr_half = to_numpy(half_tensor)
    assert arr_half.dtype == np.float32
    assert np.allclose(arr_half, 1.0)
