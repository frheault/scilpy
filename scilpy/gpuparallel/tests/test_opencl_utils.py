import pytest
from scilpy.gpuparallel.opencl_utils import cl_device_type


@pytest.mark.skip(reason="pyopencl not installed")
def test_cl_device_type():
    # This is a simple test, we can't really test the cl.device_type
    # without a working opencl environment.
    assert cl_device_type('cpu') != -1
    assert cl_device_type('gpu') != -1
    assert cl_device_type('invalid') == -1
