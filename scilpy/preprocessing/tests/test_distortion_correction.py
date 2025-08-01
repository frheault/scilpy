import numpy as np
from numpy.testing import assert_array_equal
from scilpy.preprocessing.distortion_correction import (
    create_acqparams,
    create_index,
    create_multi_topup_index,
    create_non_zero_norm_bvecs,
)


def test_create_acqparams():
    acqparams = create_acqparams(0.1, 'x', nb_rev_b0s=0)
    assert_array_equal(acqparams, [[1, 0, 0, 0.1]])


def test_create_index():
    # TODO: Implement this test
    pass


def test_create_multi_topup_index():
    # TODO: Implement this test
    pass


def test_create_non_zero_norm_bvecs():
    # TODO: Implement this test
    pass
