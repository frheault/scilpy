import numpy as np
from scilpy.stats.stats import (
    verify_normality,
    verify_homoscedasticity,
    verify_group_difference,
    verify_post_hoc,
)


def test_verify_normality():
    data = np.random.normal(0, 1, 100)
    normality, p_value = verify_normality(data)
    assert normality


def test_verify_homoscedasticity():
    # TODO: Implement this test
    pass


def test_verify_group_difference():
    # TODO: Implement this test
    pass


def test_verify_post_hoc():
    # TODO: Implement this test
    pass
