import numpy as np
from scilpy.image.volume_metrics import estimate_piesno_sigma


def test_estimate_piesno_sigma():
    data = np.zeros((10, 10, 10, 10))
    data[..., 5:] += 10
    sigma, mask = estimate_piesno_sigma(data)
    assert sigma.shape == (10, 10, 10)
    assert mask.shape == (10, 10, 10)
