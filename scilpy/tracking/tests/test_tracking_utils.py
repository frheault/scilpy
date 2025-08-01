# -*- coding: utf-8 -*-
import numpy as np
from tqdm import tqdm

from scilpy.tracking.utils import (TrackingDirection, tqdm_if_verbose,
                                   get_theta, sample_distribution)


def test_tracking_direction():
    direction = TrackingDirection([1, 0, 0], index=5)
    assert isinstance(direction, list)
    assert direction.index == 5
    assert np.array_equal(direction, [1, 0, 0])


def test_tqdm_if_verbose():
    gen = range(10)

    # Test with verbose=True
    tqdm_gen = tqdm_if_verbose(gen, verbose=True)
    assert isinstance(tqdm_gen, tqdm)

    # Test with verbose=False
    tqdm_gen = tqdm_if_verbose(gen, verbose=False)
    assert not isinstance(tqdm_gen, tqdm)
    assert list(tqdm_gen) == list(gen)


def test_get_theta():
    # Test user-provided theta
    assert get_theta(42, 'det') == 42
    assert get_theta(42, 'prob') == 42
    assert get_theta(42, 'ptt') == 42
    assert get_theta(42, 'eudx') == 42

    # Test default thetas
    assert get_theta(None, 'det') == 45
    assert get_theta(None, 'prob') == 20
    assert get_theta(None, 'ptt') == 20
    assert get_theta(None, 'eudx') == 60


def test_sample_distribution():
    dist = np.array([0.1, 0.2, 0.7])
    rng = np.random.default_rng(12345)

    # Run a few times to see if it samples correctly
    samples = [sample_distribution(dist, rng) for _ in range(100)]
    assert all(s in [0, 1, 2] for s in samples)
    # With this seed, we should get a mix of indices
    assert len(np.unique(samples)) > 1

    # Test with a zero distribution
    zero_dist = np.zeros(3)
    assert sample_distribution(zero_dist, rng) is None


def test_add_mandatory_options_tracking():
    # TODO: Implement this test
    pass


def test_add_tracking_options():
    # TODO: Implement this test
    pass


def test_add_tracking_ptt_options():
    # TODO: Implement this test
    pass


def test_add_seeding_options():
    # TODO: Implement this test
    pass


def test_add_out_options():
    # TODO: Implement this test
    pass


def test_verify_streamline_length_options():
    # TODO: Implement this test
    pass


def test_verify_seed_options():
    # TODO: Implement this test
    pass


def test_save_tractogram():
    # TODO: Implement this test
    pass


def test_get_direction_getter():
    # TODO: Implement this test
    pass
