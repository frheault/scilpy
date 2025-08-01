import numpy as np
from dipy.io.stateful_tractogram import Origin, Space
from scilpy.image.volume_space_management import (
    DataVolume,
    FibertubeDataVolume,
)


def test_data_volume_get_value_at_idx():
    data = np.arange(27).reshape((3, 3, 3))
    dv = DataVolume(data, (1, 1, 1))
    assert dv.get_value_at_idx(1, 1, 1) == 13


def test_data_volume_get_value_at_coordinate():
    data = np.arange(27).reshape((3, 3, 3))
    dv = DataVolume(data, (1, 1, 1), interpolation='nearest')
    assert dv.get_value_at_coordinate(0.6, 0.6, 0.6, Space.VOX,
                                      Origin('corner')) == 0


def test_data_volume_is_idx_in_bound():
    data = np.zeros((3, 3, 3))
    dv = DataVolume(data, (1, 1, 1))
    assert dv.is_idx_in_bound(1, 1, 1)
    assert not dv.is_idx_in_bound(3, 3, 3)


def test_data_volume_is_coordinate_in_bound():
    data = np.zeros((3, 3, 3))
    dv = DataVolume(data, (1, 1, 1))
    assert dv.is_coordinate_in_bound(1, 1, 1, Space.VOX, Origin('corner'))
    assert not dv.is_coordinate_in_bound(3, 3, 3, Space.VOX, Origin('corner'))


def test_fibertube_data_volume_get_value_at_idx():
    # TODO: Implement this test
    pass


def test_fibertube_data_volume_get_value_at_coordinate():
    # TODO: Implement this test
    pass


def test_fibertube_data_volume_is_idx_in_bound():
    # TODO: Implement this test
    pass


def test_fibertube_data_volume_is_coordinate_in_bound():
    # TODO: Implement this test
    pass


def test_fibertube_data_volume_get_absolute_direction():
    # TODO: Implement this test
    pass
