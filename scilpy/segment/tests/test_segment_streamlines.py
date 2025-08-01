# -*- coding: utf-8 -*-
import numpy as np
import pytest
from dipy.io.stateful_tractogram import StatefulTractogram
from nibabel import Nifti1Image

from scilpy.segment.streamlines import (
    streamlines_in_mask, filter_grid_roi_both, filter_grid_roi,
    filter_ellipsoid, filter_cuboid)


@pytest.fixture
def dummy_sft():
    # Streamline 1: stays in mask 1
    # Streamline 2: goes from mask 1 to mask 2
    # Streamline 3: completely outside masks
    # Streamline 4: starts in mask 1, ends outside
    streamlines = [
        np.array([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]]),
        np.array([[0., 0., 0.], [4., 4., 4.], [8., 8., 8.]]),
        np.array([[10., 10., 10.], [11., 11., 11.]]),
        np.array([[0., 1., 1.], [4., 5., 5.], [5., 5., 5.]])
    ]
    img = Nifti1Image(np.zeros((12, 12, 12)), np.eye(4))
    sft = StatefulTractogram(streamlines, img, 'rasmm')
    return sft


@pytest.fixture
def masks():
    mask1 = np.zeros((12, 12, 12), dtype=np.uint8)
    mask1[0:3, 0:3, 0:3] = 1
    mask2 = np.zeros((12, 12, 12), dtype=np.uint8)
    mask2[8:10, 8:10, 8:10] = 1
    return mask1, mask2


def test_streamlines_in_mask(dummy_sft, masks):
    mask1, _ = masks
    sft = dummy_sft.from_sft(dummy_sft.streamlines, dummy_sft)
    sft.to_vox()
    sft.to_corner()

    # Test any part in mask
    ids = streamlines_in_mask(sft, mask1, all_in=False)
    assert np.array_equal(sorted(ids), [0, 1, 3])

    # Test all in mask
    ids = streamlines_in_mask(sft, mask1, all_in=True)
    assert np.array_equal(sorted(ids), [0])


def test_filter_grid_roi_both(dummy_sft, masks):
    mask1, mask2 = masks
    new_sft, ids = filter_grid_roi_both(dummy_sft, mask1, mask2)
    assert len(new_sft) == 1
    assert np.array_equal(ids, [1])


def test_filter_grid_roi(dummy_sft, masks):
    mask1, _ = masks
    sft = dummy_sft.from_sft(dummy_sft.streamlines, dummy_sft)

    # any
    ids = filter_grid_roi(sft, mask1, 'any', False)
    assert np.array_equal(sorted(ids), [0, 1, 3])

    # all
    ids = filter_grid_roi(sft, mask1, 'all', False)
    assert np.array_equal(sorted(ids), [0])

    # either_end
    ids = filter_grid_roi(sft, mask1, 'either_end', False)
    assert np.array_equal(sorted(ids), [0, 1, 3])

    # both_ends
    ids = filter_grid_roi(sft, mask1, 'both_ends', False)
    assert np.array_equal(sorted(ids), [0])

    # exclude
    ids = filter_grid_roi(sft, mask1, 'any', True)
    assert np.array_equal(sorted(ids), [2])

    # return_sft
    ids, sft_out = filter_grid_roi(sft, mask1, 'any', False, return_sft=True)
    assert len(sft_out) == 3

    # return_rejected_sft
    ids, sft_out, rej_sft = filter_grid_roi(sft, mask1, 'any', False,
                                            return_sft=True,
                                            return_rejected_sft=True)
    assert len(sft_out) == 3
    assert len(rej_sft) == 1


def test_filter_ellipsoid_smoke(dummy_sft):
    radius = np.array([2., 2., 2.])
    center = np.array([0., 0., 0.])
    ids, new_sft = filter_ellipsoid(dummy_sft, radius, center, 'any', False)
    assert len(new_sft) <= len(dummy_sft)
    assert np.all(ids < len(dummy_sft))


def test_filter_cuboid_smoke(dummy_sft):
    radius = np.array([2., 2., 2.])
    center = np.array([0., 0., 0.])
    ids, new_sft = filter_cuboid(dummy_sft, radius, center, 'any', False)
    assert len(new_sft) <= len(dummy_sft)
    assert np.all(ids < len(dummy_sft))


def test_pre_filtering_for_geometrical_shape():
    # TODO: Implement this test
    pass
