import os
import tempfile
import h5py
import numpy as np
import nibabel as nib
import pytest
from numpy.testing import assert_array_equal, assert_almost_equal
from dipy.io.stateful_tractogram import StatefulTractogram, Space, Origin
from scilpy.io.hdf5 import (
    reconstruct_sft_from_hdf5,
    assert_header_compatible_hdf5,
    reconstruct_streamlines_from_hdf5,
    construct_hdf5_from_sft,
    construct_hdf5_header,
    construct_hdf5_group_from_streamlines,
)


def _get_small_sft():
    # SFT = 2 streamlines: [3 points, 4 points]
    fake_ref = nib.Nifti1Image(np.zeros((3, 3, 3)), affine=np.eye(4))
    fake_sft = StatefulTractogram(streamlines=[[[0.1, 0.1, 0.1],
                                                [0.2, 0.2, 0.2],
                                                [0.3, 0.3, 0.3]],
                                               [[1.1, 1.1, 1.1],
                                                [1.2, 1.2, 1.2],
                                                [1.3, 1.3, 1.3],
                                                [1.4, 1.4, 1.4]]],
                                  reference=fake_ref,
                                  space=Space.VOX, origin=Origin('corner'))
    return fake_sft


def test_reconstruct_sft_from_hdf5():
    sft = _get_small_sft()
    with tempfile.NamedTemporaryFile(suffix='.h5') as f:
        with h5py.File(f.name, 'w') as hf:
            construct_hdf5_from_sft(hf, sft)
        with h5py.File(f.name, 'r') as hf:
            reconstructed_sft, _ = reconstruct_sft_from_hdf5(hf, None)
            assert_almost_equal(sft.streamlines.get_data(),
                                reconstructed_sft.streamlines.get_data(),
                                decimal=6)


def test_assert_header_compatible_hdf5():
    sft = _get_small_sft()
    with tempfile.NamedTemporaryFile(suffix='.h5') as f:
        with h5py.File(f.name, 'w') as hf:
            construct_hdf5_header(hf, sft)
            compatible_img = nib.Nifti1Image(np.zeros((3, 3, 3)), np.eye(4))
            assert_header_compatible_hdf5(hf, compatible_img)

            incompatible_img = nib.Nifti1Image(
                np.zeros((4, 4, 4)), np.eye(4))
            with pytest.raises(IOError):
                assert_header_compatible_hdf5(hf, incompatible_img)


def test_reconstruct_streamlines_from_hdf5():
    sft = _get_small_sft()
    with tempfile.NamedTemporaryFile(suffix='.h5') as f:
        with h5py.File(f.name, 'w') as hf:
            group = hf.create_group('test_group')
            construct_hdf5_group_from_streamlines(group, sft.streamlines)
            reconstructed_streamlines = reconstruct_streamlines_from_hdf5(group)
            assert_almost_equal(sft.streamlines.get_data(),
                                reconstructed_streamlines.get_data(), decimal=6)


def test_construct_hdf5_from_sft():
    sft = _get_small_sft()
    with tempfile.NamedTemporaryFile(suffix='.h5') as f:
        with h5py.File(f.name, 'w') as hf:
            construct_hdf5_from_sft(hf, sft)
            assert 'streamlines' in hf
            assert 'affine' in hf.attrs


def test_construct_hdf5_header():
    sft = _get_small_sft()
    with tempfile.NamedTemporaryFile(suffix='.h5') as f:
        with h5py.File(f.name, 'w') as hf:
            construct_hdf5_header(hf, sft)
            assert 'affine' in hf.attrs
            assert 'dimensions' in hf.attrs
            assert 'voxel_sizes' in hf.attrs
            assert 'voxel_order' in hf.attrs


def test_construct_hdf5_group_from_streamlines():
    sft = _get_small_sft()
    with tempfile.NamedTemporaryFile(suffix='.h5') as f:
        with h5py.File(f.name, 'w') as hf:
            group = hf.create_group('test_group')
            construct_hdf5_group_from_streamlines(group, sft.streamlines)
            assert 'data' in group
            assert 'offsets' in group
            assert 'lengths' in group
