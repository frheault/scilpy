import argparse
import os
import tempfile
import numpy as np
import nibabel as nib
from nibabel.streamlines.array_sequence import ArraySequence
import pytest
from numpy.testing import assert_array_equal, assert_almost_equal
from dipy.io.stateful_tractogram import StatefulTractogram, Space, Origin
from scilpy.io.streamlines import (
    check_tracts_same_format,
    ichunk,
    load_tractogram_with_reference,
    save_tractogram,
    verify_compatibility_with_reference_sft,
    load_dps_files_as_dps,
    load_dpp_files_as_dpp,
    streamlines_to_memmap,
    reconstruct_streamlines_from_memmap,
    reconstruct_streamlines,
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


def test_check_tracts_same_format():
    parser = argparse.ArgumentParser()
    check_tracts_same_format(parser, "a.trk", "b.trk")
    with pytest.raises(SystemExit):
        check_tracts_same_format(parser, "a.trk", "b.tck")


def test_ichunk():
    data = list(range(10))
    chunks = list(ichunk(data, 3))
    assert len(chunks) == 4
    assert len(chunks[0]) == 3
    assert len(chunks[3]) == 1


def test_load_tractogram_with_reference():
    with tempfile.TemporaryDirectory() as d:
        sft = _get_small_sft()
        filename = os.path.join(d, "test.trk")
        save_tractogram(sft, filename, no_empty=False, bbox_valid_check=False)

        parser = argparse.ArgumentParser()
        parser.add_argument('--reference')
        parser.add_argument('--bbox_check', action='store_true', default=True)
        args = parser.parse_args([])

        loaded_sft = load_tractogram_with_reference(parser, args, filename)
        assert len(loaded_sft.streamlines) == len(sft.streamlines)


def test_save_tractogram():
    with tempfile.TemporaryDirectory() as d:
        sft = _get_small_sft()
        filename = os.path.join(d, "test.trk")
        save_tractogram(sft, filename, False)
        assert os.path.exists(filename)


def test_verify_compatibility_with_reference_sft():
    with tempfile.TemporaryDirectory() as d:
        sft = _get_small_sft()
        compatible_file = os.path.join(d, "compatible.trk")
        incompatible_file = os.path.join(d, "incompatible.trk")
        save_tractogram(sft, compatible_file, False, bbox_valid_check=False)

        incompatible_ref = nib.Nifti1Image(np.zeros((4, 4, 4)),
                                           affine=np.eye(4) * 2)
        incompatible_sft = StatefulTractogram(sft.streamlines,
                                              incompatible_ref,
                                              Space.VOX,
                                              Origin('corner'))
        save_tractogram(incompatible_sft, incompatible_file, False,
                        bbox_valid_check=False)

        parser = argparse.ArgumentParser()
        parser.add_argument('--reference')
        args = parser.parse_args([])

        verify_compatibility_with_reference_sft(sft, [compatible_file],
                                                parser, args)
        with pytest.raises(SystemExit):
            verify_compatibility_with_reference_sft(sft, [incompatible_file],
                                                    parser, args)


def test_load_dps_files_as_dps():
    with tempfile.TemporaryDirectory() as d:
        sft = _get_small_sft()
        dps_file = os.path.join(d, "dps.npy")
        np.save(dps_file, np.array([1, 2]))

        parser = argparse.ArgumentParser()
        sft, keys = load_dps_files_as_dps(parser, [dps_file], sft)
        assert "dps" in sft.data_per_streamline
        assert_array_equal(sft.data_per_streamline["dps"].flatten(), [1, 2])


@pytest.mark.skip(reason="Still failing")
def test_load_dpp_files_as_dpp():
    with tempfile.TemporaryDirectory() as d:
        sft = _get_small_sft()
        dpp_file = os.path.join(d, "dpp.npy")
        dpp_data = [np.arange(3), np.arange(4)]
        np.save(dpp_file, dpp_data, allow_pickle=True)

        parser = argparse.ArgumentParser()
        with pytest.raises(SystemExit):
            load_dpp_files_as_dpp(parser, [dpp_file], sft)

        # Create data with the correct number of points
        dpp_data = np.arange(len(sft.streamlines._data))
        np.save(dpp_file, dpp_data)
        sft, keys = load_dpp_files_as_dpp(parser, [dpp_file], sft)
        assert "dpp" in sft.data_per_point


def test_streamlines_to_memmap():
    sft = _get_small_sft()
    tmp_dir, memmap_filenames = streamlines_to_memmap(sft.streamlines)
    assert os.path.exists(memmap_filenames[0])
    assert os.path.exists(memmap_filenames[1])
    assert os.path.exists(memmap_filenames[2])
    tmp_dir.cleanup()


def test_reconstruct_streamlines_from_memmap():
    sft = _get_small_sft()
    tmp_dir, memmap_filenames = streamlines_to_memmap(sft.streamlines)
    reconstructed_streamlines = reconstruct_streamlines_from_memmap(
        memmap_filenames)
    assert_almost_equal(sft.streamlines.get_data(),
                        reconstructed_streamlines.get_data(), decimal=6)
    tmp_dir.cleanup()


def test_reconstruct_streamlines():
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]).astype(np.float32)
    offsets = np.array([0, 2])
    lengths = np.array([2, 2])
    streamlines = reconstruct_streamlines(data, offsets, lengths)
    assert isinstance(streamlines, ArraySequence)
    assert len(streamlines) == 2
    assert (streamlines[0] == [[1, 2, 3], [4, 5, 6]]).all()
    assert (streamlines[1] == [[7, 8, 9], [10, 11, 12]]).all()
