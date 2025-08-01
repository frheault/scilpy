import os
import tempfile

from dipy.io.stateful_tractogram import StatefulTractogram, Space
from dipy.io.streamline import load_tractogram, save_tractogram
from dipy.testing import assert_true
import nibabel as nib
import numpy as np
from numpy.testing import (assert_array_equal, assert_raises,
                           assert_allclose, assert_equal)

from scilpy.tractograms.tractogram_operations import (
    concatenate_sft,
    shuffle_streamlines,
    flip_sft,
    compress_sft,
    split_sft_by_number,
    split_sft_randomly,
    remove_invalid_streamlines,
    get_subset_streamlines,
    cut_invalid_streamlines,
    assert_sft_compatibility)


sft = None


def setup_module():
    global sft
    fake_dir = tempfile.TemporaryDirectory()
    streamlines = [[[0, 0, 0], [1, 1, 1]], [[0, 0, 0], [2, 2, 2]]]
    sft = StatefulTractogram(streamlines, 'same', Space.VOX)
    save_tractogram(sft, os.path.join(fake_dir.name, 'sft.trk'))


def test_shuffle_streamlines():
    # Shuffling pretty straightforward, not testing.
    # Verifying that initial SFT is not modified.
    sft2 = shuffle_streamlines(sft)
    assert not np.array_equal(sft2.streamlines[0], sft.streamlines[0])


def test_flip_sft():
    sft_flip_x = flip_sft(sft, ['x'])
    assert_allclose(sft_flip_x.streamlines._data[:, 0], -sft.streamlines._data[:, 0])

    sft_flip_y = flip_sft(sft, ['y'])
    assert_allclose(sft_flip_y.streamlines._data[:, 1], -sft.streamlines._data[:, 1])

    sft_flip_z = flip_sft(sft, ['z'])
    assert_allclose(sft_flip_z.streamlines._data[:, 2], -sft.streamlines._data[:, 2])

    sft_flip_xy = flip_sft(sft, ['x', 'y'])
    assert_allclose(sft_flip_xy.streamlines._data[:, 0], -sft.streamlines._data[:, 0])
    assert_allclose(sft_flip_xy.streamlines._data[:, 1], -sft.streamlines._data[:, 1])


def test_multiply_sft_affine():
    pass


def test_compress_sft():
    compressed_sft = compress_sft(sft)
    assert len(sft.streamlines) == len(compressed_sft.streamlines)
    assert len(sft.streamlines[0]) > len(compressed_sft.streamlines[0])


def test_split_sft_by_number():
    all_sfts = split_sft_by_number(sft, 1)
    assert len(all_sfts) == 2


def test_split_sft_randomly():
    all_sfts = split_sft_randomly(sft, 1, 1234)
    assert len(all_sfts) == 2


def test_concatenate_sft():
    sft_1 = StatefulTractogram(sft.streamlines[0:1], 'same', Space.VOX)
    sft_2 = StatefulTractogram(sft.streamlines[1:2], 'same', Space.VOX)

    sft_union = concatenate_sft([sft_1, sft_2])
    assert len(sft_union) == 2
    assert_array_equal(sft_union.streamlines._data, sft.streamlines._data)


def test_remove_invalid_streamlines():
    # Manual creation of invalid streamlines
    streamlines = [[[0, 0, 0], [1, 1, 1]], [[0, 0, 0], [0, 0, 0]],
                   [[1, 1, 1], [1, 1, 1]]]
    sft_w_invalid = StatefulTractogram(streamlines, 'same', Space.VOX)

    clean_sft, _ = remove_invalid_streamlines(sft_w_invalid)
    assert len(clean_sft) == 1


def test_get_subset_streamlines():
    subset_sft = get_subset_streamlines(sft, [0])
    assert len(subset_sft) == 1
    assert_array_equal(sft.streamlines[0], subset_sft.streamlines[0])


def test_cut_invalid_streamlines():
    # Manual creation of invalid streamlines
    invalid_coord = sft.dimensions[0] + 1
    streamlines = [[[0, 0, 0], [1, 1, 1]],
                   [[0, 0, 0], [invalid_coord, invalid_coord, invalid_coord]]]
    sft_w_invalid = StatefulTractogram(streamlines, 'same', Space.VOX)

    new_sft, _ = cut_invalid_streamlines(sft_w_invalid)
    assert len(new_sft) == 2
    assert len(new_sft.streamlines[1]) == 1


def test_assert_sft_compatibility():
    sft_1 = sft
    sft_2 = StatefulTractogram(sft.streamlines[0:1], 'same', Space.VOX)
    assert_true(assert_sft_compatibility([sft_1, sft_2]))

    sft_3 = StatefulTractogram(sft.streamlines[0:1], 'same', Space.RASMM)
    assert_raises(ValueError, assert_sft_compatibility, [sft_1, sft_3])


def test_upsample_tractogram():
    pass
    # resampled_sft = upsample_tractogram(sft, 10)
    # assert_equal(len(resampled_sft), 10)


def test_downsample_tractogram():
    pass
    # resampled_sft = downsample_tractogram(sft, 1)
    # assert_equal(len(resampled_sft), 1)
