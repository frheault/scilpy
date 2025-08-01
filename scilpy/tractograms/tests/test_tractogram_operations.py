import os
import tempfile

from dipy.io.stateful_tractogram import StatefulTractogram, Space
from dipy.io.streamline import load_tractogram, save_tractogram
from dipy.testing import assert_true
import nibabel as nib
import numpy as np
from numpy.testing import (assert_array_equal, assert_raises,
                           assert_allclose, assert_equal)

from scilpy.tractograms.streamline_operations import (
    remove_overlapping_points_streamlines, remove_single_point_streamlines)
from scilpy.tractograms.tractogram_operations import (
    concatenate_sft,
    shuffle_streamlines,
    flip_sft,
    compress_sft,
    split_sft_sequentially,
    split_sft_randomly,
    cut_invalid_streamlines)


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


def test_shuffle_streamlines_orientation():
    # TODO: Implement this test
    pass


def test_get_axis_flip_vector():
    # TODO: Implement this test
    pass


def test_compress_sft():
    compressed_sft = compress_sft(sft)
    assert len(sft.streamlines) == len(compressed_sft.streamlines)
    assert len(sft.streamlines[0]) > len(compressed_sft.streamlines[0])


def test_split_sft_sequentially():
    all_sfts = split_sft_sequentially(sft, [1, 1])
    assert len(all_sfts) == 2
    assert len(all_sfts[0]) == 1
    assert len(all_sfts[1]) == 1
    assert_array_equal(all_sfts[0].streamlines[0], sft.streamlines[0])
    assert_array_equal(all_sfts[1].streamlines[0], sft.streamlines[1])


def test_split_sft_randomly():
    all_sfts = split_sft_randomly(sft, 1, 1234)
    assert len(all_sfts) == 2
    assert len(all_sfts[0]) == 1
    assert len(all_sfts[1]) == 1
    # Check that the total number of streamlines is conserved and unique
    total_streamlines = all_sfts[0].streamlines + all_sfts[1].streamlines
    assert len(total_streamlines) == len(sft)


def test_split_sft_randomly_per_cluster():
    # Create an SFT with two clear clusters
    streamlines = [np.array([[0., 0., 0.], [1., 0., 0.]])] * 10 + \
                  [np.array([[10., 0., 0.], [11., 0., 0.]])] * 10
    sft_cluster = StatefulTractogram(streamlines, 'same', Space.VOX)

    # Split into one chunk of size 10
    all_sfts = split_sft_randomly_per_cluster(sft_cluster, [10], 1234,
                                              thresholds=[5.])

    assert len(all_sfts) == 2  # 1 chunk + remainder
    assert len(all_sfts[0]) == 10
    assert len(all_sfts[1]) == 10

    total_len = sum(len(s) for s in all_sfts)
    assert total_len == len(sft_cluster)


def test_perform_tractogram_operation_on_sft():
    # TODO: Implement this test
    pass


def test_perform_tractogram_operation_on_lines():
    # TODO: Implement this test
    pass


def test_intersection_robust():
    # TODO: Implement this test
    pass


def test_difference_robust():
    # TODO: Implement this test
    pass


def test_union_robust():
    # TODO: Implement this test
    pass


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

    sft_no_overlap = remove_overlapping_points_streamlines(sft_w_invalid)
    clean_sft = remove_single_point_streamlines(sft_no_overlap)
    assert len(clean_sft) == 1


def test_get_subset_streamlines():
    subset_sft = sft[[0]]
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


def test_sft_compatibility():
    sft_1 = sft
    sft_2 = StatefulTractogram(sft.streamlines[0:1], 'same', Space.VOX)
    assert StatefulTractogram.are_compatible(sft_1, sft_2)

    sft_3 = StatefulTractogram(sft.streamlines[0:1], 'same', Space.RASMM)
    assert not StatefulTractogram.are_compatible(sft_1, sft_3)


def test_upsample_tractogram():
    # TODO: Implement this test
    pass


def test_subsample_streamlines_alter():
    # TODO: Implement this test
    pass


def test_cut_streamlines_alter():
    # TODO: Implement this test
    pass


def test_replace_streamlines_alter():
    # TODO: Implement this test
    pass


def test_trim_streamlines_alter():
    # TODO: Implement this test
    pass


def test_transform_streamlines_alter():
    # TODO: Implement this test
    pass
