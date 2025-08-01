import numpy as np
import nibabel as nib
from numpy.testing import assert_almost_equal, assert_array_equal

from dipy.io.stateful_tractogram import StatefulTractogram, Space, Origin
from scilpy.utils.metrics_tools import (
    compute_lesion_stats,
    get_bundle_metrics_profiles,
    weighted_mean_std,
    get_bundle_metrics_mean_std,
    get_bundle_metrics_mean_std_per_point,
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


def test_compute_lesion_stats():
    """Test the compute_lesion_stats function."""
    map_data = np.array([[1, 1, 0],
                         [1, 1, 0],
                         [0, 0, 0]], dtype=np.int16)
    lesion_atlas = np.array([[1, 1, 0],
                             [2, 0, 0],
                             [0, 0, 3]], dtype=np.int16)

    # Test with single_label=True
    stats = compute_lesion_stats(map_data, lesion_atlas, single_label=True,
                                 voxel_sizes=[1, 1, 1], min_lesion_vol=1)
    assert stats['lesion_total_volume'] == 3.0
    assert stats['lesion_count'] == 2.0
    assert_array_equal(stats['lesion_volume'], [2.0, 1.0])

    # Test with min_lesion_vol
    stats = compute_lesion_stats(map_data, lesion_atlas, single_label=True,
                                 voxel_sizes=[1, 1, 1], min_lesion_vol=1.5)
    assert stats['lesion_total_volume'] == 2.0
    assert stats['lesion_count'] == 1.0
    assert_array_equal(stats['lesion_volume'], [2.0])

    # Test with multiple labels
    map_data_multi = np.array([[1, 1, 0],
                               [2, 2, 0],
                               [0, 0, 0]], dtype=np.int16)
    lesion_atlas_multi = np.array([[1, 1, 5],
                                   [2, 3, 5],
                                   [4, 4, 5]], dtype=np.int16)
    stats = compute_lesion_stats(map_data_multi, lesion_atlas_multi,
                                 single_label=False, voxel_sizes=[1, 1, 1],
                                 min_lesion_vol=1)

    # Label 1 overlaps with lesion 1 (vol 2)
    assert stats['lesion_total_volume']['001'] == 2.0
    assert stats['lesion_count']['001'] == 1.0
    assert_array_equal(stats['lesion_volume']['001'], [2.0])

    # Label 2 overlaps with lesion 2 (vol 1) and 3 (vol 1)
    assert stats['lesion_total_volume']['002'] == 2.0
    assert stats['lesion_count']['002'] == 2.0
    assert_array_equal(stats['lesion_volume']['002'], [1.0, 1.0])


def test_get_bundle_metrics_profiles():
    """Test the get_bundle_metrics_profiles function."""
    sft = _get_small_sft()
    metric_data = np.zeros((3, 3, 3))
    metric_data[0, 0, 0] = 1
    metric_data[1, 1, 1] = 2
    metric_file = nib.Nifti1Image(metric_data, affine=np.eye(4))
    profiles = get_bundle_metrics_profiles(sft, [metric_file])

    # 2 streamlines, 1 metric
    assert len(profiles) == 1
    # The first element of the list of profiles should contain the values for
    # each streamline for the first (and only) metric.
    assert len(profiles[0]) == 2
    assert_array_equal(profiles[0][0], [1, 1, 1])
    assert_array_equal(profiles[0][1], [2, 2, 2, 2])


def test_weighted_mean_std():
    """Test the weighted_mean_std function."""
    # Test with equal weights
    data = np.array([1, 2, 3, 4, 5])
    weights = np.array([1, 1, 1, 1, 1])
    mean, std = weighted_mean_std(weights, data)
    assert_almost_equal(mean, 3.0)
    assert_almost_equal(std, np.std(data))

    # Test with different weights
    data = np.array([1, 2, 3])
    weights = np.array([1, 0, 1])
    mean, std = weighted_mean_std(weights, data)
    assert_almost_equal(mean, 2.0)
    assert_almost_equal(std, 1.0)

    # Test with NaN and inf values in data
    data_with_nan = np.array([1, 2, np.nan, 4])
    weights = np.array([1, 1, 1, 1])
    mean, std = weighted_mean_std(weights, data_with_nan)
    # The function should ignore nan values
    valid_data = np.array([1, 2, 4])
    assert_almost_equal(mean, np.average(valid_data, weights=np.array([1,1,1])))
    assert_almost_equal(std, np.sqrt(np.average((valid_data - np.mean(valid_data))**2, weights=np.array([1,1,1]))))

    # Test with inf values
    data_with_inf = np.array([1, 2, np.inf, 4])
    mean, std = weighted_mean_std(weights, data_with_inf)
    # The function should ignore inf values
    assert_almost_equal(mean, np.average(valid_data, weights=np.array([1,1,1])))
    assert_almost_equal(std, np.sqrt(np.average((valid_data - np.mean(valid_data))**2, weights=np.array([1,1,1]))))


def test_get_bundle_metrics_mean_std():
    """Test the get_bundle_metrics_mean_std function."""
    sft = _get_small_sft()
    metric_data = np.zeros((3, 3, 3))
    metric_data[0, 0, 0] = 1
    metric_data[1, 1, 1] = 2
    metric_file = nib.Nifti1Image(metric_data, affine=np.eye(4))

    # Test without density weighting
    stats = get_bundle_metrics_mean_std(sft.streamlines, [metric_file],
                                        distance_values=None,
                                        correlation_values=None,
                                        density_weighting=False)
    mean, std = list(stats)[0]
    # The voxels traversed are (0,0,0) and (1,1,1).
    # The values are 1 and 2. Mean is 1.5, std is 0.5.
    assert_almost_equal(mean, 1.5)
    assert_almost_equal(std, 0.5)

    # Test with density weighting
    stats = get_bundle_metrics_mean_std(sft.streamlines, [metric_file],
                                        distance_values=None,
                                        correlation_values=None,
                                        density_weighting=True)
    mean, std = list(stats)[0]
    # Voxel (0,0,0) is traversed by 1 streamline (3 points).
    # Voxel (1,1,1) is traversed by 1 streamline (4 points).
    # The weights are based on the number of streamlines, not points. So the
    # weights should be equal.
    # The number of streamlines traversing each voxel is 1.
    # So the weights are equal. Mean is 1.5, std is 0.5.
    assert_almost_equal(mean, 1.5)
    assert_almost_equal(std, 0.5)


def test_get_bundle_metrics_mean_std_per_point():
    """Test the get_bundle_metrics_mean_std_per_point function."""
    sft = _get_small_sft()
    metric_data = np.zeros((3, 3, 3))
    metric_data[0, 0, 0] = 10
    metric_data[1, 1, 1] = 20
    metric_file = nib.Nifti1Image(metric_data, affine=np.eye(4),
                                  header=nib.Nifti1Header())
    metric_file.set_filename('metric.nii.gz')

    labels = np.zeros((3, 3, 3), dtype=int)
    labels[0, 0, 0] = 1
    labels[1, 1, 1] = 2

    stats = get_bundle_metrics_mean_std_per_point(sft.streamlines, 'bundle',
                                                  [metric_file], labels)

    # Check the stats for the first label
    label1_stats = stats['bundle']['metric']['001']
    assert_almost_equal(label1_stats['mean'], 10.0)
    assert_almost_equal(label1_stats['std'], 0.0)

    # Check the stats for the second label
    label2_stats = stats['bundle']['metric']['002']
    assert_almost_equal(label2_stats['mean'], 20.0)
    assert_almost_equal(label2_stats['std'], 0.0)
