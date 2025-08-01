import argparse
import numpy as np
import nibabel as nib
import os
import tempfile
from unittest.mock import patch
from numpy.testing import assert_almost_equal, assert_array_equal
from scilpy.io.mti import (
    load_and_verify_mti,
    _parse_acquisition_parameters,
    _prepare_B1_map,
)


@patch('scilpy.io.mti.process_contrast_map')
@patch('scilpy.io.mti.concatenate')
@patch('scilpy.io.mti.load_img')
@patch('scilpy.io.mti._prepare_B1_map')
@patch('scilpy.io.mti._parse_acquisition_parameters')
def test_load_and_verify_mti(mock_parse, mock_prepare, mock_load_img,
                             mock_concatenate, mock_process):
    with tempfile.TemporaryDirectory() as d:
        extended_dir = os.path.join(d, "extended")
        os.makedirs(extended_dir)
        affine = np.eye(4)
        contrast_names = ["c1", "c2"]
        input_maps_lists = [[os.path.join(d, "map1.nii.gz")],
                            [os.path.join(d, "map2.nii.gz")]]
        for f in input_maps_lists[0] + input_maps_lists[1]:
            nib.save(nib.Nifti1Image(np.ones((3, 3, 3)), affine), f)

        parser = argparse.ArgumentParser()
        parser.add_argument('--extended', action='store_true', default=True)
        parser.add_argument('--filtering', action='store_true', default=False)
        parser.add_argument('--out_prefix', default='prefix')
        parser.add_argument('--in_B1_map', default=None)
        parser.add_argument('--in_mtoff_t1', action='store_true', default=True)
        parser.add_argument('--B1_correction_method', default='empiric')
        parser.add_argument('--in_acq_parameters', nargs=4, type=float,
                            default=[10, 20, 1.0, 2.0])
        args = parser.parse_args([])

        mock_parse.return_value = ([1000., 2000.], [0.1, 0.2])
        mock_prepare.return_value = (np.ones((3, 3, 3)), [0.1, 0.2])
        mock_load_img.return_value = (nib.Nifti1Image(np.ones((3, 3, 3)),
                                                      affine), np.float32)
        mock_concatenate.return_value = nib.Nifti1Image(
            np.ones((3, 3, 3, 1)), affine)
        mock_process.return_value = np.ones((3, 3, 3))

        load_and_verify_mti(args, parser, input_maps_lists, extended_dir,
                            affine, contrast_names)
        assert os.path.exists(os.path.join(extended_dir,
                                           "prefix_c1_single_echo.nii.gz"))
        assert os.path.exists(os.path.join(extended_dir,
                                           "prefix_c2_single_echo.nii.gz"))


def test_parse_acquisition_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_acq_parameters', nargs=4, type=float)
    parser.add_argument('--in_jsons', nargs=2)
    args = parser.parse_args(
        ['--in_acq_parameters', '10', '20', '1.0', '2.0'])
    rep_times, flip_angles = _parse_acquisition_parameters(args)
    assert_almost_equal(flip_angles, [0.17453293, 0.34906585])
    assert_almost_equal(rep_times, [1000., 2000.])


def test_prepare_B1_map():
    with tempfile.TemporaryDirectory() as d:
        b1_path = os.path.join(d, "b1.nii.gz")
        extended_dir = os.path.join(d, "extended")
        os.makedirs(extended_dir)
        nib.save(nib.Nifti1Image(np.ones((3, 3, 3)) * 100, np.eye(4)), b1_path)

        parser = argparse.ArgumentParser()
        parser.add_argument('--in_B1_map', default=b1_path)
        parser.add_argument('--in_mtoff_t1', action='store_true', default=True)
        parser.add_argument('--B1_nominal', type=float, default=100.0)
        parser.add_argument('--B1_smooth_dims', type=int, default=5)
        parser.add_argument('--B1_correction_method', default='empiric')
        parser.add_argument('--extended', action='store_true', default=True)
        args = parser.parse_args([])

        flip_angles = np.array([0.1, 0.2])
        b1_map, new_flip_angles = _prepare_B1_map(args, flip_angles,
                                                  extended_dir, np.eye(4))
        assert b1_map is not None
        assert_array_equal(flip_angles, new_flip_angles)
        assert os.path.exists(os.path.join(extended_dir, "B1_map.nii.gz"))
