#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

import nibabel as nib
import numpy as np

from scilpy import SCILPY_HOME
from scilpy.io.fetcher import fetch_data, get_testing_files_dict
from scilpy.io.tensor import convert_tensor_format

# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['processing.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run(['scil_dti_convert_tensors', '--help'])
    assert ret.success


def test_execution_processing(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))

    # No tensor in the current test data! I'm running the dti_metrics
    # to create one.
    in_dwi = os.path.join(SCILPY_HOME, 'processing', 'dwi_crop_1000.nii.gz')
    in_bval = os.path.join(SCILPY_HOME, 'processing', '1000.bval')
    in_bvec = os.path.join(SCILPY_HOME, 'processing', '1000.bvec')
    script_runner.run(['scil_dti_metrics', in_dwi,
                      in_bval, in_bvec, '--not_all',
                      '--tensor', 'tensors.nii.gz', '--tensor_format', 'fsl'])

    ret = script_runner.run(['scil_dti_convert_tensors', 'tensors.nii.gz',
                            'converted_tensors.nii.gz', 'fsl', 'mrtrix'])

    assert ret.success


def test_non_ras_dti_convert_tensors(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))

    data = np.arange(600, dtype=np.float32).reshape((5, 5, 4, 6))
    aff_las = np.diag([-2.0, 2.0, 2.0, 1.0])
    aff_las[0, 3] = 20.0
    nib.save(nib.Nifti1Image(data, aff_las), 'tensors_las.nii.gz')

    ret = script_runner.run(['scil_dti_convert_tensors', 'tensors_las.nii.gz',
                             'converted_las.nii.gz', 'fsl', 'mrtrix'])
    assert ret.success

    out_img = nib.load('converted_las.nii.gz')
    assert nib.orientations.aff2axcodes(out_img.affine) == ('L', 'A', 'S')
    expected = convert_tensor_format(data, 'fsl', 'mrtrix')
    assert np.allclose(out_img.get_fdata(), expected)
