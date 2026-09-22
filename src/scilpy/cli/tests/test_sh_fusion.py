#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

import nibabel as nib
import numpy as np

from scilpy import SCILPY_HOME
from scilpy.io.fetcher import fetch_data, get_testing_files_dict

# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['processing.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run(['scil_sh_fusion', '--help'])
    assert ret.success


def test_execution_processing(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_sh_1 = os.path.join(SCILPY_HOME, 'processing',
                           'sh_1000.nii.gz')
    in_sh_2 = os.path.join(SCILPY_HOME, 'processing',
                           'sh_3000.nii.gz')
    ret = script_runner.run(['scil_sh_fusion', in_sh_1, in_sh_2, 'sh.nii.gz'])
    assert ret.success


def test_non_ras_sh_fusion(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_sh_1 = os.path.join(SCILPY_HOME, 'processing', 'sh_1000.nii.gz')
    in_sh_2 = os.path.join(SCILPY_HOME, 'processing', 'sh_3000.nii.gz')

    data1 = nib.load(in_sh_1).get_fdata(dtype=np.float32)
    data2 = nib.load(in_sh_2).get_fdata(dtype=np.float32)

    aff_ras = np.diag([2.0, 2.0, 2.0, 1.0])
    nib.save(nib.Nifti1Image(data1, aff_ras), 'sh1_ras.nii.gz')
    nib.save(nib.Nifti1Image(data2, aff_ras), 'sh2_ras.nii.gz')

    aff_las = np.diag([-2.0, 2.0, 2.0, 1.0])
    aff_las[0, 3] = 20.0
    nib.save(nib.Nifti1Image(data1[::-1].copy(), aff_las), 'sh1_las.nii.gz')
    nib.save(nib.Nifti1Image(data2[::-1].copy(), aff_las), 'sh2_las.nii.gz')

    # Run on RAS
    ret_r = script_runner.run(['scil_sh_fusion', 'sh1_ras.nii.gz',
                               'sh2_ras.nii.gz', 'out_ras.nii.gz'])
    assert ret_r.success

    # Run on LAS
    ret_l = script_runner.run(['scil_sh_fusion', 'sh1_las.nii.gz',
                               'sh2_las.nii.gz', 'out_las.nii.gz'])
    assert ret_l.success

    img_l = nib.load('out_las.nii.gz')
    assert nib.orientations.aff2axcodes(img_l.affine) == ('L', 'A', 'S')

    out_r = nib.load('out_ras.nii.gz').get_fdata()
    out_l = img_l.get_fdata()
    assert np.allclose(out_r, out_l[::-1], atol=1e-5)
