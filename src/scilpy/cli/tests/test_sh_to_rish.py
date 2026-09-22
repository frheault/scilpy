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
    ret = script_runner.run(['scil_sh_to_rish', '--help'])
    assert ret.success


def test_execution_processing(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_sh = os.path.join(SCILPY_HOME, 'processing',
                         'sh.nii.gz')
    ret = script_runner.run(['scil_sh_to_rish', in_sh, 'rish.nii.gz'])
    assert ret.success


def test_non_ras_sh_to_rish(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_sh = os.path.join(SCILPY_HOME, 'processing', 'sh.nii.gz')
    data = nib.load(in_sh).get_fdata(dtype=np.float32)

    aff_ras = np.diag([2.0, 2.0, 2.0, 1.0])
    nib.save(nib.Nifti1Image(data, aff_ras), 'sh_ras.nii.gz')

    aff_las = np.diag([-2.0, 2.0, 2.0, 1.0])
    aff_las[0, 3] = 20.0
    nib.save(nib.Nifti1Image(data[::-1].copy(), aff_las), 'sh_las.nii.gz')

    # Run on RAS
    ret_r = script_runner.run(['scil_sh_to_rish', 'sh_ras.nii.gz',
                               'rish_ras_'])
    assert ret_r.success

    # Run on LAS
    ret_l = script_runner.run(['scil_sh_to_rish', 'sh_las.nii.gz',
                               'rish_las_'])
    assert ret_l.success

    # Check output order 0
    img_l = nib.load('rish_las_0.nii.gz')
    assert nib.orientations.aff2axcodes(img_l.affine) == ('L', 'A', 'S')

    out_r = nib.load('rish_ras_0.nii.gz').get_fdata()
    out_l = img_l.get_fdata()
    assert np.allclose(out_r, out_l[::-1], atol=1e-5)
