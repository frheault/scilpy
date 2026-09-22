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
    ret = script_runner.run(['scil_sh_convert', '--help'])
    assert ret.success


def test_execution_processing(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_fodf = os.path.join(SCILPY_HOME, 'processing',
                           'fodf.nii.gz')
    ret = script_runner.run(['scil_sh_convert', in_fodf,
                             'fodf_descoteaux07.nii.gz', 'tournier07',
                             'descoteaux07_legacy', '--processes', '1'])
    assert ret.success


def test_non_ras_sh_convert(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_fodf = os.path.join(SCILPY_HOME, 'processing', 'fodf.nii.gz')
    data = nib.load(in_fodf).get_fdata(dtype=np.float32)

    # Save RAS and LAS datasets
    aff_ras = np.diag([2.0, 2.0, 2.0, 1.0])
    nib.save(nib.Nifti1Image(data, aff_ras), 'sh_ras.nii.gz')

    aff_las = np.diag([-2.0, 2.0, 2.0, 1.0])
    aff_las[0, 3] = 20.0
    data_las = data[::-1].copy()
    nib.save(nib.Nifti1Image(data_las, aff_las), 'sh_las.nii.gz')

    # Run on RAS
    ret_r = script_runner.run(['scil_sh_convert', 'sh_ras.nii.gz',
                               'out_ras.nii.gz', 'tournier07',
                               'descoteaux07_legacy', '--processes', '1'])
    assert ret_r.success

    # Run on LAS
    ret_l = script_runner.run(['scil_sh_convert', 'sh_las.nii.gz',
                               'out_las.nii.gz', 'tournier07',
                               'descoteaux07_legacy', '--processes', '1'])
    assert ret_l.success

    img_l = nib.load('out_las.nii.gz')
    assert nib.orientations.aff2axcodes(img_l.affine) == ('L', 'A', 'S')

    out_r = nib.load('out_ras.nii.gz').get_fdata()
    out_l = img_l.get_fdata()
    assert np.allclose(out_r, out_l[::-1], atol=1e-5)
