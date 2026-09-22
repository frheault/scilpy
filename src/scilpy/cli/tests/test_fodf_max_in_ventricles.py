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
    ret = script_runner.run(['scil_fodf_max_in_ventricles', '--help'])
    assert ret.success


def test_execution_processing(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_fodf = os.path.join(SCILPY_HOME, 'processing',
                           'fodf.nii.gz')
    in_fa = os.path.join(SCILPY_HOME, 'processing',
                         'fa.nii.gz')
    in_md = os.path.join(SCILPY_HOME, 'processing',
                         'md.nii.gz')
    ret = script_runner.run(['scil_fodf_max_in_ventricles', in_fodf,
                             in_fa, in_md, '--sh_basis', 'tournier07',
                             '--out_mask', 'mask.nii.gz',
                             '--max_value_output', 'max_value.txt'])
    assert ret.success


def test_non_ras_fodf_max_in_ventricles(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_fodf = os.path.join(SCILPY_HOME, 'processing', 'fodf.nii.gz')
    in_fa = os.path.join(SCILPY_HOME, 'processing', 'fa.nii.gz')
    in_md = os.path.join(SCILPY_HOME, 'processing', 'md.nii.gz')

    fodf_data = nib.load(in_fodf).get_fdata(dtype=np.float32)
    fa_data = nib.load(in_fa).get_fdata(dtype=np.float32)
    md_data = nib.load(in_md).get_fdata(dtype=np.float32)

    aff_las = np.diag([-2.0, 2.0, 2.0, 1.0])
    aff_las[0, 3] = 20.0

    nib.save(nib.Nifti1Image(fodf_data[::-1].copy(), aff_las),
             'fodf_las.nii.gz')
    nib.save(nib.Nifti1Image(fa_data[::-1].copy(), aff_las),
             'fa_las.nii.gz')
    nib.save(nib.Nifti1Image(md_data[::-1].copy(), aff_las),
             'md_las.nii.gz')

    # Run on RAS
    ret_ras = script_runner.run(['scil_fodf_max_in_ventricles', in_fodf,
                                 in_fa, in_md,
                                 '--sh_basis', 'tournier07',
                                 '--out_mask', 'mask_ras.nii.gz',
                                 '--max_value_output', 'max_value_ras.txt'])
    assert ret_ras.success

    # Run on LAS
    ret = script_runner.run(['scil_fodf_max_in_ventricles', 'fodf_las.nii.gz',
                             'fa_las.nii.gz', 'md_las.nii.gz',
                             '--sh_basis', 'tournier07',
                             '--out_mask', 'mask_las.nii.gz',
                             '--max_value_output', 'max_value_las.txt'])
    assert ret.success

    # Verify LAS output orientation is preserved
    out_mask_las = nib.load('mask_las.nii.gz')
    assert nib.orientations.aff2axcodes(out_mask_las.affine) == ('L', 'A', 'S')

    # Compare max value with RAS run
    with open('max_value_ras.txt') as f:
        val_ras = float(f.read().strip())
    with open('max_value_las.txt') as f:
        val_las = float(f.read().strip())
    assert np.isclose(val_ras, val_las, atol=1e-4)
