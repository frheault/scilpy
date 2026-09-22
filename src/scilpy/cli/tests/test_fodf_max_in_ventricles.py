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


def test_non_ras_fodf_max_in_ventricles_with_mask(script_runner, monkeypatch):
    """
    --in_mask is reoriented to match the fODF grid before use. A mask that
    is symmetric under a left-right flip would pass even if that
    reorientation were broken, so use an asymmetric (left-half-only) mask
    to actually exercise it.
    """
    # Isolate this test's relative-path output files in the shared scratch
    # dir, same as every other test in this file.
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_fodf = os.path.join(SCILPY_HOME, 'processing', 'fodf.nii.gz')
    in_fa = os.path.join(SCILPY_HOME, 'processing', 'fa.nii.gz')
    in_md = os.path.join(SCILPY_HOME, 'processing', 'md.nii.gz')

    fodf_img = nib.load(in_fodf)
    fodf_data = fodf_img.get_fdata(dtype=np.float32)
    fa_data = nib.load(in_fa).get_fdata(dtype=np.float32)
    md_data = nib.load(in_md).get_fdata(dtype=np.float32)

    # Asymmetric mask: only the first half of the X axis (in the RAS
    # array's own indexing).
    mask_data = np.zeros(fodf_data.shape[:3], dtype=np.uint8)
    mask_data[:fodf_data.shape[0] // 2] = 1
    nib.save(nib.Nifti1Image(mask_data, fodf_img.affine), 'mask_in_ras.nii.gz')

    aff_las = np.diag([-2.0, 2.0, 2.0, 1.0])
    aff_las[0, 3] = 20.0

    nib.save(nib.Nifti1Image(fodf_data[::-1].copy(), aff_las),
             'fodf_mask_las.nii.gz')
    nib.save(nib.Nifti1Image(fa_data[::-1].copy(), aff_las),
             'fa_mask_las.nii.gz')
    nib.save(nib.Nifti1Image(md_data[::-1].copy(), aff_las),
             'md_mask_las.nii.gz')
    # Same physical mask, flipped the same way as the LAS data above.
    nib.save(nib.Nifti1Image(mask_data[::-1].copy(), aff_las),
             'mask_in_las.nii.gz')

    ret_ras = script_runner.run(['scil_fodf_max_in_ventricles', in_fodf,
                                 in_fa, in_md, '--sh_basis', 'tournier07',
                                 '--in_mask', 'mask_in_ras.nii.gz',
                                 '--max_value_output',
                                 'max_value_mask_ras.txt'])
    assert ret_ras.success

    ret_las = script_runner.run(['scil_fodf_max_in_ventricles',
                                 'fodf_mask_las.nii.gz', 'fa_mask_las.nii.gz',
                                 'md_mask_las.nii.gz',
                                 '--sh_basis', 'tournier07',
                                 '--in_mask', 'mask_in_las.nii.gz',
                                 '--max_value_output',
                                 'max_value_mask_las.txt'])
    assert ret_las.success

    with open('max_value_mask_ras.txt') as f:
        val_ras = float(f.read().strip())
    with open('max_value_mask_las.txt') as f:
        val_las = float(f.read().strip())
    assert np.isclose(val_ras, val_las, atol=1e-4)
