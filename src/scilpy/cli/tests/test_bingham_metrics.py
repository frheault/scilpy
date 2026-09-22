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
    ret = script_runner.run(['scil_bingham_metrics',
                            '--help'])
    assert ret.success


def test_execution_processing(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_bingham = os.path.join(SCILPY_HOME, 'processing',
                              'fodf_bingham.nii.gz')

    ret = script_runner.run(['scil_bingham_metrics',
                             in_bingham, '--nbr_integration_steps', '10',
                             '--processes', '1'])

    assert ret.success


def test_execution_processing_mask(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_bingham = os.path.join(SCILPY_HOME, 'processing',
                              'fodf_bingham.nii.gz')
    in_mask = os.path.join(SCILPY_HOME, 'processing',
                           'seed.nii.gz')

    ret = script_runner.run(['scil_bingham_metrics',
                             in_bingham, '--nbr_integration_steps', '10',
                             '--processes', '1', '--mask', in_mask, '-f'])

    assert ret.success


def test_execution_processing_not_all(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_bingham = os.path.join(SCILPY_HOME, 'processing',
                              'fodf_bingham.nii.gz')

    ret = script_runner.run(['scil_bingham_metrics',
                             in_bingham, '--nbr_integration_steps', '10',
                             '--processes', '1', '--not_all', '--out_fs',
                             'fs.nii.gz', '-f'])

    assert ret.success


def test_non_ras_bingham_metrics(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_b = os.path.join(SCILPY_HOME, 'processing', 'fodf_bingham.nii.gz')
    data = nib.load(in_b).get_fdata(dtype=np.float32)[15:17, 15:17, 15:17]

    # Save RAS and LAS datasets
    aff_ras = np.diag([2.0, 2.0, 2.0, 1.0])
    nib.save(nib.Nifti1Image(data, aff_ras), 'b_ras.nii.gz')

    aff_las = np.diag([-2.0, 2.0, 2.0, 1.0])
    aff_las[0, 3] = 4.0
    data_las = data[::-1, :, :].copy()
    # Invert the x vector components for physical realism.
    # The spherical integration is invariant to this sign change.
    data_las[..., 1] *= -1
    data_las[..., 4] *= -1
    nib.save(nib.Nifti1Image(data_las, aff_las), 'b_las.nii.gz')

    # Run on RAS
    ret_r = script_runner.run(['scil_bingham_metrics', 'b_ras.nii.gz',
                               '--out_fd', 'fd_ras.nii.gz',
                               '--nbr_integration_steps', '10',
                               '--processes', '1', '-f'])
    assert ret_r.success

    # Run on LAS
    ret_l = script_runner.run(['scil_bingham_metrics', 'b_las.nii.gz',
                               '--out_fd', 'fd_las.nii.gz',
                               '--nbr_integration_steps', '10',
                               '--processes', '1', '-f'])
    assert ret_l.success

    # Check orientation was restored and values match
    img_l = nib.load('fd_las.nii.gz')
    assert nib.orientations.aff2axcodes(img_l.affine) == ('L', 'A', 'S')

    fd_r = nib.load('fd_ras.nii.gz').get_fdata()
    fd_l = img_l.get_fdata()
    assert np.allclose(fd_r, fd_l[::-1], atol=1e-5)
