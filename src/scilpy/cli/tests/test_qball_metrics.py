#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy import SCILPY_HOME
from scilpy.io.fetcher import fetch_data, get_testing_files_dict
from scilpy.tests.utils import check_output_existence_and_affine

fetch_data(get_testing_files_dict(), keys=['processing.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run(['scil_qball_metrics', '--help'])
    assert ret.success


def test_execution_processing(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_dwi = os.path.join(SCILPY_HOME, 'processing',
                          'dwi_crop_1000.nii.gz')
    in_bval = os.path.join(SCILPY_HOME, 'processing',
                           '1000.bval')
    in_bvec = os.path.join(SCILPY_HOME, 'processing',
                           '1000.bvec')
    ret = script_runner.run(['scil_qball_metrics', in_dwi,
                            in_bval, in_bvec])
    assert ret.success
    check_output_existence_and_affine(['gfa.nii.gz', 'peaks.nii.gz',
                                       'peaks_indices.nii.gz', 'sh.nii.gz',
                                       'nufo.nii.gz',
                                       'anisotropic_power.nii.gz'],
                                      in_dwi)


def test_execution_not_all(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_dwi = os.path.join(SCILPY_HOME, 'processing',
                          'dwi_crop_1000.nii.gz')
    in_bval = os.path.join(SCILPY_HOME, 'processing',
                           '1000.bval')
    in_bvec = os.path.join(SCILPY_HOME, 'processing',
                           '1000.bvec')
    ret = script_runner.run(['scil_qball_metrics', in_dwi,
                            in_bval, in_bvec, "--not_all", "--sh", "2.nii.gz"])
    assert ret.success
    check_output_existence_and_affine('2.nii.gz', in_dwi)

    # Test wrong b0. Current minimal b-val is 5.
    ret = script_runner.run(['scil_qball_metrics', in_dwi,
                            in_bval, in_bvec, "--not_all", "--sh", "2.nii.gz",
                            '--b0_threshold', '1', '-f'])
    assert not ret.success
