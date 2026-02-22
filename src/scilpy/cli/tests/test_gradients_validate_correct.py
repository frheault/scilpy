#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy import SCILPY_HOME
from scilpy.io.fetcher import fetch_data, get_testing_files_dict

# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['processing.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run(['scil_gradients_validate_correct', '--help'])
    assert ret.success


def test_execution_processing_dti_peaks(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_dwi = os.path.join(SCILPY_HOME, 'processing',
                          'dwi_crop_1000.nii.gz')
    in_bval = os.path.join(SCILPY_HOME, 'processing',
                           '1000.bval')
    in_bvec = os.path.join(SCILPY_HOME, 'processing',
                           '1000.bvec')

    # generate the peaks file and fa map we'll use to test our script
    script_runner.run(['scil_dti_metrics', in_dwi, in_bval, in_bvec,
                       '--not_all', '--fa', 'fa.nii.gz',
                       '--evecs', 'evecs.nii.gz'])
    # test the actual script
    ret = script_runner.run(['scil_gradients_validate_correct', 'bvec_corr',
                             '--in_bvec', in_bvec, '--peaks', 'evecs_v1.nii.gz',
                             '--fa', 'fa.nii.gz', '-v'])
    assert ret.success


def test_execution_processing_fodf_peaks(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_bvec = os.path.join(SCILPY_HOME, 'processing',
                           'dwi.bvec')
    in_peaks = os.path.join(SCILPY_HOME, 'processing',
                            'peaks.nii.gz')
    in_fa = os.path.join(SCILPY_HOME, 'processing',
                         'fa.nii.gz')

    # test the actual script
    ret = script_runner.run(['scil_gradients_validate_correct',
                             'bvec_corr_fodf', '--in_bvec', in_bvec,
                             '--peaks', in_peaks, '--fa', in_fa, '-v'])
    assert ret.success


def test_execution_processing_dwi_mode(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_dwi = os.path.join(SCILPY_HOME, 'processing',
                          'dwi_crop_1000.nii.gz')
    in_bval = os.path.join(SCILPY_HOME, 'processing',
                           '1000.bval')
    in_bvec = os.path.join(SCILPY_HOME, 'processing',
                           '1000.bvec')

    # test the actual script in DWI mode
    ret = script_runner.run(['scil_gradients_validate_correct', 'bvec_corr_dwi',
                             '--dwi', in_dwi, '--bval', in_bval,
                             '--bvec', in_bvec, '-v'])
    assert ret.success
