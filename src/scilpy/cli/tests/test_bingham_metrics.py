#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy import SCILPY_HOME
from scilpy.io.fetcher import fetch_data, get_testing_files_dict
from scilpy.tests.utils import check_output_existence_and_affine

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
    check_output_existence_and_affine(['fd.nii.gz', 'fs.nii.gz', 'ff.nii.gz'],
                                      in_bingham)


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
    check_output_existence_and_affine(['fd.nii.gz', 'fs.nii.gz', 'ff.nii.gz'],
                                      in_bingham)


def test_execution_processing_not_all(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_bingham = os.path.join(SCILPY_HOME, 'processing',
                              'fodf_bingham.nii.gz')

    ret = script_runner.run(['scil_bingham_metrics',
                             in_bingham, '--nbr_integration_steps', '10',
                             '--processes', '1', '--not_all', '--out_fs',
                             'fs.nii.gz', '-f'])

    assert ret.success
    check_output_existence_and_affine('fs.nii.gz', in_bingham)
