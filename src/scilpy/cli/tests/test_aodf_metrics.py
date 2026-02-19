#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy.io.dvc import pull_test_case_package
from scilpy.tests.utils import check_output_existence_and_affine

# If they already exist, this only takes 5 seconds (check md5sum)
test_data_root = pull_test_case_package("aodf")
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run(['scil_aodf_metrics', '--help'])
    assert ret.success


def test_execution(script_runner, monkeypatch):

    # toDo: Add --mask.
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_fodf = os.path.join(
        f"{test_data_root}/fodf_descoteaux07_sub_unified_asym.nii.gz")

    # Using a low resolution sphere for peak extraction reduces process time
    ret = script_runner.run(['scil_aodf_metrics', in_fodf,
                             '--sphere', 'repulsion100', '--processes', '1',
                             '-f'])
    assert ret.success
    check_output_existence_and_affine(['asi_map.nii.gz', 'odd_power_map.nii.gz',
                                       'asym_peaks.nii.gz',
                                       'asym_peak_values.nii.gz',
                                       'asym_peak_indices.nii.gz',
                                       'nufid.nii.gz'],
                                      in_fodf)


def test_assert_not_all(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_fodf = os.path.join(
        f"{test_data_root}/fodf_descoteaux07_sub_unified_asym.nii.gz")

    ret = script_runner.run(['scil_aodf_metrics', in_fodf,
                             '--not_all', '--processes', '1'])
    assert not ret.success


def test_execution_not_all(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_fodf = os.path.join(
        f"{test_data_root}/fodf_descoteaux07_sub_unified_asym.nii.gz")

    ret = script_runner.run(['scil_aodf_metrics', in_fodf,
                             '--not_all', '--asi_map',
                             'asi_map.nii.gz',
                             '--processes', '1',
                             '-f'])
    assert ret.success
    check_output_existence_and_affine('asi_map.nii.gz', in_fodf)


def test_assert_symmetric_input(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_fodf = os.path.join(
        f"{test_data_root}/fodf_descoteaux07_sub.nii.gz")

    # Using a low resolution sphere for peak extraction reduces process time
    ret = script_runner.run(['scil_aodf_metrics', in_fodf,
                             '--sphere', 'repulsion100',
                             '--processes', '1', '-f'])
    assert not ret.success


def test_execution_symmetric_input(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_fodf = os.path.join(
        f"{test_data_root}/fodf_descoteaux07_sub.nii.gz")

    # Using a low resolution sphere for peak extraction reduces process time
    # Using multiprocessing to test this option.
    ret = script_runner.run(['scil_aodf_metrics', in_fodf,
                             '--sphere', 'repulsion100', '--not_all',
                             '--nufid', 'nufid.nii.gz',
                             '--processes', '4', '-f'])
    assert ret.success
    check_output_existence_and_affine('nufid.nii.gz', in_fodf)
