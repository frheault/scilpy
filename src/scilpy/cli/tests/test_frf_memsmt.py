#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

import nibabel as nib
import numpy as np

from scilpy import SCILPY_HOME
from scilpy.io.fetcher import fetch_data, get_testing_files_dict

fetch_data(get_testing_files_dict(), keys=['btensor_testdata.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run(['scil_frf_memsmt', '--help'])
    assert ret.success


def test_roi_center_shape_parameter(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_dwi_lin = os.path.join(SCILPY_HOME, 'btensor_testdata',
                              'dwi_linear.nii.gz')
    in_bval_lin = os.path.join(SCILPY_HOME, 'btensor_testdata',
                               'linear.bvals')
    in_bvec_lin = os.path.join(SCILPY_HOME, 'btensor_testdata',
                               'linear.bvecs')
    in_dwi_plan = os.path.join(SCILPY_HOME, 'btensor_testdata',
                               'dwi_planar.nii.gz')
    in_bval_plan = os.path.join(SCILPY_HOME, 'btensor_testdata',
                                'planar.bvals')
    in_bvec_plan = os.path.join(SCILPY_HOME, 'btensor_testdata',
                                'planar.bvecs')
    in_dwi_sph = os.path.join(SCILPY_HOME, 'btensor_testdata',
                              'dwi_spherical.nii.gz')
    in_bval_sph = os.path.join(SCILPY_HOME, 'btensor_testdata',
                               'spherical.bvals')
    in_bvec_sph = os.path.join(SCILPY_HOME, 'btensor_testdata',
                               'spherical.bvecs')

    ret = script_runner.run(['scil_frf_memsmt', 'wm_frf.txt',
                             'gm_frf.txt', 'csf_frf.txt', '--in_dwis',
                             in_dwi_lin, in_dwi_plan, in_dwi_sph, '--in_bvals',
                             in_bval_lin, in_bval_plan, in_bval_sph,
                             '--in_bvecs', in_bvec_lin, in_bvec_plan,
                             in_bvec_sph, '--in_bdeltas', '1', '-0.5', '0',
                             '--roi_center', '1', '--min_nvox', '1', '-f'])

    assert (not ret.success)


def test_roi_radii_shape_parameter(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_dwi_lin = os.path.join(SCILPY_HOME, 'btensor_testdata',
                              'dwi_linear.nii.gz')
    in_bval_lin = os.path.join(SCILPY_HOME, 'btensor_testdata',
                               'linear.bvals')
    in_bvec_lin = os.path.join(SCILPY_HOME, 'btensor_testdata',
                               'linear.bvecs')
    in_dwi_plan = os.path.join(SCILPY_HOME, 'btensor_testdata',
                               'dwi_planar.nii.gz')
    in_bval_plan = os.path.join(SCILPY_HOME, 'btensor_testdata',
                                'planar.bvals')
    in_bvec_plan = os.path.join(SCILPY_HOME, 'btensor_testdata',
                                'planar.bvecs')
    in_dwi_sph = os.path.join(SCILPY_HOME, 'btensor_testdata',
                              'dwi_spherical.nii.gz')
    in_bval_sph = os.path.join(SCILPY_HOME, 'btensor_testdata',
                               'spherical.bvals')
    in_bvec_sph = os.path.join(SCILPY_HOME, 'btensor_testdata',
                               'spherical.bvecs')
    ret = script_runner.run(['scil_frf_memsmt', 'wm_frf.txt',
                             'gm_frf.txt', 'csf_frf.txt', '--in_dwis',
                             in_dwi_lin, in_dwi_plan, in_dwi_sph, '--in_bvals',
                             in_bval_lin, in_bval_plan, in_bval_sph,
                             '--in_bvecs', in_bvec_lin, in_bvec_plan,
                             in_bvec_sph, '--in_bdeltas', '1', '-0.5', '0',
                             '--roi_radii', '37', '--min_nvox', '1', '-f'])
    assert ret.success

    ret = script_runner.run(['scil_frf_memsmt', 'wm_frf.txt',
                             'gm_frf.txt', 'csf_frf.txt', '--in_dwis',
                             in_dwi_lin, in_dwi_plan, in_dwi_sph, '--in_bvals',
                             in_bval_lin, in_bval_plan, in_bval_sph,
                             '--in_bvecs', in_bvec_lin, in_bvec_plan,
                             in_bvec_sph, '--in_bdeltas', '1', '-0.5', '0',
                             '--roi_radii', '37', '37', '37',
                             '--min_nvox', '1', '-f'])
    assert ret.success

    ret = script_runner.run(['scil_frf_memsmt', 'wm_frf.txt',
                             'gm_frf.txt', 'csf_frf.txt', '--in_dwis',
                             in_dwi_lin, in_dwi_plan, in_dwi_sph, '--in_bvals',
                             in_bval_lin, in_bval_plan, in_bval_sph,
                             '--in_bvecs', in_bvec_lin, in_bvec_plan,
                             in_bvec_sph, '--in_bdeltas', '1', '-0.5', '0',
                             '--roi_radii', '37', '37', '37', '37', '37',
                             '--min_nvox', '1', '-f'])

    assert (not ret.success)


def test_inputs_check(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_dwi_lin = os.path.join(SCILPY_HOME, 'btensor_testdata',
                              'dwi_linear.nii.gz')
    in_bval_lin = os.path.join(SCILPY_HOME, 'btensor_testdata',
                               'linear.bvals')
    in_bvec_lin = os.path.join(SCILPY_HOME, 'btensor_testdata',
                               'linear.bvecs')
    in_dwi_plan = os.path.join(SCILPY_HOME, 'btensor_testdata',
                               'dwi_planar.nii.gz')
    in_bval_plan = os.path.join(SCILPY_HOME, 'btensor_testdata',
                                'planar.bvals')
    in_bvec_plan = os.path.join(SCILPY_HOME, 'btensor_testdata',
                                'planar.bvecs')

    ret = script_runner.run(['scil_frf_memsmt', 'wm_frf.txt',
                             'gm_frf.txt', 'csf_frf.txt', '--in_dwis',
                             in_dwi_lin, in_dwi_plan, '--in_bvals',
                             in_bval_lin, '--in_bvecs', in_bvec_lin,
                             '--in_bdeltas', '1', '--min_nvox', '1', '-f'])
    assert (not ret.success)

    ret = script_runner.run(['scil_frf_memsmt', 'wm_frf.txt',
                             'gm_frf.txt', 'csf_frf.txt', '--in_dwis',
                             in_dwi_lin, in_dwi_plan, '--in_bvals',
                             in_bval_lin, in_bval_plan, '--in_bvecs',
                             in_bvec_lin, in_bvec_plan, '--in_bdeltas',
                             '1', '-0.5', '0', '--min_nvox', '1', '-f'])
    assert (not ret.success)


def test_outputs_precision(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_dwi_lin = os.path.join(SCILPY_HOME, 'btensor_testdata',
                              'dwi_linear.nii.gz')
    in_bval_lin = os.path.join(SCILPY_HOME, 'btensor_testdata',
                               'linear.bvals')
    in_bvec_lin = os.path.join(SCILPY_HOME, 'btensor_testdata',
                               'linear.bvecs')
    in_dwi_plan = os.path.join(SCILPY_HOME, 'btensor_testdata',
                               'dwi_planar.nii.gz')
    in_bval_plan = os.path.join(SCILPY_HOME, 'btensor_testdata',
                                'planar.bvals')
    in_bvec_plan = os.path.join(SCILPY_HOME, 'btensor_testdata',
                                'planar.bvecs')
    in_dwi_sph = os.path.join(SCILPY_HOME, 'btensor_testdata',
                              'dwi_spherical.nii.gz')
    in_bval_sph = os.path.join(SCILPY_HOME, 'btensor_testdata',
                               'spherical.bvals')
    in_bvec_sph = os.path.join(SCILPY_HOME, 'btensor_testdata',
                               'spherical.bvecs')
    ret = script_runner.run(['scil_frf_memsmt', 'wm_frf.txt',
                             'gm_frf.txt', 'csf_frf.txt', '--in_dwis',
                             in_dwi_lin, in_dwi_plan, in_dwi_sph, '--in_bvals',
                             in_bval_lin, in_bval_plan, in_bval_sph,
                             '--in_bvecs', in_bvec_lin, in_bvec_plan,
                             in_bvec_sph, '--in_bdeltas', '1', '-0.5', '0',
                             '--min_nvox', '1', '--precision', '4', '-f'])

    assert ret.success

    for frf_file in ['wm_frf.txt', 'gm_frf.txt', 'csf_frf.txt']:
        with open(frf_file, "r") as f:
            for item in f.readline().strip("\n").split(" "):
                assert len(item.split(".")[1]) == 4


def test_execution_processing(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_dwi_lin = os.path.join(SCILPY_HOME, 'btensor_testdata',
                              'dwi_linear.nii.gz')
    in_bval_lin = os.path.join(SCILPY_HOME, 'btensor_testdata',
                               'linear.bvals')
    in_bvec_lin = os.path.join(SCILPY_HOME, 'btensor_testdata',
                               'linear.bvecs')
    in_dwi_plan = os.path.join(SCILPY_HOME, 'btensor_testdata',
                               'dwi_planar.nii.gz')
    in_bval_plan = os.path.join(SCILPY_HOME, 'btensor_testdata',
                                'planar.bvals')
    in_bvec_plan = os.path.join(SCILPY_HOME, 'btensor_testdata',
                                'planar.bvecs')
    in_dwi_sph = os.path.join(SCILPY_HOME, 'btensor_testdata',
                              'dwi_spherical.nii.gz')
    in_bval_sph = os.path.join(SCILPY_HOME, 'btensor_testdata',
                               'spherical.bvals')
    in_bvec_sph = os.path.join(SCILPY_HOME, 'btensor_testdata',
                               'spherical.bvecs')
    ret = script_runner.run(['scil_frf_memsmt', 'wm_frf.txt',
                             'gm_frf.txt', 'csf_frf.txt', '--in_dwis',
                             in_dwi_lin, in_dwi_plan, in_dwi_sph, '--in_bvals',
                             in_bval_lin, in_bval_plan, in_bval_sph,
                             '--in_bvecs', in_bvec_lin, in_bvec_plan,
                             in_bvec_sph, '--in_bdeltas', '1', '-0.5', '0',
                             '--min_nvox', '1', '-f'])
    assert ret.success


def test_non_ras_mask_frf_memsmt(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_dwi_lin = os.path.join(SCILPY_HOME, 'btensor_testdata',
                              'dwi_linear.nii.gz')
    in_bval_lin = os.path.join(SCILPY_HOME, 'btensor_testdata',
                               'linear.bvals')
    in_bvec_lin = os.path.join(SCILPY_HOME, 'btensor_testdata',
                               'linear.bvecs')
    in_dwi_plan = os.path.join(SCILPY_HOME, 'btensor_testdata',
                               'dwi_planar.nii.gz')
    in_bval_plan = os.path.join(SCILPY_HOME, 'btensor_testdata',
                                'planar.bvals')
    in_bvec_plan = os.path.join(SCILPY_HOME, 'btensor_testdata',
                                'planar.bvecs')
    in_dwi_sph = os.path.join(SCILPY_HOME, 'btensor_testdata',
                              'dwi_spherical.nii.gz')
    in_bval_sph = os.path.join(SCILPY_HOME, 'btensor_testdata',
                               'spherical.bvals')
    in_bvec_sph = os.path.join(SCILPY_HOME, 'btensor_testdata',
                               'spherical.bvecs')

    aff_las = np.diag([-1.0, 1.0, 1.0, 1.0])
    aff_las[0, 3] = 4.0

    dwi_lin_las = (
        nib.load(in_dwi_lin).get_fdata(dtype=np.float32)[::-1].copy())
    dwi_plan_las = (
        nib.load(in_dwi_plan).get_fdata(dtype=np.float32)[::-1].copy())
    dwi_sph_las = (
        nib.load(in_dwi_sph).get_fdata(dtype=np.float32)[::-1].copy())
    nib.save(nib.Nifti1Image(dwi_lin_las, aff_las), 'dwi_lin_las.nii.gz')
    nib.save(nib.Nifti1Image(dwi_plan_las, aff_las), 'dwi_plan_las.nii.gz')
    nib.save(nib.Nifti1Image(dwi_sph_las, aff_las), 'dwi_sph_las.nii.gz')

    bvecs_lin = np.loadtxt(in_bvec_lin)
    bvecs_lin_las = bvecs_lin.copy()
    bvecs_lin_las[0] *= -1
    np.savetxt('lin_las.bvecs', bvecs_lin_las)

    bvecs_plan = np.loadtxt(in_bvec_plan)
    bvecs_plan_las = bvecs_plan.copy()
    bvecs_plan_las[0] *= -1
    np.savetxt('plan_las.bvecs', bvecs_plan_las)

    bvecs_sph = np.loadtxt(in_bvec_sph)
    bvecs_sph_las = bvecs_sph.copy()
    bvecs_sph_las[0] *= -1
    np.savetxt('sph_las.bvecs', bvecs_sph_las)

    mask_las = np.ones((5, 1, 1), dtype=np.uint8)
    nib.save(nib.Nifti1Image(mask_las, aff_las), 'mask_las.nii.gz')

    ret_las = script_runner.run(['scil_frf_memsmt', 'wm_frf_las.txt',
                                 'gm_frf_las.txt', 'csf_frf_las.txt',
                                 '--in_dwis', 'dwi_lin_las.nii.gz',
                                 'dwi_plan_las.nii.gz', 'dwi_sph_las.nii.gz',
                                 '--in_bvals', in_bval_lin, in_bval_plan,
                                 in_bval_sph, '--in_bvecs', 'lin_las.bvecs',
                                 'plan_las.bvecs', 'sph_las.bvecs',
                                 '--in_bdeltas', '1', '-0.5', '0',
                                 '--mask', 'mask_las.nii.gz',
                                 '--wm_frf_mask', 'wm_mask_las.nii.gz',
                                 '--min_nvox', '1', '-f'])
    assert ret_las.success

    out_mask = nib.load('wm_mask_las.nii.gz')
    assert nib.orientations.aff2axcodes(out_mask.affine) == ('L', 'A', 'S')

    # Run on RAS
    aff_ras = np.diag([1.0, 1.0, 1.0, 1.0])
    dwi_lin_ras = nib.load(in_dwi_lin).get_fdata(dtype=np.float32)
    dwi_plan_ras = nib.load(in_dwi_plan).get_fdata(dtype=np.float32)
    dwi_sph_ras = nib.load(in_dwi_sph).get_fdata(dtype=np.float32)
    nib.save(nib.Nifti1Image(dwi_lin_ras, aff_ras), 'dwi_lin_ras.nii.gz')
    nib.save(nib.Nifti1Image(dwi_plan_ras, aff_ras), 'dwi_plan_ras.nii.gz')
    nib.save(nib.Nifti1Image(dwi_sph_ras, aff_ras), 'dwi_sph_ras.nii.gz')
    mask_ras = np.ones((5, 1, 1), dtype=np.uint8)
    nib.save(nib.Nifti1Image(mask_ras, aff_ras), 'mask_ras.nii.gz')

    ret_ras = script_runner.run(['scil_frf_memsmt', 'wm_frf_ras.txt',
                                 'gm_frf_ras.txt', 'csf_frf_ras.txt',
                                 '--in_dwis', 'dwi_lin_ras.nii.gz',
                                 'dwi_plan_ras.nii.gz', 'dwi_sph_ras.nii.gz',
                                 '--in_bvals', in_bval_lin, in_bval_plan,
                                 in_bval_sph, '--in_bvecs', in_bvec_lin,
                                 in_bvec_plan, in_bvec_sph,
                                 '--in_bdeltas', '1', '-0.5', '0',
                                 '--mask', 'mask_ras.nii.gz',
                                 '--wm_frf_mask', 'wm_mask_ras.nii.gz',
                                 '--min_nvox', '1', '-f'])
    assert ret_ras.success

    mask_ras_data = nib.load('wm_mask_ras.nii.gz').get_fdata()
    mask_las_data = out_mask.get_fdata()
    assert np.allclose(mask_ras_data, mask_las_data[::-1])

    frf_ras = np.loadtxt('wm_frf_ras.txt')
    frf_las = np.loadtxt('wm_frf_las.txt')
    assert np.allclose(frf_ras, frf_las, atol=1e-5)
