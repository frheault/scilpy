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
    ret = script_runner.run(['scil_fodf_memsmt', '--help'])
    assert ret.success


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
    in_wm_frf = os.path.join(SCILPY_HOME, 'btensor_testdata',
                             'wm_frf.txt')
    in_gm_frf = os.path.join(SCILPY_HOME, 'btensor_testdata',
                             'gm_frf.txt')
    in_csf_frf = os.path.join(SCILPY_HOME, 'btensor_testdata',
                              'csf_frf.txt')

    ret = script_runner.run(['scil_fodf_memsmt', in_wm_frf,
                             in_gm_frf, in_csf_frf, '--in_dwis',
                             in_dwi_lin, in_dwi_plan, '--in_bvals',
                             in_bval_lin, '--in_bvecs', in_bvec_lin,
                             '--in_bdeltas', '1',
                             '--wm_out_fODF', 'wm_fodf.nii.gz',
                             '--gm_out_fODF', 'gm_fodf.nii.gz',
                             '--csf_out_fODF', 'csf_fodf.nii.gz', '--vf',
                             'vf.nii.gz', '--sh_order', '4', '--sh_basis',
                             'tournier07', '--processes', '1', '-f'])
    assert (not ret.success)

    ret = script_runner.run(['scil_fodf_memsmt', in_wm_frf,
                             in_gm_frf, in_csf_frf, '--in_dwis',
                             in_dwi_lin, in_dwi_plan, '--in_bvals',
                             in_bval_lin, in_bval_plan, '--in_bvecs',
                             in_bvec_lin, in_bvec_plan, '--in_bdeltas',
                             '1', '-0.5', '0',
                             '--wm_out_fODF', 'wm_fodf.nii.gz',
                             '--gm_out_fODF', 'gm_fodf.nii.gz',
                             '--csf_out_fODF', 'csf_fodf.nii.gz', '--vf',
                             'vf.nii.gz', '--sh_order', '4', '--sh_basis',
                             'tournier07', '--processes', '1', '-f'])
    assert (not ret.success)


def test_execution_processing(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_dwi_lin = os.path.join(SCILPY_HOME, 'btensor_testdata',
                              'dwi_linear.nii.gz')
    in_bval_lin = os.path.join(SCILPY_HOME, 'btensor_testdata',
                               'linear.bvals')
    in_bvec_lin = os.path.join(SCILPY_HOME, 'btensor_testdata',
                               'linear.bvecs')
    in_dwi_sph = os.path.join(SCILPY_HOME, 'btensor_testdata',
                              'dwi_spherical.nii.gz')
    in_bval_sph = os.path.join(SCILPY_HOME, 'btensor_testdata',
                               'spherical.bvals')
    in_bvec_sph = os.path.join(SCILPY_HOME, 'btensor_testdata',
                               'spherical.bvecs')
    in_wm_frf = os.path.join(SCILPY_HOME, 'btensor_testdata',
                             'wm_frf.txt')
    in_gm_frf = os.path.join(SCILPY_HOME, 'btensor_testdata',
                             'gm_frf.txt')
    in_csf_frf = os.path.join(SCILPY_HOME, 'btensor_testdata',
                              'csf_frf.txt')

    ret = script_runner.run(['scil_fodf_memsmt', in_wm_frf,
                             in_gm_frf, in_csf_frf, '--in_dwis',
                             in_dwi_lin, in_dwi_sph, '--in_bvals',
                             in_bval_lin, in_bval_sph,
                             '--in_bvecs', in_bvec_lin,
                             in_bvec_sph, '--in_bdeltas', '1', '0',
                             '--sh_order', '8', '--processes', '8', '-f'])
    assert ret.success


def test_non_ras_mask_fodf_memsmt(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_dwi_lin = os.path.join(SCILPY_HOME, 'btensor_testdata',
                              'dwi_linear.nii.gz')
    in_bval_lin = os.path.join(SCILPY_HOME, 'btensor_testdata',
                               'linear.bvals')
    in_bvec_lin = os.path.join(SCILPY_HOME, 'btensor_testdata',
                               'linear.bvecs')
    in_dwi_sph = os.path.join(SCILPY_HOME, 'btensor_testdata',
                              'dwi_spherical.nii.gz')
    in_bval_sph = os.path.join(SCILPY_HOME, 'btensor_testdata',
                               'spherical.bvals')
    in_bvec_sph = os.path.join(SCILPY_HOME, 'btensor_testdata',
                               'spherical.bvecs')
    in_wm_frf = os.path.join(SCILPY_HOME, 'btensor_testdata',
                             'wm_frf.txt')
    in_gm_frf = os.path.join(SCILPY_HOME, 'btensor_testdata',
                             'gm_frf.txt')
    in_csf_frf = os.path.join(SCILPY_HOME, 'btensor_testdata',
                              'csf_frf.txt')

    aff_las = np.diag([-1.0, 1.0, 1.0, 1.0])
    aff_las[0, 3] = 4.0

    dwi_lin_las = nib.load(in_dwi_lin).get_fdata(dtype=np.float32)[::-1].copy()
    dwi_sph_las = nib.load(in_dwi_sph).get_fdata(dtype=np.float32)[::-1].copy()
    nib.save(nib.Nifti1Image(dwi_lin_las, aff_las), 'dwi_lin_las.nii.gz')
    nib.save(nib.Nifti1Image(dwi_sph_las, aff_las), 'dwi_sph_las.nii.gz')

    bvecs_lin = np.loadtxt(in_bvec_lin)
    bvecs_lin_las = bvecs_lin.copy()
    bvecs_lin_las[0] *= -1
    np.savetxt('lin_las.bvecs', bvecs_lin_las)

    bvecs_sph = np.loadtxt(in_bvec_sph)
    bvecs_sph_las = bvecs_sph.copy()
    bvecs_sph_las[0] *= -1
    np.savetxt('sph_las.bvecs', bvecs_sph_las)

    mask_las = np.ones((5, 1, 1), dtype=np.uint8)
    nib.save(nib.Nifti1Image(mask_las, aff_las), 'mask_las.nii.gz')

    ret_las = script_runner.run(['scil_fodf_memsmt', in_wm_frf,
                                 in_gm_frf, in_csf_frf, '--in_dwis',
                                 'dwi_lin_las.nii.gz', 'dwi_sph_las.nii.gz',
                                 '--in_bvals', in_bval_lin, in_bval_sph,
                                 '--in_bvecs', 'lin_las.bvecs',
                                 'sph_las.bvecs',
                                 '--in_bdeltas', '1', '0',
                                 '--mask', 'mask_las.nii.gz',
                                 '--wm_out_fODF', 'wm_las.nii.gz',
                                 '--vf', 'vf_las.nii.gz',
                                 '--sh_order', '4', '--processes', '1', '-f'])
    assert ret_las.success

    out_img = nib.load('wm_las.nii.gz')
    assert nib.orientations.aff2axcodes(out_img.affine) == ('L', 'A', 'S')

    # Run on RAS
    aff_ras = np.diag([1.0, 1.0, 1.0, 1.0])
    dwi_lin_ras = nib.load(in_dwi_lin).get_fdata(dtype=np.float32)
    dwi_sph_ras = nib.load(in_dwi_sph).get_fdata(dtype=np.float32)
    nib.save(nib.Nifti1Image(dwi_lin_ras, aff_ras), 'dwi_lin_ras.nii.gz')
    nib.save(nib.Nifti1Image(dwi_sph_ras, aff_ras), 'dwi_sph_ras.nii.gz')
    mask_ras = np.ones((5, 1, 1), dtype=np.uint8)
    nib.save(nib.Nifti1Image(mask_ras, aff_ras), 'mask_ras.nii.gz')

    ret_ras = script_runner.run(['scil_fodf_memsmt', in_wm_frf,
                                 in_gm_frf, in_csf_frf, '--in_dwis',
                                 'dwi_lin_ras.nii.gz', 'dwi_sph_ras.nii.gz',
                                 '--in_bvals', in_bval_lin, in_bval_sph,
                                 '--in_bvecs', in_bvec_lin, in_bvec_sph,
                                 '--in_bdeltas', '1', '0',
                                 '--mask', 'mask_ras.nii.gz',
                                 '--wm_out_fODF', 'wm_ras.nii.gz',
                                 '--vf', 'vf_ras.nii.gz',
                                 '--sh_order', '4', '--processes', '1', '-f'])
    assert ret_ras.success

    # Harmonic power per degree l and volume fractions are invariant
    # under spatial reflection.
    wm_ras = nib.load('wm_ras.nii.gz').get_fdata()
    wm_las = out_img.get_fdata()
    assert np.allclose(wm_ras[..., 0], wm_las[::-1, ..., 0], atol=1e-4)

    power_r_l2 = np.sum(wm_ras[..., 1:6]**2, axis=-1)
    power_l_l2 = np.sum(wm_las[::-1, ..., 1:6]**2, axis=-1)
    assert np.allclose(power_r_l2, power_l_l2, atol=1e-3)

    power_r_l4 = np.sum(wm_ras[..., 6:15]**2, axis=-1)
    power_l_l4 = np.sum(wm_las[::-1, ..., 6:15]**2, axis=-1)
    assert np.allclose(power_r_l4, power_l_l4, atol=1e-3)

    vf_ras = nib.load('vf_ras.nii.gz').get_fdata()
    vf_las = nib.load('vf_las.nii.gz').get_fdata()
    assert np.allclose(vf_ras, vf_las[::-1], atol=1e-3)
