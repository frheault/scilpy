#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import nibabel as nib
import numpy as np

from scilpy.cli.scil_viz_dti_screenshot import prepare_data_for_actors
from scilpy.io.stateful_image import StatefulImage


def test_help_option(script_runner):
    ret = script_runner.run(['scil_viz_dti_screenshot', '--help'])
    assert ret.success


def test_prepare_data_for_actors_ras_vs_las(tmp_path):
    np.random.seed(42)
    dwi = np.random.rand(8, 8, 8, 7).astype(np.float32) * 100 + 50
    bvals = np.array([0, 1000, 1000, 1000, 1000, 1000, 1000])
    bvecs = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0.707, 0.707, 0],
        [0.707, 0, 0.707],
        [0, 0.707, 0.707]
    ])
    template = np.random.rand(8, 8, 8).astype(np.float32) * 50 + 20

    aff_ras = np.eye(4)
    aff_las = np.diag([-1.0, 1.0, 1.0, 1.0])
    aff_las[0, 3] = 7.0

    dwi_path_ras = str(tmp_path / 'dwi_ras.nii.gz')
    dwi_path_las = str(tmp_path / 'dwi_las.nii.gz')
    bval_path = str(tmp_path / 'bvals')
    bvec_ras_path = str(tmp_path / 'bvecs_ras')
    bvec_las_path = str(tmp_path / 'bvecs_las')
    tpl_path = str(tmp_path / 'template.nii.gz')
    tpl_las_path = str(tmp_path / 'template_las.nii.gz')

    nib.save(nib.Nifti1Image(dwi, aff_ras), dwi_path_ras)
    nib.save(nib.Nifti1Image(dwi[::-1].copy(), aff_las), dwi_path_las)

    np.savetxt(bval_path, bvals)
    np.savetxt(bvec_ras_path, bvecs)
    bvecs_las = bvecs.copy()
    bvecs_las[:, 0] *= -1
    np.savetxt(bvec_las_path, bvecs_las)

    nib.save(nib.Nifti1Image(template, aff_ras), tpl_path)
    nib.save(nib.Nifti1Image(template[::-1].copy(), aff_las), tpl_las_path)

    slices = (4, 4, 4)
    fa_ras, evals_ras, _ = prepare_data_for_actors(
        dwi_path_ras, bval_path, bvec_ras_path, tpl_path, slices)
    fa_las, evals_las, _ = prepare_data_for_actors(
        dwi_path_las, bval_path, bvec_las_path, tpl_path, slices)

    # FA alone is invariant to many axis flips even when orientation
    # handling is broken; also compare evals, which are not. (evecs are
    # not compared directly: with only 6 random gradient directions,
    # near-degenerate eigenvalues make individual eigenvector components
    # numerically ambiguous up to sign/ordering, independent of any
    # orientation bug.)
    assert np.allclose(fa_ras, fa_las, atol=1e-4)
    assert np.allclose(evals_ras, evals_las, atol=1e-4)

    # Verify template orientation variation (LAS template)
    fa_las_tpl, evals_las_tpl, _ = prepare_data_for_actors(
        dwi_path_ras, bval_path, bvec_ras_path, tpl_las_path, slices)
    assert np.allclose(fa_ras, fa_las_tpl, atol=1e-4)
    assert np.allclose(evals_ras, evals_las_tpl, atol=1e-4)

    # Verify both DWI and template in LAS
    fa_both_las, evals_both_las, _ = prepare_data_for_actors(
        dwi_path_las, bval_path, bvec_las_path, tpl_las_path, slices)
    assert np.allclose(fa_ras, fa_both_las, atol=1e-4)
    assert np.allclose(evals_ras, evals_both_las, atol=1e-4)

    # Verify passing already-loaded StatefulImage template
    simg_tpl = StatefulImage.load(tpl_path)
    simg_tpl.to_ras()
    fa_simg, evals_simg, _ = prepare_data_for_actors(
        dwi_path_ras, bval_path, bvec_ras_path, simg_tpl, slices)
    assert np.allclose(fa_ras, fa_simg, atol=1e-4)
    assert np.allclose(evals_ras, evals_simg, atol=1e-4)

    # Verify passing an already-loaded StatefulImage template that has NOT
    # been reoriented yet (still LAS): prepare_data_for_actors() must
    # reorient it itself instead of assuming the caller already did.
    simg_tpl_las = StatefulImage.load(tpl_las_path, to_orientation=None)
    assert simg_tpl_las.axcodes[:3] == ('L', 'A', 'S')
    fa_simg_las, evals_simg_las, _ = prepare_data_for_actors(
        dwi_path_ras, bval_path, bvec_ras_path, simg_tpl_las, slices)
    assert np.allclose(fa_ras, fa_simg_las, atol=1e-4)
    assert np.allclose(evals_ras, evals_simg_las, atol=1e-4)
