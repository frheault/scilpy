#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import nibabel as nib
import numpy as np
import tempfile


tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run(['scil_volume_modify_voxel_order', '--help'])
    assert ret.success


def test_execution(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))
    in_file = 'input.nii.gz'
    img = nib.Nifti1Image(np.zeros((10, 20, 30)), np.eye(4))
    nib.save(img, in_file)

    # Test with character-based voxel order
    out_file_lps = 'output_lps.nii.gz'
    ret = script_runner.run(['scil_volume_modify_voxel_order', in_file,
                             out_file_lps, '--new_voxel_order=LPS', '-f'])
    assert ret.success
    lps_img = nib.load(out_file_lps)
    assert nib.aff2axcodes(lps_img.affine) == ('L', 'P', 'S')

    # Test with numeric voxel order
    out_file_asr = 'output_asr.nii.gz'
    ret = script_runner.run(['scil_volume_modify_voxel_order', in_file,
                             out_file_asr, '--new_voxel_order=3,1,2', '-f'])
    assert ret.success
    asr_img = nib.load(out_file_asr)
    assert nib.aff2axcodes(asr_img.affine) == ('S', 'R', 'A')

    # Test with negative numeric voxel order
    out_file_lai = 'output_lai.nii.gz'
    ret = script_runner.run(['scil_volume_modify_voxel_order', in_file,
                             out_file_lai, '--new_voxel_order=-1,2,-3',
                             '-f'])
    assert ret.success
    lai_img = nib.load(out_file_lai)
    assert nib.aff2axcodes(lai_img.affine) == ('L', 'A', 'I')

    # Test with invalid input
    ret = script_runner.run(['scil_volume_modify_voxel_order', in_file,
                             'output.nii.gz', '--new_voxel_order=invalid',
                             '-f'])
    assert not ret.success


def test_execution_with_bvecs(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))

    # 1. Setup RAS data
    in_file = 'ras.nii.gz'
    data = np.zeros((10, 10, 10, 2))
    affine = np.eye(4)
    img = nib.Nifti1Image(data, affine)
    nib.save(img, in_file)

    # Vector 1: [1, 0, 0] (Right)
    # Vector 2: [0, 1, 0] (Anterior)
    bvals = np.array([1000, 1000])
    bvecs = np.array([[1, 0, 0], [0, 1, 0]])
    np.savetxt('ras.bval', bvals[None, :], fmt='%d')
    np.savetxt('ras.bvec', bvecs.T, fmt='%.8f')

    # 2. Run reorient to LPI
    # LPI: x=L, y=P, z=I (all 3 axes flipped compared to RAS)
    out_file = 'lpi.nii.gz'
    out_bvec = 'lpi.bvec'
    ret = script_runner.run(['scil_volume_modify_voxel_order', in_file,
                             out_file, '--new_voxel_order=LPI',
                             '--in_bval', 'ras.bval', '--in_bvec', 'ras.bvec',
                             '--out_bvec', out_bvec, '-f'])
    assert ret.success

    # 3. Verify results
    lpi_img = nib.load(out_file)
    assert nib.aff2axcodes(lpi_img.affine) == ('L', 'P', 'I')

    lpi_bvecs = np.loadtxt(out_bvec).T  # (N, 3)
    # Expected for LPI (det < 0):
    # R_fsl = diag([1, -1, -1])  (x NOT flipped in FSL bvec file for det < 0
    # if we follow the MRtrix importing convention correctly)
    # Let's check our actual implementation logic:
    # World [1, 0, 0] (Right) -> v_fsl = R_fsl.T * [1, 0, 0] = [1, 0, 0]
    # World [0, 1, 0] (Anterior) -> v_fsl = R_fsl.T * [0, 1, 0] = [0, -1, 0]
    expected_lpi = np.array([[1, 0, 0], [0, -1, 0]])
    assert np.allclose(lpi_bvecs, expected_lpi)
