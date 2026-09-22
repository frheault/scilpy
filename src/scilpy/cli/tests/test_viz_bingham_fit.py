#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

import nibabel as nib
import numpy as np
from PIL import Image

from scilpy import SCILPY_HOME
from scilpy.io.fetcher import fetch_data, get_testing_files_dict

fetch_data(get_testing_files_dict(), keys=['processing.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run(['scil_viz_bingham_fit', '--help'])
    assert ret.success


def test_silent_without_output(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))

    in_dummy = os.path.join(SCILPY_HOME, 'processing', 'fodf_bingham.nii.gz')
    out = os.path.join(tmp_dir.name, 'test_bingham.png')
    ret = script_runner.run(['scil_viz_bingham_fit', in_dummy,
                             '--silent', '--output', out])

    assert ret.success


def test_non_ras_viz_bingham_fit(script_runner, monkeypatch):
    monkeypatch.chdir(os.path.expanduser(tmp_dir.name))

    in_dummy = os.path.join(SCILPY_HOME, 'processing', 'fodf_bingham.nii.gz')
    img = nib.load(in_dummy)
    data = img.get_fdata(dtype=np.float32)

    # Save LAS copy
    aff_las = np.diag([-2.0, 2.0, 2.0, 1.0])
    aff_las[0, 3] = 20.0
    data_las = data[::-1].copy()
    nib.save(nib.Nifti1Image(data_las, aff_las), 'bingham_las.nii.gz')

    out = os.path.join(tmp_dir.name, 'test_bingham_las.png')
    ret = script_runner.run(['scil_viz_bingham_fit', 'bingham_las.nii.gz',
                             '--silent', '--output', out])
    assert ret.success
    assert os.path.exists(out)

    # Assert non-trivial pixel variance to confirm glyphs are drawn
    rendered = np.asarray(Image.open(out))
    assert rendered.var() > 10.0
