import os
import pytest
import tempfile

import nibabel as nib
import numpy as np

from scilpy import SCILPY_HOME
from scilpy.io.fetcher import fetch_data, get_testing_files_dict

# If they already exist, this only takes 5 seconds (check md5sum)
fetch_data(get_testing_files_dict(), keys=['tracking.zip'])
tmp_dir = tempfile.TemporaryDirectory()


@pytest.fixture(scope="session")
def las_fodf(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("las_fodf_data")
    in_fodf = os.path.join(SCILPY_HOME, 'tracking', 'fodf.nii.gz')
    img = nib.load(in_fodf)
    cur_ornt = nib.orientations.axcodes2ornt(
        nib.orientations.aff2axcodes(img.affine))
    las_ornt = nib.orientations.axcodes2ornt(('L', 'A', 'S'))
    transform = nib.orientations.ornt_transform(cur_ornt, las_ornt)
    las_img = img.as_reoriented(transform)
    out_path = str(tmp_path / 'fodf_las.nii.gz')
    nib.save(las_img, out_path)
    return out_path


def test_help_option(script_runner, monkeypatch):
    ret = script_runner.run(['scil_fodf_bundleparc', '--help'])
    assert ret.success


@pytest.mark.ml
@pytest.mark.serial
def test_execution(script_runner, monkeypatch, las_fodf):
    ret = script_runner.run(['scil_fodf_bundleparc', las_fodf, '-f',
                             '--bundles', 'FX_left'])
    assert ret.success


@pytest.mark.ml
@pytest.mark.serial
def test_execution_100_labels(script_runner, monkeypatch, las_fodf):
    ret = script_runner.run(['scil_fodf_bundleparc', las_fodf,
                             '--nb_pts', '100', '-f', '--bundles',
                             'IFO_right'])
    assert ret.success


@pytest.mark.ml
@pytest.mark.serial
def test_execution_keep_biggest_blob(script_runner, monkeypatch, las_fodf):
    ret = script_runner.run(['scil_fodf_bundleparc', las_fodf,
                             '--keep_biggest_blob', '-f', '--bundles',
                             'CA'])
    assert ret.success


@pytest.mark.ml
@pytest.mark.serial
def test_execution_invalid_bundle(script_runner, monkeypatch, las_fodf):
    ret = script_runner.run(['scil_fodf_bundleparc', las_fodf,
                             '-f', '--bundles', 'CC'])
    assert not ret.success


@pytest.mark.ml
@pytest.mark.serial
def test_execution_mm(script_runner, monkeypatch, las_fodf):
    ret = script_runner.run(['scil_fodf_bundleparc', las_fodf,
                             '--mm', '10',
                             '--bundles', 'IFO_right', '-f'])
    assert ret.success


@pytest.mark.ml
@pytest.mark.serial
def test_execution_cont(script_runner, monkeypatch, las_fodf):
    ret = script_runner.run(['scil_fodf_bundleparc', las_fodf,
                             '--continuous',
                             '--bundles', 'IFO_right', '-f'])
    assert ret.success


def test_execution_non_las_rejection(script_runner):
    # Tracking fodf is in RAS orientation; bundleparc must reject it
    in_fodf = os.path.join(SCILPY_HOME, 'tracking', 'fodf.nii.gz')
    ret = script_runner.run(['scil_fodf_bundleparc', in_fodf,
                             '--bundles', 'FX_left', '-f'])
    assert not ret.success
    assert "BundleParc expects fODF input in LAS orientation" in ret.stderr


def test_execution_3d_volume_error(tmp_path, script_runner):
    in_3d = str(tmp_path / 'volume_3d.nii.gz')
    nib.save(nib.Nifti1Image(np.zeros((10, 10, 10), dtype=np.float32),
                             np.eye(4)), in_3d)

    ret = script_runner.run(['scil_fodf_bundleparc', in_3d,
                             '--bundles', 'FX_left', '-f'])
    assert not ret.success
    assert "must be 4D" in ret.stderr
