import os
import tempfile
import numpy as np
import nibabel as nib
import pytest
from scilpy.io.image import (
    load_img,
    assert_same_resolution,
    get_data_as_mask,
)


def test_load_img():
    with tempfile.NamedTemporaryFile(suffix='.nii.gz') as f:
        data = np.zeros((3, 3, 3), dtype=np.uint8)
        nib.save(nib.Nifti1Image(data, np.eye(4)), f.name)
        img, dtype = load_img(f.name)
        assert isinstance(img, nib.Nifti1Image)
        assert dtype == np.uint8

    img, dtype = load_img("1.0")
    assert isinstance(img, float)
    assert dtype == np.float64


def test_assert_same_resolution():
    with tempfile.TemporaryDirectory() as d:
        img1_path = os.path.join(d, "img1.nii.gz")
        img2_path = os.path.join(d, "img2.nii.gz")
        img3_path = os.path.join(d, "img3.nii.gz")
        nib.save(nib.Nifti1Image(np.zeros((3, 3, 3)), np.eye(4)), img1_path)
        nib.save(nib.Nifti1Image(np.zeros((3, 3, 3)), np.eye(4)), img2_path)
        affine = np.eye(4)
        affine[0, 0] = 2
        nib.save(nib.Nifti1Image(np.zeros((3, 3, 3)), affine), img3_path)

        assert_same_resolution([img1_path, img2_path])
        with pytest.raises(Exception):
            assert_same_resolution([img1_path, img3_path])


def test_get_data_as_mask():
    with tempfile.NamedTemporaryFile(suffix='.nii.gz') as f:
        data = np.array([0, 1, 0, 1]).reshape((2, 2, 1))
        img = nib.Nifti1Image(data.astype(np.uint8), np.eye(4))
        nib.save(img, f.name)
        mask_data = get_data_as_mask(nib.load(f.name))
        assert (mask_data == data).all()

    with tempfile.NamedTemporaryFile(suffix='.nii.gz') as f:
        data = np.random.rand(2, 2, 1)
        img = nib.Nifti1Image(data, np.eye(4))
        nib.save(img, f.name)
        with pytest.raises(IOError):
            get_data_as_mask(nib.load(f.name))
