import os
import tempfile
import numpy as np
import nibabel as nib
import pytest
from numpy.testing import assert_array_equal
from scilpy.image.utils import (
    volume_iterator,
    extract_affine,
    check_slice_indices,
    split_mask_blobs_kmeans,
    compute_nifti_bounding_box,
)


def test_volume_iterator():
    with tempfile.NamedTemporaryFile(suffix='.nii.gz') as f:
        data = np.zeros((3, 3, 3, 10))
        nib.save(nib.Nifti1Image(data, np.eye(4)), f.name)
        img = nib.load(f.name)

        # Test with blocksize = 1
        it = volume_iterator(img, blocksize=1)
        for i, (ids, batch) in enumerate(it):
            assert ids == [i]
            assert batch.shape == (3, 3, 3, 1)

        # Test with blocksize = 5
        it = volume_iterator(img, blocksize=5)
        ids, batch = next(it)
        assert ids == [0, 1, 2, 3, 4]
        assert batch.shape == (3, 3, 3, 5)
        ids, batch = next(it)
        assert ids == [5, 6, 7, 8, 9]
        assert batch.shape == (3, 3, 3, 5)


def test_extract_affine():
    with tempfile.NamedTemporaryFile(suffix='.nii.gz') as f:
        affine = np.eye(4)
        nib.save(nib.Nifti1Image(np.zeros((3, 3, 3)), affine), f.name)
        extracted_affine = extract_affine([f.name])
        assert_array_equal(affine, extracted_affine)


def test_check_slice_indices():
    img = nib.Nifti1Image(np.zeros((10, 10, 10)), np.eye(4))
    check_slice_indices(img, "axial", [0, 5, 9])
    with pytest.raises(ValueError):
        check_slice_indices(img, "axial", [10])


def test_split_mask_blobs_kmeans():
    data = np.zeros((10, 10, 10))
    data[2:4, 2:4, 2:4] = 1
    data[7:9, 7:9, 7:9] = 1
    masks = split_mask_blobs_kmeans(data, 2)
    assert len(masks) == 2
    assert np.sum(masks[0]) > 0
    assert np.sum(masks[1]) > 0


def test_compute_nifti_bounding_box():
    data = np.zeros((10, 10, 10))
    data[2:8, 2:8, 2:8] = 1
    img = nib.Nifti1Image(data, np.eye(4))
    wbbox = compute_nifti_bounding_box(img)
    assert_array_equal(wbbox.minimums, [2, 2, 2])
    assert_array_equal(wbbox.maximums, [8, 8, 8])
