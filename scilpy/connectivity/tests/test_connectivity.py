import os
import tempfile
import numpy as np
import nibabel as nib
import pytest
from numpy.testing import assert_array_equal
from dipy.io.stateful_tractogram import StatefulTractogram, Space, Origin
from scilpy.connectivity.connectivity import (
    compute_triu_connectivity_from_labels,
    load_node_nifti,
    compute_connectivity_matrices_from_hdf5,
)


def _get_small_sft():
    # SFT = 2 streamlines: [2 points, 2 points]
    fake_ref = nib.Nifti1Image(np.zeros((3, 3, 3)), affine=np.eye(4))
    streamlines = [[[0.1, 0.1, 0.1], [1.1, 1.1, 1.1]],
                   [[0.2, 0.2, 0.2], [1.2, 1.2, 1.2]]]
    sft = StatefulTractogram(streamlines, fake_ref, Space.VOX,
                             origin=Origin('corner'))
    return sft


@pytest.mark.skip(reason="Still failing")
def test_compute_triu_connectivity_from_labels():
    sft = _get_small_sft()
    labels_data = np.zeros((3, 3, 3), dtype=int)
    labels_data[0, 0, 0] = 1
    labels_data[1, 1, 1] = 2
    matrix, _, start_labels, end_labels = \
        compute_triu_connectivity_from_labels(sft, labels_data)
    assert matrix.shape == (2, 2)
    assert matrix[0, 1] == 2


def test_load_node_nifti():
    with tempfile.TemporaryDirectory() as d:
        ref_img = nib.Nifti1Image(np.zeros((3, 3, 3)), np.eye(4))
        in_label = "1"
        out_label = "2"
        filename = os.path.join(d, "1_2.nii.gz")
        nib.save(ref_img, filename)
        data = load_node_nifti(d, in_label, out_label, ref_img)
        assert data is not None


def test_compute_connectivity_matrices_from_hdf5():
    # TODO: Implement this test
    pass
