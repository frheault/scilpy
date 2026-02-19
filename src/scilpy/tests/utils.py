# -*- coding: utf-8 -*-

import os
import nibabel as nib
import numpy as np


def nan_array_equal(a, b):
    a = np.asarray(a)
    b = np.asarray(b)

    nan_a = np.argwhere(np.isnan(a))
    nan_b = np.argwhere(np.isnan(a))

    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    return np.array_equal(a, b) and np.array_equal(nan_a, nan_b)


def check_output_existence_and_affine(output_files, input_ref):
    """
    Check if output files exist and if their affine is the same as the
    input reference.

    Parameters
    ----------
    output_files: list of str or str
        The output files to check.
    input_ref: str
        The input reference file to compare the affine with.
    """
    if isinstance(output_files, str):
        output_files = [output_files]

    ref_img = nib.load(input_ref)
    ref_affine = ref_img.affine

    for output_file in output_files:
        assert os.path.isfile(output_file), \
            "Output file {} does not exist.".format(output_file)
        out_img = nib.load(output_file)
        assert np.allclose(out_img.affine, ref_affine), \
            "Affine of {} is not the same as reference {}.".format(
                output_file, input_ref)
