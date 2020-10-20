#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate binary classification measures between gold standard and bundles.
All tractograms must be in the same space (aligned to one reference)
The measures can be applied to voxel-wise or streamline-wise representation.

A gold standard must be provided for the desired representation.
A gold standard would be a segmentation from an expert or a group of experts.
If only the streamline-wise representation is provided without a voxel-wise
gold standard, it will be computed from the provided streamlines.
At least one of the two representations is required.

The gold standard tractogram is the tractogram (whole brain most likely) from
which the segmentation is performed.
The gold standard tracking mask is the tracking mask used by the tractography
algorighm to generate the gold standard tractogram.

The computed binary classification measures are:
sensitivity, specificity, precision, accuracy, dice, kappa, youden for both
the streamline and voxel representation (if provided).
"""

import argparse
import itertools
import json
import logging
import multiprocessing
import os

from dipy.io.streamline import load_tractogram
from dipy.io.utils import is_header_compatible
import nibabel as nib
import numpy as np

from scilpy.io.image import get_data_as_mask
from scilpy.io.utils import (add_json_args,
                             add_overwrite_arg,
                             add_reference_arg,
                             assert_inputs_exist,
                             assert_outputs_exist)
from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.tractanalysis.reproducibility_measures import ICC_rep_anova
from scilpy.tractanalysis.streamlines_metrics import compute_tract_counts_map

from scipy.ndimage import gaussian_filter
from scilpy.image.resample_volume import resample_volume

import rpy2
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import r, NA_Integer, numpy2ri
importr('irr')
rpy2.robjects.numpy2ri.activate()

import warnings
from rpy2.rinterface import RRuntimeWarning
warnings.filterwarnings("ignore", category=RRuntimeWarning)
from rpy2.rinterface_lib.callbacks import logger as rpy2_logger
import logging
rpy2_logger.setLevel(logging.ERROR)

def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('--in_subject_name', nargs='+', required=True,
                   help='Path of the input bundles.')
    p.add_argument('--in_session_name', nargs='+', required=True,
                   help='Path of the input bundles.')
    p.add_argument('--filename', required=True,
                   help='Path of the input bundles.')
    p.add_argument('out_map',
                   help='Path of the output json.')

    p.add_argument('--binary', action='store_true',
                   help='Path of the output json.')

    add_reference_arg(p)
    add_json_args(p)
    add_overwrite_arg(p)

    return p

from time import time

def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    # assert_inputs_exist(parser, args.in_bundles)
    assert_outputs_exist(parser, args, args.out_map)

    # nbr_cpu = validate_nbr_processes(parser, args)

    ref_img = nib.load(args.reference)
    mask = np.zeros(ref_img.shape)

    subjects_list = []
    data = np.zeros(ref_img.shape)
    for folder_1 in args.in_subject_name:
        sessions_list = []
        sessions_list_missing = []
        for folder_2 in args.in_session_name:
            _, ext = os.path.splitext(args.filename)
            filename = os.path.join(folder_1, folder_2, args.filename)
            print(filename)
            if ext in ['.tck', '.trk', '.fib']:
                if os.path.isfile(filename):
                    sft = load_tractogram_with_reference(
                        parser, args, filename)
                    if not is_header_compatible(ref_img, sft):
                        parser.error("Not compatible.")
                    sft.to_vox()
                    sft.to_corner()
                    data = compute_tract_counts_map(sft.streamlines,
                                                    sft.dimensions)
                else:
                    data = 'NA'
            else:
                if os.path.isfile(filename):
                    img = nib.load(filename, mmap=True)
                    if not is_header_compatible(ref_img, img):
                        parser.error("Not compatible.")
                    data = img.get_fdata(dtype=np.float32)
                else:
                    data = 'NA'
            if isinstance(data, np.ndarray):
                data = gaussian_filter(data, sigma=1)
                mask[data > 0] = 1
            if args.binary and isinstance(data, np.ndarray):
                data[data > 0] = 1
            sessions_list.append(data)
        subjects_list.append(sessions_list)

    results = np.zeros(ref_img.shape)
    len_mask = len(np.argwhere(mask))
    for k, ind in enumerate(np.argwhere(mask)):
        np_array = np.zeros((len(args.in_subject_name), len(args.in_session_name)))
        # timer = time()
        print(k, len_mask)
        ind = tuple(ind)
        # tmp_subjects_list = []
        for i in range(len(subjects_list)):
            # tmp_sessions_list = []
            for j in range(len(subjects_list[i])):
                if subjects_list[i][j] != 'NA':
                    np_array[i,j] = subjects_list[i][j][ind]
                else:
                    np_array[i,j] = NA_Integer
            # tmp_subjects_list.append((tmp_sessions_list))

        # np_array = np.array(tmp_subjects_list)
        nr, nc = np_array.shape
        matrix = ro.r.matrix(np_array, nrow=nr, ncol=nc)
        if args.binary:
            mode = 'ordinal'
        else:
            mode = 'interval'
        # print('a', time() - timer)
        timer = time()
        krippendorff_alpha = r['kripp.alpha'](matrix, mode)
        results[ind] = float(np.array((dict(zip(krippendorff_alpha.names,
                                                list(krippendorff_alpha)))['value'])))
        # print('b', time() - timer)

    print(np.average(results[mask > 0]))
    nib.save(nib.Nifti1Image(results, ref_img.affine,
                             header=ref_img.header), args.out_map)


if __name__ == "__main__":
    main()
