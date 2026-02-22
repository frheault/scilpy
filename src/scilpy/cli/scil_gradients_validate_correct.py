#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Detect sign flips and/or axes swaps in the gradients table from a fiber
coherence index [1]. The script can work in two modes:

1) Peaks mode: Takes as input the principal direction(s) at each voxel,
   the b-vectors and the fractional anisotropy map.
   >>> scil_gradients_validate_correct bvec peaks_v1.nii.gz fa.nii.gz bvec_corr

2) DWI mode: Takes as input the DWI, the b-values and the b-vectors.
   A quick DTI fit is performed internally on a central sub-volume.
   >>> scil_gradients_validate_correct dwi.nii.gz bval bvec bvec_corr

Note that in peaks mode, peaks_v1.nii.gz is the file containing the
direction associated to the highest eigenvalue at each voxel.

It is also possible to use a file containing multiple principal directions per
voxel, given that they are sorted by decreasing amplitude. In that case, the
first direction (with the highest amplitude) will be chosen for validation.
Only 4D data is supported, so the directions must be stored in a single
dimension. For example, peaks.nii.gz from scil_fodf_metrics could be used.

------------------------------------------------------------------------------
Reference:
[1] Schilling KG, Yeh FC, Nath V, Hansen C, Williams O, Resnick S, Anderson AW,
    Landman BA. A fiber coherence index for quality control of B-table
    orientation in diffusion MRI scans. Magn Reson Imaging. 2019 May;58:82-89.
    doi: 10.1016/j.mri.2019.01.018.
------------------------------------------------------------------------------
"""

import argparse
import logging

from dipy.core.gradients import gradient_table
from dipy.reconst.dti import TensorModel, fractional_anisotropy
import numpy as np

from scilpy.io.gradients import read_bvals_bvecs
from scilpy.io.stateful_image import StatefulImage
from scilpy.io.stateful_gradient import StatefulGradient
from scilpy.io.utils import (add_overwrite_arg, assert_inputs_exist,
                             assert_outputs_exist, add_verbose_arg,
                             assert_headers_compatible)
from scilpy.io.image import get_data_as_mask
from scilpy.reconst.fiber_coherence import compute_coherence_table_for_transforms
from scilpy.version import version_string


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter,
                                epilog=version_string)

    p.add_argument('out_bvec',
                   help='Path to corrected bvec file (FSL format).')

    dwi_group = p.add_argument_group('DWI Mode (performs internal DTI fit)')
    dwi_group.add_argument('--dwi', help='Path to DWI nifti file.')
    dwi_group.add_argument('--bval', help='Path to bval file.')
    dwi_group.add_argument('--bvec', help='Path to bvec file.')

    peaks_group = p.add_argument_group('Peaks Mode')
    peaks_group.add_argument('--peaks', help='Path to peaks file.')
    peaks_group.add_argument('--fa', help='Path to the FA file.')
    peaks_group.add_argument('--in_bvec', help='Path to bvec file.')

    p.add_argument('--mask',
                   help='Path to an optional mask. If set, FA and Peaks will '
                        'only be used inside the mask.')
    p.add_argument('--fa_threshold', default=0.2, type=float,
                   help='FA threshold. Only voxels with FA higher '
                        'than fa_threshold will be considered. [%(default)s]')
    p.add_argument('--column_wise', action='store_true',
                   help='Specify if input peaks are column-wise (..., 3, N) '
                        'instead of row-wise (..., N, 3).')

    add_verbose_arg(p)
    add_overwrite_arg(p)
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    # Mode detection and validation
    if args.dwi:
        if not args.bval or not args.bvec:
            parser.error('--dwi mode requires --bval and --bvec.')
        if args.peaks or args.fa or args.in_bvec:
            parser.error('--dwi mode is incompatible with --peaks/--fa/--in_bvec.')

        logging.info('DWI mode selected.')
        assert_inputs_exist(parser, [args.dwi, args.bval, args.bvec],
                            optional=args.mask)
        assert_outputs_exist(parser, args, args.out_bvec)

        dwi_simg = StatefulImage.load(args.dwi)
        sgrad = read_bvals_bvecs(args.bval, args.bvec, simg=dwi_simg)
        print(sgrad.bvecs[0:3, :])
        data = dwi_simg.get_fdata().astype(np.float32)
        if len(data.shape) != 4:
            parser.error('Input data must be DWI (4D) in DWI mode.')

        # Quick DTI fit internally in a 1/4 dimensions of the nifti cube
        # around the center
        logging.info('Performing quick DTI fit on central sub-volume...')
        gtab = gradient_table(sgrad.bvals, bvecs=sgrad.bvecs)
        tenmodel = TensorModel(gtab, fit_method='WLS',
                               min_signal=np.min(data[data > 0]))

        mask = np.zeros(data.shape[:3], dtype=bool)
        interval_i = slice(data.shape[0] // 2 - data.shape[0] // 2,
                           data.shape[0] // 2 + data.shape[0] // 2)
        interval_j = slice(data.shape[1] // 2 - data.shape[1] // 2,
                           data.shape[1] // 2 + data.shape[1] // 2)
        interval_k = slice(data.shape[2] // 2 - data.shape[2] // 2,
                           data.shape[2] // 2 + data.shape[2] // 2)
        mask[interval_i, interval_j, interval_k] = 1

        tenfit = tenmodel.fit(data, mask=mask)
        fa = fractional_anisotropy(tenfit.evals)
        peaks = tenfit.evecs.astype(np.float32)[..., 0]
        peaks_simg = dwi_simg  # Reference image for saving

    elif args.peaks:
        if not args.fa or not args.in_bvec:
            parser.error('--peaks mode requires --fa and --in_bvec.')

        logging.info('Peaks mode selected.')
        assert_inputs_exist(parser, [args.peaks, args.fa, args.in_bvec],
                            optional=args.mask)
        assert_outputs_exist(parser, args, args.out_bvec)
        assert_headers_compatible(parser, [args.peaks, args.fa],
                                  optional=args.mask)

        peaks_simg = StatefulImage.load(args.peaks)
        fa_simg = StatefulImage.load(args.fa)

        # Load bvecs relative to the peaks image
        sgrad = read_bvals_bvecs(None, args.in_bvec, simg=peaks_simg)

        fa = fa_simg.get_fdata()
        peaks = peaks_simg.get_fdata()

        if peaks.shape[-1] > 3:
            logging.info('More than one principal direction per voxel was given.')
            peaks = peaks[..., 0:3]
            logging.info('The first peak is assumed to be the biggest.')

        # convert peaks to a volume of shape (H, W, D, N, 3)
        if args.column_wise:
            peaks = np.reshape(peaks, peaks.shape[:3] + (3, -1))
            peaks = np.transpose(peaks, axes=(0, 1, 2, 4, 3))
        else:
            peaks = np.reshape(peaks, peaks.shape[:3] + (-1, 3))

        peaks = np.squeeze(peaks)

    else:
        parser.error('Either --dwi or --peaks mode must be selected.')

    if args.mask:
        mask_data = get_data_as_mask(StatefulImage.load(args.mask),
                                     ref_shape=peaks.shape[:3])
        fa[np.logical_not(mask_data)] = 0
        peaks[np.logical_not(mask_data)] = 0

    peaks[fa < args.fa_threshold] = 0
    coherence, transform = compute_coherence_table_for_transforms(peaks, fa)

    best_t = transform[np.argmax(coherence)]
    if (best_t == np.eye(3)).all():
        logging.info('b-vectors are already correct.')
        correct_bvecs_ras = sgrad.to_ras()
    else:
        logging.info('Applying correction to b-vectors. '
                     'Transform is: \n{0}.'.format(best_t))
        # Transform is applied to world-space vectors
        correct_bvecs_ras = np.dot(sgrad.to_ras(), best_t)

    logging.info('Saving bvecs to file: {0}.'.format(args.out_bvec))

    # Save corrected bvecs using original affine of peaks_simg
    final_sgrad = StatefulGradient(np.zeros(len(correct_bvecs_ras)),
                                   correct_bvecs_ras, peaks_simg, space='ras')
    final_sgrad.save('/tmp/dummy.bval', args.out_bvec)


if __name__ == "__main__":
    main()
