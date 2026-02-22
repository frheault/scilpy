#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Detect when data strides are different from [1, 2, 3] and correct them.
The script takes as input a nifti file and outputs a nifti file with the
corrected strides if needed.

Input file can be 3D or 4D. Only the first 3 dimensions are considered
for the stride correction. In the case of DWI data, we recommand to also input
the b-values and b-vectors files to correct the b-vectors accordingly. If the
--validate_bvecs is set, the script first detects sign flips and/or axes swaps
in the b-vectors from a fiber coherence index [1] and corrects the b-vectors.
Then, the b-vectors are permuted and sign flipped to match the new strides.

A typical pipeline could be:
>>> scil_volume_validate_correct_strides t1.nii.gz t1_restride.nii.gz
>>> scil_volume_validate_correct_strides dwi.nii.gz dwi_restride.nii.gz
    --in_bvec dwi.bvec --out_bvec dwi_restride.bvec --validate_bvec
    --in_bval dwi.bval

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

from scilpy.gradients.bvec_bval_tools import check_b0_threshold
from scilpy.io.stateful_image import StatefulImage
from scilpy.io.stateful_gradient import StatefulGradient
from scilpy.io.utils import (add_b0_thresh_arg, add_overwrite_arg,
                             add_skip_b0_check_arg, add_stateful_gradient_args,
                             add_verbose_arg, assert_inputs_exist,
                             assert_outputs_exist,
                             get_stateful_gradient_from_args)
from scilpy.reconst.fiber_coherence import compute_coherence_table_for_transforms
from scilpy.version import version_string


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter,
                                epilog=version_string)

    p.add_argument('in_data',
                   help='Path to input nifti file.')
    p.add_argument('out_data',
                   help='Path to output nifti file with corrected strides.')

    add_stateful_gradient_args(p, mandatory=False)
    p.add_argument('--out_bvec',
                   help='Path to output bvec file (FSL format). Must be '
                        'provided if --in_bvec is used.')
    p.add_argument('--validate_bvec', action='store_true',
                   help='If set, the script first detects sign flips and/or '
                        'axes swaps \nin the b-vectors from a fiber coherence '
                        'index [1] and corrects \nthe b-vectors before '
                        'saving them matching the new strides.')

    add_b0_thresh_arg(p)
    add_skip_b0_check_arg(p, will_overwrite_with_min=True)
    add_verbose_arg(p)
    add_overwrite_arg(p)
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    assert_inputs_exist(parser, [args.in_data],
                        optional=[args.in_bvec, args.in_bval])
    assert_outputs_exist(parser, args, args.out_data, optional=args.out_bvec)

    if args.in_bvec and not args.out_bvec:
        parser.error('--out_bvec must be provided if --in_bvec is used.')
    if args.validate_bvec and (not args.in_bvec or not args.in_bval):
        parser.error('--in_bvec and --in_bval must be provided if '
                     '--validate_bvecs is set.')

    # Load image. StatefulImage.load() automatically standardizes to RAS
    # and keeps the original orientation info.
    simg = StatefulImage.load(args.in_data)

    # Correction: Original script wanted strides [1, 2, 3] which corresponds
    # to RAS order but maybe different from the standardised in-memory RAS.
    # Actually, "correct strides" usually means the identity orientation.
    # Since StatefulImage standardizes to RAS, saving it will revert
    # to original. To FORCE new strides, we must update the state.

    # We want to save the data in RAS orientation (strides 1,2,3)
    # The in-memory data is already RAS. We just need to tell the simg
    # that its "original" state is now RAS.
    simg._original_axcodes = ('R', 'A', 'S')
    simg._original_affine = simg.affine.copy()

    simg.save(args.out_data)

    if args.in_bvec:
        # Load gradients relative to the image
        sgrad = get_stateful_gradient_from_args(args, simg)

        if args.validate_bvec:
            logging.info('Validating b-vectors from fiber coherence index...')
            data = simg.get_fdata().astype(np.float32)
            if len(data.shape) != 4:
                parser.error('Input data must be DWI (4D) when --validate_bvec '
                             'is set.')

            args.b0_threshold = check_b0_threshold(sgrad.bvals.min(),
                                                   b0_thr=args.b0_threshold,
                                                   skip_b0_check=args.skip_b0_check)
            gtab = gradient_table(sgrad.bvals, bvecs=sgrad.bvecs,
                                  b0_threshold=args.b0_threshold)

            tenmodel = TensorModel(gtab, fit_method='WLS',
                                   min_signal=np.min(data[data > 0]))

            mask = np.zeros(data.shape[:3], dtype=bool)
            interval_i = slice(data.shape[0] // 2 - data.shape[0] // 4,
                               data.shape[0] // 2 + data.shape[0] // 4)
            interval_j = slice(data.shape[1] // 2 - data.shape[1] // 4,
                               data.shape[1] // 2 + data.shape[1] // 4)
            interval_k = slice(data.shape[2] // 2 - data.shape[2] // 4,
                               data.shape[2] // 2 + data.shape[2] // 4)
            mask[interval_i, interval_j, interval_k] = 1

            tenfit = tenmodel.fit(data, mask=mask)
            fa = fractional_anisotropy(tenfit.evals)
            evecs = tenfit.evecs.astype(np.float32)[..., 0]
            evecs[fa < 0.2] = 0
            coherence, transform = compute_coherence_table_for_transforms(evecs,
                                                                          fa)

            best_t = transform[np.argmax(coherence)]
            if (best_t == np.eye(3)).all():
                logging.info('The b-vectors are aligned with the original data.')
                final_bvecs_rasmm = sgrad.to_rasmm()
            else:
                logging.warning('Applying correction to b-vectors.')
                logging.info('Transform is: \n{0}.'.format(best_t))
                # Apply correction in World space
                final_bvecs_rasmm = np.dot(sgrad.to_rasmm(), best_t)
        else:
            final_bvecs_rasmm = sgrad.to_rasmm()

        # Save corrected/permuted bvecs
        # Since we changed simg._original_affine to RAS,
        # saving through StatefulGradient will export them in RAS.
        final_sgrad = StatefulGradient(sgrad.bvals, final_bvecs_rasmm,
                                       simg, space='rasmm')
        # We only need to save the bvecs here as requested by --out_bvec
        final_sgrad.save('/tmp/dummy.bval', args.out_bvec)


if __name__ == "__main__":
    main()
